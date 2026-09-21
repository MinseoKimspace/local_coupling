import contextlib
import io
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import torch
import yaml

import coupling
import diagnose
import eval as eval_checkerboard
import eval_horse
from experiment import load_model
from model import PointSetTransformer
from separation import separation_penalty, separation_settings
import train
import train_horse


METHOD = "target_guided_separation"


class SeparationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        self.source = torch.tensor([[[-3., -1.], [-3., 1.], [3., -1.], [3., 1.]]], dtype=torch.float64)
        self.target = torch.tensor([[[-1., 0.], [-.1, -2.], [1., 0.], [.1, 2.]]], dtype=torch.float64)
        self.options = dict(time=.5, margin_fraction=.25, weight=4., min_target_gap=1e-6)

    def run_variant(self, source=None, target=None, k=2, options=None):
        source = self.source if source is None else source
        target = self.target if target is None else target
        diagnostics = {}
        permutation = coupling.coupling_permutation(
            source, target, coupling=METHOD, num_regions=k,
            separation=self.options if options is None else options,
            separation_diagnostics=diagnostics,
            generator=torch.Generator(device=source.device).manual_seed(42),
        )
        return permutation, diagnostics

    def test_toy_changes_source_assignment_and_achieves_margin(self):
        with patch.object(coupling, "pair_within_regions", wraps=coupling.pair_within_regions) as pairing:
            permutation, diagnostics = self.run_variant()
        source_labels, target_labels = pairing.call_args.args[2:4]
        torch.testing.assert_close(source_labels, torch.tensor([[0, 1, 0, 1]]))
        torch.testing.assert_close(target_labels, torch.tensor([[0, 0, 1, 1]]))
        self.assertEqual(pairing.call_args.kwargs["local"], "random")
        torch.testing.assert_close(permutation.sort(1).values, torch.arange(4).unsqueeze(0))
        pair, = diagnostics["pairs"]
        self.assertAlmostEqual(pair["target_gap"], .09987523388778446, places=10)
        self.assertAlmostEqual(pair["required_source_gap"], -.04993761694389223, places=10)
        self.assertAlmostEqual(pair["threshold"], 0., places=10)
        self.assertAlmostEqual(pair["baseline"]["source_gap"], -1.6978789760923358, places=10)
        self.assertAlmostEqual(pair["corrected"]["source_gap"], 1.6978789760923358, places=10)
        self.assertFalse(pair["baseline"]["margin_satisfied"])
        self.assertTrue(pair["corrected"]["margin_satisfied"])
        self.assertGreater(pair["baseline"]["margin_time"], self.options["time"])
        self.assertEqual(pair["corrected"]["margin_time"], 0.)
        self.assertEqual(diagnostics["summary"]["eligible_pairs"], 1)
        self.assertEqual(diagnostics["summary"]["corrected_solve_clouds"], 1)
        self.assertAlmostEqual(diagnostics["summary"]["changed_source_fraction"], .5)
        json.dumps(diagnostics, allow_nan=False)

    def test_unary_cost_matches_explicit_squared_hinges(self):
        anchors = self.target[:, [1, 3]]
        labels = torch.tensor([[0, 0, 1, 1]])
        penalty, _ = separation_penalty(
            self.source, self.target, anchors, labels, labels, self.options)
        normal = torch.tensor([.2, 4.], dtype=torch.float64)
        normal /= normal.norm()
        projection = self.source[0] @ normal
        target_gap = (self.target[0, 2:] @ normal).min() - (self.target[0, :2] @ normal).max()
        required = (.25 - .5) * target_gap / (1 - .5)
        expected = torch.stack(((projection + required / 2).clamp_min(0).square(),
                                (required / 2 - projection).clamp_min(0).square()), -1)
        torch.testing.assert_close(penalty[0], expected, rtol=0, atol=1e-12)

    def test_zero_weight_is_identical_to_tg_without_rng_changes(self):
        first = torch.Generator().manual_seed(42)
        second = torch.Generator().manual_seed(42)
        global_state = torch.get_rng_state().clone()
        expected = coupling.coupling_permutation(
            self.source, self.target, coupling="target_guided", num_regions=2, generator=first)
        diagnostics = {}
        actual = coupling.coupling_permutation(
            self.source, self.target, coupling=METHOD, num_regions=2, generator=second,
            separation=dict(self.options, weight=0.), separation_diagnostics=diagnostics)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(first.get_state(), second.get_state(), rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), global_state, rtol=0, atol=0)
        self.assertEqual(diagnostics["summary"]["corrected_solve_clouds"], 0)
        self.assertEqual(diagnostics["summary"]["changed_source_fraction"], 0.)

    def test_target_partition_points_and_uneven_3d_capacities_unchanged(self):
        source, target = torch.randn(2, 17, 3), torch.rand(2, 17, 3)
        original_source, original_target = source.clone(), target.clone()
        with patch.object(coupling, "pair_within_regions", wraps=coupling.pair_within_regions) as pairing:
            coupling.coupling_permutation(source, target, coupling="target_guided", num_regions=4)
            expected_target_labels = pairing.call_args.args[3].clone()
            permutation, diagnostics = self.run_variant(source, target, k=4)
            source_labels, target_labels = pairing.call_args.args[2:4]
        torch.testing.assert_close(target_labels, expected_target_labels, rtol=0, atol=0)
        for labels in (*source_labels, *target_labels):
            torch.testing.assert_close(torch.bincount(labels, minlength=4), torch.tensor([5, 4, 4, 4]))
        torch.testing.assert_close(permutation.sort(1).values, torch.arange(17).expand(2, -1))
        torch.testing.assert_close(source, original_source, rtol=0, atol=0)
        torch.testing.assert_close(target, original_target, rtol=0, atol=0)
        json.dumps(diagnostics, allow_nan=False)

    def test_no_pairs_k1_duplicates_and_gap_filter_return_baseline(self):
        for target, k, options in (
            (self.target, 1, self.options),
            (torch.zeros_like(self.target), 2, self.options),
            (self.target, 2, dict(self.options, min_target_gap=100.)),
        ):
            with self.subTest(k=k, options=options):
                expected = coupling.coupling_permutation(
                    self.source, target, coupling="target_guided", num_regions=k,
                    generator=torch.Generator().manual_seed(42))
                actual, diagnostics = self.run_variant(target=target, k=k, options=options)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(diagnostics["pairs"], [])
                self.assertEqual(diagnostics["summary"]["eligible_pairs"], 0)
                self.assertEqual(diagnostics["summary"]["corrected_solve_clouds"], 0)
                self.assertEqual(diagnostics["summary"]["clouds_without_pairs"], 1)
                json.dumps(diagnostics, allow_nan=False)

    def test_gap_bound_is_attained_over_all_local_permutations(self):
        _, diagnostics = self.run_variant()
        pair, = diagnostics["pairs"]
        normal = torch.tensor(pair["normal"], dtype=self.source.dtype)
        for name, labels in (("baseline", [0, 0, 1, 1]), ("corrected", [0, 1, 0, 1])):
            source_groups = [self.source[0, torch.tensor(labels) == region] for region in (0, 1)]
            target_groups = [self.target[0, :2], self.target[0, 2:]]
            record = pair[name]
            for time in (0., .2, .5, 1.):
                bound = (1 - time) * record["source_gap"] + time * pair["target_gap"]
                actual_gaps = []
                for left, right in itertools.product(itertools.permutations(range(2)), repeat=2):
                    paths = [(1 - time) * x + time * y[list(order)]
                             for x, y, order in zip(source_groups, target_groups, (left, right))]
                    gap = ((paths[1] @ normal).min() - (paths[0] @ normal).max()).item()
                    actual_gaps.append(gap)
                    self.assertGreaterEqual(gap + 1e-12, bound)
                self.assertAlmostEqual(min(actual_gaps), bound, places=10)
            violation_bound = (pair["required_source_gap"]
                               - record["left_violation"] - record["right_violation"])
            self.assertGreaterEqual(record["source_gap"] + 1e-12, violation_bound)
            gap_at_time = ((1 - self.options["time"]) * record["source_gap"]
                           + self.options["time"] * pair["target_gap"])
            self.assertAlmostEqual(record["gap_at_time"], gap_at_time, places=10)
            self.assertAlmostEqual(record["margin_deficit"], max(pair["margin"] - gap_at_time, 0.), places=10)
            source_gap, target_gap = record["source_gap"], pair["target_gap"]
            separation_time = -source_gap / (target_gap - source_gap) if source_gap < 0 else 0.
            margin_time = ((pair["margin"] - source_gap) / (target_gap - source_gap)
                           if source_gap < pair["margin"] else 0.)
            self.assertAlmostEqual(record["separation_time"], separation_time, places=10)
            self.assertAlmostEqual(record["margin_time"], margin_time, places=10)

    def test_impossible_margin_is_reported_not_claimed_guaranteed(self):
        source = torch.zeros(1, 4, 1, dtype=torch.float64)
        target = torch.tensor([[[-1.], [-.8], [.8], [1.]]], dtype=torch.float64)
        _, diagnostics = self.run_variant(source, target,
            options=dict(self.options, time=.1, margin_fraction=.5, weight=1e6))
        pair, = diagnostics["pairs"]
        record = pair["corrected"]
        self.assertEqual(record["source_gap"], 0.)
        self.assertFalse(record["margin_satisfied"])
        self.assertGreater(record["margin_deficit"], 0.)
        self.assertAlmostEqual(record["margin_time"], .5)
        self.assertEqual(diagnostics["summary"]["corrected"]["margin_satisfied_fraction"], 0.)
        json.dumps(diagnostics, allow_nan=False)

    def test_setting_validation_and_metadata(self):
        for key, values in {
            "time": (-.1, 1., float("nan"), float("inf")),
            "margin_fraction": (0., -.1, 1.01, float("nan")),
            "weight": (-1., float("inf"), float("nan")),
            "min_target_gap": (-1., float("inf"), float("nan")),
        }.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    separation_settings({key: value})
        with self.assertRaises(ValueError):
            separation_settings({"unknown_option": 1})
        settings = separation_settings(self.options)
        self.assertEqual(settings, self.options)
        info = coupling.coupling_info(METHOD)
        self.assertEqual(info["target_partition"], "balanced")
        self.assertEqual(info["local_pairing"], "random")
        self.assertEqual(info["exact_solver"], "POT/network_simplex")
        self.assertNotEqual(info["cost"], "squared_euclidean")

    def test_train_eval_and_diagnose_preserve_separation_metadata(self):
        root, previous = Path(__file__).resolve().parents[1], Path.cwd()
        template = (root / "checkerboard_experiments" / "target_guided.yaml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            try:
                os.chdir(temp)
                for dataset in ("checkerboard", "horse"):
                    config = yaml.safe_load(template)
                    config.update(device="cpu", coupling=METHOD, num_regions=2,
                                  separation={"time": .5, "weight": 4.}, checkpoint="separation.pt")
                    config["data"].update(batch_size=2, n_points=16)
                    config["evaluation"].update(batch_size=2, histogram_bins=8)
                    config["model"].update(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
                    config["training"]["num_steps"] = 1
                    if dataset == "horse":
                        config["data"].pop("grid_size")
                    Path("input.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
                    trainer = train_horse if dataset == "horse" else train
                    evaluator = eval_horse if dataset == "horse" else eval_checkerboard
                    model_class = train_horse.HorsePointSetTransformer if dataset == "horse" else PointSetTransformer
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        run = trainer.main("input.yaml")
                        artifact = evaluator.main(run / "config.yaml", 2)
                    training_record = json.loads((run / "training.json").read_text(encoding="utf-8"))
                    saved_config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
                    evaluation_record = json.loads(artifact.with_suffix(".json").read_text(encoding="utf-8"))
                    expected_settings = separation_settings(config["separation"])
                    self.assertEqual(saved_config["separation"], expected_settings)
                    self.assertEqual(training_record["config"], saved_config)
                    self.assertEqual(evaluation_record["config"], saved_config)
                    self.assertTrue(evaluation_record["training_config_verified"])
                    self.assertEqual(evaluation_record["coupling_details"], coupling.coupling_info(METHOD))
                    history = training_record["coupling_diagnostics"]["records"]
                    self.assertEqual(len(history), 1)
                    self.assertEqual(history[0]["step"], 1)
                    self.assertIn("margin_deficit_mean", history[0]["corrected"])
                    self.assertIn("NOT learned ODE", training_record["coupling_diagnostics"]["scope"])
                    if dataset == "checkerboard":
                        with patch.object(diagnose, "NFES", (1, 2)), \
                                patch.object(diagnose, "TIMES", (0., .5, 1.)), \
                                patch.object(diagnose, "BINS", ((0., .5), (.5, 1.))), \
                                patch.object(diagnose, "render") as render:
                            directory = diagnose.diagnose(run / "config.yaml", dataset,
                                batches=1, batch_size=2, reference_nfe=2, output="analysis")
                        render.assert_called_once()
                        payload = json.loads((directory / "diagnostics.json").read_text(encoding="utf-8"))
                        self.assertEqual(payload["config"], saved_config)
                        self.assertEqual(len(payload["separation_diagnostics"]), 1)
                        diagnostic = payload["separation_diagnostics"][0]
                        self.assertEqual(diagnostic["settings"], expected_settings)
                        self.assertEqual(diagnostic["summary"]["clouds"], 2)
                        self.assertIn("pairs", diagnostic)
                        self.assertEqual([row["nfe"] for row in payload["endpoint_errors"]], [1, 2])
                    saved_config["separation"]["time"] = .75
                    mismatch = run / "mismatch.yaml"
                    mismatch.write_text(yaml.safe_dump(saved_config), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "separation"):
                        load_model(mismatch, model_class, dataset)
            finally:
                os.chdir(previous)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_preserves_bijection_and_serializable_diagnostics(self):
        source, target = self.source.cuda(), self.target.cuda()
        permutation, diagnostics = self.run_variant(source, target)
        self.assertEqual(permutation.device, source.device)
        torch.testing.assert_close(permutation.sort(1).values.cpu(), torch.arange(4).unsqueeze(0))
        self.assertTrue(diagnostics["pairs"][0]["corrected"]["margin_satisfied"])
        json.dumps(diagnostics, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
