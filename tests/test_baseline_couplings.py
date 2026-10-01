import contextlib
import io
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

import coupling
import diagnose
import eval as eval_checkerboard
import eval_horse
from experiment import save_training
from model import PointSetTransformer
from third_party.equivariant_flow_matching import coupling as efm
from third_party.torchcfm.optimal_transport import OTPlanSampler
import train
import train_horse


class BaselineCouplingTests(unittest.TestCase):
    def setUp(self):
        self.numpy_state = np.random.get_state()
        generator = torch.Generator().manual_seed(31)
        self.source = torch.randn(3, 4, 2, generator=generator, dtype=torch.float64)
        self.target = torch.randn(3, 4, 2, generator=generator, dtype=torch.float64)

    def tearDown(self):
        np.random.set_state(self.numpy_state)

    @staticmethod
    def optimal_point_cost(source, target):
        return min((source - target[list(order)]).square().sum().item()
                   for order in itertools.permutations(range(len(target))))

    def test_minibatch_matches_upstream_and_uses_cloud_cost(self):
        np.random.seed(42)
        expected = OTPlanSampler(method="exact").sample_plan(self.source, self.target)
        np.random.seed(42)
        with patch("third_party.torchcfm.optimal_transport.pot.emd",
                   wraps=coupling.ot.emd) as solve:
            actual = coupling.coupled_points(self.source, self.target, coupling="minibatch_ot")
        for observed, reference in zip(actual, expected):
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        cost = solve.call_args.args[2]
        expected_cost = ((self.source[:, None] - self.target[None, :]) ** 2).sum((2, 3))
        self.assertEqual(cost.shape, (3, 3))
        np.testing.assert_allclose(cost, expected_cost.numpy(), rtol=1e-12, atol=1e-12)
        for selected in actual[0]:
            self.assertTrue(any(torch.equal(selected, original) for original in self.source))
        for selected in actual[1]:
            self.assertTrue(any(torch.equal(selected, original) for original in self.target))

    def test_minibatch_b1_does_not_match_individual_points(self):
        source = torch.tensor([[[0., 0.], [1., 0.], [0., 2.]]])
        target = source[:, [2, 0, 1]].clone()
        paired_source, paired_target = coupling.coupled_points(source, target, coupling="minibatch_ot")
        torch.testing.assert_close(paired_source, source, rtol=0, atol=0)
        torch.testing.assert_close(paired_target, target, rtol=0, atol=0)
        self.assertGreater((paired_source - paired_target).square().sum().item(), 0)

    def test_efm_b1_point_cost_is_optimal_without_geometry_transform(self):
        source, target = self.source[:1], self.target[:1]
        paired_source, paired_target = coupling.coupled_points(
            source, target, coupling="equivariant_ot_permutation")
        torch.testing.assert_close(paired_source, source, rtol=0, atol=0)
        self.assertAlmostEqual((paired_source - paired_target).square().sum().item(),
                               self.optimal_point_cost(source[0], target[0]), places=12)
        self.assertEqual(sorted(map(tuple, paired_target[0].tolist())),
                         sorted(map(tuple, target[0].tolist())))

    def test_efm_cloud_plan_uses_permutation_cost_and_sampling_with_replacement(self):
        point_costs = np.array([[self.optimal_point_cost(x, y) for y in self.target]
                               for x in self.source])
        captured = {}

        def sample_indices(count, *, p, size, replace=True):
            captured.update(count=count, size=size, replace=replace, p=p)
            choices = np.flatnonzero(p > 0)
            # Deliberately repeat a supported cloud pair; all three draws are valid.
            return np.repeat(choices[0], size)

        with patch.object(efm.pot, "emd", wraps=coupling.ot.emd) as solve, \
                patch.object(efm.np.random, "choice", side_effect=sample_indices):
            actual_source, actual_target = coupling.coupled_points(
                self.source, self.target, coupling="equivariant_ot_permutation")
        normalized_cost = solve.call_args.args[2]
        np.testing.assert_allclose(normalized_cost, point_costs / point_costs.max(), rtol=2e-6, atol=1e-7)
        self.assertEqual((captured["count"], captured["size"], captured["replace"]), (9, 3, True))
        plan = captured["p"].reshape(3, 3)
        np.testing.assert_allclose(plan.sum(0), np.full(3, 1 / 3))
        np.testing.assert_allclose(plan.sum(1), np.full(3, 1 / 3))
        best_cloud_cost = min(sum(point_costs[i, j] for i, j in enumerate(order)) / 3
                              for order in itertools.permutations(range(3)))
        self.assertAlmostEqual((plan * point_costs).sum(), best_cloud_cost, places=10)
        i, j = np.divmod(np.flatnonzero(captured["p"] > 0)[0], 3)
        for source, target in zip(actual_source, actual_target):
            torch.testing.assert_close(source, self.source[i], rtol=0, atol=0)
            self.assertEqual(sorted(map(tuple, target.tolist())), sorted(map(tuple, self.target[j].tolist())))
            self.assertAlmostEqual((source - target).square().sum().item(), point_costs[i, j], places=12)

    def test_minibatch_sampling_keeps_upstream_replacement_default(self):
        def sample_indices(count, *, p, size, replace=True):
            self.assertTrue(replace)
            return np.repeat(np.flatnonzero(p > 0)[0], size)

        with patch("third_party.torchcfm.optimal_transport.np.random.choice", side_effect=sample_indices):
            source, target = coupling.coupled_points(self.source, self.target, coupling="minibatch_ot")
        torch.testing.assert_close(source, source[:1].expand_as(source), rtol=0, atol=0)
        torch.testing.assert_close(target, target[:1].expand_as(target), rtol=0, atol=0)

    def test_inputs_and_dtype_are_preserved(self):
        for method in coupling.CLOUD_METHODS:
            for dtype in (torch.float32, torch.float64):
                with self.subTest(method=method, dtype=dtype):
                    source, target = self.source.to(dtype), self.target.to(dtype)
                    original_source, original_target = source.clone(), target.clone()
                    outputs = coupling.coupled_points(source, target, coupling=method)
                    torch.testing.assert_close(source, original_source, rtol=0, atol=0)
                    torch.testing.assert_close(target, original_target, rtol=0, atol=0)
                    for output in outputs:
                        self.assertEqual(output.shape, source.shape)
                        self.assertEqual(output.dtype, dtype)
                        self.assertEqual(output.device, source.device)
                        self.assertTrue(torch.isfinite(output).all())
        zeros = torch.zeros(2, 4, 2)
        for method in coupling.CLOUD_METHODS:
            for output in coupling.coupled_points(zeros, zeros, coupling=method):
                torch.testing.assert_close(output, zeros, rtol=0, atol=0)

    def test_existing_pairing_path_is_unchanged(self):
        for method in ("independent", "target_guided", "target_guided_exact_optimized", "global_ot"):
            with self.subTest(method=method):
                permutation = coupling.coupling_permutation(
                    self.source, self.target, coupling=method, num_regions=2,
                    generator=torch.Generator().manual_seed(17))
                expected_target = (self.target if permutation is None else
                                   self.target.gather(1, permutation.unsqueeze(-1).expand_as(self.target)))
                source, target = coupling.coupled_points(
                    self.source, self.target, coupling=method, num_regions=2,
                    generator=torch.Generator().manual_seed(17))
                torch.testing.assert_close(source, self.source, rtol=0, atol=0)
                torch.testing.assert_close(target, expected_target, rtol=0, atol=0)
        for method in coupling.CLOUD_METHODS:
            with self.assertRaisesRegex(ValueError, "coupled_points"):
                coupling.coupling_permutation(self.source, self.target, coupling=method)

    def test_configs_match_dataset_baselines(self):
        root = Path(__file__).resolve().parents[1]
        settings = (
            ("checkerboard_experiments", "target_guided_exact_optimized.yaml", "{method}.yaml"),
            ("horse_experiments", "horse_target_guided_exact_optimized_k8_n256_seed0.yaml",
             "horse_{method}_n256_seed0.yaml"),
        )
        def training_settings(config):
            return {k: v for k, v in config.items() if k not in ("coupling", "num_regions", "checkpoint")}
        for directory, baseline_name, pattern in settings:
            baseline = yaml.safe_load((root / directory / baseline_name).read_text())
            for method in coupling.CLOUD_METHODS:
                with self.subTest(dataset=directory, method=method):
                    config = yaml.safe_load((root / directory / pattern.format(method=method)).read_text())
                    self.assertEqual(training_settings(config), training_settings(baseline))
                    self.assertEqual(config["coupling"], method)
                    self.assertNotIn("num_regions", config)
                    self.assertNotEqual(config["checkpoint"], baseline["checkpoint"])
                    info = coupling.coupling_info(method)
                    self.assertEqual(info["plan_sampling"], "with_replacement")
                    self.assertFalse(info["target_rotation"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_outputs_and_inputs(self):
        source, target = self.source.cuda().float(), self.target.cuda().float()
        original_source, original_target = source.clone(), target.clone()
        for method in coupling.CLOUD_METHODS:
            with self.subTest(method=method):
                for output in coupling.coupled_points(source, target, coupling=method):
                    self.assertEqual(output.device, source.device)
                    self.assertEqual(output.dtype, source.dtype)
                    self.assertEqual(output.shape, source.shape)
                    self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(source, original_source, rtol=0, atol=0)
        torch.testing.assert_close(target, original_target, rtol=0, atol=0)


class BaselineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.numpy_state = np.random.get_state()
        self.original_directory = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        os.chdir(self.temp.name)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        os.chdir(self.original_directory)
        self.temp.cleanup()
        torch.set_num_threads(self.previous_threads)
        np.random.set_state(self.numpy_state)

    @staticmethod
    def config(method, dataset):
        config = {
            "seed": 0, "device": "cpu", "dtype": "float32", "coupling": method,
            "data": {"batch_size": 2, "n_points": 8},
            "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                      "dim_feedforward": 16, "dropout": 0.0},
            "training": {"num_steps": 9, "learning_rate": 0.001, "weight_decay": 0.01,
                         "log_every": 1},
            "evaluation": {"batch_size": 2, "histogram_bins": 8},
            "checkpoint": f"{dataset}_{method}.pt",
        }
        if dataset == "checkerboard":
            config["data"]["grid_size"] = 4
        return config

    def test_both_datasets_train_save_load_and_evaluate(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for dataset, trainer, evaluator in (
                ("checkerboard", train.main, eval_checkerboard.main),
                ("horse", train_horse.main, eval_horse.main),
            ):
                for method in coupling.CLOUD_METHODS:
                    with self.subTest(dataset=dataset, method=method):
                        config = self.config(method, dataset)
                        input_path = Path(f"{dataset}_{method}.yaml")
                        input_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                        run = trainer(input_path, steps=1)
                        saved_config = yaml.safe_load((run / "config.yaml").read_text())
                        self.assertEqual(saved_config["training"]["num_steps"], 1)
                        self.assertEqual(yaml.safe_load(input_path.read_text())["training"]["num_steps"], 9)
                        training = json.loads((run / "training.json").read_text())
                        self.assertTrue(np.isfinite(training["final_loss"]))
                        self.assertEqual(training["coupling_details"]["method"], method)
                        if dataset == "checkerboard":
                            output = evaluator(run / "config.yaml", num_steps=1, render=False)
                        else:
                            with patch.object(eval_horse, "render_comparison"):
                                output = evaluator(run / "config.yaml", num_steps=1)
                        result = json.loads(output.with_suffix(".json").read_text())
                        self.assertTrue(result["training_config_verified"])
                        self.assertEqual(result["config"]["training"]["num_steps"], 1)
                        self.assertEqual(result["coupling_details"], training["coupling_details"])
                        self.assertEqual(result["euler_steps"], 1)
                        for metric in ("chamfer", "leakage", "histogram_js"):
                            self.assertTrue(np.isfinite(result[metric]), metric)

    def test_diagnostics_keep_raw_rollout_noise_and_training_matching_batch(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for method in coupling.CLOUD_METHODS:
                with self.subTest(method=method):
                    config = self.config(method, "checkerboard")
                    model = PointSetTransformer(**config["model"])
                    run = save_training(model, config, "checkerboard", "fixture.yaml", 0.0, 0.0)
                    captured = {}

                    def resample(source, target, **options):
                        captured["raw"] = source.clone()
                        captured["paired"] = source[:1].expand_as(source).clone()
                        self.assertFalse(torch.equal(captured["raw"], captured["paired"]))
                        return captured["paired"], target

                    def check_fm(model, source, target, time):
                        torch.testing.assert_close(source, captured["paired"], rtol=0, atol=0)
                        self.assertEqual(source.shape[0], 2)
                        return source.new_zeros(2), source.new_ones(2)

                    def check_rollout(model, noise, steps, keep_path=False):
                        torch.testing.assert_close(noise, captured["raw"], rtol=0, atol=0)
                        path = noise[0].unsqueeze(0).repeat(steps + 1, 1, 1) if keep_path else None
                        return noise + 1, noise.new_ones(2), path

                    with patch.object(coupling, "coupled_points", side_effect=resample) as pair, \
                            patch.object(diagnose, "fm_errors", side_effect=check_fm) as fm, \
                            patch.object(diagnose, "rollout", side_effect=check_rollout) as rollout, \
                            patch.object(diagnose, "render"):
                        directory = diagnose.diagnose(run / "config.yaml", "checkerboard", batches=1)
                    self.assertEqual(pair.call_count, 1)
                    self.assertGreater(fm.call_count, 0)
                    self.assertGreater(rollout.call_count, 1)
                    result = json.loads((directory / "diagnostics.json").read_text())
                    self.assertEqual(result["batch_size"], config["data"]["batch_size"])
                    with self.assertRaisesRegex(ValueError, "training data.batch_size"):
                        diagnose.diagnose(run / "config.yaml", "checkerboard", batches=1, batch_size=3)


if __name__ == "__main__":
    unittest.main()
