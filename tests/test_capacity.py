import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from adaptive_capacity import (CAPACITY_MODES, adaptive_capacities, capacity_profile,
                               local_spectral_scores, ranked_capacities, _directions)
import coupling
import diagnose
import eval as eval_checkerboard
import eval_horse
import train
import train_horse
from experiment import load_model
from summarize_results import signature


ROOT = Path(__file__).resolve().parents[1]
PROFILE = [16, 24, 24, 32, 32, 40, 40, 48]


class CapacityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)

    def test_profile_validation_and_remainders(self):
        for n, k in ((1, 1), (17, 4), (8, 8), (256, 8)):
            values = capacity_profile(n, k)
            self.assertEqual(len(values), k)
            self.assertEqual(sum(values), n)
            self.assertGreaterEqual(min(values), 1)
        self.assertEqual(capacity_profile(256, 8, PROFILE), PROFILE)
        for profile in ([8, 8], [0, 16], [8.0, 9], [-1, 18], [True, 16]):
            with self.assertRaises(ValueError):
                capacity_profile(17, 2, profile)

    def test_identical_multiset_rank_and_entropy(self):
        scores = torch.tensor([[0.2, 0.9, 0.1, 0.8, 0.7, 0.3, 0.4, 0.5]])
        results = {}
        for mode in ("large", "small", "random"):
            result = ranked_capacities(scores, PROFILE, mode, generator=torch.Generator().manual_seed(7))
            results[mode] = result
            self.assertEqual(result.sort().values.tolist(), [PROFILE])
        order = scores.argsort()
        self.assertEqual(results["large"].gather(1, order).tolist(), [PROFILE])
        self.assertEqual(results["small"].gather(1, order).tolist(), [PROFILE[::-1]])
        shuffled_scores = scores.flip(1)
        torch.testing.assert_close(results["random"], ranked_capacities(
            shuffled_scores, PROFILE, "random", generator=torch.Generator().manual_seed(7)))
        for counts in results.values():
            rho = counts.double() / 256
            entropy = (rho * counts.double().log()).sum()
            kl = (rho * (rho * 8).log()).sum()
            torch.testing.assert_close(entropy - math.log(256 / 8), kl)

    def test_score_matches_explicit_weighted_cross_pairs(self):
        target = torch.randn(1, 20, 2, dtype=torch.float64) * 0.2
        anchors = target[:, [0, 3]]
        radius, low, high, directions = 0.35, (0.5,), (2.0,), 4
        actual = local_spectral_scores(target, anchors, window_radius=radius,
            low_frequencies=low, high_frequencies=high, num_directions=directions,
            min_effective_points=2)
        expected = []
        vectors = _directions(2, directions, target)
        for anchor in anchors[0]:
            w = torch.softmax(-(target[0] - anchor).square().sum(-1) / (2 * radius**2), 0)
            pairs = w[:, None] * w[None, :]
            pairs.fill_diagonal_(0)
            difference = target[0, :, None] - target[0, None, :]
            energies = []
            for frequency in low + high:
                phase = 2 * math.pi * frequency * (difference @ vectors.T)
                energies.append((pairs[..., None] * phase.cos()).sum((0, 1)).mean() / pairs.sum())
            l, h = (v.clamp_min(0) for v in energies)
            expected.append(h / (l + h))
        torch.testing.assert_close(actual, torch.stack(expected).unsqueeze(0), rtol=1e-10, atol=1e-10)

    def test_score_order_translation_invariance_3d_and_sparse_fallback(self):
        for dim in (2, 3):
            target = torch.randn(2, 64, dim, dtype=torch.float64) * 0.3
            anchors = target[:, [0, 10, 20, 30]]
            score = local_spectral_scores(target, anchors)
            perm = torch.randperm(64)
            torch.testing.assert_close(score, local_spectral_scores(target[:, perm], anchors))
            torch.testing.assert_close(score, local_spectral_scores(target + 7, anchors + 7))
            torch.testing.assert_close(score[:, [2, 0, 3, 1]], local_spectral_scores(target, anchors[:, [2, 0, 3, 1]]))
            self.assertTrue(((score >= 0) & (score <= 1)).all())
            torch.testing.assert_close(local_spectral_scores(target, anchors, min_effective_points=1000),
                                       torch.zeros_like(score))
        singleton = torch.zeros(1, 1, 2)
        torch.testing.assert_close(local_spectral_scores(singleton, singleton), torch.zeros(1, 1))
        with self.assertRaises(ValueError):
            local_spectral_scores(target, anchors, low_frequencies=[4], high_frequencies=[2])

    def test_finite_sampling_diagonal_not_mistaken_for_signal(self):
        # Uniform ring with a broad window has less high-band energy than a tiny cluster.
        angle = torch.arange(256, dtype=torch.float64) * (2 * math.pi / 256)
        ring = torch.stack((angle.cos(), angle.sin()), -1).unsqueeze(0)
        anchor = torch.zeros(1, 1, 2, dtype=torch.float64)
        broad = local_spectral_scores(ring, anchor, window_radius=10)
        narrow = local_spectral_scores(ring * 0.01, anchor, window_radius=10)
        self.assertGreater(narrow.item(), broad.item())

    def test_actual_capacity_bijection_unchanged_anchors_and_inputs(self):
        source, target = torch.randn(2, 17, 3), torch.randn(2, 17, 3)
        originals = source.clone(), target.clone()
        options = {"profile": [2, 3, 5, 7]}
        captures = []
        original = coupling.assign_regions
        def capture(points, centers, capacities, **kwargs):
            labels = original(points, centers, capacities, **kwargs)
            captures.append((centers.clone(), capacities.clone(), labels.clone(), kwargs["solver"]))
            return labels
        for name in CAPACITY_MODES:
            with patch.object(coupling, "assign_regions", side_effect=capture), \
                    patch.object(coupling, "pair_within_regions", wraps=coupling.pair_within_regions) as local:
                state = torch.get_rng_state().clone()
                result = coupling.coupling_permutation(source, target, coupling=name, num_regions=4,
                    capacity=options, generator=torch.Generator().manual_seed(8),
                    capacity_generator=torch.Generator().manual_seed(9))
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                self.assertEqual(local.call_args.kwargs["local"], "random")
            torch.testing.assert_close(result.sort().values, torch.arange(17).expand(2, -1))
            target_call, source_call = captures[-2:]
            torch.testing.assert_close(target_call[1], source_call[1])
            for call in (target_call, source_call):
                self.assertEqual(call[3], "exact")
                for labels, capacity in zip(call[2], call[1]):
                    torch.testing.assert_close(torch.bincount(labels, minlength=4), capacity)
            torch.testing.assert_close(target_call[0], target[torch.arange(2)[:, None],
                coupling.farthest_point_sample(target, 4)])
        torch.testing.assert_close(source, originals[0])
        torch.testing.assert_close(target, originals[1])

    def test_equal_profile_recovers_existing_tg_with_fresh_pairing_rng(self):
        source, target = torch.randn(2, 16, 2), torch.rand(2, 16, 2)
        expected = coupling.coupling_permutation(source, target, coupling="target_guided", num_regions=4,
                                                generator=torch.Generator().manual_seed(8))
        for name in CAPACITY_MODES:
            result = coupling.coupling_permutation(source, target, coupling=name, num_regions=4,
                capacity={"profile": [4, 4, 4, 4]}, generator=torch.Generator().manual_seed(8),
                capacity_generator=torch.Generator().manual_seed(9))
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "capacity settings"):
            coupling.coupling_permutation(source, target, coupling="target_guided", num_regions=4,
                                          capacity={"profile": [4, 4, 4, 4]})

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda(self):
        target = torch.randn(2, 32, 3, device="cuda")
        source = torch.randn_like(target)
        anchors = target[:, :4]
        torch.testing.assert_close(local_spectral_scores(target, anchors).cpu(),
                                   local_spectral_scores(target.cpu(), anchors.cpu()), atol=1e-5, rtol=1e-4)
        for name in CAPACITY_MODES:
            result = coupling.coupling_permutation(source, target, coupling=name, num_regions=4,
                generator=torch.Generator(device="cuda").manual_seed(8),
                capacity_generator=torch.Generator(device="cuda").manual_seed(9))
            torch.testing.assert_close(result.sort().values.cpu(), torch.arange(32).expand(2, -1))


class CapacityIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_all_configs_preserve_experiment_controls(self):
        paths = list((ROOT / "capacity_experiments").rglob("*.yaml"))
        self.assertEqual(len(paths), 24)
        names = set()
        for dataset in ("checkerboard", "horse"):
            baseline_path = (ROOT / "horse_experiments/horse_target_guided_k8_n256_seed0.yaml" if dataset == "horse"
                             else ROOT / "checkerboard_experiments/target_guided.yaml")
            baseline = yaml.safe_load(baseline_path.read_text(encoding="utf-8"))
            for path in (ROOT / "capacity_experiments" / dataset).glob("*.yaml"):
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                for key in ("data", "model", "training", "evaluation", "num_regions", "device", "dtype"):
                    self.assertEqual(config[key], baseline[key])
                self.assertIn(config["seed"], (0, 1, 2))
                self.assertNotIn(config["checkpoint"], names)
                names.add(config["checkpoint"])
                if config["coupling"] == "target_guided":
                    self.assertNotIn("capacity", config)
                else:
                    self.assertEqual(config["capacity"]["profile"], PROFILE)
                    self.assertEqual(coupling.coupling_info(config["coupling"])["local_pairing"], "random")

    def test_train_eval_diagnostics_and_config_verification(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            try:
                os.chdir(temp)
                for dataset in ("checkerboard", "horse"):
                    for mode in ("uniform", "large", "small", "random"):
                        path = ROOT / "capacity_experiments" / dataset / f"tg_capacity_{mode}_k8_n256_seed0.yaml"
                        config = yaml.safe_load(path.read_text(encoding="utf-8"))
                        config["device"] = "cpu"
                        config["data"].update(batch_size=2, n_points=16)
                        config["evaluation"].update(batch_size=2, histogram_bins=8)
                        config["num_regions"] = 4
                        config["model"].update(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
                        config["training"]["num_steps"] = 1
                        if mode != "uniform":
                            config["capacity"]["profile"] = [2, 3, 5, 6]
                        Path("config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
                        trainer = train_horse if dataset == "horse" else train
                        evaluator = eval_horse if dataset == "horse" else eval_checkerboard
                        run = trainer.main("config.yaml")
                        artifact = evaluator.main(run / "config.yaml", 2)
                        record = json.loads(artifact.with_suffix(".json").read_text())
                        self.assertTrue(record["training_config_verified"])
                        self.assertEqual(record["config"].get("capacity"), config.get("capacity"))
                        if mode == "large":
                            directory = diagnose.diagnose(run / "config.yaml", dataset, batches=1, batch_size=2)
                            diagnostic = json.loads((directory / "diagnostics.json").read_text())
                            self.assertEqual(diagnostic["config"]["capacity"], config["capacity"])
                            changed = yaml.safe_load((run / "config.yaml").read_text())
                            changed["capacity"]["profile"] = [1, 4, 5, 6]
                            (run / "changed.yaml").write_text(yaml.safe_dump(changed))
                            cls = trainer.HorsePointSetTransformer if dataset == "horse" else trainer.PointSetTransformer
                            with self.assertRaisesRegex(ValueError, "capacity"):
                                load_model(run / "changed.yaml", cls, dataset)
            finally:
                os.chdir(previous)

    def test_summary_keeps_capacity_settings_in_seed_group_signature(self):
        record = {"dataset": "horse", "config": {"seed": 0, "coupling": "target_guided_capacity_large",
                  "capacity": {"profile": [2, 3, 5, 6]}}}
        other = copy.deepcopy(record)
        other["config"]["capacity"]["profile"] = [1, 4, 5, 6]
        self.assertNotEqual(signature(record), signature(other))


if __name__ == "__main__":
    unittest.main()
