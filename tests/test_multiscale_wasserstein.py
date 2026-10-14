import contextlib
import copy
import io
from itertools import permutations
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch
import yaml

import multiscale_wasserstein as msw
import nsot


class MultiscaleWassersteinTests(unittest.TestCase):
    def setUp(self):
        self.previous = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        os.chdir(self.temp.name)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        os.chdir(self.previous)
        self.temp.cleanup()
        torch.set_num_threads(self.threads)

    @staticmethod
    def config(dataset="checkerboard"):
        config = {
            "seed": 0, "device": "cpu", "dtype": "float32", "coupling": msw.METHOD,
            "multiscale_wasserstein": {
                "cache": f"{dataset}.npz", "superset_size": 16, "cache_seed": 5,
                "partition_counts": [2, 4], "scale_weights": [1., 1.], "max_cost_mib": 512,
            },
            "data": {"batch_size": 2, "n_points": 4},
            "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                      "dim_feedforward": 16, "dropout": 0.0},
            "training": {"num_steps": 2, "learning_rate": .001, "weight_decay": .01, "log_every": 1},
            "evaluation": {"batch_size": 2, "histogram_bins": 8},
            "checkpoint": f"{dataset}_multiscale.pt",
        }
        if dataset == "checkerboard":
            config["data"]["grid_size"] = 4
        return config

    def prepare(self, config, dataset="checkerboard"):
        with contextlib.redirect_stdout(io.StringIO()):
            return msw.prepare(config, dataset)

    @staticmethod
    def direct_w2(source, target):
        cost = cdist(source, target, "sqeuclidean")
        rows, columns = linear_sum_assignment(cost)
        return float(cost[rows, columns].mean())

    def test_balanced_partitions_are_nested_deterministic_and_coordinate_preserving(self):
        rng = np.random.default_rng(17)
        # Repeated coordinates exercise deterministic tie handling.
        points = rng.integers(-2, 3, size=(24, 2)).astype(np.float32)
        before = points.copy()
        levels = msw.balanced_partitions(points, [2, 6, 12])
        again = msw.balanced_partitions(points, [2, 6, 12])
        self.assertEqual(len(levels), 3)
        for count, labels, repeated in zip([2, 6, 12], levels, again):
            self.assertEqual(labels.shape, (24,))
            np.testing.assert_array_equal(labels, repeated)
            np.testing.assert_array_equal(np.unique(labels), np.arange(count))
            np.testing.assert_array_equal(np.bincount(labels), np.full(count, 24 // count))
        for coarse, fine in zip(levels[:-1], levels[1:]):
            for child in np.unique(fine):
                self.assertEqual(len(np.unique(coarse[fine == child])), 1)
        np.testing.assert_array_equal(points, before)

    def test_patch_distance_is_exact_empirical_w2_not_centroid_distance(self):
        source = np.array([[-2., 0.], [2., 0.]])
        target = np.array([[-1., 0.], [1., 0.]])
        self.assertEqual(msw.patch_wasserstein_squared(source, source), 0.)
        self.assertAlmostEqual(msw.patch_wasserstein_squared(source, target), 1.)
        self.assertEqual(np.square(source.mean(0) - target.mean(0)).sum(), 0.)
        rng = np.random.default_rng(11)
        source, target = rng.normal(size=(4, 2)), rng.normal(size=(4, 2))
        cost = cdist(source, target, "sqeuclidean")
        expected = min(cost[np.arange(4), list(order)].mean() for order in permutations(range(4)))
        self.assertAlmostEqual(msw.patch_wasserstein_squared(source, target), expected, places=12)

    def test_identity_translation_and_single_scale_global_exact_limit(self):
        rng = np.random.default_rng(3)
        source = rng.normal(size=(16, 2))
        for offset in (np.zeros(2), np.array([.7, -1.2])):
            target = source + offset
            order, diagnostics = msw.multiscale_permutation(source, target, [2, 4], [1., 1.])
            np.testing.assert_array_equal(order, np.arange(16))
            self.assertIsInstance(diagnostics, dict)
            json.dumps(diagnostics, allow_nan=False)
        target = rng.normal(size=(16, 2))
        order, _ = msw.multiscale_permutation(source, target, [1], [1.])
        self.assertAlmostEqual(np.square(source - target[order]).sum(1).mean(),
                               self.direct_w2(source, target), places=12)

    def test_multiscale_leaf_objective_and_selected_point_assignments_are_exact(self):
        rng = np.random.default_rng(31)
        source, target = rng.normal(size=(16, 2)), rng.normal(size=(16, 2))
        original = source.copy(), target.copy()
        counts = [2, 4]
        source_levels = msw.balanced_partitions(source, counts)
        target_levels = msw.balanced_partitions(target, counts)
        source_leaves, target_leaves = source_levels[-1], target_levels[-1]
        for weights in ([1., 0.], [0., 1.], [1., 3.]):
            with self.subTest(weights=weights):
                weights = np.asarray(weights) / np.sum(weights)
                leaf_cost = np.zeros((4, 4))
                for weight, count, slabels, tlabels in zip(weights, counts, source_levels, target_levels):
                    patch_cost = np.array([[self.direct_w2(source[slabels == i], target[tlabels == j])
                                            for j in range(count)] for i in range(count)])
                    source_ancestors = [slabels[np.flatnonzero(source_leaves == i)[0]] for i in range(4)]
                    target_ancestors = [tlabels[np.flatnonzero(target_leaves == j)[0]] for j in range(4)]
                    leaf_cost += weight * patch_cost[np.ix_(source_ancestors, target_ancestors)]
                order, diagnostics = msw.multiscale_permutation(source, target, counts, weights.tolist())
                np.testing.assert_array_equal(np.sort(order), np.arange(16))
                selected_leaves = []
                for i in range(4):
                    members = np.flatnonzero(source_leaves == i)
                    selected = np.unique(target_leaves[order[members]])
                    self.assertEqual(len(selected), 1)
                    j = int(selected[0])
                    selected_leaves.append(j)
                    actual_point_cost = np.square(source[members] - target[order[members]]).sum(1).mean()
                    self.assertAlmostEqual(actual_point_cost,
                                           self.direct_w2(source[members], target[target_leaves == j]), places=12)
                np.testing.assert_array_equal(np.sort(selected_leaves), np.arange(4))
                optimum = min(leaf_cost[np.arange(4), list(p)].sum() for p in permutations(range(4)))
                self.assertAlmostEqual(leaf_cost[np.arange(4), selected_leaves].sum(), optimum, places=12)
                self.assertAlmostEqual(diagnostics["multiscale_objective"], optimum / 4, places=12)
                self.assertAlmostEqual(diagnostics["point_transport_cost"],
                                       np.square(source - target[order]).sum(1).mean(), places=12)
        np.testing.assert_array_equal(source, original[0])
        np.testing.assert_array_equal(target, original[1])

    def test_invalid_points_partitions_weights_and_memory_are_rejected(self):
        points = np.zeros((16, 2))
        for invalid in (np.empty((0, 2)), np.zeros((16, 3)), np.full((16, 2), np.nan),
                        np.full((16, 2), np.inf), np.zeros(16)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                msw.multiscale_permutation(points, invalid, [2, 4], [1., 1.])
        with self.assertRaises(ValueError):
            msw.patch_wasserstein_squared(points, points[:8])
        for counts in ([], [2, 2], [4, 2], [2, 3], [2, 8, 32], [0, 4], [True, 4], [2., 4]):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                msw.multiscale_permutation(points, points, counts, [1.] * len(counts))
        for weights in ([1.], [0., 0.], [-1., 2.], [float("nan"), 1.], [1., float("inf")]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                msw.multiscale_permutation(points, points, [2, 4], weights)
        with self.assertRaises(ValueError):
            msw.multiscale_permutation(points, points, [2, 4], [1., 1.], max_cost_mib=.00001)

    def test_settings_normalize_weights_and_reject_unsupported_options(self):
        config = self.config()
        config["multiscale_wasserstein"]["scale_weights"] = [2., 6.]
        np.testing.assert_allclose(msw.settings(config)["scale_weights"], [.25, .75])
        for key, value in (("superset_size", 15), ("superset_size", True), ("cache_seed", -1),
                           ("cache_seed", True), ("partition_counts", [2, 3]), ("scale_weights", [0., 0.]),
                           ("max_cost_mib", 0), ("max_cost_mib", float("inf")), ("cache", ""),
                           ("online_noise", .2)):
            changed = self.config()
            changed["multiscale_wasserstein"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                msw.settings(changed)
        for section, key, value in (("model", "point_dim", 3), ("data", "n_points", 17),
                                    ("data", "n_points", 0)):
            changed = self.config()
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                msw.settings(changed)

    def test_cache_endpoints_match_nsot_for_both_datasets_and_rng_is_isolated(self):
        for dataset in ("checkerboard", "horse"):
            with self.subTest(dataset=dataset):
                config = self.config(dataset)
                torch.manual_seed(93)
                state = torch.get_rng_state().clone()
                self.prepare(config, dataset)
                torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
                source, target, order, metadata, digest = msw.load_cache(config, dataset)
                baseline = copy.deepcopy(config)
                baseline["coupling"] = "nsot"
                baseline.pop("multiscale_wasserstein")
                baseline["nsot"] = {"cache": "unused.npz", "superset_size": 16, "cache_seed": 5,
                                    "beta": 0., "solver": nsot.SOLVER}
                expected_source, expected_target = nsot.draw_supersets(baseline, dataset)
                np.testing.assert_array_equal(source, expected_source)
                np.testing.assert_array_equal(target, expected_target)
                np.testing.assert_array_equal(np.sort(order), np.arange(16))
                self.assertEqual(source.dtype, np.float32)
                self.assertEqual(target.dtype, np.float32)
                self.assertEqual(len(digest), 64)
                self.assertIsInstance(metadata, dict)

    def test_cache_reuse_mismatch_and_hash_do_not_overwrite_existing_cache(self):
        config = self.config()
        with self.assertRaises(FileNotFoundError):
            msw.load_cache(config, "checkerboard")
        path, _ = self.prepare(config)
        original = path.read_bytes()
        with patch.object(msw, "multiscale_permutation", side_effect=AssertionError("recomputed")):
            self.prepare(config)
        self.assertEqual(path.read_bytes(), original)
        for key, value in (("cache_seed", 9), ("partition_counts", [1, 4]),
                           ("scale_weights", [1., 3.]), ("cache_sha256", "bad")):
            changed = copy.deepcopy(config)
            changed["multiscale_wasserstein"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.prepare(changed)
            self.assertEqual(path.read_bytes(), original)
        changed = copy.deepcopy(config)
        changed["data"]["grid_size"] = 6
        with self.assertRaises(ValueError):
            msw.load_cache(changed, "checkerboard")

    def test_cache_rejects_nonbijective_and_nonfinite_arrays_and_unknown_formats(self):
        config = self.config()
        path, _ = self.prepare(config)
        source, target, order, metadata, _ = msw.load_cache(config, "checkerboard")
        for corruption in ("permutation", "valid_permutation", "source", "target", "finite_source", "format", "coordinates", "extra"):
            with self.subTest(corruption=corruption):
                arrays = {"source": source.copy(), "target": target.copy(), "permutation": order.copy()}
                changed_metadata = copy.deepcopy(metadata)
                if corruption == "permutation":
                    arrays["permutation"][:] = 0
                elif corruption == "valid_permutation":
                    arrays["permutation"][[0, 1]] = arrays["permutation"][[1, 0]]
                elif corruption in ("source", "target"):
                    arrays[corruption][0, 0] = np.nan
                elif corruption == "finite_source":
                    arrays["source"][0, 0] += 1.
                elif corruption == "format":
                    changed_metadata["format_version"] = -1
                elif corruption == "coordinates":
                    changed_metadata["target_coordinates"] = "unsupported sampling"
                else:
                    arrays["unexpected"] = np.zeros(1)
                np.savez_compressed(path, **arrays, metadata=np.array(json.dumps(changed_metadata)))
                before = path.read_bytes()
                with self.assertRaises(ValueError):
                    self.prepare(config)
                self.assertEqual(path.read_bytes(), before)

    def test_iid_pair_subsampling_preserves_original_coordinates_without_noise(self):
        config = self.config()
        self.prepare(config)
        source, target, order, _, _ = msw.load_cache(config, "checkerboard")
        sampler = msw.MultiscaleWassersteinSampler(config, "checkerboard", "cpu", torch.float32)
        generator = torch.Generator().manual_seed(42)
        actual_source, actual_target = sampler.sample(500, generator=generator)
        reference_generator = torch.Generator().manual_seed(42)
        indices = torch.randint(16, (500, 4), generator=reference_generator)
        torch.testing.assert_close(actual_source, torch.from_numpy(source)[indices], rtol=0, atol=0)
        torch.testing.assert_close(actual_target, torch.from_numpy(target[order])[indices], rtol=0, atol=0)
        torch.testing.assert_close(generator.get_state(), reference_generator.get_state(), rtol=0, atol=0)
        self.assertTrue(any(len(torch.unique(row)) < len(row) for row in indices))
        self.assertIsInstance(sampler.details(), dict)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(batch_size=invalid), self.assertRaises(ValueError):
                sampler.sample(invalid)

    def test_both_datasets_train_load_diagnose_and_evaluate_without_training_cache(self):
        import audit_generation
        import audit_jacobian
        from diagnostic_data import DiagnosticData
        import eval as checker_evaluator
        import eval_horse
        from experiment import load_model
        from model import PointSetTransformer
        import train
        import train_horse

        with contextlib.redirect_stdout(io.StringIO()):
            for dataset, trainer, evaluator, model_class in (
                ("checkerboard", train.main, checker_evaluator.main, PointSetTransformer),
                ("horse", train_horse.main, eval_horse.main, train_horse.HorsePointSetTransformer),
            ):
                with self.subTest(dataset=dataset):
                    config = self.config(dataset)
                    self.prepare(config, dataset)
                    path = Path(f"{dataset}.yaml")
                    with path.open("w", encoding="utf-8") as stream:
                        yaml.safe_dump(config, stream)
                    with patch.object(train, "coupled_points", side_effect=AssertionError("online matching called")):
                        run = trainer(path, steps=1)
                    model, trained, checkpoint, metadata = load_model(run / "config.yaml", model_class, dataset)
                    self.assertTrue(metadata["training_config_verified"])
                    self.assertEqual(trained[msw.METHOD]["cache_sha256"], nsot.file_sha256(config[msw.METHOD]["cache"]))
                    self.assertNotIn("cache_sha256", yaml.safe_load(path.read_text())[msw.METHOD])
                    data = DiagnosticData(trained, dataset, "cpu", torch.float32, 2026)
                    fm = audit_generation.fm_summary(model, trained, data, 1, 2, 2026)
                    self.assertIn("fixed training cache", fm["sampling_scope"])
                    self.assertTrue(all(np.isfinite(row["mse"]["mean"]) for row in fm["time_errors"]))
                    source, target, details = audit_jacobian._cached_pairs(trained, dataset, data, 2)
                    sampler = msw.MultiscaleWassersteinSampler(trained, dataset, "cpu", torch.float32)
                    for index in range(2):
                        generator = torch.Generator().manual_seed(data.draw_seed("jacobian_pairs", index))
                        expected_source, expected_target = sampler.sample(1, generator=generator)
                        torch.testing.assert_close(source[index], expected_source[0], rtol=0, atol=0)
                        torch.testing.assert_close(target[index], expected_target[0], rtol=0, atol=0)
                    self.assertEqual(details["cache_sha256"], sampler.cache_sha256)
                    cache = Path(config[msw.METHOD]["cache"])
                    cache.rename(cache.with_suffix(".backup"))
                    if dataset == "checkerboard":
                        output = evaluator(run / "config.yaml", 2, render=False)
                    else:
                        with patch.object(eval_horse, "render_comparison"), patch("horse_regions.HorseRegions.render"):
                            output = evaluator(run / "config.yaml", 2)
                    result = json.loads(output.with_suffix(".json").read_text())
                    self.assertTrue(result["training_config_verified"])
                    self.assertTrue(np.isfinite(result["chamfer"]))
                    self.assertEqual(result["coupling_details"]["cache_sha256"], sampler.cache_sha256)


if __name__ == "__main__":
    unittest.main()
