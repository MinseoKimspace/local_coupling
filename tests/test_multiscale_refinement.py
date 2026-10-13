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
from scipy.spatial.distance import cdist
import torch
import yaml

import multiscale_refinement as refinement
import multiscale_wasserstein as baseline


class MultiscaleRefinementTests(unittest.TestCase):
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
        result = {
            "seed": 0, "device": "cpu", "dtype": "float32", "coupling": refinement.METHOD,
            "multiscale_wasserstein": {
                "cache": f"{dataset}_base.npz", "superset_size": 16, "cache_seed": 5,
                "partition_counts": [2, 4], "scale_weights": [1., 1.], "max_cost_mib": 512,
            },
            "multiscale_refined": {"cache": f"{dataset}_refined.npz", "neighbors": 2, "max_edges": 2000000},
            "data": {"batch_size": 2, "n_points": 4},
            "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                      "dim_feedforward": 16, "dropout": 0.0},
            "training": {"num_steps": 2, "learning_rate": .001, "weight_decay": .01, "log_every": 1},
            "evaluation": {"batch_size": 2, "histogram_bins": 8},
            "checkpoint": f"{dataset}_refined.pt",
        }
        if dataset == "checkerboard":
            result["data"]["grid_size"] = 4
        return result

    def prepare(self, config, dataset="checkerboard"):
        with contextlib.redirect_stdout(io.StringIO()):
            base_config = refinement.base_config(config)
            base_path, _ = baseline.prepare(base_config, dataset)
            config[baseline.METHOD]["cache_sha256"] = baseline.file_sha256(base_path)
            original = base_path.read_bytes()
            torch_state = torch.get_rng_state().clone()
            result = refinement.prepare(config, dataset)
            self.assertEqual(base_path.read_bytes(), original)
            torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
            return result

    @staticmethod
    def point_objective_matrix(source, target, partition_counts, scale_weights):
        weights = np.asarray(scale_weights, dtype=np.float64)
        weights = weights / weights.sum()
        source_levels = baseline.balanced_partitions(source, partition_counts)
        target_levels = baseline.balanced_partitions(target, partition_counts)
        cost = weights[-1] * cdist(source, target, "sqeuclidean")
        for level, count in enumerate(partition_counts[:-1]):
            left, right = source_levels[level], target_levels[level]
            distances = np.array([
                [baseline.patch_wasserstein_squared(source[left == i], target[right == j])
                 for j in range(count)] for i in range(count)
            ])
            cost += weights[level] * distances[left[:, None], right[None, :]]
        return cost

    def test_zero_neighbors_preserves_base_permutation_and_objective_exactly(self):
        rng = np.random.default_rng(45)
        source, target = rng.normal(size=(16, 2)), rng.normal(size=(16, 2))
        original = source.copy(), target.copy()
        for weights in ([1., 1.], [2., 5.], [1., 0.], [0., 1.]):
            with self.subTest(weights=weights):
                base_order, base_details = baseline.multiscale_permutation(source, target, [2, 4], weights)
                order, details = refinement.refine_permutation(
                    source, target, base_order, [2, 4], weights, neighbors=0,
                    expected_base_objective=base_details["multiscale_objective"])
                np.testing.assert_array_equal(order, base_order)
                expected = self.point_objective_matrix(source, target, [2, 4], weights)[np.arange(16), order].mean()
                self.assertAlmostEqual(details["objective_before"], expected, places=12)
                self.assertAlmostEqual(details["objective_after"], expected, places=12)
                self.assertAlmostEqual(expected, base_details["multiscale_objective"], places=12)
                self.assertEqual(details["changed_fraction"], 0.)
                self.assertGreaterEqual(details["edges"], 16)
        np.testing.assert_array_equal(source, original[0])
        np.testing.assert_array_equal(target, original[1])

    def test_sparse_candidates_retain_baseline_and_are_unique_deterministic_symmetric(self):
        source = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        target = source.copy()
        base_order = np.array([2, 3, 0, 1])
        labels = np.array([0, 0, 1, 1])
        rows, columns, details = refinement.candidate_edges(
            source, target, base_order, labels, labels, neighbors=1)
        edges = set(zip(rows.tolist(), columns.tolist()))
        self.assertEqual(len(edges), len(rows))
        self.assertTrue(set(enumerate(base_order.tolist())).issubset(edges))
        for i in range(4):
            self.assertTrue({(i, j) for j in range(4) if labels[j] == labels[base_order[i]]}.issubset(edges))
        np.testing.assert_array_equal(rows * 4 + columns, np.sort(rows * 4 + columns))
        inverse = np.argsort(base_order)
        for i, j in edges:
            self.assertIn((int(inverse[j]), int(base_order[i])), edges)
        again = refinement.candidate_edges(source, target, base_order, labels, labels, neighbors=1)
        np.testing.assert_array_equal(rows, again[0])
        np.testing.assert_array_equal(columns, again[1])
        self.assertIsInstance(details, dict)

    def test_cross_leaf_refinement_can_change_assignment_without_losing_any_target(self):
        source = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
        target = source.copy()
        base_order = np.array([2, 3, 0, 1])
        originals = source.copy(), target.copy(), base_order.copy()
        order, details = refinement.refine_permutation(
            source, target, base_order, [1, 2], [0., 1.], neighbors=3)
        np.testing.assert_array_equal(order, np.arange(4))
        self.assertLess(details["objective_after"], details["objective_before"])
        self.assertAlmostEqual(details["objective_after"], 0., places=12)
        self.assertEqual(details["changed_fraction"], 1.)
        self.assertEqual(details["cross_leaf_fraction"], 1.)
        np.testing.assert_array_equal(np.sort(order), np.arange(4))
        for actual, expected in zip((source, target, base_order), originals):
            np.testing.assert_array_equal(actual, expected)

    def test_sparse_solution_matches_independent_brute_force_on_small_graph(self):
        rng = np.random.default_rng(9)
        source, target = rng.normal(size=(6, 2)), rng.normal(size=(6, 2))
        counts, weights = [2, 6], [2., 3.]
        base_order, base_details = baseline.multiscale_permutation(source, target, counts, weights)
        source_labels = baseline.balanced_partitions(source, counts)[-1]
        target_labels = baseline.balanced_partitions(target, counts)[-1]
        rows, cols, _ = refinement.candidate_edges(
            source, target, base_order, source_labels, target_labels, neighbors=1)
        edges = set(zip(rows.tolist(), cols.tolist()))
        cost = self.point_objective_matrix(source, target, counts, weights)
        feasible = [order for order in permutations(range(6))
                    if all((i, j) in edges for i, j in enumerate(order))]
        self.assertTrue(feasible)
        expected = min(cost[np.arange(6), list(order)].mean() for order in feasible)
        order, details = refinement.refine_permutation(
            source, target, base_order, counts, weights, neighbors=1,
            expected_base_objective=base_details["multiscale_objective"])
        self.assertTrue(all((i, j) in edges for i, j in enumerate(order)))
        np.testing.assert_array_equal(np.sort(order), np.arange(6))
        self.assertAlmostEqual(cost[np.arange(6), order].mean(), expected, places=12)
        self.assertAlmostEqual(details["objective_after"], expected, places=12)
        self.assertLessEqual(details["objective_after"], details["objective_before"] + 1e-12)

    def test_zero_cost_duplicate_points_and_ties_remain_feasible(self):
        source = np.zeros((8, 2))
        target = np.zeros((8, 2))
        base_order = np.arange(7, -1, -1)
        for neighbors in (0, 1, 7):
            with self.subTest(neighbors=neighbors):
                order, details = refinement.refine_permutation(
                    source, target, base_order, [2, 4], [1., 1.], neighbors=neighbors)
                np.testing.assert_array_equal(np.sort(order), np.arange(8))
                self.assertEqual(details["objective_before"], 0.)
                self.assertEqual(details["objective_after"], 0.)
                json.dumps(details, allow_nan=False)
        rows, cols, _ = refinement.candidate_edges(
            source, target, np.arange(8), np.arange(8), np.arange(8), neighbors=1)
        expected = {(i, i) for i in range(8)} | {(0, i) for i in range(1, 8)} | {(i, 0) for i in range(1, 8)}
        self.assertEqual(set(zip(rows.tolist(), cols.tolist())), expected)

    def test_invalid_inputs_and_graph_memory_guards_fail_before_work(self):
        source = np.zeros((16, 2))
        order = np.arange(16)
        for bad_order in (np.zeros(16, dtype=int), np.arange(15), np.arange(16, dtype=float),
                          np.arange(1, 17)):
            with self.subTest(order=bad_order.shape), self.assertRaises(ValueError):
                refinement.refine_permutation(source, source, bad_order, [2, 4], [1., 1.])
        for kwargs in ({"neighbors": -1}, {"neighbors": True}, {"neighbors": 1.5},
                       {"max_edges": 0}, {"max_edges": True}, {"max_cost_mib": .000001},
                       {"neighbors": 2, "max_edges": 191}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                refinement.refine_permutation(source, source, order, [2, 4], [1., 1.], **kwargs)
        invalid = source.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            refinement.refine_permutation(source, invalid, order, [2, 4], [1., 1.])
        with self.assertRaises(ValueError):
            refinement.refine_permutation(source, source, order, [2, 4], [1., 1.],
                                          expected_base_objective=10.)

    def test_refinement_does_not_allocate_a_dense_superset_cost_matrix(self):
        rng = np.random.default_rng(12)
        count = 64
        source, target = rng.normal(size=(count, 2)), rng.normal(size=(count, 2))
        order, _ = baseline.multiscale_permutation(source, target, [8, 16], [1., 1.])

        def guarded(original):
            def allocate(shape, *args, **kwargs):
                if isinstance(shape, (list, tuple)) and tuple(shape) == (count, count):
                    raise AssertionError("Dense M-by-M matrix allocated")
                return original(shape, *args, **kwargs)
            return allocate

        def guarded_cdist(a, b, *args, **kwargs):
            if len(a) == count and len(b) == count:
                raise AssertionError("Dense superset pairwise distances requested")
            return cdist(a, b, *args, **kwargs)

        with contextlib.ExitStack() as stack:
            for name in ("empty", "zeros", "ones", "full"):
                stack.enter_context(patch.object(np, name, side_effect=guarded(getattr(np, name))))
            stack.enter_context(patch.object(baseline, "cdist", side_effect=guarded_cdist))
            if hasattr(refinement, "cdist"):
                stack.enter_context(patch.object(refinement, "cdist", side_effect=guarded_cdist))
            refined, details = refinement.refine_permutation(
                source, target, order, [8, 16], [1., 1.], neighbors=2)
        np.testing.assert_array_equal(np.sort(refined), np.arange(count))
        self.assertLess(details["edges"], count * count)

    def test_settings_reject_unknown_and_invalid_refinement_settings(self):
        for key, value in (("neighbors", -1), ("neighbors", True), ("neighbors", 1.5),
                           ("max_edges", 0), ("max_edges", 1.5), ("cache", ""), ("beta", .2)):
            config = self.config()
            config[refinement.METHOD][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                refinement.settings(config)
        config = self.config()
        config[refinement.METHOD]["neighbors"] = 0
        self.assertEqual(refinement.settings(config)["neighbors"], 0)
        restored = refinement.base_config(config)
        self.assertEqual(restored["coupling"], baseline.METHOD)
        self.assertEqual(config["coupling"], refinement.METHOD)

    def test_cache_reuse_hashes_spec_mismatch_and_independence_from_base_file(self):
        config = self.config()
        with self.assertRaises(FileNotFoundError):
            refinement.prepare(config, "checkerboard")
        path, metadata = self.prepare(config)
        source, target, order, _, digest = refinement.load_cache(config, "checkerboard")
        original = path.read_bytes()
        with patch.object(refinement, "refine_permutation", side_effect=AssertionError("recomputed")):
            with contextlib.redirect_stdout(io.StringIO()):
                refinement.prepare(config, "checkerboard")
        self.assertEqual(path.read_bytes(), original)
        base_source, base_target, _, _, base_digest = baseline.load_cache(refinement.base_config(config), "checkerboard")
        np.testing.assert_array_equal(source, base_source)
        np.testing.assert_array_equal(target, base_target)
        np.testing.assert_array_equal(np.sort(order), np.arange(16))
        self.assertEqual(metadata["base_cache_sha256"], base_digest)
        self.assertEqual(len(digest), 64)
        for section, key, value in ((refinement.METHOD, "neighbors", 1),
                                    (refinement.METHOD, "cache_sha256", "bad"),
                                    (baseline.METHOD, "cache_sha256", "bad"),
                                    (baseline.METHOD, "scale_weights", [1., 3.]),
                                    ("data", "grid_size", 6)):
            changed = copy.deepcopy(config)
            changed[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                refinement.prepare(changed, "checkerboard")
            self.assertEqual(path.read_bytes(), original)
        base_path = Path(config[baseline.METHOD]["cache"])
        base_path.rename(base_path.with_suffix(".backup"))
        reloaded = refinement.load_cache(config, "checkerboard")
        np.testing.assert_array_equal(reloaded[2], order)

    def test_cache_tampering_is_rejected_without_overwriting(self):
        config = self.config()
        path, _ = self.prepare(config)
        with np.load(path, allow_pickle=False) as archive:
            originals = {key: archive[key].copy() for key in archive.files}
        for corruption in ("source", "target", "permutation", "valid_permutation", "format", "extra"):
            arrays = {key: value.copy() for key, value in originals.items()}
            if corruption in ("source", "target"):
                arrays[corruption][0, 0] += 1.
            elif corruption == "permutation":
                arrays["permutation"][:] = 0
            elif corruption == "valid_permutation":
                arrays["permutation"][[0, 1]] = arrays["permutation"][[1, 0]]
            elif corruption == "format":
                metadata = json.loads(str(arrays["metadata"].item()))
                metadata["format_version"] = -1
                arrays["metadata"] = np.array(json.dumps(metadata))
            else:
                arrays["unexpected"] = np.zeros(1)
            np.savez_compressed(path, **arrays)
            before = path.read_bytes()
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                refinement.prepare(config, "checkerboard")
            self.assertEqual(path.read_bytes(), before)

    def test_sampler_source_and_generator_draws_match_base_exactly(self):
        config = self.config()
        self.prepare(config)
        original = baseline.MultiscaleWassersteinSampler(refinement.base_config(config), "checkerboard", "cpu", torch.float32)
        sampler = refinement.MultiscaleRefinedSampler(config, "checkerboard", "cpu", torch.float32)
        source, target, order, _, _ = refinement.load_cache(config, "checkerboard")
        generator = torch.Generator().manual_seed(22)
        reference = torch.Generator().manual_seed(22)
        actual_source, actual_target = sampler.sample(32, generator=generator)
        base_source, _ = original.sample(32, generator=reference)
        torch.testing.assert_close(actual_source, base_source, rtol=0, atol=0)
        torch.testing.assert_close(generator.get_state(), reference.get_state(), rtol=0, atol=0)
        indices = torch.randint(16, (32, 4), generator=torch.Generator().manual_seed(22))
        torch.testing.assert_close(actual_source, torch.from_numpy(source)[indices], rtol=0, atol=0)
        torch.testing.assert_close(actual_target, torch.from_numpy(target[order])[indices], rtol=0, atol=0)
        self.assertIsInstance(sampler.details(), dict)
        for invalid in (0, -1, True, 1.5):
            with self.subTest(batch_size=invalid), self.assertRaises(ValueError):
                sampler.sample(invalid)

    def test_tiny_both_dataset_training_diagnostics_and_cache_free_evaluation(self):
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
                    self.assertEqual(trained[refinement.METHOD]["cache_sha256"],
                                     baseline.file_sha256(config[refinement.METHOD]["cache"]))
                    data = DiagnosticData(trained, dataset, "cpu", torch.float32, 2026)
                    fm = audit_generation.fm_summary(model, trained, data, 1, 2, 2026)
                    self.assertIn("fixed training cache", fm["sampling_scope"])
                    self.assertTrue(all(np.isfinite(row["mse"]["mean"]) for row in fm["time_errors"]))
                    source, target, details = audit_jacobian._cached_pairs(trained, dataset, data, 2)
                    sampler = refinement.MultiscaleRefinedSampler(trained, dataset, "cpu", torch.float32)
                    for index in range(2):
                        generator = torch.Generator().manual_seed(data.draw_seed("jacobian_pairs", index))
                        expected_source, expected_target = sampler.sample(1, generator=generator)
                        torch.testing.assert_close(source[index], expected_source[0], rtol=0, atol=0)
                        torch.testing.assert_close(target[index], expected_target[0], rtol=0, atol=0)
                    self.assertEqual(details["cache_sha256"], sampler.cache_sha256)
                    for method in (baseline.METHOD, refinement.METHOD):
                        cache = Path(config[method]["cache"])
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
