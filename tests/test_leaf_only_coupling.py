"""Leaf-only reuses the MS-W2 hierarchy and omits only inactive scale costs."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch

import multiscale_wasserstein as msw


class LeafOnlyCouplingTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    @staticmethod
    def config(cache):
        return {
            "coupling": msw.METHOD,
            msw.METHOD: {"cache": str(cache), "superset_size": 16, "cache_seed": 7,
                         "partition_counts": [2, 4], "scale_weights": [0., 1.]},
            "data": {"batch_size": 2, "n_points": 4, "grid_size": 4},
            "model": {"point_dim": 2},
        }

    @staticmethod
    def original_mapping(source, target, counts, weights, source_method="axis", target_method="axis"):
        """Independent reference to the old evaluate-all-scales implementation."""
        source_labels = msw.balanced_partitions(source, counts, strategy=source_method)
        target_labels = msw.balanced_partitions(target, counts, strategy=target_method)
        leaves = counts[-1]
        source_ids = [np.flatnonzero(source_labels[-1] == i) for i in range(leaves)]
        target_ids = [np.flatnonzero(target_labels[-1] == i) for i in range(leaves)]
        combined = np.zeros((leaves, leaves), dtype=np.float64)
        for level, (count, weight) in enumerate(zip(counts, np.array(weights) / sum(weights))):
            matrix = np.empty((count, count), dtype=np.float64)
            for i in range(count):
                for j in range(count):
                    cost = cdist(source[source_labels[level] == i],
                                 target[target_labels[level] == j], "sqeuclidean")
                    rows, cols = linear_sum_assignment(cost)
                    matrix[i, j] = cost[rows, cols].mean()
            ancestors_s = np.array([source_labels[level][ids[0]] for ids in source_ids])
            ancestors_t = np.array([target_labels[level][ids[0]] for ids in target_ids])
            combined += weight * matrix[ancestors_s[:, None], ancestors_t[None, :]]
        rows, cols = linear_sum_assignment(combined)
        result = np.empty(len(source), dtype=np.int64)
        for i, j in zip(rows, cols):
            local_cost = cdist(source[source_ids[i]], target[target_ids[j]], "sqeuclidean")
            local_rows, local_cols = linear_sum_assignment(local_cost)
            result[source_ids[i][local_rows]] = target_ids[j][local_cols]
        return result, float(combined[rows, cols].mean())

    def test_skips_zero_weight_costs_but_keeps_all_nested_partitions(self):
        rng = np.random.default_rng(19)
        source, target = rng.normal(size=(24, 2)), rng.normal(size=(24, 2))
        counts = [2, 6, 12]
        source_before, target_before = source.copy(), target.copy()
        progress = []
        with patch.object(msw, "balanced_partitions", wraps=msw.balanced_partitions) as partition, \
                patch.object(msw, "patch_wasserstein_squared", wraps=msw.patch_wasserstein_squared) as cost:
            mapping, report = msw.multiscale_permutation(source, target, counts, [0, 0, 1], progress=progress.append)
        self.assertEqual(partition.call_count, 2)
        for call in partition.call_args_list:
            self.assertEqual(call.args[1], counts)
        self.assertEqual(cost.call_count, 12 ** 2)
        for call in cost.call_args_list:
            self.assertEqual(call.args[0].shape, (2, 2))
            self.assertEqual(call.args[1].shape, (2, 2))
        for row in report["scales"][:2]:
            self.assertEqual(row["cost_status"], "skipped_zero_weight")
            self.assertEqual(row["exact_patch_solves"], 0)
            self.assertEqual(row["cost_seconds"], 0.)
            self.assertIsNone(row["selected_mean_w2_squared"])
        self.assertEqual(report["scales"][-1]["exact_patch_solves"], 144)
        self.assertTrue(any("skipped_zero_weight" in message for message in progress))
        json.dumps(report, allow_nan=False)
        np.testing.assert_array_equal(np.sort(mapping), np.arange(24))
        np.testing.assert_array_equal(source, source_before)
        np.testing.assert_array_equal(target, target_before)

    def test_mapping_and_objective_equal_original_including_zero_leaf_weight(self):
        rng = np.random.default_rng(31)
        source, target = rng.normal(size=(32, 2)), rng.normal(size=(32, 2))
        for weights in ([1, 1, 1], [0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 2]):
            for source_method, target_method in (("axis", "axis"), ("axis", "pca"), ("pca", "pca")):
                with self.subTest(weights=weights, source=source_method, target=target_method):
                    expected, expected_cost = self.original_mapping(
                        source, target, [2, 4, 8], weights, source_method, target_method)
                    actual, report = msw.multiscale_permutation(
                        source, target, [2, 4, 8], weights,
                        source_partition=source_method, target_partition=target_method)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(report["multiscale_objective"], expected_cost)

    def test_memory_guard_uses_only_active_patch_costs_and_mandatory_leaf_costs(self):
        rng = np.random.default_rng(43)
        source, target = rng.normal(size=(128, 2)), rng.normal(size=(128, 2))
        _, report = msw.multiscale_permutation(source, target, [2, 8, 16], [0, 0, 1], max_cost_mib=.005)
        self.assertEqual(report["largest_dense_cost_mib"], 16 ** 2 * 8 / 2 ** 20)
        with self.assertRaisesRegex(ValueError, "Largest dense cost matrix"):
            msw.multiscale_permutation(source, target, [2, 8, 16], [1, 1, 1], max_cost_mib=.005)
        with self.assertRaisesRegex(ValueError, "Largest dense cost matrix"):
            msw.multiscale_permutation(source, target, [2, 8, 16], [0, 0, 1], max_cost_mib=.001)
        # Baseline guard behavior is unchanged when no optional weights are supplied.
        self.assertEqual(msw._memory_guard(128, [2, 8, 16], 512), 64 ** 2 * 8 / 2 ** 20)

    def test_settings_allow_leaf_only_without_bounding_unused_coarse_matrices(self):
        config = self.config("unused.npz")
        config[msw.METHOD].update(superset_size=10000, partition_counts=[4, 20, 100],
                                 scale_weights=[0, 0, 1], max_cost_mib=1)
        self.assertEqual(msw.settings(config)["partition_counts"], [4, 20, 100])
        config[msw.METHOD]["scale_weights"] = [1, 1, 1]
        with self.assertRaisesRegex(ValueError, "Largest dense cost matrix"):
            msw.settings(config)

    def test_baseline_cache_identity_is_unchanged_and_zero_weight_cache_is_distinct(self):
        config = self.config("unused.npz")
        baseline = copy.deepcopy(config)
        baseline[msw.METHOD]["scale_weights"] = [1, 1]
        for source_method, target_method in (("axis", "axis"), ("axis", "pca"), ("pca", "pca")):
            with self.subTest(source=source_method, target=target_method):
                for cfg in (config, baseline):
                    cfg[msw.METHOD].update(source_partition=source_method, target_partition=target_method)
                old_spec = msw.cache_spec(baseline, "checkerboard")
                new_spec = msw.cache_spec(config, "checkerboard")
                self.assertEqual(old_spec.keys(), new_spec.keys())
                if "pca" in (source_method, target_method):
                    self.assertEqual(old_spec["implementation"], msw.PCA_IMPLEMENTATION)
                    self.assertEqual(new_spec["implementation"], msw.PCA_ZERO_WEIGHT_IMPLEMENTATION)
                else:
                    self.assertEqual(old_spec["implementation"], msw.IMPLEMENTATION)
                    self.assertEqual(new_spec["implementation"], msw.ZERO_WEIGHT_IMPLEMENTATION)

    def test_old_zero_weight_cache_timings_are_not_reused_and_existing_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory) / "leaf.npz"
            config = self.config(path)
            msw.prepare(config, "checkerboard")
            source, target, order, metadata, _ = msw.load_cache(config, "checkerboard")
            self.assertEqual(metadata["implementation"], msw.ZERO_WEIGHT_IMPLEMENTATION)
            metadata["implementation"] = msw.IMPLEMENTATION
            np.savez_compressed(path, source=source, target=target, permutation=order,
                                metadata=np.array(json.dumps(metadata)))
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "implementation"):
                msw.prepare(config, "checkerboard")
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
