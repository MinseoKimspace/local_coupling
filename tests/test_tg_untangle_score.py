"""Independent numerical and invariant checks for offline conflict search."""

import copy
import unittest

import numpy as np

import tg_untangle


def brute_score(source, target, permutation, edges, options):
    """Scalar-loop oracle, deliberately independent of vectorized deltas."""
    if not len(edges):
        return 0.0
    scores = []
    velocity = target[permutation] - source
    for left, right in edges:
        delta_velocity = velocity[left] - velocity[right]
        for time, budget in zip(options["times"], options["lipschitz"]):
            position = (1.0 - time) * source + time * target[permutation]
            excess = max(float(np.linalg.norm(delta_velocity))
                         - budget * float(np.linalg.norm(position[left] - position[right])), 0.0)
            scores.append(excess ** 2)
    return float(np.mean(scores))


class ConflictScoreTests(unittest.TestCase):
    @staticmethod
    def fixture():
        source = np.array([[-1., 0.], [-.1, .1], [.2, -.1],
                           [1., 0.], [.1, .2], [-.2, -.2]])
        target = np.array([[-.7, .9], [.8, .7], [-.9, -.4],
                           [.9, -.8], [-.2, .8], [.4, -.9]])
        source_labels = np.array([0, 0, 0, 1, 1, 1])
        target_labels = np.array([0, 1, 0, 1, 0, 1])
        options = {"enabled": True, "neighbors": 2, "times": [.1, .35, .7],
                   "lipschitz": [.8, 1., 1.3], "permutations": 3,
                   "swap_steps": 100, "seed": 4}
        return source, target, source_labels, target_labels, options

    def assert_valid_pool(self, pool, source_labels, target_labels):
        self.assertEqual(pool.dtype, np.dtype(np.int32))
        for permutation in pool:
            np.testing.assert_array_equal(np.sort(permutation), np.arange(len(source_labels)))
            np.testing.assert_array_equal(target_labels[permutation], source_labels)

    def test_options_broadcast_disable_and_do_not_mutate(self):
        supplied = {"times": [.2, .6], "lipschitz": .75}
        before = copy.deepcopy(supplied)
        options = tg_untangle.normalize_options(supplied)
        self.assertEqual(supplied, before)
        self.assertTrue(options["enabled"])
        self.assertEqual(options["lipschitz"], [.75, .75])
        self.assertEqual(tg_untangle.normalize_options({"times": [.5]})["lipschitz"], [1.])
        self.assertFalse(tg_untangle.normalize_options(None)["enabled"])
        self.assertFalse(tg_untangle.normalize_options(False)["enabled"])
        self.assertTrue(tg_untangle.normalize_options(True)["enabled"])

    def test_score_matches_scalar_edge_time_oracle_without_mutation(self):
        source, target, _, _, options = self.fixture()
        permutation = np.array([2, 0, 4, 3, 1, 5])
        edges = tg_untangle.build_source_edges(source, options["neighbors"])
        inputs = [array.copy() for array in (source, target, permutation, edges)]
        expected = brute_score(source, target, permutation, edges, options)
        actual = tg_untangle.conflict_score(source, target, permutation, edges, options)
        self.assertGreater(expected, 0.)
        self.assertAlmostEqual(actual, expected, places=13)
        for array, before in zip((source, target, permutation, edges), inputs):
            np.testing.assert_array_equal(array, before)

    def test_rigid_translation_has_zero_conflict(self):
        source, _, _, _, options = self.fixture()
        target = source + np.array([2., -3.])
        edges = tg_untangle.build_source_edges(source, 2)
        score = tg_untangle.conflict_score(source, target, np.arange(len(source)), edges, options)
        self.assertEqual(score, 0.)

    def test_contraction_conflict_is_not_a_path_crossing_detector(self):
        source = np.array([[-1., 0.], [1., 0.]])
        target = np.zeros_like(source)
        options = {"times": [.25, .75], "lipschitz": 1.}
        score = tg_untangle.conflict_score(source, target, np.arange(2),
                                           np.array([[0, 1]]), options)
        # Straight non-crossing contraction has excess 2t, hence mean(4t^2).
        self.assertAlmostEqual(score, 1.25, places=14)

    def test_crossing_at_midpoint_has_positive_conflict(self):
        source = np.array([[-1., 0.], [1., 0.]])
        score = tg_untangle.conflict_score(source, source, np.array([1, 0]),
                                           np.array([[0, 1]]), {"times": [.5], "lipschitz": 1.})
        self.assertAlmostEqual(score, 16., places=14)

    def test_empty_edge_score_and_deterministic_tie_breaking(self):
        source = np.array([[0., 0.], [1., 0.], [-1., 0.], [8., 0.]])
        edges = tg_untangle.build_source_edges(source, 1)
        np.testing.assert_array_equal(edges, np.array([[0, 1], [0, 2], [1, 3]]))
        self.assertEqual(tg_untangle.conflict_score(source, source, np.arange(4),
                                                   np.empty((0, 2), dtype=int), True), 0.)

    def test_source_graph_includes_cross_patch_neighbors(self):
        source = np.array([[0., 0.], [.01, 0.], [10., 0.], [10.01, 0.]])
        labels = np.array([0, 1, 0, 1])
        edges = tg_untangle.build_source_edges(source, 1)
        np.testing.assert_array_equal(edges, np.array([[0, 1], [2, 3]]))
        pool, stats = tg_untangle.optimize_permutations(
            source, source.copy(), labels, labels, 2,
            {"neighbors": 1, "permutations": 2, "swap_steps": 8}, 7)
        self.assertEqual(stats["cross_patch_edge_count"], 2)
        self.assertEqual(stats["edge_count"], 2)
        self.assert_valid_pool(pool, labels, labels)

    def test_optimization_preserves_bijection_membership_and_exact_score(self):
        source, target, source_labels, target_labels, options = self.fixture()
        before = [array.copy() for array in (source, target, source_labels, target_labels)]
        pool, stats = tg_untangle.optimize_permutations(
            source, target, source_labels, target_labels, 2, options, 9)
        self.assertEqual(pool.shape, (options["permutations"], len(source)))
        self.assert_valid_pool(pool, source_labels, target_labels)
        self.assertGreater(stats["accepted_swaps"], 0)
        self.assertEqual(stats["proposed_swaps"], options["permutations"] * options["swap_steps"])
        edges = tg_untangle.build_source_edges(source, options["neighbors"])
        for permutation, record in zip(pool, stats["per_permutation"]):
            self.assertLessEqual(record["final_score"], record["initial_score"] + 1e-12)
            self.assertAlmostEqual(record["final_score"],
                                   brute_score(source, target, permutation, edges, options), places=13)
        self.assertAlmostEqual(stats["final_score_mean"],
                               np.mean([r["final_score"] for r in stats["per_permutation"]]), places=13)
        for array, original in zip((source, target, source_labels, target_labels), before):
            np.testing.assert_array_equal(array, original)

    def test_determinism_and_numpy_global_rng_isolation(self):
        source, target, source_labels, target_labels, options = self.fixture()
        global_state = np.random.get_state()
        pool, stats = tg_untangle.optimize_permutations(
            source, target, source_labels, target_labels, 2, options, 9)
        after = np.random.get_state()
        self.assertEqual(global_state[0], after[0])
        np.testing.assert_array_equal(global_state[1], after[1])
        self.assertEqual(global_state[2:], after[2:])
        repeated, repeated_stats = tg_untangle.optimize_permutations(
            source, target, source_labels, target_labels, 2, options, 9)
        np.testing.assert_array_equal(pool, repeated)
        self.assertEqual(stats, repeated_stats)

    def test_zero_swaps_pool_exposes_same_initial_scores_as_search(self):
        source, target, source_labels, target_labels, options = self.fixture()
        no_swaps = {**options, "swap_steps": 0}
        pool, stats = tg_untangle.optimize_permutations(
            source, target, source_labels, target_labels, 2, no_swaps, 9)
        _, optimized = tg_untangle.optimize_permutations(
            source, target, source_labels, target_labels, 2, options, 9)
        self.assert_valid_pool(pool, source_labels, target_labels)
        self.assertEqual(stats["accepted_swaps"], 0)
        self.assertEqual(stats["proposed_swaps"], 0)
        edges = tg_untangle.build_source_edges(source, options["neighbors"])
        for permutation, record, refined in zip(pool, stats["per_permutation"], optimized["per_permutation"]):
            self.assertEqual(record["initial_score"], record["final_score"])
            self.assertEqual(record["initial_score"], refined["initial_score"])
            self.assertAlmostEqual(record["final_score"],
                                   brute_score(source, target, permutation, edges, no_swaps), places=13)

    def test_singleton_patches_cannot_make_swaps(self):
        source = np.array([[0., 0.], [1., 0.], [0., 1.]])
        source_labels, target_labels = np.arange(3), np.array([2, 0, 1])
        pool, stats = tg_untangle.optimize_permutations(
            source, source[::-1].copy(), source_labels, target_labels, 3,
            {"neighbors": 1, "permutations": 2, "swap_steps": 30}, 3)
        self.assert_valid_pool(pool, source_labels, target_labels)
        self.assertEqual(stats["proposed_swaps"], 0)
        self.assertEqual(stats["accepted_swaps"], 0)

    def test_n256_k8_pool_has_finite_nonincreasing_objective(self):
        rng = np.random.default_rng(102)
        source = rng.normal(size=(256, 2)).astype(np.float32)
        source_labels = np.repeat(np.arange(8), 32)
        target_labels = source_labels[::-1].copy()
        angles = target_labels * np.pi / 4
        target = np.column_stack((np.cos(angles), np.sin(angles))) + .12 * rng.normal(size=(256, 2))
        pool, stats = tg_untangle.optimize_permutations(
            source, target, source_labels, target_labels, 8,
            {"permutations": 2, "swap_steps": 16}, 5)
        self.assert_valid_pool(pool, source_labels, target_labels)
        self.assertTrue(np.isfinite(stats["final_score_mean"]))
        for record in stats["per_permutation"]:
            self.assertLessEqual(record["final_score"], record["initial_score"] + 1e-12)

    def test_options_validation_rejects_ambiguous_or_invalid_settings(self):
        invalid = [0, "true", {"enabled": 1}, {"unknown": 2}, {"neighbors": 0},
                   {"neighbors": True}, {"permutations": 0}, {"swap_steps": -1},
                   {"seed": -1}, {"times": []}, {"times": [0.]}, {"times": [1.]},
                   {"times": [.2, .2]}, {"times": [float("nan")]},
                   {"lipschitz": -1.}, {"lipschitz": float("inf")},
                   {"times": [.2, .4], "lipschitz": [1.]}]
        for value in invalid:
            with self.subTest(options=value), self.assertRaises(ValueError):
                tg_untangle.normalize_options(value)

    def test_score_validation_rejects_bad_arrays_permutations_and_edges(self):
        source, target, _, _, options = self.fixture()
        valid = np.arange(len(source))
        edges = tg_untangle.build_source_edges(source, 2)
        invalid_points = source.copy()
        invalid_points[0, 0] = np.nan
        bad_inputs = [(invalid_points, target, valid, edges),
                      (source, target[:-1], valid, edges),
                      (source, target, np.zeros(len(source), dtype=int), edges),
                      (source, target, valid.astype(float), edges),
                      (source, target, valid, np.array([[0, 0]])),
                      (source, target, valid, np.array([[0, 1], [1, 0]])),
                      (source, target, valid, np.array([[0, len(source)]]))]
        for args in bad_inputs:
            with self.subTest(inputs=args), self.assertRaises(ValueError):
                tg_untangle.conflict_score(*args, options)
        for neighbors in (0, len(source), True):
            with self.subTest(neighbors=neighbors), self.assertRaises(ValueError):
                tg_untangle.build_source_edges(source, neighbors)

    def test_search_validation_rejects_bad_labels_and_disabled_search(self):
        source, target, source_labels, target_labels, options = self.fixture()
        invalid = [(source_labels.astype(float), target_labels, 2),
                   (source_labels[:-1], target_labels[:-1], 2),
                   (source_labels, np.zeros_like(target_labels), 2),
                   (source_labels, np.full_like(target_labels, 2), 2),
                   (source_labels, target_labels, 0),
                   (source_labels, target_labels, len(source) + 1)]
        for labels_a, labels_b, k in invalid:
            with self.subTest(labels=(labels_a, labels_b, k)), self.assertRaises(ValueError):
                tg_untangle.optimize_permutations(source, target, labels_a, labels_b, k, options, 9)
        with self.assertRaises(ValueError):
            tg_untangle.optimize_permutations(source, target, source_labels, target_labels,
                                              2, False, 9)
        with self.assertRaises(ValueError):
            tg_untangle.optimize_permutations(source, target, source_labels, target_labels,
                                              2, options, -1)


if __name__ == "__main__":
    unittest.main()
