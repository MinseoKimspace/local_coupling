"""Behavioral checks for path-affine coupling, independent of cache IO."""

import json
import unittest

import numpy as np

import tg_local


class LocalAffineScoreTests(unittest.TestCase):
    def test_affine_contraction_rotation_and_translation_are_explained(self):
        rng = np.random.default_rng(6)
        points = rng.normal(size=(64, 2))
        affine = np.array([[-3.0, -0.7], [0.7, -0.4]])
        velocity = points @ affine.T + np.array([10.0, -8.0])
        score, coverage = tg_local.local_affine_score(points, velocity, ridge=1e-8)
        random_score, _ = tg_local.local_affine_score(points, rng.normal(size=(64, 2)), ridge=1e-8)
        self.assertLess(score, 1e-9)
        self.assertGreater(random_score, 0.1)
        self.assertEqual(coverage, 1.0)
        # A common-velocity penalty would heavily penalize this necessary field.
        self.assertGreater(np.mean(np.sum((velocity - velocity.mean(0)) ** 2, axis=1)), 1.0)

    def test_actual_paths_are_scored_before_average_and_times_are_used(self):
        rng = np.random.default_rng(43)
        source = rng.normal(size=(32, 2))
        target = rng.normal(size=(32, 2))
        permutations = np.stack([rng.permutation(32), rng.permutation(32)])
        options = tg_local.settings({"times": [0., .5, .9], "neighbors": 8})
        score, by_time, coverage = tg_local.path_score(source, target, permutations, options)
        normalization = np.mean(np.sum((target - target.mean(0)) ** 2, axis=1))
        explicit = []
        for permutation in permutations:
            velocity = target[permutation] - source
            explicit.append([tg_local.local_affine_score(
                source + time * velocity, velocity, neighbors=8,
                ridge=options["ridge"], normalization=normalization)[0]
                             for time in options["times"]])
        self.assertAlmostEqual(score, np.mean(explicit), places=12)
        np.testing.assert_allclose(by_time, np.mean(explicit, axis=0))
        self.assertEqual(coverage, [1., 1., 1.])
        self.assertGreater(np.ptp(by_time), 1e-3)
        mean_velocity = target[permutations].mean(0) - source
        averaged_first = tg_local.local_affine_score(
            source, mean_velocity, neighbors=8, ridge=options["ridge"], normalization=normalization)[0]
        self.assertGreater(abs(by_time[0] - averaged_first), 1e-3)

    def test_translation_is_unbiased_and_small_neighborhoods_are_reported(self):
        points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        velocity = np.tile([4., -2.], (4, 1))
        score, coverage = tg_local.local_affine_score(points, velocity)
        self.assertAlmostEqual(score, 0.)
        self.assertEqual(coverage, 1.)
        gate = np.zeros((4, 4), dtype=bool)
        score, coverage = tg_local.local_affine_score(points, velocity, allowed=gate)
        self.assertEqual((score, coverage), (0., 0.))

    def test_fixed_target_gate_coverage_is_invariant_across_actual_bijections(self):
        rng = np.random.default_rng(45)
        source, target = rng.normal(size=(32, 2)), rng.normal(size=(32, 2))
        gate = rng.random((32, 32)) < .18
        gate &= gate.T
        np.fill_diagonal(gate, False)
        permutations = np.stack([rng.permutation(32) for _ in range(5)])
        coverage = [tg_local.path_score(source, target, row[None], target_gate=gate)[2]
                    for row in permutations]
        np.testing.assert_array_equal(coverage, np.broadcast_to(coverage[0], (5, 4)))


class LocalCouplingTests(unittest.TestCase):
    @staticmethod
    def cloud():
        axis = np.linspace(-1., 1., 16)
        source = np.column_stack((np.repeat([-.04, .04], 16), np.tile(axis, 2))).astype(np.float32)
        target = np.column_stack((np.repeat([-.4, .4], 16), np.tile(.6 * axis, 2))).astype(np.float32)
        labels = np.repeat([0, 1], 16).astype(np.int32)
        return source, target, labels, labels.copy()

    @staticmethod
    def options(**overrides):
        return tg_local.settings({"num_tables": 3, "neighbors": 8,
                                  "score_permutations": 2, "candidate_count": 4,
                                  "max_swaps": 2, "max_cost_increase": .5,
                                  "destination_budget": 1., **overrides})

    def test_table_capacities_cost_budget_rng_isolation_and_determinism(self):
        for subpatches in (1, 2):
            args = self.cloud()
            original = tuple(item.copy() for item in args)
            options = self.options(subpatches=subpatches)
            np.random.seed(123)
            state = np.random.get_state()
            first = tg_local.build_tables(*args, 2, options, seed=7)
            repeated = tg_local.build_tables(*args, 2, options, seed=7)
            tables, target_labels, stats = first
            np.testing.assert_array_equal(tables, repeated[0])
            np.testing.assert_array_equal(target_labels, repeated[1])
            self.assertEqual(stats, repeated[2])
            for item, before in zip(args, original):
                np.testing.assert_array_equal(item, before)
            after = np.random.get_state()
            self.assertEqual(after[0], state[0])
            np.testing.assert_array_equal(after[1], state[1])
            self.assertEqual(after[2:], state[2:])
            self.assertEqual(tables.dtype, np.int32)
            self.assertEqual(target_labels.dtype, np.int32)
            self.assertEqual(tables.shape, (3, 32))
            np.testing.assert_array_equal(tables[0] // subpatches, args[2])
            np.testing.assert_array_equal(target_labels // subpatches, args[3])
            counts = np.bincount(target_labels, minlength=2 * subpatches)
            self.assertTrue(np.all(counts > 0))
            for row in tables:
                np.testing.assert_array_equal(np.bincount(row, minlength=len(counts)), counts)
            for trace in stats["score_trace_per_table"]:
                self.assertTrue(np.all(np.diff(trace) <= 1e-10))
            self.assertLessEqual(stats["guided_score"], stats["baseline_score"] + 1e-10)
            self.assertLessEqual(max(stats["centroid_cost_per_table"]),
                                 stats["baseline_centroid_cost"] * (1 + options["max_cost_increase"]) + 1e-10)
            self.assertLessEqual(stats["max_destination_rms"], stats["destination_rms_limit"] + 1e-10)
            self.assertLessEqual(max(stats["expected_fine_cost_per_table"]),
                                 stats["baseline_expected_fine_cost"] * (1 + options["max_cost_increase"]) + 1e-10)
            np.testing.assert_array_equal(stats["scored_fraction_by_time"],
                                          stats["baseline_scored_fraction_by_time"])
            self.assertEqual(stats["swaps_per_table"][0], 0)
            self.assertTrue(np.isfinite(stats["heldout_guided_score"]))
            json.dumps(stats, allow_nan=False)

    def test_rank_children_match_unequal_target_capacities_without_point_matching(self):
        target = np.array([[3., 0.], [1., 0.], [2., 0.], [0., 0.], [4., 0.]])
        source = np.array([[8., 3.], [-2., 1.], [2., 0.], [9., 4.], [1., 2.]])
        parent = np.zeros(5, dtype=np.int32)
        child_target = tg_local.target_children(target, parent, 1, 2)
        child_source = tg_local.source_children(source, parent, target, child_target, 1, 2)
        np.testing.assert_array_equal(np.bincount(child_target), [3, 2])
        np.testing.assert_array_equal(np.bincount(child_source), [3, 2])
        np.testing.assert_array_equal(child_target, [1, 0, 0, 0, 1])
        np.testing.assert_array_equal(child_source, [1, 0, 0, 1, 0])
        self.assertLess(tg_local._expected_fine_cost(source, target, child_source, child_target, 2),
                        tg_local._expected_fine_cost(source, target, parent, parent, 1))

    def test_scoring_samples_are_bijections_and_not_a_fixed_pairing(self):
        source, target, source_labels, target_labels = self.cloud()
        tables, fine_target, _ = tg_local.build_tables(
            source, target, source_labels, target_labels, 2,
            self.options(subpatches=2, max_swaps=0), seed=2)
        permutations = tg_local._permutations(tables[0], fine_target, 4, 20, 55)
        self.assertGreater(len(set(map(tuple, permutations))), 1)
        for permutation in permutations:
            np.testing.assert_array_equal(np.sort(permutation), np.arange(32))
            np.testing.assert_array_equal(fine_target[permutation], tables[0])
        # Uniform target choices are not selected by their score in this draw.
        repeated = tg_local._permutations(tables[0], fine_target, 4, 2000, 5)
        eligible = np.flatnonzero(fine_target == tables[0, 0])
        observed = np.bincount(repeated[:, 0], minlength=32)[eligible]
        self.assertTrue(np.all(np.abs(observed - 250) < 65))

    def test_transport_retains_unchanged_pairs_and_uniform_new_bijection_law(self):
        old = np.repeat([0, 1], 4).astype(np.int32)
        new = old.copy()
        new[[0, 1, 4, 5]] = new[[4, 5, 0, 1]]
        before = tg_local._permutations(old, old, 2, 4000, 19)
        after = tg_local._transport_permutations(before, old, new, 2, 20)
        np.testing.assert_array_equal(after[:, old == new], before[:, old == new])
        np.testing.assert_array_equal(np.sort(after, axis=1), np.broadcast_to(np.arange(8), after.shape))
        np.testing.assert_array_equal(old[after], np.broadcast_to(new, after.shape))
        # An incoming member remains uniform over all targets of its new group.
        observed = np.bincount(after[:, 0], minlength=8)[4:]
        self.assertTrue(np.all(np.abs(observed - 1000) < 120))
        for row in range(8):
            for group in (0, 1):
                released = before[row, (old != new) & (old == group)]
                received = after[row, (old != new) & (new == group)]
                np.testing.assert_array_equal(np.sort(released), np.sort(received))

    def test_target_gap_gate_prevents_cross_patch_coarse_swaps(self):
        source, target, source_labels, target_labels = self.cloud()
        target[:, 0] = np.repeat([-1., 1.], 16)
        target[:, 1] *= 1e-4
        tables, _, stats = tg_local.build_tables(
            source, target, source_labels, target_labels, 2,
            self.options(max_cost_increase=10., destination_budget=10.), seed=8)
        self.assertEqual(stats["allowed_target_patch_pairs"], [])
        self.assertEqual(stats["eligible_source_edges"], 0)
        self.assertEqual(stats["accepted_swaps"], 0)
        np.testing.assert_array_equal(tables, np.broadcast_to(source_labels, tables.shape))

    def test_zero_budget_or_swaps_returns_explicit_fallback(self):
        for override in ({"max_swaps": 0}, {"destination_budget": 0.}):
            args = self.cloud()
            tables, _, stats = tg_local.build_tables(*args, 2, self.options(**override), seed=7)
            np.testing.assert_array_equal(tables, np.broadcast_to(args[2], tables.shape))
            self.assertEqual(stats["accepted_swaps"], 0)
            self.assertTrue(stats["no_feasible_change"])
            self.assertAlmostEqual(stats["baseline_score"], stats["guided_score"])
            self.assertAlmostEqual(stats["heldout_baseline_score"], stats["heldout_guided_score"])

    def test_degenerate_single_point_and_duplicate_coordinates_are_finite(self):
        for n in (1, 8):
            source, target = np.zeros((n, 2)), np.zeros((n, 2))
            labels = np.zeros(n, dtype=np.int32)
            tables, fine_target, stats = tg_local.build_tables(source, target, labels, labels, 1)
            self.assertEqual(stats["accepted_swaps"], 0)
            self.assertEqual(stats["guided_score"], 0.)
            np.testing.assert_array_equal(tables, np.zeros_like(tables))
            np.testing.assert_array_equal(fine_target, labels)
            json.dumps(stats, allow_nan=False)

    def test_bad_configuration_and_unbalanced_inputs_fail(self):
        bad = ({"unknown": 3}, {"subpatches": 3}, {"neighbors": 2},
               {"score_permutations": True}, {"num_tables": 1}, {"max_swaps": -1},
               {"ridge": 0}, {"ridge": float("nan")}, {"destination_budget": -1},
               {"times": []}, {"times": [0., 1.]}, {"times": [.5, 0.]},
               {"times": [0., 0.]}, {"times": [True]})
        for options in bad:
            with self.subTest(options=options), self.assertRaises(ValueError):
                tg_local.settings(options)
        args = list(self.cloud())
        args[3][0] = 1
        with self.assertRaisesRegex(ValueError, "capacities"):
            tg_local.build_tables(*args, 2)
        source = target = np.zeros((1, 2))
        labels = np.zeros(1, dtype=np.int32)
        with self.assertRaisesRegex(ValueError, "subpatches"):
            tg_local.build_tables(source, target, labels, labels, 1, {"subpatches": 2})


if __name__ == "__main__":
    unittest.main()
