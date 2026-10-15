import unittest

import numpy as np

from msw2_coupling_geometry import gaussian_routing_audit, geometry_audit


class GeometryAuditTests(unittest.TestCase):
    def setUp(self):
        self.source = np.array([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]])
        self.levels = [np.array([0, 0, 1, 1]), np.arange(4)]

    def test_constant_translation_has_no_velocity_jumps(self):
        summary, rows, arrays = geometry_audit(
            self.source, self.source + [3., 2.], self.levels,
            neighbors=2, times=[0., .5, 1.], distance_bins=2)
        np.testing.assert_array_equal(arrays["velocity_jump"], 0.)
        np.testing.assert_array_equal(arrays["point_velocity_ratio_mean"], 0.)
        self.assertEqual(summary["source_edges"]["velocity_jump"]["max"], 0.)
        self.assertTrue(all(r["velocity_jump_max"] == 0 for r in rows["time_conflicts"]))
        self.assertTrue(np.all(arrays["edge_i"] < arrays["edge_j"]))
        edges = np.column_stack((arrays["edge_i"], arrays["edge_j"]))
        self.assertEqual(len(edges), len(np.unique(edges, axis=0)))

    def test_scaling_known_norm_and_point_arrays(self):
        summary, _, arrays = geometry_audit(
            self.source, 2 * self.source, self.levels, neighbors=2, times=[0.])
        np.testing.assert_allclose(arrays["target_distance"], 2 * arrays["source_distance"])
        np.testing.assert_allclose(arrays["velocity_jump"], arrays["source_distance"])
        np.testing.assert_allclose(arrays["velocity_ratio"], 1., atol=1e-10)
        np.testing.assert_array_equal(arrays["point_boundary_fraction"], 1.)
        self.assertGreater(summary["ratio_epsilon"], 0.)

    def test_distance_strata_do_not_invent_missing_groups(self):
        summary, rows, _ = geometry_audit(
            self.source, 2 * self.source, self.levels, neighbors=2,
            times=[0.], distance_bins=3)
        coarse, finest = summary["partition_comparisons"]
        self.assertEqual(finest["same_edges"], 0)
        self.assertEqual(finest["matched_edge_fraction"], 0.)
        self.assertIsNone(finest["distance_standardized_means"]["velocity_jump"]["same"])
        self.assertTrue(finest["unmatched_distance_bins"])
        self.assertEqual(sum(r["same_count"] + r["cross_count"]
                             for r in rows["distance_strata"] if r["level"] == 0),
                         coarse["same_edges"] + coarse["cross_edges"])

    def test_interpolant_coincidences_counted_and_not_self_edges(self):
        summary, rows, arrays = geometry_audit(
            self.source, -self.source, self.levels, neighbors=2, times=[.5])
        self.assertEqual(rows["time_conflicts"][0]["zero_distance_edge_count"],
                         rows["time_conflicts"][0]["edge_count"])
        self.assertTrue(np.all(arrays["time_0_edge_i"] != arrays["time_0_edge_j"]))
        self.assertTrue(np.isfinite(arrays["time_0_velocity_ratio"]).all())
        self.assertIsNone(rows["time_conflicts"][0]["velocity_ratio_max"])
        self.assertEqual(rows["time_conflicts"][0]["velocity_ratio_count"], 0)
        self.assertGreater(rows["time_conflicts"][0]["coincident_velocity_jump_max"], 0.)
        self.assertGreater(arrays["time_0_velocity_ratio"].max(), 1e10)
        self.assertGreater(summary["ratio_epsilon"], 0.)

    def test_inputs_unchanged_and_local_state(self):
        before = self.source.copy()
        labels_before = [a.copy() for a in self.levels]
        geometry_audit(self.source, 2 * self.source, self.levels, neighbors=2, times=[0.])
        np.testing.assert_array_equal(self.source, before)
        for a, b in zip(self.levels, labels_before):
            np.testing.assert_array_equal(a, b)

    def test_invalid_geometry_inputs(self):
        for kwargs in ({"neighbors": True}, {"neighbors": 4},
                       {"distance_bins": 0}, {"times": []},
                       {"times": [.5, .5]}, {"times": [np.inf]},
                       {"times": ["bad"]}, {"times": [1j]},
                       {"times": [True]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                geometry_audit(self.source, self.source, self.levels, **kwargs)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            geometry_audit(np.zeros((4, 2)), self.source, self.levels, neighbors=2)
        with self.assertRaisesRegex(ValueError, "nested"):
            geometry_audit(self.source, self.source,
                           [self.levels[0], np.array([0, 1, 0, 1])], neighbors=2)
        with self.assertRaisesRegex(ValueError, "labels"):
            geometry_audit(self.source, self.source, [np.zeros(4, dtype=bool)], neighbors=2)


class GaussianRoutingTests(unittest.TestCase):
    def setUp(self):
        self.source = np.array([[-1., 0.], [1., 0.]])

    def test_weight_conservation_and_region_query_contributions(self):
        summary, arrays = gaussian_routing_audit(
            self.source, self.source, draws=2000, neighbors=(1, 2), seed=10, chunk_size=143)
        for k in (1, 2):
            self.assertEqual(arrays[f"count{k}"].sum(), 2000 * k)
            self.assertAlmostEqual(arrays[f"q{k}"].sum(), 1.)
            self.assertEqual(summary["allocations"][str(k)]["incidence_sum"], 2000 * k)
            query_contribution = (arrays["query_neighbor_indices"][:, :k] == 0).mean(1)
            self.assertAlmostEqual(query_contribution.mean(), arrays[f"q{k}"][0])
        np.testing.assert_array_equal(arrays["q2"], [.5, .5])
        self.assertAlmostEqual(summary["allocations"]["2"]["effective_bank_points"], 2.)
        self.assertEqual(summary["allocations"]["2"]["js_nats_vs_uniform"], 0.)
        self.assertAlmostEqual(arrays["q1"][0], .5, delta=.05)

    def test_chunk_size_does_not_change_draws_or_routes(self):
        source = np.random.default_rng(2).standard_normal((20, 2))
        a, aa = gaussian_routing_audit(source, source, draws=1000,
                                     neighbors=(1, 8), seed=21, chunk_size=1)
        b, bb = gaussian_routing_audit(source, source, draws=1000,
                                     neighbors=(1, 8), seed=21, chunk_size=113)
        for key in aa:
            np.testing.assert_array_equal(aa[key], bb[key])
        self.assertEqual(a["gaussian_query_sha256"], b["gaussian_query_sha256"])
        self.assertEqual(a["allocations"], b["allocations"])

    def test_does_not_reset_global_numpy_rng(self):
        np.random.seed(91)
        before = np.random.get_state()
        gaussian_routing_audit(self.source, self.source, draws=20, neighbors=(1, 2))
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_invalid_routing_parameters(self):
        for kwargs in ({"draws": 1}, {"draws": True}, {"neighbors": ()},
                       {"neighbors": (1, 1)}, {"neighbors": (3,)},
                       {"neighbors": (True,)}, {"seed": -1}, {"chunk_size": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                gaussian_routing_audit(self.source, self.source, **kwargs)
        with self.assertRaisesRegex(ValueError, "same shape"):
            gaussian_routing_audit(self.source, self.source[:1], neighbors=(1,))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            gaussian_routing_audit(np.zeros((2, 2)), self.source, neighbors=(1,))


if __name__ == "__main__":
    unittest.main()
