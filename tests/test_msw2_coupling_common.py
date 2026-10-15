import unittest

import numpy as np
import torch

from audit_ms_w2_coupling import _validate_options
from metrics import checkerboard_metrics
from msw2_coupling_common import SpatialRegions, routing_regions
from train_horse import load_horse_mask


class SpatialCouplingDiagnosticsTests(unittest.TestCase):
    def test_checker_mass_uses_all_points_and_matches_original_support(self):
        spatial = SpatialRegions({"data": {"grid_size": 4}}, "checkerboard", grid_bins=4)
        points = np.array([[-.75, .75], [.25, .75], [-.25, .75], [2., 0.], [1., -1.]])
        leakage, _, _ = checkerboard_metrics(torch.tensor(points), 4, 8)
        self.assertAlmostEqual(1 - spatial.support(points).mean(), leakage, places=6)
        masses = dict(zip(spatial.region_names, spatial.mass(points)))
        self.assertAlmostEqual(masses["cell_0_0"], .2)
        self.assertAlmostEqual(sum(masses[n] for n in spatial.region_names if n.startswith("cell_")), masses["foreground"])
        self.assertAlmostEqual(masses["foreground"] + masses["leakage"], 1.)
        self.assertAlmostEqual(spatial.expected_grid_mass.sum(), 1.)
        self.assertEqual(spatial.expected_grid_mass[-1], 0.)
        self.assertAlmostEqual(spatial.grid_hist(points)[-1], .2)

    def test_wide_path_grid_distinguishes_gaussian_tails(self):
        spatial = SpatialRegions({"data": {"grid_size": 4}}, "checkerboard")
        first, second = np.array([[-3., 0.]]), np.array([[-2., 0.]])
        np.testing.assert_array_equal(spatial.grid_hist(first), spatial.grid_hist(second))
        self.assertAlmostEqual(.5 * np.abs(spatial.path_hist(first) - spatial.path_hist(second)).sum(), 1.)
        self.assertAlmostEqual(spatial.path_hist(np.array([[5., 0.]]))[-1], 1.)
        self.assertEqual(spatial.definition()["path_histogram"]["range"], [-4., 4.])

    def test_horse_fixed_roi_measurements_match_existing_metrics(self):
        mask = load_horse_mask("cpu", torch.float32)
        spatial = SpatialRegions({"data": {}}, "horse", mask, grid_bins=16)
        points = np.random.default_rng(14).uniform(-1., 1., (300, 2))
        masses = dict(zip(spatial.region_names, spatial.mass(points)))
        scores = spatial.horse.score(torch.tensor(points))
        for roi in spatial.horse.regions:
            suffix = "_mass" if roi["kind"] == "foreground" else "_leakage"
            self.assertAlmostEqual(masses[roi["name"]], scores["roi_" + roi["name"] + suffix])
        self.assertAlmostEqual(spatial.expected_grid_mass.sum(), 1.)
        self.assertAlmostEqual(spatial.expected_mass[spatial.region_names.index("front_legs")], .05949967750852301)

    def test_routing_se_uses_query_level_contributions(self):
        spatial = SpatialRegions({"data": {"grid_size": 4}}, "checkerboard", grid_bins=4)
        points = np.array([[-.75, .75], [.25, .75]])
        routed = {"q1": np.array([.5, .5]), "q2": np.array([.5, .5]),
                  "query_neighbor_indices": np.array([[0, 1], [0, 1], [1, 0], [1, 0]])}
        summary, rows, arrays = routing_regions(points, points, spatial, routed, (1, 2))
        first = next(r for r in rows if r["region"] == "cell_0_0" and r["neighbors"] == 1)
        second = next(r for r in rows if r["region"] == "cell_0_0" and r["neighbors"] == 2)
        self.assertAlmostEqual(first["query_mc_se"], np.std([1, 1, 0, 0], ddof=1) / 2)
        self.assertEqual(second["query_mc_se"], 0.)
        self.assertEqual(first["bank_point_mass"], .5)
        self.assertEqual(first["target_expected_mass"], .125)
        self.assertAlmostEqual(arrays["routing_grid_k2"].sum(), 1.)
        self.assertEqual(summary["1"]["grid_tv_vs_empirical_target_bank"], 0.)

    def test_invalid_options_fail_in_preflight(self):
        options = dict(clouds=2, batch_size=1, neighbors=2, distance_bins=3,
                       routing_draws=32, routing_chunk_size=7, anchors=4, contexts=2,
                       path_steps=2, grid_bins=4, seed=2026, routing_neighbors=[1, 2],
                       nfes=[1, 2], times=[0., .5, 1.])
        _validate_options(options, 16)
        for key, bad in (("contexts", 1), ("neighbors", 16), ("nfes", [0]),
                         ("routing_neighbors", [2]), ("times", [0., 0.]), ("seed", -1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                _validate_options({**options, key: bad}, 16)


if __name__ == "__main__":
    unittest.main()
