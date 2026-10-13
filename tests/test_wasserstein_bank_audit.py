import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import wasserstein_bank as bank


class WassersteinBankAuditTests(unittest.TestCase):
    @staticmethod
    def config():
        return {
            "dataset": "checkerboard", "seed": 0, "coupling": bank.METHOD,
            "data": {"n_points": 4, "grid_size": 4},
            "model": {"point_dim": 2},
            "wasserstein_bank": {
                "cache_seed": 9, "superset_size": 12, "pool_per_shape": 8,
                "subset_size": 3, "scales": [2, 4], "repetitions": 1,
                "alpha_values": [0.25, 1., 4.], "variance_times": [0., 0.2, 1.],
                "bandwidths": [0.1], "variance_bandwidth": 0.1,
                "candidates": [{"kind": "random"}, {"kind": "exact"},
                               {"kind": "perturbed_exact", "noise_std": 0.1},
                               {"kind": "perturbed_exact", "noise_std": 0.7}],
            },
        }

    def test_time_integral_weights_nonuniform_grid(self):
        curve = [{"time": t, "mse_per_coordinate": t} for t in [0., 0.1, 1.]]
        self.assertAlmostEqual(bank.integrated_curve(curve), 0.5)
        restricted = [{"time": t, "value": t} for t in [0.2, 0.3, 0.8]]
        self.assertAlmostEqual(bank.integrated_curve(restricted, "value"), 0.5)
        for bad in (curve[:1], [curve[1], curve[0]], [curve[0], curve[0]],
                    [{"time": 0., "value": 0.}, {"time": 1., "value": np.nan}]):
            with self.subTest(curve=bad), self.assertRaises(ValueError):
                bank.integrated_curve(bad, "value" if "value" in bad[0] else "mse_per_coordinate")

    def test_exact_t0_floor_and_excess_risk_identity(self):
        source = np.array([[0., 0.], [1., 0.], [0., 1.]])
        speed = np.array([2., -1.])
        velocities = np.stack([np.tile(speed, (3, 1)), np.tile(-speed, (3, 1))])
        floor = bank.fixed_bank_t0_bayes_mse(source, velocities)
        self.assertAlmostEqual(floor, np.square(speed).mean())
        prediction = np.array([[0.2, 0.7], [1.2, -2.], [0., 0.]])
        observed_risk = np.square(prediction[None] - velocities).mean()
        excess = np.square(prediction - velocities.mean(0)).mean()
        self.assertAlmostEqual(observed_risk - floor, excess)
        # One finite deterministic map has zero irreducible risk even if rough.
        self.assertEqual(bank.fixed_bank_t0_bayes_mse(source, prediction[None]), 0.)

    def test_t0_floor_pools_duplicate_source_coordinates(self):
        source = np.array([[0., 0.], [0., 0.]])
        velocities = np.array([[[1., 0.], [-1., 0.]]])
        self.assertAlmostEqual(bank.fixed_bank_t0_bayes_mse(source, velocities), 0.5)

    def test_exact_joint_alpha_limits_have_correct_scaling(self):
        source = np.array([[0., 0.], [2., 1.], [-1., 3.], [0.5, -2.]])
        velocity = np.array([[2., -1.], [-1., 0.], [1., 4.], [-3., 1.]])
        subset = np.array([0, 2])
        def joint(alpha):
            features = np.concatenate([source, alpha * velocity], axis=1)
            return bank.wasserstein_squared(features[subset], features)
        x_score = bank.wasserstein_squared(source[subset], source)
        u_score = bank.wasserstein_squared(velocity[subset], velocity)
        self.assertAlmostEqual(joint(1e-6), x_score, places=8)
        self.assertAlmostEqual(joint(1e5) / 1e10, u_score, places=8)

    def test_score_audit_transport_identity_and_shared_endpoint_mean(self):
        config = self.config()
        opts = bank.settings(config)
        source, targets = bank.draw_supersets(config, "checkerboard", 17)
        permutations = np.stack([np.stack([bank.candidate_permutation(source, targets[0], candidate,
                                                                        np.random.default_rng(3))])
                                 for candidate in opts["candidates"]])
        report = bank.score_banks(source, targets, permutations, opts, 21)
        expected_mean = targets.astype(np.float64).mean((0, 1)) - source.astype(np.float64).mean(0)
        for row in report["candidates"]:
            np.testing.assert_allclose(row["mean_displacement"], expected_mean, atol=1e-14)
            self.assertAlmostEqual(row["transport"], row["displacement_covariance_trace"] + np.square(expected_mean).sum())
            self.assertEqual(row["t0_fixed_bank_bayes_mse_per_coordinate"], 0.)
            self.assertFalse(row["t0_floor_informative"])
            self.assertEqual(row["path_joint"], bank.integrated_curve(row["path_joint_by_time"], "score"))
            self.assertEqual(row["joint"], row["joint_t0_by_alpha"]["1.0"])
        self.assertIn("spearman_with_transport", report["correlations"]["displacement"])
        self.assertEqual(set(report["alpha_rank_correlations"]), {"0.25", "1.0", "4.0"})

    def test_candidate_override_validation_and_cache_identity(self):
        config = self.config()
        expected_spec = bank.cache_spec(config, "checkerboard")
        for candidate_id in range(4):
            config["wasserstein_bank"]["candidate_id"] = candidate_id
            self.assertEqual(bank.cache_spec(config, "checkerboard"), expected_spec)
        for invalid in [-1, 4, True, 1.0, None]:
            config["wasserstein_bank"]["candidate_id"] = invalid
            with self.subTest(candidate=invalid), self.assertRaises(ValueError):
                bank.settings(config)
        for invalid in [[0.], [-1.], [1., 1.], [np.nan], []]:
            config = self.config()
            config["wasserstein_bank"]["alpha_values"] = invalid
            with self.subTest(alpha=invalid), self.assertRaises(ValueError):
                bank.settings(config)

    def test_prepared_cost_noise_is_shared_and_override_selects_explicit_bank(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as folder:
            config["wasserstein_bank"]["cache"] = str(Path(folder) / "bank.npz")
            with contextlib.redirect_stdout(io.StringIO()):
                bank.prepare(config, "checkerboard")
            arrays, metadata, _ = bank.load_cache(config, "checkerboard")
            self.assertEqual(metadata["spec"]["format_version"], 2)
            for prefix, offset in [("", 0), ("validation_", 100003)]:
                for candidate_id in [2, 3]:
                    candidate = config["wasserstein_bank"]["candidates"][candidate_id]
                    expected = bank.candidate_permutation(arrays[prefix + "source"], arrays[prefix + "targets"][0],
                                                          candidate, np.random.default_rng(9 + offset + 1500))
                    np.testing.assert_array_equal(arrays[prefix + "permutations"][candidate_id, 0], expected)
            # Choose a bank deliberately different from the default joint winner.
            config["wasserstein_bank"]["candidate_id"] = (metadata["selected"]["joint"] + 1) % 4
            sampler = bank.WassersteinBankSampler(config, "checkerboard", "cpu", torch.float32)
            self.assertEqual(sampler.selected, config["wasserstein_bank"]["candidate_id"])
            self.assertEqual(sampler.details()["selector"], "candidate")
            self.assertEqual(sampler.details()["selection_mode"], "explicit_candidate")
            self.assertEqual(sampler.details()["selected_candidate"], sampler.selected)
            _, target = sampler.sample(2, generator=torch.Generator().manual_seed(4))
            generator = torch.Generator().manual_seed(4)
            shapes = torch.randint(1, (2,), generator=generator)
            indices = torch.randint(12, (2, 4), generator=generator)
            destination = sampler.arrays["permutations"][sampler.selected, shapes[:, None], indices]
            torch.testing.assert_close(target, sampler.arrays["targets"][shapes[:, None], destination])


if __name__ == "__main__":
    unittest.main()
