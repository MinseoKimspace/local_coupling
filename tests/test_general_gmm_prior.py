"""Ordinary learned-weight/full-covariance GMM sampling and preparation checks."""

import copy
import contextlib
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import gmm_prior


def config(dataset="checkerboard"):
    value = {"coupling": "nsot", "dtype": "float32", "num_regions": 2,
             "data": {"n_points": 128}, "model": {"point_dim": 2},
             "anchor_flow": {"mode": "gmm_prior", "reference_points": 256, "seed": 11,
                             "n_init": 1, "max_iter": 300, "tol": 1e-3, "reg_covar": 1e-6}}
    if dataset == "checkerboard":
        value["data"]["grid_size"] = 4
    return value


def parameters():
    return {"weights": [.25, .75], "means": [[-1., .2], [2., -.5]],
            "covariances": [[[.09, .06], [.06, .16]], [[.25, -.08], [-.08, .04]]]}


class GeneralGMMValidationTests(unittest.TestCase):
    def test_defaults_and_other_modes(self):
        value = config()
        value["anchor_flow"] = {"mode": "gmm_prior"}
        opts = gmm_prior.settings(value)
        self.assertEqual(opts, {"mode": "gmm_prior", "reference_points": 4096, "seed": 0,
                                "n_init": 5, "max_iter": 300, "tol": 1e-4, "reg_covar": 1e-6})
        self.assertIsNone(gmm_prior.settings({}))
        self.assertIsNone(gmm_prior.settings({"anchor_flow": {"mode": "anchor_prior", "sigma": .1}}))
        with self.assertRaisesRegex(ValueError, "mapping"):
            gmm_prior.settings({"anchor_flow": "gmm_prior"})

    def test_invalid_settings_and_forbidden_anchor_fields(self):
        for key, invalid in (("sigma", .1), ("centers", [[0., 0.], [1., 1.]]),
                             ("reference_points", 1), ("n_init", True), ("max_iter", 0),
                             ("tol", float("nan")), ("reg_covar", 0), ("seed", -1)):
            value = config()
            value["anchor_flow"][key] = invalid
            with self.subTest(key=key), self.assertRaises(ValueError):
                gmm_prior.settings(value)
        for key, invalid in (("coupling", "target_guided_cached"), ("dtype", "float64"), ("num_regions", 129)):
            value = config()
            value[key] = invalid
            with self.subTest(key=key), self.assertRaises(ValueError):
                gmm_prior.settings(value)
        value = config()
        value["model"]["point_dim"] = 3
        with self.assertRaises(ValueError):
            gmm_prior.settings(value)

    def test_resolved_parameters_must_be_complete_normalized_and_spd(self):
        for key in parameters():
            value = config()
            value["anchor_flow"][key] = parameters()[key]
            with self.assertRaisesRegex(ValueError, "together"):
                gmm_prior.settings(value)
        invalid_values = [
            ("weights", [0., 1.]), ("weights", [-1., 2.]), ("weights", [.5, .6]),
            ("weights", [1e-100, 1.]), ("means", [[float("nan"), 0.], [0., 0.]]),
            ("means", [[1e100, 0.], [0., 0.]]), ("means", [[0., 0.]]),
            ("covariances", [[[1., 2.], [0., 1.]], [[1., 0.], [0., 1.]]]),
            ("covariances", [[[1., 2.], [2., 1.]], [[1., 0.], [0., 1.]]]),
            ("covariances", [[[1e-100, 0.], [0., 1e-100]], [[1., 0.], [0., 1.]]]),
            ("covariances", [[[1e100, 0.], [0., 1.]], [[1., 0.], [0., 1.]]]),
        ]
        for key, invalid in invalid_values:
            value = parameters()
            value[key] = invalid
            with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                gmm_prior.bind(config(), value)

    def test_binding_does_not_replace_saved_prior_and_cache_spec_excludes_parameters(self):
        value = config()
        before = gmm_prior.cache_spec(value)
        resolved = gmm_prior.bind(value, parameters())
        self.assertEqual(gmm_prior.bind(value, resolved), resolved)
        self.assertEqual(gmm_prior.cache_spec(value), before)
        different = parameters()
        different["means"][0][0] += .1
        with self.assertRaisesRegex(ValueError, "differ"):
            gmm_prior.bind(value, different)


class GeneralGMMSamplingTests(unittest.TestCase):
    def test_empirical_weights_means_covariances_include_correlation(self):
        value = config()
        learned = gmm_prior.bind(value, parameters())
        source, labels = gmm_prior.sample_source_with_components(
            value, 1024, device="cpu", dtype=torch.float32,
            generator=torch.Generator().manual_seed(109))
        self.assertEqual(source.shape, (1024, 128, 2))
        self.assertEqual(labels.shape, (1024, 128))
        for component in range(2):
            selected = source[labels == component].double()
            self.assertAlmostEqual((labels == component).double().mean().item(),
                                   learned["weights"][component], delta=.005)
            torch.testing.assert_close(selected.mean(0), torch.tensor(learned["means"][component], dtype=torch.float64),
                                       rtol=0, atol=.012)
            torch.testing.assert_close(torch.cov(selected.T),
                                       torch.tensor(learned["covariances"][component], dtype=torch.float64),
                                       rtol=0, atol=.004)
        self.assertGreater(labels.sum(1).unique().numel(), 10)

    def test_sampler_matches_label_then_correlated_noise_formula_and_rng(self):
        value = config()
        learned = gmm_prior.bind(value, parameters())
        generator = torch.Generator().manual_seed(44)
        labels = torch.multinomial(torch.tensor(learned["weights"]), 3 * 128,
                                   replacement=True, generator=generator).reshape(3, 128)
        noise = torch.randn(3, 128, 2, generator=generator)
        factors = torch.linalg.cholesky(torch.tensor(learned["covariances"]))
        expected = torch.tensor(learned["means"])[labels] + (factors[labels] @ noise.unsqueeze(-1)).squeeze(-1)
        actual_generator = torch.Generator().manual_seed(44)
        actual, actual_labels = gmm_prior.sample_source_with_components(
            value, 3, device="cpu", dtype=torch.float32, generator=actual_generator)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_labels, labels, rtol=0, atol=0)
        torch.testing.assert_close(generator.get_state(), actual_generator.get_state(), rtol=0, atol=0)

    def test_unresolved_or_wrong_dtype_sampling_fails_and_metadata_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "resolved"):
            gmm_prior.sample_source_with_components(config(), 2, device="cpu", dtype=torch.float32)
        value = config()
        gmm_prior.bind(value, parameters())
        with self.assertRaisesRegex(ValueError, "float32"):
            gmm_prior.sample_source_with_components(value, 2, device="cpu", dtype=torch.float64)
        details = gmm_prior.experiment_details(value)
        self.assertFalse(details["quality_guarantee"])
        self.assertIn("not coupling-only", details["limitation"])
        self.assertIn("not original", details["paper_variant"])
        self.assertIn("Sigma", details["hybrid"])


class GeneralGMMFitTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_both_datasets_deterministic_training_reference_and_rng_restoration(self):
        from data import sample_checkerboard
        from train_horse import load_horse_mask, sample_horse
        for dataset in ("checkerboard", "horse"):
            value = config(dataset)
            original = copy.deepcopy(value)
            torch.manual_seed(73)
            state = torch.get_rng_state().clone()
            numpy_state = np.random.get_state()
            params, details = gmm_prior.fit(value, dataset)
            torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
            after_numpy_state = np.random.get_state()
            self.assertEqual(numpy_state[0], after_numpy_state[0])
            np.testing.assert_array_equal(numpy_state[1], after_numpy_state[1])
            self.assertEqual(numpy_state[2:], after_numpy_state[2:])
            self.assertEqual(torch.get_num_threads(), 1)
            again, again_details = gmm_prior.fit(value, dataset)
            self.assertEqual(params, again)
            self.assertEqual(details, again_details)
            self.assertEqual(value, original)
            self.assertTrue(details["converged"])
            self.assertTrue(details["learned_weights"])
            self.assertTrue(details["learned_covariances"])
            self.assertEqual(details["covariance_type"], "full")
            saved = copy.deepcopy(value)
            gmm_prior.bind(saved, params)
            self.assertEqual(gmm_prior.bind(saved, params), params)
            self.assertEqual(gmm_prior.settings(saved), saved["anchor_flow"])
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(value["anchor_flow"]["seed"])
                if dataset == "horse":
                    reference = sample_horse(load_horse_mask("cpu", torch.float32), 1, 256)
                else:
                    reference = sample_checkerboard(1, 256, "cpu", torch.float32, 4)
            self.assertEqual(details["training_reference_sha256"],
                             hashlib.sha256(reference[0].numpy().tobytes()).hexdigest())

    def test_validation_preserves_normalized_weight_roundoff_idempotently(self):
        value = config()
        value["num_regions"] = 3
        learned = {"weights": [.1, .2, .7000000000000001], "means": [[0., 0.]] * 3,
                   "covariances": [[[1., 0.], [0., 1.]]] * 3}
        first = gmm_prior.bind(value, learned)
        self.assertEqual(gmm_prior.bind(value, first), first)
        self.assertEqual(gmm_prior.settings(value), value["anchor_flow"])

    def test_random_weights_repeated_binding_and_actual_nsot_cache_roundtrip(self):
        import nsot
        value = config()
        value["num_regions"] = value["data"]["n_points"] = 8
        random = np.random.default_rng(2026)
        learned = {"weights": random.dirichlet(np.ones(8)).tolist(),
                   "means": random.normal(size=(8, 2)).tolist(),
                   "covariances": [[[.02, .005], [.005, .01]]] * 8}
        gmm_prior.bind(value, learned)
        canonical = copy.deepcopy(value["anchor_flow"])
        for _ in range(100):
            gmm_prior.bind(value, {key: canonical[key] for key in ("weights", "means", "covariances")})
            self.assertEqual(gmm_prior.settings(value), canonical)
        with tempfile.TemporaryDirectory() as temporary:
            value["nsot"] = {"cache": str(Path(temporary) / "general_gmm.npz"),
                             "superset_size": 64, "cache_seed": 5, "beta": .2, "solver": nsot.SOLVER}
            with contextlib.redirect_stdout(io.StringIO()):
                cache, metadata = nsot.prepare(value, "checkerboard")
                saved_bytes = cache.read_bytes()
                nsot.load_cache(value, "checkerboard")
                self.assertEqual(value["anchor_flow"], canonical)
                unresolved = copy.deepcopy(value)
                for key in ("weights", "means", "covariances"):
                    unresolved["anchor_flow"].pop(key)
                nsot.load_cache(unresolved, "checkerboard")
                self.assertEqual(unresolved["anchor_flow"], canonical)
                nsot.prepare(unresolved, "checkerboard")
                self.assertEqual(cache.read_bytes(), saved_bytes)
            self.assertEqual(metadata["gmm_parameters"],
                             {key: canonical[key] for key in ("weights", "means", "covariances")})

    def test_resolved_fit_does_not_import_or_fit_sklearn(self):
        value = config()
        learned = gmm_prior.bind(value, parameters())
        with patch("sklearn.mixture.GaussianMixture", side_effect=AssertionError("refit")):
            reused, details = gmm_prior.fit(value, "checkerboard")
        self.assertEqual(reused, learned)
        self.assertTrue(details["parameters_reused"])
        self.assertFalse(details["fit_performed"])

    def test_nonconverged_fit_is_rejected_and_restores_threads_and_rng(self):
        from sklearn.mixture import GaussianMixture

        class NonconvergedMixture(GaussianMixture):
            def fit(self, points, y=None):
                super().fit(points, y)
                self.converged_ = False
                return self

        torch.manual_seed(99)
        before = torch.get_rng_state().clone()
        torch.set_num_threads(2)
        with patch("sklearn.mixture.GaussianMixture", NonconvergedMixture), \
                self.assertRaisesRegex(RuntimeError, "did not converge"):
            gmm_prior.fit(config(), "checkerboard")
        self.assertEqual(torch.get_num_threads(), 2)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
