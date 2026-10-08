"""Directional NSOT is a coupling change, not a changed generation prior.

The Gaussian identity is a population statement.  Integration tests separately
check that the finite OT bank, target points, RNG order and inference stay fixed.
"""

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

import numpy as np
import torch
import yaml

import anchor_flow
import eval as eval_checkerboard
import eval_horse
import experiment
from model import PointSetTransformer
import nsot
import nsot_directional
import train
import train_horse


def tiny_config(dataset="checkerboard", *, directional=True):
    config = {
        "seed": 0, "device": "cpu", "dtype": "float32", "coupling": "nsot",
        "num_regions": 2,
        "anchor_flow": {"mode": "anchor_prior", "sigma": .1,
                        "reference_points": 64, "seed": 11},
        "nsot": {"cache": f"{dataset}_anchor_pairs.npz", "superset_size": 64,
                 "cache_seed": 5, "beta": .2, "solver": nsot.SOLVER},
        "data": {"batch_size": 2, "n_points": 8},
        "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                  "dim_feedforward": 16, "dropout": 0.0},
        "training": {"num_steps": 1, "learning_rate": .001, "weight_decay": .01,
                     "log_every": 1},
        "evaluation": {"batch_size": 2, "histogram_bins": 8},
        "checkpoint": f"{dataset}_directional.pt",
    }
    if directional:
        config["nsot"]["directional_hybrid"] = {
            "artifact": f"{dataset}_directional.json", "strength": .9,
            "ridge": .001, "min_points": 16,
            "min_target_anisotropy": 1.5, "min_normal_r2": .25,
        }
    if dataset == "checkerboard":
        config["data"]["grid_size"] = 4
    return config


def rotation(angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def thin_pairs(count=4096):
    """One reliable rotated thin target; target/source axes are different."""
    rng = np.random.default_rng(63)
    centers = np.array([[.7, -.4]], dtype=np.float64)
    source_rotation, target_rotation = rotation(.61), rotation(-.83)
    jacobian = target_rotation @ np.diag([2., .15]) @ source_rotation.T
    source = centers + .2 * rng.standard_normal((count, 2))
    target = np.array([[-.3, .4]]) + (source - centers) @ jacobian.T
    target += .0001 * rng.standard_normal(target.shape)
    return source, target, np.zeros(count, dtype=np.int64), centers, jacobian, target_rotation[:, 1]


class DirectionalHybridMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def options(**overrides):
        config = tiny_config()
        config["nsot"]["directional_hybrid"].update(overrides)
        return nsot_directional.settings(config)

    def fitted(self, *, beta=.2, options=None):
        source, target, labels, centers, _, _ = thin_pairs()
        return nsot_directional.fit(source, target, labels, centers, sigma=.2,
                                    beta=beta, options=options or self.options())

    def test_trace_budget_eigenvalues_and_covariance_identity(self):
        result = self.fitted()
        matrices = np.asarray(result["matrices"])
        self.assertEqual(matrices.shape, (1, 2, 2))
        np.testing.assert_allclose(matrices, matrices.swapaxes(-1, -2), atol=1e-14)
        np.testing.assert_allclose(np.linalg.eigvalsh(matrices), [[.02, .38]], atol=1e-12)
        np.testing.assert_allclose(np.trace(matrices, axis1=-2, axis2=-1), [.4], atol=1e-12)
        shrink, refresh = nsot_directional.factors(matrices)
        np.testing.assert_allclose(shrink @ shrink.swapaxes(-1, -2)
                                   + refresh @ refresh.swapaxes(-1, -2),
                                   np.eye(2)[None], atol=1e-12)
        np.testing.assert_allclose(refresh @ refresh.swapaxes(-1, -2), matrices, atol=1e-12)
        self.assertFalse(result["components"][0]["fallback"])
        self.assertEqual(result["components"][0]["reason"], "directional")

    def test_noise_is_reduced_in_source_direction_that_changes_target_normal(self):
        source, target, labels, centers, jacobian, normal = thin_pairs()
        result = nsot_directional.fit(source, target, labels, centers, sigma=.2,
                                     beta=.2, options=self.options())
        matrix = np.asarray(result["matrices"])[0]
        sensitive = jacobian.T @ normal
        sensitive /= np.linalg.norm(sensitive)
        weakest = np.linalg.eigh(matrix)[1][:, 0]
        self.assertGreater(abs(weakest @ sensitive), .999)
        # The same trace/noise budget as beta*I; only orientation changes.
        normal_variance = .2**2 * normal @ jacobian @ matrix @ jacobian.T @ normal
        isotropic_variance = .2**2 * .2 * np.linalg.norm(jacobian.T @ normal)**2
        self.assertLess(normal_variance, .11 * isotropic_variance)
        self.assertGreater(result["components"][0]["normal_r2"], .99)

    def test_population_component_mean_covariance_and_mixture_weights_preserved(self):
        # Fresh population draws independent of the fitted calibration bank.
        centers = torch.tensor([[-2., -.7], [.4, 1.2], [2.3, -.3]], dtype=torch.float64)
        generator = torch.Generator().manual_seed(830)
        labels = torch.randint(3, (120000,), generator=generator)
        means = centers[labels]
        source = means + .2 * torch.randn(120000, 2, dtype=torch.float64, generator=generator)
        noise = torch.randn(source.shape, dtype=torch.float64, generator=generator)
        matrices = np.stack([rotation(angle) @ np.diag([.02, .38]) @ rotation(angle).T
                             for angle in (.2, -.7, 1.1)])
        shrink, refresh = (torch.as_tensor(value, dtype=torch.float64)
                           for value in nsot_directional.factors(matrices))
        actual = nsot_directional.apply(source, means, sigma=.2,
                                        shrink=shrink[labels], refresh=refresh[labels], noise=noise)
        for component in range(3):
            residual = actual[labels == component] - centers[component]
            self.assertLess(residual.mean(0).abs().max().item(), .0035)
            centered = residual - residual.mean(0)
            covariance = centered.T @ centered / len(residual)
            torch.testing.assert_close(covariance, .04 * torch.eye(2, dtype=torch.float64),
                                       rtol=0, atol=.0015)
            self.assertLess(abs((labels == component).double().mean().item() - 1 / 3), .006)

    def test_zero_strength_and_beta_endpoints_recover_isotropic_hybrid(self):
        generator = torch.Generator().manual_seed(5)
        source = torch.randn(3, 8, 2, dtype=torch.float64, generator=generator)
        centers = torch.randn(source.shape, dtype=torch.float64, generator=generator)
        noise = torch.randn(source.shape, dtype=torch.float64, generator=generator)
        for beta, strength in ((0., .9), (.2, 0.), (1., .9)):
            with self.subTest(beta=beta, strength=strength):
                result = self.fitted(beta=beta, options=self.options(strength=strength))
                matrix = np.asarray(result["matrices"])[0]
                np.testing.assert_allclose(matrix, beta * np.eye(2), atol=1e-14)
                shrink, refresh = (torch.as_tensor(value, dtype=source.dtype)
                                   for value in nsot_directional.factors(matrix))
                actual = nsot_directional.apply(source, centers, sigma=.2, shrink=shrink,
                                                refresh=refresh, noise=noise)
                expected = nsot.component_centered_hybrid(source, centers, sigma=.2,
                                                          beta=beta, noise=noise)
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)

    def test_rotated_factors_broadcast_over_batches_and_remain_deterministic(self):
        generator = torch.Generator().manual_seed(7)
        matrix = rotation(.37) @ np.diag([.02, .38]) @ rotation(.37).T
        shrink, refresh = (torch.tensor(value) for value in nsot_directional.factors(matrix))
        source = torch.randn(2, 9, 2, dtype=torch.float64, generator=generator)
        centers = torch.zeros_like(source)
        noise = torch.randn(source.shape, dtype=source.dtype, generator=generator)
        actual = nsot_directional.apply(source, centers, sigma=.1, shrink=shrink,
                                        refresh=refresh, noise=noise)
        expected = source @ shrink.T + .1 * noise @ refresh.T
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)

    def test_unreliable_components_fall_back_without_changing_trace(self):
        rng = np.random.default_rng(71)
        for reason in ("insufficient_points", "target_near_isotropic", "poor_normal_fit"):
            count = 8 if reason == "insufficient_points" else 8192
            source = .2 * rng.standard_normal((count, 2))
            if reason == "target_near_isotropic":
                target = source.copy()
            else:
                target = np.column_stack((2 * source[:, 0], .03 * rng.standard_normal(count)))
            result = nsot_directional.fit(source, target, np.zeros(count, dtype=np.int64),
                                         np.zeros((1, 2)), sigma=.2, beta=.2, options=self.options())
            with self.subTest(reason=reason):
                np.testing.assert_allclose(result["matrices"], [.2 * np.eye(2)], atol=1e-14)
                self.assertTrue(result["components"][0]["fallback"])
                self.assertEqual(result["components"][0]["reason"], reason)

    def test_empty_and_degenerate_components_fall_back(self):
        source = np.zeros((64, 2))
        target = np.column_stack((np.arange(64), np.zeros(64)))
        result = nsot_directional.fit(source, target, np.zeros(64, dtype=np.int64),
                                     np.array([[0., 0.], [1., 1.]]), sigma=.2,
                                     beta=.2, options=self.options())
        self.assertEqual(result["components"][0]["reason"], "degenerate_source")
        self.assertEqual(result["components"][1]["reason"], "insufficient_points")
        np.testing.assert_allclose(result["matrices"], np.broadcast_to(.2 * np.eye(2), (2, 2, 2)))

    def test_invalid_settings_fail_loudly(self):
        self.assertIsNone(nsot_directional.settings(tiny_config(directional=False)))
        invalid = {
            "strength": (-.1, 1.1, True, float("nan")),
            "ridge": (-.1, True, float("inf")),
            "min_points": (0, 2.5, True),
            "min_target_anisotropy": (.9, True, float("nan")),
            "min_normal_r2": (-.1, 1.1, True),
            "artifact": ("", None, True),
            "artifact_sha256": ("bad", "z" * 64, True),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.options(**{field: value})
        for changed in ("prior", "coupling"):
            config = tiny_config()
            if changed == "prior":
                config.pop("anchor_flow")
            else:
                config["coupling"] = "independent"
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                nsot_directional.settings(config)

    def test_invalid_matrix_and_tensor_shapes_fail_loudly(self):
        for matrix in (np.zeros((2, 3)), [[.2, .1], [0., .2]],
                       [[-.1, 0.], [0., .2]], [[1.1, 0.], [0., .2]],
                       [[float("nan"), 0.], [0., .2]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                nsot_directional.factors(matrix)
        source = torch.zeros(4, 2)
        identity = torch.eye(2)
        common = {"sigma": .1, "shrink": identity, "refresh": identity, "noise": source}
        for change in ({"sigma": 0.}, {"sigma": True}, {"sigma": float("nan")},
                       {"noise": source.double()}, {"noise": torch.zeros(4, 3)},
                       {"shrink": torch.zeros(3, 3)}, {"refresh": identity.double()}):
            arguments = {**common, **change}
            with self.subTest(change=change), self.assertRaises(ValueError):
                nsot_directional.apply(source, source, **arguments)

    def test_invalid_fit_arrays_or_component_labels_fail_loudly(self):
        source, target, labels, centers, _, _ = thin_pairs(64)
        invalid = (
            (source[:, :1], target, labels, centers),
            (source, target[:-1], labels, centers),
            (source, target, labels.astype(float), centers),
            (source, target, np.full(64, 1, dtype=np.int64), centers),
            (source, target, np.full(64, -1, dtype=np.int64), centers),
            (source, np.full_like(target, np.nan), labels, centers),
        )
        for values in invalid:
            with self.subTest(shapes=[value.shape for value in values]), self.assertRaises(ValueError):
                nsot_directional.fit(*values, sigma=.2, beta=.2, options=self.options())


class DirectionalHybridIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.previous = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        os.chdir(self.temp.name)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        os.chdir(self.previous)
        self.temp.cleanup()
        torch.set_num_threads(self.previous_threads)

    @staticmethod
    def prepare(config, dataset="checkerboard"):
        with contextlib.redirect_stdout(io.StringIO()):
            return nsot.prepare(config, dataset)

    def prepared_from_existing_bank(self, dataset="checkerboard"):
        baseline = tiny_config(dataset, directional=False)
        path, metadata = self.prepare(baseline, dataset)
        original = path.read_bytes()
        directional = tiny_config(dataset)
        with patch.object(nsot, "exact_superset_permutation", side_effect=AssertionError("OT recomputed")):
            self.prepare(directional, dataset)
        self.assertEqual(path.read_bytes(), original)
        return baseline, directional, metadata

    def test_sidecar_reuses_parent_bank_without_any_pair_or_target_changes(self):
        baseline, config, parent_metadata = self.prepared_from_existing_bank()
        artifact_path = Path(config["nsot"]["directional_hybrid"]["artifact"])
        artifact_bytes = artifact_path.read_bytes()
        artifact = json.loads(artifact_bytes)
        self.assertEqual(artifact["pair_cache_sha256"], nsot.file_sha256(config["nsot"]["cache"]))
        self.assertEqual(artifact["prior_centers"], parent_metadata["prior_centers"])
        self.assertEqual(artifact["beta"], .2)
        self.assertEqual(len(artifact["matrices"]), config["num_regions"])
        baseline_arrays = nsot.load_cache(baseline, "checkerboard")[:3]
        directional_arrays = nsot.load_cache(config, "checkerboard")[:3]
        for left, right in zip(baseline_arrays, directional_arrays):
            np.testing.assert_array_equal(left, right)
        with patch.object(nsot_directional, "fit", side_effect=AssertionError("artifact refit")):
            self.prepare(config)
        self.assertEqual(artifact_path.read_bytes(), artifact_bytes)

    def test_sampler_replays_original_indices_noise_and_component_labels(self):
        baseline, config, _ = self.prepared_from_existing_bank()
        sampler = nsot.NSOTPairSampler(config, "checkerboard", "cpu", torch.float32)
        original = nsot.NSOTPairSampler(baseline, "checkerboard", "cpu", torch.float32)
        actual_source, actual_target = sampler.sample(3, generator=torch.Generator().manual_seed(42))
        replay = torch.Generator().manual_seed(42)
        index = torch.randint(64, (3, 8), generator=replay)
        noise = torch.randn(3, 8, 2, generator=replay)
        labels = sampler.source_components[index]
        expected = nsot_directional.apply(
            sampler.source[index], sampler.prior_centers[labels], sigma=.1,
            shrink=sampler.directional_shrink[labels], refresh=sampler.directional_refresh[labels], noise=noise)
        torch.testing.assert_close(actual_source, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_target, sampler.target[index], rtol=0, atol=0)
        _, original_target = original.sample(3, generator=torch.Generator().manual_seed(42))
        torch.testing.assert_close(actual_target, original_target, rtol=0, atol=0)
        expected_hash = nsot.file_sha256(config["nsot"]["directional_hybrid"]["artifact"])
        self.assertEqual(config["nsot"]["directional_hybrid"]["artifact_sha256"], expected_hash)
        self.assertEqual(sampler.artifact_sha256, expected_hash)
        self.assertIn("directional_hybrid", sampler.details())
        # Models still receive only coordinates and time, not component labels.
        self.assertNotIn("anchor_embedding", config["model"])

    def test_online_sampling_does_not_fit_factorize_or_read_any_artifact(self):
        _, config, _ = self.prepared_from_existing_bank()
        sampler = nsot.NSOTPairSampler(config, "checkerboard", "cpu", torch.float32)
        expected = sampler.sample(2, generator=torch.Generator().manual_seed(98))
        with patch.object(nsot_directional, "fit", side_effect=AssertionError("online fit")), \
                patch.object(nsot_directional, "factors", side_effect=AssertionError("online factorization")), \
                patch.object(nsot, "_load_cache", side_effect=AssertionError("online pair cache IO")), \
                patch.object(nsot, "_load_directional_artifact", side_effect=AssertionError("online sidecar IO")), \
                patch.object(Path, "open", side_effect=AssertionError("online file IO")):
            actual = sampler.sample(2, generator=torch.Generator().manual_seed(98))
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_preparation_respects_pinned_parent_cache_and_missing_artifact(self):
        baseline = tiny_config(directional=False)
        pair_path, _ = self.prepare(baseline)
        pair_bytes = pair_path.read_bytes()
        config = tiny_config()
        config["nsot"]["cache_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.prepare(config)
        self.assertFalse(Path(config["nsot"]["directional_hybrid"]["artifact"]).exists())
        self.assertEqual(pair_path.read_bytes(), pair_bytes)
        config["nsot"]["cache_sha256"] = nsot.file_sha256(pair_path)
        config["nsot"]["directional_hybrid"]["artifact_sha256"] = "0" * 64
        with self.assertRaisesRegex(FileNotFoundError, "Pinned"):
            self.prepare(config)
        self.assertEqual(pair_path.read_bytes(), pair_bytes)

    def test_configuration_mismatch_and_bound_artifact_tampering_are_rejected(self):
        _, config, _ = self.prepared_from_existing_bank()
        nsot.NSOTPairSampler(config, "checkerboard", "cpu", torch.float32)
        for section, key, value in (("directional_hybrid", "strength", .5),
                                    ("directional_hybrid", "ridge", .01),
                                    ("directional_hybrid", "min_points", 17),
                                    ("nsot", "beta", .3)):
            changed = copy.deepcopy(config)
            destination = changed["nsot"] if section == "nsot" else changed["nsot"][section]
            destination[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                nsot.NSOTPairSampler(changed, "checkerboard", "cpu", torch.float32)
        path = Path(config["nsot"]["directional_hybrid"]["artifact"])
        artifact = json.loads(path.read_text(encoding="utf-8"))
        artifact["fit_seconds"] += 1.
        path.write_text(json.dumps(artifact), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            nsot.NSOTPairSampler(config, "checkerboard", "cpu", torch.float32)

    def test_unbound_sidecar_rejects_invalid_matrix_and_parent_cache_hash(self):
        _, config, _ = self.prepared_from_existing_bank()
        path = Path(config["nsot"]["directional_hybrid"]["artifact"])
        original = json.loads(path.read_text(encoding="utf-8"))
        for field, value in (("pair_cache_sha256", "0" * 64),
                             ("matrices", [[[1.2, 0.], [0., .2]]] * 2),
                             ("matrices", [[[.3, 0.], [0., .3]]] * 2)):
            changed = copy.deepcopy(original)
            changed[field] = value
            path.write_text(json.dumps(changed), encoding="utf-8")
            unbound = copy.deepcopy(config)
            unbound["nsot"]["directional_hybrid"].pop("artifact_sha256", None)
            with self.subTest(field=field), self.assertRaises(ValueError):
                nsot.NSOTPairSampler(unbound, "checkerboard", "cpu", torch.float32)

    def test_unbound_sidecar_rejects_malformed_provenance_and_fit_reports(self):
        _, config, _ = self.prepared_from_existing_bank()
        path = Path(config["nsot"]["directional_hybrid"]["artifact"])
        original = json.loads(path.read_text(encoding="utf-8"))
        cases = []
        missing_provenance = copy.deepcopy(original)
        missing_provenance.pop("fit_source_sha256")
        cases.append(("missing_provenance", missing_provenance))
        malformed_hash = copy.deepcopy(original)
        malformed_hash["fit_source_sha256"]["nsot.py"] = "not-a-source-hash"
        cases.append(("malformed_source_hash", malformed_hash))
        for field, value in (("normal_r2", float("nan")), ("component", 1),
                             ("count", -1), ("beta_eigenvalues", [.1, .9]),
                             ("fallback", "yes")):
            malformed = copy.deepcopy(original)
            malformed["components"][0][field] = value
            cases.append((field, malformed))
        wrong_total = copy.deepcopy(original)
        wrong_total["components"][0]["count"] += 1
        cases.append(("wrong_total_count", wrong_total))
        wrong_quality_guarantee = copy.deepcopy(original)
        wrong_quality_guarantee["quality_guarantee"] = True
        cases.append(("wrong_quality_guarantee", wrong_quality_guarantee))
        for name, artifact in cases:
            path.write_text(json.dumps(artifact), encoding="utf-8")
            unbound = copy.deepcopy(config)
            unbound["nsot"]["directional_hybrid"].pop("artifact_sha256", None)
            with self.subTest(case=name), self.assertRaises(ValueError):
                nsot.NSOTPairSampler(unbound, "checkerboard", "cpu", torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_sampler_replay_original_component_indexing_and_backward(self):
        _, config, _ = self.prepared_from_existing_bank()
        device = torch.device("cuda")
        sampler = nsot.NSOTPairSampler(config, "checkerboard", device, torch.float32)
        generator = torch.Generator(device=device).manual_seed(42)
        source, target = sampler.sample(2, generator=generator)
        replay = torch.Generator(device=device).manual_seed(42)
        index = torch.randint(64, (2, 8), device=device, generator=replay)
        noise = torch.randn(2, 8, 2, device=device, generator=replay)
        labels = sampler.source_components[index]
        expected = nsot_directional.apply(
            sampler.source[index], sampler.prior_centers[labels], sigma=.1,
            shrink=sampler.directional_shrink[labels], refresh=sampler.directional_refresh[labels], noise=noise)
        torch.testing.assert_close(source, expected, rtol=0, atol=0)
        torch.testing.assert_close(target, sampler.target[index], rtol=0, atol=0)
        self.assertEqual(sampler.directional_shrink.device.type, "cuda")
        self.assertEqual(sampler.directional_shrink.dtype, torch.float32)
        self.assertEqual(sampler.source_components.dtype, torch.int64)
        model = PointSetTransformer(**config["model"]).to(device)
        loss = train.train_step(model, torch.optim.AdamW(model.parameters()), target,
                                coupling="nsot", paired_noise=source, anchor_config=config)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(source.device.type, "cuda")

    def test_both_datasets_train_save_reload_and_generate_without_training_artifacts(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for dataset, trainer, evaluator, model_class in (
                    ("checkerboard", train.main, eval_checkerboard.main, PointSetTransformer),
                    ("horse", train_horse.main, eval_horse.main, train_horse.HorsePointSetTransformer)):
                config = tiny_config(dataset)
                path = Path(f"{dataset}.yaml")
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                self.prepare(config, dataset)
                with patch.object(train, "coupled_points", side_effect=AssertionError("online OT called")):
                    run = trainer(path, steps=1)
                model, saved, checkpoint, metadata = experiment.load_model(run / "config.yaml", model_class, dataset)
                self.assertTrue(metadata["training_config_verified"])
                directional = saved["nsot"]["directional_hybrid"]
                self.assertEqual(directional["artifact_sha256"], nsot.file_sha256(directional["artifact"]))
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.assertEqual(payload["config"]["nsot"], saved["nsot"])
                self.assertIn("directional_hybrid", metadata["coupling_details"])
                # Source sampling is deliberately unchanged for generation.
                isotropic = copy.deepcopy(saved)
                isotropic["nsot"].pop("directional_hybrid")
                torch.manual_seed(71)
                expected = anchor_flow.sample_source(isotropic, 2, device="cpu", dtype=torch.float32)
                torch.manual_seed(71)
                actual = anchor_flow.sample_source(saved, 2, device="cpu", dtype=torch.float32)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for artifact in (Path(saved["nsot"]["cache"]), Path(directional["artifact"])):
                    artifact.rename(artifact.with_suffix(".backup"))
                with patch.object(nsot, "load_cache", side_effect=AssertionError("cache at inference")), \
                        patch.object(anchor_flow, "fit_prior_centers", side_effect=AssertionError("fit at inference")):
                    if dataset == "checkerboard":
                        output = evaluator(run / "config.yaml", 1, render=False)
                    else:
                        with patch.object(eval_horse, "render_comparison"), patch("horse_regions.HorseRegions.render"):
                            output = evaluator(run / "config.yaml", 1)
                result = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
                self.assertTrue(np.isfinite(result["chamfer"]))
                self.assertTrue(np.isfinite(result["source_chamfer"]))
                self.assertEqual(result["coupling_details"], metadata["coupling_details"])


if __name__ == "__main__":
    unittest.main()
