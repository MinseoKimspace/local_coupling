"""Component-preserving NSOT with a fixed training-derived anchor GMM.

Population moment checks verify the hybrid formula; finite cached supersets
remain empirical approximations, not exact population Gaussian mixtures.
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
import audit_generation
from diagnostic_data import DiagnosticData, tensor_sha256
import eval as eval_checkerboard
import eval_horse
import experiment
from model import PointSetTransformer
import nsot
import prepare_nsot
import train
import train_horse


def tiny_config(dataset="checkerboard", *, prior=True):
    config = {
        "seed": 0, "device": "cpu", "dtype": "float32", "coupling": "nsot",
        "nsot": {"cache": f"{dataset}_{'anchor_prior' if prior else 'gaussian'}.npz",
                 "superset_size": 64, "cache_seed": 5, "beta": .2, "solver": nsot.SOLVER},
        "data": {"batch_size": 2, "n_points": 8},
        "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                  "dim_feedforward": 16, "dropout": 0.0},
        "training": {"num_steps": 2, "learning_rate": .001, "weight_decay": .01, "log_every": 1},
        "evaluation": {"batch_size": 2, "histogram_bins": 8},
        "checkpoint": f"{dataset}_nsot_anchor_prior.pt",
    }
    if prior:
        config["num_regions"] = 2
        config["anchor_flow"] = {"mode": "anchor_prior", "sigma": .1,
                                 "reference_points": 64, "seed": 11}
    if dataset == "checkerboard":
        config["data"]["grid_size"] = 4
    return config


class ComponentCenteredHybridTests(unittest.TestCase):
    def test_centered_hybrid_preserves_component_mean_and_variance(self):
        generator = torch.Generator().manual_seed(201)
        labels = torch.randint(2, (60000,), generator=generator)
        anchors = torch.tensor([[-3., -.7], [2., .6]], dtype=torch.float64)
        centers = anchors[labels]
        source = centers + .2 * torch.randn(60000, 2, dtype=torch.float64, generator=generator)
        noise = torch.randn(source.shape, dtype=torch.float64, generator=generator)
        for beta in (0., .2, 1.):
            actual = nsot.component_centered_hybrid(source, centers, sigma=.2, beta=beta, noise=noise)
            expected = centers + math.sqrt(1 - beta) * (source - centers) + .2 * math.sqrt(beta) * noise
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-14)
            for index in (0, 1):
                residual = actual[labels == index] - anchors[index]
                self.assertLess(residual.mean(0).abs().max().item(), .005)
                self.assertLess((residual.var(0, unbiased=False) - .04).abs().max().item(), .0015)
            if beta == 0:
                torch.testing.assert_close(actual, source, rtol=0, atol=1e-14)
            elif beta == 1:
                torch.testing.assert_close(actual, centers + .2 * noise, rtol=0, atol=0)
        incorrect = math.sqrt(.8) * source + math.sqrt(.2) * noise
        self.assertGreater((incorrect[labels == 0].mean(0) - anchors[0]).abs().max().item(), .25)

    def test_gmm_components_reconstruct_points_and_are_sampled_iid(self):
        config = tiny_config()
        config["data"]["n_points"] = 64
        anchor_flow.bind_prior_centers(config, [[-2., -.3], [2., .3]])
        generator = torch.Generator().manual_seed(123)
        source, labels = anchor_flow.sample_source_with_components(
            config, 256, device="cpu", dtype=torch.float32, generator=generator)
        replay = torch.Generator().manual_seed(123)
        expected_labels = torch.randint(2, (256, 64), generator=replay)
        noise = torch.randn(256, 64, 2, generator=replay)
        expected = torch.tensor(config["anchor_flow"]["centers"])[expected_labels] + .1 * noise
        torch.testing.assert_close(labels, expected_labels, rtol=0, atol=0)
        torch.testing.assert_close(source, expected, rtol=0, atol=0)
        source_only = anchor_flow.sample_source(config, 256, device="cpu", dtype=torch.float32,
                                               generator=torch.Generator().manual_seed(123))
        torch.testing.assert_close(source_only, source, rtol=0, atol=0)
        self.assertLess(abs(labels.float().mean().item() - .5), .015)
        self.assertGreater(labels.sum(1).unique().numel(), 5)

    def test_new_yaml_matches_gaussian_baseline_hyperparameters(self):
        root = Path(__file__).resolve().parents[1]
        for directory, prior_name, base_name in (
                ("checkerboard_experiments", "nsot_anchor_prior_k8_n256_seed0.yaml", "nsot.yaml"),
                ("horse_experiments", "horse_nsot_anchor_prior_k8_n256_seed0.yaml", "horse_nsot_n256_seed0.yaml")):
            config = yaml.safe_load((root / directory / prior_name).read_text())
            baseline = yaml.safe_load((root / directory / base_name).read_text())
            excluded = {"anchor_flow", "num_regions", "nsot", "checkpoint"}
            self.assertEqual({k: v for k, v in config.items() if k not in excluded},
                             {k: v for k, v in baseline.items() if k not in excluded})
            self.assertEqual({k: v for k, v in config["nsot"].items() if k != "cache"},
                             {k: v for k, v in baseline["nsot"].items() if k != "cache"})
            self.assertNotEqual(config["nsot"]["cache"], baseline["nsot"]["cache"])
            self.assertEqual(config["num_regions"], 8)
            self.assertEqual(config["anchor_flow"]["mode"], "anchor_prior")


class NSOTAnchorPriorIntegrationTests(unittest.TestCase):
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
    def prepare(config, dataset="checkerboard"):
        with contextlib.redirect_stdout(io.StringIO()):
            return nsot.prepare(config, dataset)

    def test_exact_ot_is_recomputed_on_gmm_source_not_replaced_after_ot(self):
        config = tiny_config()
        anchor_flow.bind_prior_centers(config, [[-2., -.2], [2., .3]])
        captures = []
        original = nsot.exact_superset_permutation

        def capture(source, target):
            captures.append((source.copy(), target.copy()))
            return original(source, target)

        torch.manual_seed(19)
        before = torch.get_rng_state()
        with patch.object(nsot, "exact_superset_permutation", side_effect=capture):
            path, metadata = self.prepare(config)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(len(captures), 1)
        source, target, permutation, loaded, digest = nsot.load_cache(config, "checkerboard")
        np.testing.assert_array_equal(source, captures[0][0])
        np.testing.assert_array_equal(target, captures[0][1])
        expected_permutation, cost = original(source, target)
        np.testing.assert_array_equal(permutation, expected_permutation)
        self.assertAlmostEqual(metadata["cost_after"], cost, places=12)
        self.assertEqual(metadata["format_version"], 2)
        self.assertEqual(loaded, metadata)
        self.assertEqual(digest, nsot.file_sha256(path))
        with np.load(path, allow_pickle=False) as archive:
            labels = archive["source_components"]
        self.assertEqual(labels.dtype, np.int64)
        self.assertEqual(labels.shape, (64,))
        anchors = np.asarray(config["anchor_flow"]["centers"])
        residual = source - anchors[labels]
        self.assertLess(np.mean(residual**2), .025)
        self.assertGreater(source[:, 0].var(), 3.5)
        baseline = tiny_config(prior=False)
        base_source, _ = nsot.draw_supersets(baseline, "checkerboard")
        self.assertFalse(np.array_equal(source, base_source))
        with patch.object(nsot, "exact_superset_permutation", side_effect=AssertionError("recomputed")):
            self.prepare(config)

    def test_sampler_uses_correct_component_indices_and_centered_beta_formula(self):
        config = tiny_config()
        anchor_flow.bind_prior_centers(config, [[-.8, -.2], [.7, .3]])
        self.prepare(config)
        for beta in (0., .2, 1.):
            config["nsot"]["beta"] = beta
            sampler = nsot.NSOTPairSampler(config, "checkerboard", "cpu", torch.float32)
            actual_source, actual_target = sampler.sample(3, generator=torch.Generator().manual_seed(42))
            replay = torch.Generator().manual_seed(42)
            indices = torch.randint(64, (3, 8), generator=replay)
            noise = torch.randn(3, 8, 2, generator=replay)
            selected = sampler.source[indices]
            centers = sampler.prior_centers[sampler.source_components[indices]]
            expected_source = nsot.component_centered_hybrid(selected, centers, sigma=.1, beta=beta, noise=noise)
            torch.testing.assert_close(actual_source, expected_source, rtol=0, atol=0)
            torch.testing.assert_close(actual_target, sampler.target[indices], rtol=0, atol=0)
            details = sampler.details()
            self.assertEqual(details["anchor_flow"]["mode"], "anchor_prior")
            self.assertIn("NOT independent coupling", details["beta_one"])

    def test_cache_prior_mismatch_guards_labels_integrity_and_binding(self):
        config = tiny_config()
        path, metadata = self.prepare(config)
        self.assertNotIn("centers", config["anchor_flow"])
        loaded_config = copy.deepcopy(config)
        nsot.load_cache(loaded_config, "checkerboard")
        self.assertEqual(loaded_config["anchor_flow"]["centers"], metadata["prior_centers"])
        for field, value in (("sigma", .2), ("seed", 12), ("reference_points", 128),
                             ("centers", [[-10., 0.], [10., 0.]])):
            mismatch = copy.deepcopy(config)
            mismatch["anchor_flow"][field] = value
            with self.assertRaises(ValueError):
                nsot.load_cache(mismatch, "checkerboard")
        baseline = tiny_config(prior=False)
        self.prepare(baseline)
        changed = copy.deepcopy(config)
        changed["nsot"]["cache"] = baseline["nsot"]["cache"]
        with self.assertRaises(ValueError):
            nsot.load_cache(changed, "checkerboard")
        changed = copy.deepcopy(baseline)
        changed["nsot"]["cache"] = str(path)
        with self.assertRaises(ValueError):
            nsot.load_cache(changed, "checkerboard")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        original_label = int(arrays["source_components"][0])
        arrays["source_components"][0] = 2
        np.savez_compressed(path, **arrays)
        with self.assertRaises(ValueError):
            nsot.load_cache(config, "checkerboard")
        arrays["source_components"][0] = 1 - original_label
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            nsot.load_cache(config, "checkerboard")

    def test_existing_gaussian_nsot_draw_and_hybrid_rng_are_unchanged(self):
        config = tiny_config(prior=False)
        source, target = nsot.draw_supersets(config, "checkerboard")
        from data import sample_checkerboard
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            expected_source = torch.randn(64, 2)
            expected_target = sample_checkerboard(1, 64, "cpu", torch.float32, 4)[0]
        np.testing.assert_array_equal(source, expected_source.numpy())
        np.testing.assert_array_equal(target, expected_target.numpy())
        _, metadata = self.prepare(config)
        self.assertEqual(metadata["format_version"], 1)
        sampler = nsot.NSOTPairSampler(config, "checkerboard", "cpu", torch.float32)
        actual = sampler.sample(3, generator=torch.Generator().manual_seed(42))
        replay = torch.Generator().manual_seed(42)
        indices = torch.randint(64, (3, 8), generator=replay)
        noise = torch.randn(3, 8, 2, generator=replay)
        expected = (math.sqrt(.8) * sampler.source[indices] + math.sqrt(.2) * noise, sampler.target[indices])
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_component_indexing_sampler_and_backward(self):
        config = tiny_config()
        self.prepare(config)
        device = torch.device("cuda")
        sampler = nsot.NSOTPairSampler(config, "checkerboard", device, torch.float32)
        source, target = sampler.sample(2, generator=torch.Generator(device=device).manual_seed(42))
        replay = torch.Generator(device=device).manual_seed(42)
        indices = torch.randint(64, (2, 8), device=device, generator=replay)
        noise = torch.randn(2, 8, 2, device=device, generator=replay)
        centers = sampler.prior_centers[sampler.source_components[indices]]
        expected = nsot.component_centered_hybrid(sampler.source[indices], centers,
                                                  sigma=.1, beta=.2, noise=noise)
        torch.testing.assert_close(source, expected, rtol=0, atol=0)
        torch.testing.assert_close(target, sampler.target[indices], rtol=0, atol=0)
        self.assertEqual(sampler.source_components.dtype, torch.int64)
        self.assertEqual(sampler.source_components.device.type, "cuda")
        model = PointSetTransformer(**config["model"]).to(device)
        loss = train.train_step(model, torch.optim.AdamW(model.parameters()), target,
                                coupling="nsot", paired_noise=source, anchor_config=config)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(source.device.type, "cuda")

    def test_both_datasets_tiny_train_checkpoint_eval_and_diagnostics(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for dataset, trainer, evaluator, model_class in (
                    ("checkerboard", train.main, eval_checkerboard.main, PointSetTransformer),
                    ("horse", train_horse.main, eval_horse.main, train_horse.HorsePointSetTransformer)):
                config = tiny_config(dataset)
                path = Path(f"{dataset}.yaml")
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                prepare_nsot.main(path, dataset)
                run = trainer(path, steps=1)
                self.assertIn("anchor_prior", run.name)
                model, trained, checkpoint, metadata = experiment.load_model(run / "config.yaml", model_class, dataset)
                self.assertTrue(metadata["training_config_verified"])
                self.assertEqual(len(trained["anchor_flow"]["centers"]), 2)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.assertEqual(payload["config"]["anchor_flow"], trained["anchor_flow"])
                data = DiagnosticData(trained, dataset, "cpu", torch.float32, 2026)
                fm = audit_generation.fm_summary(model, trained, data, 1, 2, 2026)
                self.assertTrue(all(np.isfinite(row["mse"]["mean"]) for row in fm["time_errors"]))
                with patch.object(audit_generation, "render"), patch("horse_regions.HorseRegions.render"):
                    directory = audit_generation.audit(run / "config.yaml", dataset, clouds=2, batch_size=2,
                                                       nfes=(1, 2), reference_nfe=2, max_reference_nfe=8,
                                                       fm_batches=1, matching_batch_size=2, seed=2026)
                diagnostics = json.loads((directory / "diagnostics.json").read_text(encoding="utf-8"))
                self.assertEqual(diagnostics["noise_sha256"], tensor_sha256(data.bank(2)[0]))
                self.assertEqual(diagnostics["target_sha256"], tensor_sha256(data.bank(2)[1]))
                self.assertTrue(np.isfinite(diagnostics["initial_source_quality"]["chamfer"]))
                with patch.object(nsot, "load_cache", side_effect=AssertionError("cache at inference")), \
                        patch.object(anchor_flow, "fit_prior_centers", side_effect=AssertionError("fit at inference")):
                    torch.manual_seed(138)
                    expected_source = anchor_flow.sample_source(trained, 2, device="cpu", dtype=torch.float32)
                    torch.manual_seed(138)
                    noise, prediction, _ = experiment.sample_for_evaluation(model, trained, 2)
                    torch.testing.assert_close(noise, expected_source, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(prediction).all())
                    if dataset == "checkerboard":
                        output = evaluator(run / "config.yaml", 2, render=False)
                    else:
                        with patch.object(eval_horse, "render_comparison"), patch("horse_regions.HorseRegions.render"):
                            output = evaluator(run / "config.yaml", 2)
                result = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
                self.assertEqual(result["config"]["anchor_flow"], trained["anchor_flow"])
                self.assertTrue(np.isfinite(result["chamfer"]))
                self.assertTrue(np.isfinite(result["source_chamfer"]))


if __name__ == "__main__":
    unittest.main()
