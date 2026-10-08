"""Tiny CPU checks for fixed anchor-GMM priors and unchanged Hard Bank TG."""

import contextlib
import copy
import io
import json
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
from diagnostic_data import DiagnosticData
import eval as eval_checkerboard
import eval_horse
import experiment
from model import PointSetTransformer
import tg_cache
import train
import train_horse


def tiny_config(mode=None, dataset="checkerboard"):
    config = {
        "seed": 0, "device": "cpu", "dtype": "float32", "coupling": "target_guided_cached",
        "num_regions": 2,
        "tg_cache": {"path": f"{dataset}_{mode or 'baseline'}_bank", "sampling": "bank",
                     "num_clouds": 8, "seed": 5, "prepare_batch_size": 4, "num_workers": 0},
        "data": {"batch_size": 2, "n_points": 8},
        "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                  "dim_feedforward": 16, "dropout": 0.0},
        "training": {"num_steps": 2, "learning_rate": .001, "weight_decay": .01, "log_every": 1},
        "evaluation": {"batch_size": 2, "histogram_bins": 8},
        "checkpoint": f"{dataset}_{mode or 'baseline'}.pt",
    }
    if dataset == "checkerboard":
        config["data"]["grid_size"] = 4
    if mode is not None:
        config["anchor_flow"] = {"mode": mode, "sigma": .1, "reference_points": 64, "seed": 11}
    return config


class CaptureVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.2))

    def forward(self, points, times):
        return self.scale * points


class AnchorFlowMathTests(unittest.TestCase):
    def test_unmodified_source_rng_is_bitwise_identical(self):
        config = tiny_config()
        expected_generator = torch.Generator().manual_seed(13)
        actual_generator = torch.Generator().manual_seed(13)
        expected = torch.randn(3, 8, 2, generator=expected_generator)
        actual = anchor_flow.sample_source(config, 3, device="cpu", dtype=torch.float32,
                                           generator=actual_generator)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_generator.get_state(), expected_generator.get_state(), rtol=0, atol=0)

    def test_prior_is_iid_gmm_not_whitened_or_fixed_counts(self):
        config = tiny_config("anchor_prior")
        config["data"]["n_points"] = 64
        anchor_flow.bind_prior_centers(config, [[-2., 0.], [2., 0.]])
        source = anchor_flow.sample_source(config, 256, device="cpu", dtype=torch.float32,
                                          generator=torch.Generator().manual_seed(123))
        right = source[..., 0] > 0
        self.assertLess(abs(right.double().mean().item() - .5), .015)
        selected = torch.where(right, 2., -2.).to(torch.float32)
        residual = source - torch.stack([selected, torch.zeros_like(selected)], dim=-1)
        self.assertLess(abs(residual.mean().item()), .005)
        self.assertLess(abs(residual.var(unbiased=False).item() - .01), .0008)
        self.assertGreater(source[..., 0].var().item(), 3.9)
        self.assertGreater(right.sum(1).unique().numel(), 5)
        repeated = anchor_flow.sample_source(config, 256, device="cpu", dtype=torch.float32,
                                             generator=torch.Generator().manual_seed(123))
        torch.testing.assert_close(source, repeated, rtol=0, atol=0)

    def test_nonfinite_and_invalid_experimental_settings_fail(self):
        for sigma in (float("nan"), float("inf"), -.1):
            config = tiny_config("anchor_prior")
            config["anchor_flow"]["sigma"] = sigma
            with self.assertRaises(ValueError):
                anchor_flow.settings(config)
        config = tiny_config("misspelled_mode")
        with self.assertRaises(ValueError):
            anchor_flow.settings(config)
        config = tiny_config("anchor_prior")
        for centers in ([[float("nan"), 0.], [1., 1.]], [[1., 2., 3.], [4., 5., 6.]]):
            with self.assertRaises(ValueError):
                anchor_flow.bind_prior_centers(config, centers)

    def test_baseline_fm_loss_is_bitwise_identical_with_optional_config(self):
        config = tiny_config()
        source, target = torch.randn(2, 8, 2), torch.randn(2, 8, 2)
        model = CaptureVelocity()
        torch.manual_seed(937)
        times = train.sample_time(2, device=source.device, dtype=source.dtype)
        expected = train.flow_matching_loss(model, target, source, times)
        expected_state = torch.get_rng_state()
        torch.manual_seed(937)
        actual = train.coupled_flow_matching_loss(
            model, target, coupling="target_guided_cached", paired_noise=source, anchor_config=config)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), expected_state, rtol=0, atol=0)

    def test_prior_loss_requires_actual_configured_offline_coupling_and_pairs(self):
        config = tiny_config("anchor_prior")
        source, target = torch.randn(2, 8, 2), torch.randn(2, 8, 2)
        model = CaptureVelocity()
        for coupling in ("independent", "nsot"):
            with self.assertRaisesRegex(ValueError, "coupling"):
                train.coupled_flow_matching_loss(model, target, coupling=coupling,
                                                 paired_noise=source, anchor_config=config)
        with self.assertRaisesRegex(ValueError, "prepared offline pairs"):
            train.coupled_flow_matching_loss(model, target, coupling=config["coupling"], anchor_config=config)

    def test_experiment_templates_keep_baseline_model_and_training_settings(self):
        root = Path(__file__).resolve().parents[1]
        for directory, prefix in (("checkerboard_experiments", ""), ("horse_experiments", "horse_")):
            baseline = yaml.safe_load((root / directory / f"{prefix}target_guided_cached_k8_n256_seed0.yaml").read_text())
            config = yaml.safe_load((root / directory / f"{prefix}target_guided_cached_anchor_prior_k8_n256_seed0.yaml").read_text())
            for key in ("model", "training", "data", "seed", "device", "dtype", "coupling", "num_regions"):
                self.assertEqual(config[key], baseline[key])
            self.assertEqual(config["anchor_flow"]["mode"], "anchor_prior")
            self.assertEqual(config["anchor_flow"]["sigma"], .1)
            self.assertNotEqual(config["tg_cache"]["path"], baseline["tg_cache"]["path"])


class AnchorFlowCacheIntegrationTests(unittest.TestCase):
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
    def quiet_prepare(config, dataset="checkerboard"):
        with contextlib.redirect_stdout(io.StringIO()):
            return tg_cache.prepare(config, dataset)

    def test_prior_fit_is_deterministic_training_only_and_rng_isolated(self):
        for dataset in ("checkerboard", "horse"):
            config = tiny_config("anchor_prior", dataset)
            torch.manual_seed(73)
            before = torch.get_rng_state()
            first = anchor_flow.fit_prior_centers(config, dataset)
            torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
            second = anchor_flow.fit_prior_centers(config, dataset)
            torch.testing.assert_close(torch.as_tensor(first), torch.as_tensor(second), rtol=0, atol=0)
            self.assertEqual(tuple(torch.as_tensor(first).shape), (2, 2))
            self.assertTrue(torch.isfinite(torch.as_tensor(first)).all())
            self.assertLessEqual(torch.as_tensor(first).abs().max().item(), 1.1)

    def test_prior_cache_is_separate_and_mismatched_sigma_or_centers_rejected(self):
        baseline = tiny_config()
        base_path, base_meta = self.quiet_prepare(baseline)
        prior = tiny_config("anchor_prior")
        anchor_flow.bind_prior_centers(prior, [[-.7, -.2], [.6, .3]])
        prior_path, prior_meta = self.quiet_prepare(prior)
        self.assertNotEqual(base_path, prior_path)
        self.assertNotEqual(base_meta["array_sha256"]["source"], prior_meta["array_sha256"]["source"])
        source = np.load(prior_path / "source.npy")
        anchors = np.asarray(prior["anchor_flow"]["centers"])
        self.assertLess(((source[..., None, :] - anchors) ** 2).sum(-1).min(-1).mean(), .035)
        for key, value in (("sigma", .2), ("centers", [[-.6, -.2], [.6, .3]])):
            mismatch = copy.deepcopy(prior)
            mismatch["anchor_flow"][key] = value
            with self.assertRaises(ValueError):
                tg_cache.load_cache(mismatch, "checkerboard")
        mismatch = copy.deepcopy(prior)
        mismatch["tg_cache"]["path"] = str(base_path)
        with self.assertRaises(ValueError):
            tg_cache.load_cache(mismatch, "checkerboard")

    def test_tiny_train_save_load_and_cache_free_generation_both_datasets(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for dataset, trainer, evaluator, model_class in (
                    ("checkerboard", train.main, eval_checkerboard.main, PointSetTransformer),
                    ("horse", train_horse.main, eval_horse.main, train_horse.HorsePointSetTransformer)):
                config = tiny_config("anchor_prior", dataset)
                self.quiet_prepare(config, dataset)
                path = Path(f"{dataset}_anchor_prior.yaml")
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                run = trainer(path, steps=1)
                self.assertIn("anchor_prior", run.name)
                model, trained, checkpoint, metadata = experiment.load_model(run / "config.yaml", model_class, dataset)
                self.assertTrue(metadata["training_config_verified"])
                self.assertEqual(len(trained["anchor_flow"]["centers"]), 2)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.assertEqual(payload["config"]["anchor_flow"]["centers"], trained["anchor_flow"]["centers"])
                with patch.object(tg_cache, "load_cache", side_effect=AssertionError("cache at inference")), \
                        patch.object(anchor_flow, "fit_prior_centers", side_effect=AssertionError("prior refit at inference")):
                    torch.manual_seed(138)
                    expected_source = anchor_flow.sample_source(trained, 2, device="cpu", dtype=torch.float32)
                    torch.manual_seed(138)
                    noise, prediction, seconds = experiment.sample_for_evaluation(model, trained, 2)
                    torch.testing.assert_close(noise, expected_source, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(prediction).all())
                    self.assertGreaterEqual(seconds, 0.)
                    data = DiagnosticData(trained, dataset, "cpu", torch.float32, 2026)
                    expected_diagnostic = anchor_flow.sample_source(
                        trained, 1, device="cpu", dtype=torch.float32,
                        generator=torch.Generator().manual_seed(data.draw_seed("evaluation:source", 0)))[0]
                    torch.testing.assert_close(data.source(0), expected_diagnostic, rtol=0, atol=0)
                    if dataset == "checkerboard":
                        output = evaluator(run / "config.yaml", 2, render=False)
                    else:
                        with patch.object(eval_horse, "render_comparison"), patch("horse_regions.HorseRegions.render"):
                            output = evaluator(run / "config.yaml", 2)
                    result = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
                    self.assertTrue(np.isfinite(result["chamfer"]))
                    self.assertTrue(np.isfinite(result["source_chamfer"]))
                    self.assertEqual(result["config"]["anchor_flow"], trained["anchor_flow"])
                    with patch.object(audit_generation, "render"), patch("horse_regions.HorseRegions.render"):
                        audit_dir = audit_generation.audit(
                            run / "config.yaml", dataset, clouds=2, batch_size=2,
                            nfes=(1, 2), reference_nfe=2, max_reference_nfe=8, fm_batches=0, seed=2026)
                    audit_result = json.loads((audit_dir / "diagnostics.json").read_text(encoding="utf-8"))
                    self.assertTrue(np.isfinite(audit_result["initial_source_quality"]["chamfer"]))
                    self.assertIsNone(audit_result["fm"])

    def test_windows_worker_and_cuda_backward_for_anchor_prior(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = tiny_config("anchor_prior")
        config["tg_cache"]["num_workers"] = 1
        self.quiet_prepare(config)
        sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", device, torch.float32, training=True)
        try:
            source, target = sampler.sample(2)
            model = PointSetTransformer(**config["model"]).to(device)
            loss = train.train_step(model, torch.optim.AdamW(model.parameters()), target,
                                    coupling=config["coupling"], paired_noise=source, anchor_config=config,
                                    coupling_generator=torch.Generator(device=device).manual_seed(4))
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(source.device.type, device.type)
        finally:
            sampler.close()


if __name__ == "__main__":
    unittest.main()
