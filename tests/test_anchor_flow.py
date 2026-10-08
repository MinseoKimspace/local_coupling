"""Regression checks for matched anchor-GMM priors and anchor-waypoint paths.

These tests use tiny CPU clouds; they check the experimental laws and wiring,
not whether either experiment improves generation quality.
"""

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
import audit_mean_field
import diagnose
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
        config["anchor_flow"] = {"mode": mode, "sigma": .1}
        if mode == "anchor_prior":
            config["anchor_flow"].update(reference_points=64, seed=11)
    return config


class CaptureVelocity(torch.nn.Module):
    def __init__(self, velocity=None):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.2))
        self.velocity = velocity
        self.seen = None

    def forward(self, points, times):
        self.seen = (points.detach().clone(), times.detach().clone())
        if self.velocity is not None:
            return self.velocity.to(points) + self.scale * 0
        return self.scale * points


class AnchorFlowMathTests(unittest.TestCase):
    def test_waypoint_endpoints_midpoint_and_exact_velocity(self):
        generator = torch.Generator().manual_seed(7)
        source = torch.randn(2, 5, 2, dtype=torch.float64, generator=generator)
        target = torch.randn(2, 5, 2, dtype=torch.float64, generator=generator)
        waypoint = torch.randn(2, 5, 2, dtype=torch.float64, generator=generator)
        for value, expected in ((0., source), (.5, waypoint), (1., target)):
            time = source.new_full((2, 1, 1), value)
            position, velocity = anchor_flow.path_and_velocity(source, target, time, waypoint=waypoint)
            torch.testing.assert_close(position, expected, rtol=0, atol=1e-15)
            self.assertTrue(torch.isfinite(velocity).all())
            if value in (0., 1.):
                torch.testing.assert_close(velocity, target - source, rtol=0, atol=0)
        time = source.new_tensor([.17, .73]).reshape(2, 1, 1)
        position, velocity = anchor_flow.path_and_velocity(source, target, time, waypoint=waypoint)
        midpoint_offset = waypoint - .5 * (source + target)
        torch.testing.assert_close(position, (1 - time) * source + time * target
                                   + 16 * time**2 * (1 - time)**2 * midpoint_offset)
        torch.testing.assert_close(velocity, target - source + 32 * time * (1 - time) * (1 - 2 * time) * midpoint_offset)
        _, derivative = torch.autograd.functional.jvp(
            lambda times: anchor_flow.path_and_velocity(source, target, times, waypoint=waypoint)[0],
            (time,), (torch.ones_like(time),))
        torch.testing.assert_close(derivative, velocity, rtol=1e-14, atol=1e-14)

    def test_unmodified_linear_path_and_source_rng_are_bitwise_identical(self):
        source, target = torch.randn(2, 8, 2), torch.randn(2, 8, 2)
        times = torch.rand(2, 1, 1)
        position, velocity = anchor_flow.path_and_velocity(source, target, times)
        torch.testing.assert_close(position, train.linear_path(target, source, times), rtol=0, atol=0)
        torch.testing.assert_close(velocity, target - source, rtol=0, atol=0)
        for mode in (None, "anchor_waypoint"):
            config = tiny_config(mode)
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
        self.assertGreater(source[..., 0].var().item(), 3.9)  # no mean/cov whitening
        counts = right.sum(1)
        self.assertGreater(counts.unique().numel(), 5)  # not 32 points forced per anchor
        repeated = anchor_flow.sample_source(config, 256, device="cpu", dtype=torch.float32,
                                             generator=torch.Generator().manual_seed(123))
        torch.testing.assert_close(source, repeated, rtol=0, atol=0)

    def test_waypoint_noise_has_declared_scale_and_preserves_centers(self):
        config = tiny_config("anchor_waypoint")
        centers = torch.tensor([[[-.8, .2], [.6, -.3]]], dtype=torch.float64).expand(5000, -1, -1).clone()
        original = centers.clone()
        waypoint = anchor_flow.make_waypoint(config, centers, generator=torch.Generator().manual_seed(29))
        residual = waypoint - centers
        torch.testing.assert_close(centers, original, rtol=0, atol=0)
        self.assertLess(abs(residual.mean().item()), .003)
        self.assertLess(abs(residual.var(unbiased=False).item() - .01), .0008)
        repeated = anchor_flow.make_waypoint(config, centers, generator=torch.Generator().manual_seed(29))
        torch.testing.assert_close(waypoint, repeated, rtol=0, atol=0)

    def test_nonfinite_and_invalid_experimental_settings_fail(self):
        for mode in ("anchor_prior", "anchor_waypoint"):
            for sigma in (float("nan"), float("inf"), -.1):
                config = tiny_config(mode)
                config["anchor_flow"]["sigma"] = sigma
                with self.assertRaises(ValueError):
                    anchor_flow.settings(config)
        config = tiny_config("misspelled_mode")
        with self.assertRaises(ValueError):
            anchor_flow.settings(config)
        config = tiny_config("anchor_prior")
        with self.assertRaises(ValueError):
            anchor_flow.bind_prior_centers(config, [[float("nan"), 0.], [1., 1.]])
        with self.assertRaises(ValueError):
            anchor_flow.bind_prior_centers(config, [[1., 2., 3.], [4., 5., 6.]])

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

    def test_training_waypoint_uses_curved_position_and_midpoint_velocity(self):
        config = tiny_config("anchor_waypoint")
        source, target = torch.randn(2, 8, 2), torch.randn(2, 8, 2)
        centers = torch.full_like(source, 10.)
        model = CaptureVelocity(target - source)
        with patch.object(train, "sample_time", return_value=source.new_full((2, 1, 1), .5)):
            loss = train.coupled_flow_matching_loss(
                model, target, coupling="target_guided_cached", paired_noise=source,
                anchor_config=config, waypoint_centers=centers)
        torch.testing.assert_close(loss, torch.zeros_like(loss), rtol=0, atol=0)
        self.assertLess((model.seen[0] - centers).abs().mean().item(), .2)
        self.assertGreater((model.seen[0] - .5 * (source + target)).abs().mean().item(), 5.)

    def test_training_waypoint_uses_derivative_not_displacement_away_from_midpoint(self):
        config = tiny_config("anchor_waypoint")
        source, target, waypoint = torch.randn(2, 8, 2), torch.randn(2, 8, 2), torch.randn(2, 8, 2)
        time = source.new_full((2, 1, 1), .25)
        state, velocity = anchor_flow.path_and_velocity(source, target, time, waypoint=waypoint)
        self.assertGreater((velocity - (target - source)).abs().mean().item(), .1)
        model = CaptureVelocity(velocity)
        with patch.object(train, "sample_time", return_value=time), \
                patch.object(anchor_flow, "make_waypoint", return_value=waypoint):
            loss = train.coupled_flow_matching_loss(
                model, target, coupling="target_guided_cached", paired_noise=source,
                anchor_config=config, waypoint_centers=torch.zeros_like(source))
        torch.testing.assert_close(loss, torch.zeros_like(loss), rtol=0, atol=0)
        torch.testing.assert_close(model.seen[0], state, rtol=0, atol=0)

    def test_experiment_templates_keep_baseline_model_and_training_settings(self):
        root = Path(__file__).resolve().parents[1]
        for directory, prefix in (("checkerboard_experiments", ""), ("horse_experiments", "horse_")):
            baseline = yaml.safe_load((root / directory / f"{prefix}target_guided_cached_k8_n256_seed0.yaml").read_text())
            for mode in ("anchor_prior", "anchor_waypoint"):
                config = yaml.safe_load((root / directory / f"{prefix}target_guided_cached_{mode}_k8_n256_seed0.yaml").read_text())
                for key in ("model", "training", "data", "seed", "device", "dtype", "coupling", "num_regions"):
                    self.assertEqual(config[key], baseline[key])
                self.assertEqual(config["anchor_flow"]["mode"], mode)
                self.assertEqual(config["anchor_flow"]["sigma"], .1)
                # Templates use separate paths to protect existing artifacts.
                # Manual waypoint reuse is checked separately below.
                self.assertNotEqual(config["tg_cache"]["path"], baseline["tg_cache"]["path"])

    def test_waypoint_rejects_linear_path_diagnostics(self):
        config = tiny_config("anchor_waypoint")
        source, target = torch.randn(2, 8, 2), torch.randn(2, 8, 2)
        with self.assertRaisesRegex(ValueError, "linear|Linear"):
            diagnose.fm_errors(None, source, target, torch.zeros(2, 1, 1), config=config)
        with self.assertRaisesRegex(ValueError, "cached coupling"):
            audit_mean_field.conditional_moments(source, target, config)
        with self.assertRaisesRegex(ValueError, "skip-fm"):
            audit_generation.fm_summary(None, config, None, 1, 2, 2026)


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
        squared = ((source[..., None, :] - anchors) ** 2).sum(-1).min(-1)
        self.assertLess(squared.mean(), .035)
        for key, value in (("sigma", .2), ("centers", [[-.6, -.2], [.6, .3]])):
            mismatch = copy.deepcopy(prior)
            mismatch["anchor_flow"][key] = value
            with self.assertRaises(ValueError):
                tg_cache.load_cache(mismatch, "checkerboard")
        mismatch = copy.deepcopy(prior)
        mismatch["tg_cache"]["path"] = str(base_path)
        with self.assertRaises(ValueError):
            tg_cache.load_cache(mismatch, "checkerboard")

    def test_waypoint_reuses_baseline_cache_and_centers_follow_source_patch(self):
        baseline = tiny_config()
        path, meta = self.quiet_prepare(baseline)
        config = copy.deepcopy(baseline)
        config["anchor_flow"] = {"mode": "anchor_waypoint", "sigma": .1}
        reused, reused_meta = self.quiet_prepare(config)
        self.assertEqual(path, reused)
        self.assertEqual(meta, reused_meta)
        target = np.load(path / "target.npy")[0]
        source_labels = np.load(path / "source_labels.npy")[0]
        target_labels = np.load(path / "target_labels.npy")[0]
        expected_centers = np.stack([target[target_labels == k].mean(0) for k in range(2)])[source_labels]
        sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32)
        try:
            rng, seen = np.random.default_rng(23), set()
            for _ in range(10):
                source, permuted, centers = sampler.dataset.draw(0, rng)
                torch.testing.assert_close(centers, torch.from_numpy(expected_centers), rtol=1e-6, atol=1e-7)
                np.testing.assert_array_equal(source.numpy(), np.load(path / "source.npy")[0])
                self.assertEqual(sorted(map(tuple, target)), sorted(map(tuple, permuted.numpy())))
                seen.add(tuple(permuted.flatten().tolist()))
            self.assertGreater(len(seen), 1)
            batch = sampler.sample(2, generator=torch.Generator().manual_seed(3))
            self.assertEqual(len(batch), 3)
            self.assertEqual(batch[2].shape, (2, 8, 2))
        finally:
            sampler.close()

    def test_tiny_train_save_load_and_cache_free_generation_both_datasets_modes(self):
        audit_targets = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for dataset, trainer, evaluator, model_class in (
                    ("checkerboard", train.main, eval_checkerboard.main, PointSetTransformer),
                    ("horse", train_horse.main, eval_horse.main, train_horse.HorsePointSetTransformer)):
                for mode in ("anchor_prior", "anchor_waypoint"):
                    config = tiny_config(mode, dataset)
                    self.quiet_prepare(config, dataset)
                    path = Path(f"{dataset}_{mode}.yaml")
                    path.write_text(yaml.safe_dump(config), encoding="utf-8")
                    run = trainer(path, steps=1)
                    self.assertIn(mode, run.name)
                    model, trained, checkpoint, metadata = experiment.load_model(
                        run / "config.yaml", model_class, dataset)
                    self.assertTrue(metadata["training_config_verified"])
                    self.assertEqual(trained["anchor_flow"]["mode"], mode)
                    if mode == "anchor_prior":
                        self.assertEqual(len(trained["anchor_flow"]["centers"]), 2)
                        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                        self.assertEqual(payload["config"]["anchor_flow"]["centers"],
                                         trained["anchor_flow"]["centers"])
                    with patch.object(tg_cache, "load_cache", side_effect=AssertionError("cache at inference")), \
                            patch.object(anchor_flow, "fit_prior_centers", side_effect=AssertionError("prior refit at inference")):
                        torch.manual_seed(138)
                        expected_source = anchor_flow.sample_source(trained, 2, device="cpu", dtype=torch.float32)
                        torch.manual_seed(138)
                        noise, prediction, seconds = experiment.sample_for_evaluation(model, trained, 2)
                        torch.testing.assert_close(noise, expected_source, rtol=0, atol=0)
                        self.assertEqual(noise.shape, (2, 8, 2))
                        self.assertTrue(torch.isfinite(prediction).all())
                        self.assertGreaterEqual(seconds, 0.)
                        data = DiagnosticData(trained, dataset, "cpu", torch.float32, 2026)
                        self.assertEqual(data.source(0).shape, (8, 2))
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
                        self.assertEqual(result["config"]["anchor_flow"], trained["anchor_flow"])
                        self.assertTrue(np.isfinite(result["source_chamfer"]))
                        with patch.object(audit_generation, "render"), patch("horse_regions.HorseRegions.render"):
                            audit_dir = audit_generation.audit(
                                run / "config.yaml", dataset, clouds=2, batch_size=2,
                                nfes=(1, 2), reference_nfe=2, max_reference_nfe=8,
                                fm_batches=0, seed=2026)
                        audit_result = json.loads((audit_dir / "diagnostics.json").read_text(encoding="utf-8"))
                        self.assertTrue(np.isfinite(audit_result["initial_source_quality"]["chamfer"]))
                        self.assertIsNone(audit_result["fm"])
                        if dataset in audit_targets:
                            self.assertEqual(audit_result["target_sha256"], audit_targets[dataset])
                        audit_targets[dataset] = audit_result["target_sha256"]

    def test_windows_worker_and_cuda_backward_for_both_variants(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for mode in ("anchor_prior", "anchor_waypoint"):
            config = tiny_config(mode)
            config["tg_cache"]["num_workers"] = 1
            self.quiet_prepare(config)
            sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", device, torch.float32, training=True)
            try:
                source, target, *extra = sampler.sample(2)
                model = PointSetTransformer(**config["model"]).to(device)
                loss = train.train_step(
                    model, torch.optim.AdamW(model.parameters()), target,
                    coupling=config["coupling"], paired_noise=source, anchor_config=config,
                    waypoint_centers=extra[0] if extra else None,
                    coupling_generator=torch.Generator(device=device).manual_seed(4))
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(source.device.type, device.type)
            finally:
                sampler.close()


if __name__ == "__main__":
    unittest.main()
