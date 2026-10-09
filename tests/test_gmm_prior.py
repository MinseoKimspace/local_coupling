"""Controlled fixed-variance GMM-EM versus balanced-anchor NSOT priors.

The comparison changes center fitting only: equal component weights, sigma,
source/target RNG streams, backbone, hybrid kernel and OT solver stay fixed.
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
from diagnostic_data import DiagnosticData
import eval as eval_checkerboard
import eval_horse
import experiment
from model import PointSetTransformer
import nsot
import train
import train_horse


def tiny_config(dataset="checkerboard", *, gmm=True):
    variant = "gmm_em" if gmm else "anchor"
    config = {
        "seed": 0, "device": "cpu", "dtype": "float32", "coupling": "nsot",
        "num_regions": 2,
        "anchor_flow": {"mode": "anchor_prior", "sigma": .1,
                        "reference_points": 64, "seed": 11},
        "nsot": {"cache": f"{dataset}_{variant}.npz", "superset_size": 64,
                 "cache_seed": 5, "beta": .2, "solver": nsot.SOLVER},
        "data": {"batch_size": 2, "n_points": 8},
        "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                  "dim_feedforward": 16, "dropout": 0.0},
        "training": {"num_steps": 2, "learning_rate": .001,
                     "weight_decay": .01, "log_every": 1},
        "evaluation": {"batch_size": 2, "histogram_bins": 8},
        "checkpoint": f"{dataset}_{variant}.pt",
    }
    if dataset == "checkerboard":
        config["data"]["grid_size"] = 4
    if gmm:
        config["anchor_flow"].update(center_fit="gmm_em", gmm_n_init=2,
                                     gmm_max_iter=30, gmm_tol=1e-6)
    return config


def mixture_objective(points, centers, sigma):
    points, centers = points.double(), torch.as_tensor(centers).double()
    logits = -((points[:, None] - centers[None]) ** 2).sum(-1) / (2 * sigma ** 2)
    # Constants do not depend on fitted centers. Higher is better.
    return torch.logsumexp(logits, dim=1).mean().item()


class GMMSettingsTests(unittest.TestCase):
    def test_legacy_cache_spec_is_unchanged_even_when_method_is_explicit(self):
        config = tiny_config(gmm=False)
        expected = {"mode": "anchor_prior", "sigma": .1,
                    "reference_points": 64, "seed": 11}
        self.assertEqual(anchor_flow.cache_spec(config), expected)
        config["anchor_flow"]["center_fit"] = "balanced_fps_ot"
        self.assertEqual(anchor_flow.cache_spec(config), expected)
        anchor_flow.bind_prior_centers(config, [[-.4, 0.], [.4, 0.]])
        self.assertEqual(anchor_flow.cache_spec(config), expected)
        self.assertEqual(anchor_flow.variant_suffix(config), "anchor_prior")

    def test_gmm_spec_identity_and_metadata_are_explicit(self):
        config = tiny_config()
        opts = anchor_flow.settings(config)
        self.assertEqual(opts["center_fit"], "gmm_em")
        self.assertEqual(anchor_flow.cache_spec(config)["gmm_n_init"], 2)
        anchor_flow.bind_prior_centers(config, [[-.4, 0.], [.4, 0.]])
        self.assertNotIn("centers", anchor_flow.cache_spec(config))
        details = anchor_flow.experiment_details(config)
        self.assertIn("em", details["prior_center_fit"].lower())
        self.assertEqual(anchor_flow.variant_suffix(config), "anchor_prior_gmm_em")
        self.assertIn("1/K", details["source_prior"])

    def test_unknown_method_invalid_parameters_and_tg_gmm_are_rejected(self):
        invalid = {
            "center_fit": (None, "kmeans", "GMM", 1),
            "gmm_n_init": (0, -1, 1.5, True),
            "gmm_max_iter": (0, -1, 1.5, True),
            "gmm_tol": (-1., 0., float("nan"), float("inf"), True, "small"),
        }
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    config = tiny_config()
                    config["anchor_flow"][field] = value
                    with self.assertRaises(ValueError):
                        anchor_flow.settings(config)
        config = tiny_config()
        config["coupling"] = "target_guided_cached"
        with self.assertRaises(ValueError):
            anchor_flow.settings(config)

    def test_yaml_comparison_changes_only_center_fitting_and_output_paths(self):
        root = Path(__file__).resolve().parents[1]
        for directory, prefix in (("checkerboard_experiments", ""),
                                  ("horse_experiments", "horse_")):
            baseline = yaml.safe_load((root / directory /
                f"{prefix}nsot_anchor_prior_k8_n256_seed0.yaml").read_text(encoding="utf-8"))
            gmm = yaml.safe_load((root / directory /
                f"{prefix}nsot_gmm_prior_k8_n256_seed0.yaml").read_text(encoding="utf-8"))
            fitting = {"center_fit", "gmm_n_init", "gmm_max_iter", "gmm_tol"}
            self.assertEqual({k: v for k, v in gmm["anchor_flow"].items() if k not in fitting},
                             baseline["anchor_flow"])
            self.assertEqual(gmm["anchor_flow"]["center_fit"], "gmm_em")
            self.assertEqual(gmm["num_regions"], 8)
            self.assertEqual(gmm["anchor_flow"]["sigma"], .1)
            self.assertNotEqual(gmm["nsot"]["cache"], baseline["nsot"]["cache"])
            self.assertNotEqual(gmm["checkpoint"], baseline["checkpoint"])
            normalized = copy.deepcopy(gmm)
            normalized["anchor_flow"] = baseline["anchor_flow"]
            normalized["nsot"]["cache"] = baseline["nsot"]["cache"]
            normalized["checkpoint"] = baseline["checkpoint"]
            self.assertEqual(normalized, baseline)


class GMMFitTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_deterministic_training_fits_restore_rng_and_thread_count(self):
        for dataset in ("checkerboard", "horse"):
            config = tiny_config(dataset)
            torch.manual_seed(79)
            before = torch.get_rng_state()
            first = anchor_flow.fit_prior_centers(config, dataset)
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            self.assertEqual(torch.get_num_threads(), 1)
            torch.manual_seed(977)
            second = anchor_flow.fit_prior_centers(config, dataset)
            torch.testing.assert_close(torch.as_tensor(first), torch.as_tensor(second), rtol=0, atol=0)
            self.assertEqual(tuple(torch.as_tensor(first).shape), (2, 2))
            self.assertTrue(torch.isfinite(torch.as_tensor(first)).all())
            self.assertLess(torch.as_tensor(first).abs().max().item(), 1.1)
            anchor_flow.bind_prior_centers(config, first)
            with patch.object(anchor_flow, "_fit_gmm_centers", side_effect=AssertionError("refit")):
                self.assertEqual(anchor_flow.fit_prior_centers(config, dataset), first)

    def test_both_fitters_receive_the_exact_same_reference_sample(self):
        for dataset in ("checkerboard", "horse"):
            captured = {}

            def balanced(reference, k, **kwargs):
                captured["balanced"] = reference.detach().clone()
                return None, None, torch.zeros(1, k, 2), None

            def gmm(reference, k, *args, **kwargs):
                captured["gmm"] = reference.detach().clone()
                return torch.zeros(k, 2)

            with patch("coupling.balanced_target_partition", side_effect=balanced), \
                    patch.object(anchor_flow, "_fit_gmm_centers", side_effect=gmm):
                anchor_flow.fit_prior_centers(tiny_config(dataset, gmm=False), dataset)
                anchor_flow.fit_prior_centers(tiny_config(dataset), dataset)
            torch.testing.assert_close(captured["balanced"].reshape(-1, 2),
                                       captured["gmm"].reshape(-1, 2), rtol=0, atol=0)

    def test_fit_failure_restores_caller_rng_and_threads(self):
        torch.manual_seed(113)
        state = torch.get_rng_state()
        with patch.object(anchor_flow, "_fit_gmm_centers", side_effect=RuntimeError("test failure")):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                anchor_flow.fit_prior_centers(tiny_config(), "checkerboard")
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
        self.assertEqual(torch.get_num_threads(), 1)

    def test_fixed_sigma_em_recovers_separated_means(self):
        generator = torch.Generator().manual_seed(123)
        points = torch.cat((torch.tensor([[-1.5, -.2]]) + .1 * torch.randn(300, 2, generator=generator),
                            torch.tensor([[1.5, .3]]) + .1 * torch.randn(300, 2, generator=generator)))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(57)
            centers = anchor_flow._fit_gmm_centers(points, 2, .1, 3, 100, 1e-6)
        centers = torch.as_tensor(centers)
        centers = centers[torch.argsort(centers[:, 0])]
        torch.testing.assert_close(centers.float(), torch.tensor([[-1.5, -.2], [1.5, .3]]),
                                   rtol=0, atol=.025)
        self.assertGreater(mixture_objective(points, centers, .1),
                           mixture_objective(points, torch.zeros(2, 2), .1) + 50)

    def test_more_em_iterations_do_not_worsen_the_fixed_mixture_objective(self):
        generator = torch.Generator().manual_seed(112)
        points = torch.randn(200, 2, generator=generator) * .3
        fitted = []
        for iterations in (1, 40):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(71)
                fitted.append(anchor_flow._fit_gmm_centers(points, 3, .2, 1, iterations, 1e-12))
        self.assertGreaterEqual(mixture_objective(points, fitted[1], .2),
                                mixture_objective(points, fitted[0], .2) - 1e-10)

    def test_fit_choice_does_not_change_target_or_label_noise_rng_streams(self):
        for dataset in ("checkerboard", "horse"):
            arrays = []
            for use_gmm in (False, True):
                config = tiny_config(dataset, gmm=use_gmm)
                torch.manual_seed(419)
                before = torch.get_rng_state()
                arrays.append(nsot._draw_prior_supersets(config, dataset))
                torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            source_a, target_a, labels_a, centers_a, details_a = arrays[0]
            source_b, target_b, labels_b, centers_b, details_b = arrays[1]
            np.testing.assert_array_equal(target_a, target_b)
            np.testing.assert_array_equal(labels_a, labels_b)
            residual_a = source_a - np.asarray(centers_a, dtype=np.float32)[labels_a]
            residual_b = source_b - np.asarray(centers_b, dtype=np.float32)[labels_b]
            # Addition of distinct centers rounds at float32 precision.
            np.testing.assert_allclose(residual_a, residual_b, rtol=0, atol=1.2e-7)
            self.assertEqual(details_a["source_draw_seed"], details_b["source_draw_seed"])
            self.assertEqual(details_a["target_draw_seed"], details_b["target_draw_seed"])


class GMMCacheIntegrationTests(unittest.TestCase):
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
    def prepare(config, dataset):
        with contextlib.redirect_stdout(io.StringIO()):
            return nsot.prepare(config, dataset)

    def test_cache_identity_rejects_cross_fit_reuse_and_changed_fit_options(self):
        anchor = tiny_config(gmm=False)
        gmm = tiny_config()
        anchor_path, anchor_meta = self.prepare(anchor, "checkerboard")
        gmm_path, gmm_meta = self.prepare(gmm, "checkerboard")
        self.assertNotEqual(anchor_path, gmm_path)
        self.assertNotEqual(anchor_meta["anchor_prior"], gmm_meta["anchor_prior"])
        for requested, wrong in ((gmm, anchor_path), (anchor, gmm_path)):
            config = copy.deepcopy(requested)
            config["nsot"]["cache"] = str(wrong)
            with self.assertRaisesRegex(ValueError, "anchor_prior"):
                nsot.load_cache(config, "checkerboard")
        for field, value in (("gmm_n_init", 3), ("gmm_max_iter", 31), ("gmm_tol", 1e-5)):
            config = copy.deepcopy(gmm)
            config["anchor_flow"][field] = value
            with self.assertRaisesRegex(ValueError, "anchor_prior"):
                nsot.load_cache(config, "checkerboard")
        loaded = copy.deepcopy(gmm)
        nsot.load_cache(loaded, "checkerboard")
        self.assertEqual(loaded["anchor_flow"]["centers"], gmm_meta["prior_centers"])
        with patch.object(anchor_flow, "fit_prior_centers", side_effect=AssertionError("refit existing cache")):
            self.prepare(loaded, "checkerboard")

    def test_both_datasets_tiny_train_save_load_and_cache_free_inference(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for dataset, trainer, evaluator, model_class in (
                    ("checkerboard", train.main, eval_checkerboard.main, PointSetTransformer),
                    ("horse", train_horse.main, eval_horse.main, train_horse.HorsePointSetTransformer)):
                config = tiny_config(dataset)
                self.prepare(config, dataset)
                path = Path(f"{dataset}_gmm.yaml")
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                run = trainer(path, steps=1)
                self.assertIn("anchor_prior_gmm_em", run.name)
                model, trained, checkpoint, metadata = experiment.load_model(run / "config.yaml", model_class, dataset)
                self.assertTrue(metadata["training_config_verified"])
                self.assertEqual(trained["anchor_flow"]["center_fit"], "gmm_em")
                self.assertEqual(len(trained["anchor_flow"]["centers"]), 2)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.assertEqual(payload["config"]["anchor_flow"], trained["anchor_flow"])
                with patch.object(nsot, "load_cache", side_effect=AssertionError("cache at inference")), \
                        patch.object(anchor_flow, "fit_prior_centers", side_effect=AssertionError("fit at inference")):
                    torch.manual_seed(138)
                    expected = anchor_flow.sample_source(trained, 2, device="cpu", dtype=torch.float32)
                    torch.manual_seed(138)
                    source, prediction, seconds = experiment.sample_for_evaluation(model, trained, 2)
                    torch.testing.assert_close(source, expected, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(prediction).all())
                    self.assertGreaterEqual(seconds, 0.)
                    diagnostic = DiagnosticData(trained, dataset, "cpu", torch.float32, 2026)
                    self.assertTrue(torch.isfinite(diagnostic.source(0)).all())
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
