"""Full-GMM NSOT cache, hybrid, matched evaluation and training integration."""

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
import diagnostic_data
import eval as eval_checkerboard
import eval_horse
import experiment
import gmm_prior
from model import PointSetTransformer
import nsot
import train
import train_horse


def config(dataset="checkerboard", *, gmm=True):
    value = {
        "seed": 0, "device": "cpu", "dtype": "float32", "coupling": "nsot", "num_regions": 2,
        "nsot": {"cache": f"{dataset}_{'full_gmm' if gmm else 'anchor'}.npz", "superset_size": 64,
                 "cache_seed": 5, "beta": .2, "solver": nsot.SOLVER},
        "data": {"batch_size": 2, "n_points": 8},
        "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                  "dim_feedforward": 16, "dropout": 0.0},
        "training": {"num_steps": 2, "learning_rate": .001, "weight_decay": .01, "log_every": 1},
        "evaluation": {"batch_size": 2, "histogram_bins": 8},
        "checkpoint": f"{dataset}_{'full_gmm' if gmm else 'anchor'}.pt",
    }
    if dataset == "checkerboard":
        value["data"]["grid_size"] = 4
    value["anchor_flow"] = ({"mode": "gmm_prior", "reference_points": 256, "seed": 11,
                             "n_init": 1, "max_iter": 300, "tol": 1e-3, "reg_covar": 1e-6}
                            if gmm else {"mode": "anchor_prior", "reference_points": 256, "seed": 11, "sigma": .1})
    return value


def parameters():
    return {"weights": [.25, .75], "means": [[-.6, .1], [.4, -.2]],
            "covariances": [[[.015, .007], [.007, .03]], [[.04, -.008], [-.008, .01]]]}


class HybridAndTargetTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_hybrid_preserves_correlated_component_covariances_at_all_betas(self):
        generator = torch.Generator().manual_seed(14)
        means = torch.tensor(parameters()["means"], dtype=torch.float64)
        covariance = torch.tensor(parameters()["covariances"], dtype=torch.float64)
        factors = torch.linalg.cholesky(covariance)
        labels = torch.multinomial(torch.tensor([.25, .75]), 80000, replacement=True, generator=generator)
        residual = torch.matmul(factors[labels], torch.randn(80000, 2, 1, dtype=torch.float64, generator=generator)).squeeze(-1)
        source = means[labels] + residual
        noise = torch.randn(source.shape, dtype=source.dtype, generator=generator)
        for beta in (0., .2, 1.):
            actual = nsot.full_covariance_hybrid(source, means[labels], factors[labels], beta=beta, noise=noise)
            for k in range(2):
                points = actual[labels == k]
                centered = points - points.mean(0)
                torch.testing.assert_close(points.mean(0), means[k], rtol=0, atol=.004)
                torch.testing.assert_close(centered.T @ centered / len(points), covariance[k], rtol=0, atol=.001)
            if beta == 0:
                torch.testing.assert_close(actual, source, rtol=0, atol=1e-15)
            if beta == 1:
                expected = means[labels] + torch.matmul(factors[labels], noise.unsqueeze(-1)).squeeze(-1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_hybrid_shape_beta_dtype(self):
        source = torch.zeros(2, 8, 2)
        means = torch.zeros_like(source)
        factors = torch.eye(2).expand(2, 8, 2, 2)
        for beta in (-1, 2, True, float("nan")):
            with self.assertRaises(ValueError):
                nsot.full_covariance_hybrid(source, means, factors, beta=beta, noise=source)
        with self.assertRaises(ValueError):
            nsot.full_covariance_hybrid(source, means, factors.double(), beta=.2, noise=source)

    def test_matched_targets_are_independent_of_prior_rng_and_restore_rng(self):
        for dataset in ("checkerboard", "horse"):
            draws = []
            for use_gmm in (False, True):
                value = config(dataset, gmm=use_gmm)
                if use_gmm:
                    gmm_prior.bind(value, parameters())
                else:
                    anchor_flow.bind_prior_centers(value, parameters()["means"])
                torch.manual_seed(24)
                anchor_flow.sample_source(value, 2, device="cpu", dtype=torch.float32)
                before = torch.get_rng_state().clone()
                draws.append(experiment.sample_evaluation_target(value, dataset, 2, device="cpu",
                                                                  dtype=torch.float32, seed=2026))
                torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            torch.testing.assert_close(draws[0], draws[1], rtol=0, atol=0)

    def test_legacy_target_draw_stays_bitwise_identical(self):
        for dataset in ("checkerboard", "horse"):
            value = config(dataset)
            torch.manual_seed(44)
            state = torch.get_rng_state().clone()
            if dataset == "horse":
                expected = train_horse.sample_horse(train_horse.load_horse_mask("cpu", torch.float32), 2, 8)
            else:
                from data import sample_checkerboard
                expected = sample_checkerboard(2, 8, "cpu", torch.float32, 4)
            torch.set_rng_state(state)
            actual = experiment.sample_evaluation_target(value, dataset, 2, device="cpu", dtype=torch.float32)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class GeneralGMMIntegrationTests(unittest.TestCase):
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

    def prepare(self, value, dataset):
        with contextlib.redirect_stdout(io.StringIO()):
            return nsot.prepare(value, dataset)

    def test_same_training_reference_and_target_superset_as_anchor(self):
        from coupling import balanced_target_partition
        for dataset in ("checkerboard", "horse"):
            captured = []

            def record(reference, k, **kwargs):
                captured.append(reference[0].numpy().copy())
                return balanced_target_partition(reference, k, **kwargs)

            with patch("coupling.balanced_target_partition", side_effect=record):
                anchor = nsot._draw_prior_supersets(config(dataset, gmm=False), dataset)
            general = nsot._draw_prior_supersets(config(dataset), dataset)
            np.testing.assert_array_equal(anchor[1], general[1])
            import hashlib
            self.assertEqual(hashlib.sha256(captured[0].tobytes()).hexdigest(),
                             general[4]["gmm_fit_details"]["training_reference_sha256"])

    def test_cache_roundtrip_wrong_family_and_changed_parameters_rejected(self):
        value = config()
        path, metadata = self.prepare(value, "checkerboard")
        self.assertEqual(metadata["format_version"], 3)
        self.assertIn("gmm_parameters", metadata)
        self.assertNotIn("prior_centers", metadata)
        loaded = copy.deepcopy(value)
        source, target, permutation, metadata, digest = nsot.load_cache(loaded, "checkerboard")
        self.assertEqual(loaded["anchor_flow"]["means"], metadata["gmm_parameters"]["means"])
        self.assertEqual(digest, nsot.file_sha256(path))
        expected, _ = nsot.exact_superset_permutation(source, target)
        np.testing.assert_array_equal(permutation, expected)
        changed = copy.deepcopy(loaded)
        changed["anchor_flow"]["means"][0][0] += .1
        with self.assertRaisesRegex(ValueError, "differ"):
            nsot.load_cache(changed, "checkerboard")
        changed = copy.deepcopy(value)
        changed["anchor_flow"]["reg_covar"] = .01
        with self.assertRaisesRegex(ValueError, "gmm_prior"):
            nsot.load_cache(changed, "checkerboard")
        wrong_family = config(gmm=False)
        wrong_family["nsot"]["cache"] = str(path)
        with self.assertRaisesRegex(ValueError, "format_version"):
            nsot.load_cache(wrong_family, "checkerboard")
        original_bytes = path.read_bytes()
        with patch.object(gmm_prior, "fit", side_effect=AssertionError("refit")):
            self.prepare(value, "checkerboard")
        self.assertEqual(path.read_bytes(), original_bytes)

    def test_sampler_replays_original_labels_and_covariance_hybrid(self):
        value = config()
        gmm_prior.bind(value, parameters())
        self.prepare(value, "checkerboard")
        for beta in (0., .2, 1.):
            value["nsot"]["beta"] = beta
            sampler = nsot.NSOTPairSampler(value, "checkerboard", "cpu", torch.float32)
            source, target = sampler.sample(3, generator=torch.Generator().manual_seed(42))
            replay = torch.Generator().manual_seed(42)
            indices = torch.randint(64, (3, 8), generator=replay)
            noise = torch.randn(3, 8, 2, generator=replay)
            labels = sampler.source_components[indices]
            expected = nsot.full_covariance_hybrid(sampler.source[indices], sampler.prior_centers[labels],
                                                    sampler.prior_cholesky[labels], beta=beta, noise=noise)
            torch.testing.assert_close(source, expected, rtol=0, atol=0)
            torch.testing.assert_close(target, sampler.target[indices], rtol=0, atol=0)

    def test_both_datasets_train_save_cache_free_inference_eval_and_diagnostics(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for dataset, trainer, model_class, evaluator in (
                    ("checkerboard", train.main, PointSetTransformer, eval_checkerboard.main),
                    ("horse", train_horse.main, train_horse.HorsePointSetTransformer, eval_horse.main)):
                value = config(dataset)
                self.prepare(value, dataset)
                template = Path(f"{dataset}.yaml")
                template.write_text(yaml.safe_dump(value), encoding="utf-8")
                seen = []
                original_step = train.train_step

                def record_step(*args, **kwargs):
                    seen.append(copy.deepcopy(kwargs["anchor_config"]))
                    return original_step(*args, **kwargs)

                with patch.object(train, "train_step", side_effect=record_step):
                    run = trainer(template, steps=2)
                self.assertEqual(len(seen), 2)
                self.assertNotIn("covariances", seen[0]["anchor_flow"])
                saved = experiment.read_config(run / "config.yaml")
                self.assertIn("covariances", saved["anchor_flow"])
                self.assertIn("gmm_prior", run.name)
                with patch.object(gmm_prior, "fit", side_effect=AssertionError("refit at inference")), \
                        patch.object(nsot, "_load_cache", side_effect=AssertionError("cache at inference")):
                    model, saved, _, metadata = experiment.load_model(run / "config.yaml", model_class, dataset)
                    self.assertTrue(metadata["training_config_verified"])
                    data = diagnostic_data.DiagnosticData(saved, dataset, "cpu", torch.float32, 2026)
                    self.assertTrue(torch.isfinite(data.source(0)).all())
                    if dataset == "horse":
                        with patch.object(eval_horse, "render_comparison"), patch("horse_regions.HorseRegions.render"):
                            output = evaluator(run / "config.yaml", 2, target_seed=2026)
                    else:
                        output = evaluator(run / "config.yaml", 2, render=False, target_seed=2026)
                result = json.loads(output.with_suffix(".json").read_text())
                self.assertEqual(result["evaluation_target_seed"], 2026)
                self.assertEqual(len(result["evaluation_target_sha256"]), 64)
                self.assertTrue(math.isfinite(result["chamfer"]))
                self.assertTrue(math.isfinite(result["source_chamfer"]))
                self.assertEqual(result["config"]["anchor_flow"], saved["anchor_flow"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_sampler_backward_and_matched_target(self):
        value = config()
        self.prepare(value, "checkerboard")
        sampler = nsot.NSOTPairSampler(value, "checkerboard", "cuda", torch.float32)
        source, target = sampler.sample(2, generator=torch.Generator(device="cuda").manual_seed(19))
        model = PointSetTransformer(**value["model"]).cuda()
        loss = train.train_step(model, torch.optim.AdamW(model.parameters()), target, coupling="nsot",
                                paired_noise=source, anchor_config=value)
        self.assertTrue(torch.isfinite(loss))
        state = torch.cuda.get_rng_state().clone()
        first = experiment.sample_evaluation_target(value, "checkerboard", 2, device="cuda", dtype=torch.float32, seed=2026)
        torch.testing.assert_close(torch.cuda.get_rng_state(), state, rtol=0, atol=0)
        torch.randn(500, device="cuda")
        second = experiment.sample_evaluation_target(value, "checkerboard", 2, device="cuda", dtype=torch.float32, seed=2026)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_real_four_job_smoke_runner(self):
        import run_prior_comparison as runner
        root = Path(self.temp.name).resolve()
        project = Path(__file__).resolve().parents[1]
        for _, _, relative in runner.JOBS:
            template = root / relative
            template.parent.mkdir(parents=True, exist_ok=True)
            template.write_text((project / relative).read_text(encoding="utf-8"), encoding="utf-8")
        with patch.object(runner, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
            manifest_path = runner.main(["--smoke"])
        manifest = runner.read_json(manifest_path)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["training_steps"], 4)
        self.assertEqual(len(manifest["jobs"]), 4)
        summary = runner.read_json(manifest["summary_json"])
        for dataset in ("checkerboard", "horse"):
            digests = {row["evaluation_target_sha256"] for job in summary["jobs"]
                       if job["dataset"] == dataset for row in job["evaluations"]}
            self.assertEqual(len(digests), 1)
            self.assertNotIn(None, digests)
        for job in summary["jobs"]:
            self.assertEqual([row["nfe"] for row in job["evaluations"]], [1, 2])
            self.assertEqual(job["source_baseline"]["nfe"], 0)
            self.assertIsNotNone(job["training_seconds"])
            self.assertIsNotNone(job["ot_seconds"])
            self.assertTrue(Path(job["saved_config"]).is_file())


if __name__ == "__main__":
    unittest.main()
