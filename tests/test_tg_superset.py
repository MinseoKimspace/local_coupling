import contextlib
import copy
import io
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

import audit_generation
import audit_jacobian
import coupling
from diagnostic_data import DiagnosticData
import eval as eval_checkerboard
import eval_horse
from experiment import load_model
from model import PointSetTransformer
import run_tg_superset as runner
import tg_cache
import tg_superset
import train
import train_horse


ROOT = Path(__file__).resolve().parents[1]


class SubsetTests(unittest.TestCase):
    def test_unique_indices_same_counts_no_fixed_quota_and_fresh_fine(self):
        labels = np.array([0, 1, 0, 2, 2, 1, 2, 0, 1])
        target_labels = labels[::-1]
        groups = [np.flatnonzero(target_labels == k) for k in range(3)]
        rng, counts, bijections = np.random.default_rng(41), set(), set()
        for _ in range(100):
            source, target = tg_superset.subset_indices(labels, groups, 5, rng)
            self.assertEqual(len(np.unique(source)), 5)
            self.assertEqual(len(np.unique(target)), 5)
            np.testing.assert_array_equal(labels[source], target_labels[target])
            counts.add(tuple(np.bincount(labels[source], minlength=3)))
            source, target = tg_superset.subset_indices(labels, groups, len(labels), rng)
            bijections.add(tuple(target[np.argsort(source)]))
        self.assertGreater(len(counts), 3)
        self.assertGreater(len(bijections), 3)

    def test_uniform_source_AND_target_subset_law_with_unequal_capacities(self):
        labels = np.array([0, 0, 1, 2, 2, 2])
        target_labels = labels[::-1]
        groups = [np.flatnonzero(target_labels == k) for k in range(3)]
        subsets = list(itertools.combinations(range(6), 2))
        source_counts = dict.fromkeys(subsets, 0)
        target_counts = dict.fromkeys(subsets, 0)
        rng = np.random.default_rng(4)
        for _ in range(15000):
            source, target = tg_superset.subset_indices(labels, groups, 2, rng)
            source_counts[tuple(sorted(source))] += 1
            target_counts[tuple(sorted(target))] += 1
        for counts in (source_counts, target_counts):
            self.assertLess(max(abs(v - 1000) for v in counts.values()), 160)

    def test_hybrid_formula_endpoints_and_rng(self):
        source = torch.arange(24, dtype=torch.float32).reshape(2, 6, 2)
        generator = torch.Generator().manual_seed(8)
        state = generator.get_state().clone()
        self.assertIs(tg_cache.hybrid_source(source, 0, generator=generator), source)
        torch.testing.assert_close(generator.get_state(), state)
        noise = torch.randn(source.shape, generator=torch.Generator().manual_seed(8))
        actual = tg_cache.hybrid_source(source, .2, generator=generator)
        torch.testing.assert_close(actual, np.sqrt(.8) * source + np.sqrt(.2) * noise)
        actual = tg_cache.hybrid_source(source, 1, generator=torch.Generator().manual_seed(8))
        torch.testing.assert_close(actual, noise)


class DenseCacheTests(unittest.TestCase):
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
    def config(dataset="checkerboard", beta=0):
        value = {"seed": 0, "device": "cpu", "dtype": "float32", "coupling": "target_guided_superset",
                 "num_regions": 2, "tg_superset": {"path": f"{dataset}_dense", "superset_size": 33,
                                                     "num_supersets": 2, "seed": 5, "beta": beta},
                 "data": {"n_points": 8, "batch_size": 2},
                 "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                           "dim_feedforward": 16, "dropout": 0.0},
                 "training": {"num_steps": 2, "log_every": 1, "learning_rate": .001, "weight_decay": .01},
                 "evaluation": {"batch_size": 2, "histogram_bins": 8}, "checkpoint": f"{dataset}.pt"}
        if dataset == "checkerboard":
            value["data"]["grid_size"] = 4
        return value

    def prepare(self, config, dataset="checkerboard"):
        with contextlib.redirect_stdout(io.StringIO()):
            return tg_superset.prepare(config, dataset)

    def test_dense_coarse_matches_baseline_and_original_coordinates_rng_preserved(self):
        config = self.config()
        state = torch.get_rng_state().clone()
        self.prepare(config)
        torch.testing.assert_close(torch.get_rng_state(), state)
        arrays, meta, _ = tg_superset.load_cache(config, "checkerboard")
        source = torch.from_numpy(arrays["source"])
        target = torch.from_numpy(arrays["target"])
        _, labels, centers, caps = coupling.balanced_target_partition(target, 2, solver="exact_batched")
        expected = coupling.assign_regions(source, centers, caps, solver="exact_batched")
        np.testing.assert_array_equal(labels.numpy(), arrays["target_labels"])
        np.testing.assert_array_equal(expected.numpy(), arrays["source_labels"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(5)
            torch.testing.assert_close(torch.randn(1, 33, 2)[0], source[0], rtol=0, atol=0)
        self.assertNotIn("permutation", arrays)
        self.assertIn("approximate", meta["marginals"])

    def test_cache_reuse_beta_shared_hash_mismatch_and_corruption(self):
        config = self.config()
        self.prepare(config)
        _, _, original_hash = tg_superset.load_cache(config, "checkerboard")
        changed = copy.deepcopy(config)
        changed["tg_superset"]["beta"] = .1
        with patch.object(tg_superset, "assign_regions", side_effect=AssertionError("recomputed")):
            self.prepare(changed)
        self.assertEqual(tg_superset.load_cache(changed, "checkerboard")[2], original_hash)
        for section, key, value in (("tg_superset", "seed", 8), ("tg_superset", "num_supersets", 3),
                                    ("tg_superset", "cache_sha256", "bad"), ("data", "grid_size", 6)):
            changed = copy.deepcopy(config)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                tg_superset.load_cache(changed, "checkerboard")
        file = Path(config["tg_superset"]["path"]) / "source.npy"
        source = np.load(file)
        source[0, 0, 0] += 1
        np.save(file, source)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.prepare(config)

    def test_beta_ablations_same_subsets_and_no_online_OT(self):
        config = self.config()
        self.prepare(config)
        changed = copy.deepcopy(config)
        changed["tg_superset"]["beta"] = .2
        plain = tg_superset.TGSupersetPairSampler(config, "checkerboard", "cpu", torch.float32, training=True)
        noisy = tg_superset.TGSupersetPairSampler(changed, "checkerboard", "cpu", torch.float32, training=True)
        a_rng, b_rng = torch.Generator().manual_seed(12), torch.Generator().manual_seed(12)
        expected_rng = torch.Generator().manual_seed(12)
        with patch.object(coupling.ot, "emd", side_effect=AssertionError("online OT")):
            for _ in range(4):
                source, target = plain.sample(3, generator=a_rng)
                jittered, other_target = noisy.sample(3, generator=b_rng)
                torch.testing.assert_close(target, other_target, rtol=0, atol=0)
                noise = torch.randn(source.shape, generator=expected_rng)
                torch.testing.assert_close(jittered, np.sqrt(.8) * source + np.sqrt(.2) * noise)
        diagnostic = tg_superset.TGSupersetPairSampler(changed, "checkerboard", "cpu", torch.float32)
        a = diagnostic.sample(2, generator=torch.Generator().manual_seed(5))
        b = diagnostic.sample(2, generator=torch.Generator().manual_seed(5))
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_invalid_settings_before_creation(self):
        for key, value in (("beta", True), ("beta", float("nan")), ("beta", -1),
                           ("superset_size", 4), ("num_supersets", 0), ("obsolete", True)):
            config = self.config()
            config["tg_superset"][key] = value
            with self.assertRaises(ValueError):
                self.prepare(config)
            self.assertFalse(Path(config["tg_superset"]["path"]).exists())
        for function in (coupling.coupled_points, coupling.coupling_permutation):
            with self.assertRaisesRegex(ValueError, "precomputed|offline cache"):
                function(torch.zeros(1, 8, 2), torch.zeros(1, 8, 2), coupling="target_guided_superset")

    def test_train_eval_and_diagnostics_both_datasets_beta_conditions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for dataset, trainer, model_class in (("checkerboard", train.main, PointSetTransformer),
                                                  ("horse", train_horse.main, train_horse.HorsePointSetTransformer)):
                for beta in (0, .1):
                    config = self.config(dataset, beta)
                    self.prepare(config, dataset)
                    file = Path(f"{dataset}_{beta}.yaml")
                    file.write_text(yaml.safe_dump(config), encoding="utf-8")
                    with patch.object(train, "coupled_points", side_effect=AssertionError("online coupling")):
                        run = trainer(file, steps=1)
                    saved = run / "config.yaml"
                    model, trained, _, metadata = load_model(saved, model_class, dataset)
                    self.assertEqual(metadata["coupling_details"]["beta"], beta)
                    self.assertIn("cache_sha256", trained["tg_superset"])
                    data = DiagnosticData(trained, dataset, "cpu", torch.float32, 2026)
                    report = audit_generation.fm_summary(model, trained, data, 1, 2, 2026)
                    self.assertTrue(all(np.isfinite(row["mse"]["mean"]) for row in report["time_errors"]))
                    pair_source, pair_target, details = audit_jacobian._cached_pairs(trained, dataset, data, 1)
                    self.assertEqual(pair_source.shape, (1, 8, 2))
                    self.assertEqual(details["beta"], beta)
                    with patch.object(tg_superset, "load_cache", side_effect=AssertionError("generation uses cache")):
                        if dataset == "checkerboard":
                            output = eval_checkerboard.main(saved, 2, render=False)
                        else:
                            with patch.object(eval_horse, "render_comparison"), patch("horse_regions.HorseRegions.render"):
                                output = eval_horse.main(saved, 2)
                    result = json.loads(output.with_suffix(".json").read_text())
                    self.assertTrue(np.isfinite(result["chamfer"]))

    def test_stream_hybrid_keeps_same_clouds_and_fine_draws(self):
        config = self.config()
        config["coupling"] = "target_guided_cached"
        config.pop("tg_superset")
        config["tg_cache"] = {"path": "stream", "sampling": "stream", "num_clouds": None,
                              "seed": 5, "prepare_batch_size": 2, "num_workers": 0, "beta": 0}
        with contextlib.redirect_stdout(io.StringIO()):
            tg_cache.prepare(config, "checkerboard")
        changed = copy.deepcopy(config)
        changed["tg_cache"]["beta"] = .2
        plain = tg_cache.make_pair_sampler(config, "checkerboard", "cpu", torch.float32, training=True)
        noisy = tg_cache.make_pair_sampler(changed, "checkerboard", "cpu", torch.float32, training=True)
        self.assertEqual(plain.cache_sha256, noisy.cache_sha256)
        expected_rng, jitter_rng = torch.Generator().manual_seed(4), torch.Generator().manual_seed(4)
        try:
            for _ in range(2):
                source, target = plain.sample(2)
                jittered, other_target = noisy.sample(2, generator=jitter_rng)
                torch.testing.assert_close(target, other_target, rtol=0, atol=0)
                noise = torch.randn(source.shape, generator=expected_rng)
                torch.testing.assert_close(jittered, np.sqrt(.8) * source + np.sqrt(.2) * noise)
        finally:
            plain.close()
            noisy.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_sampling_and_backward(self):
        config = self.config(beta=.1)
        self.prepare(config)
        sampler = tg_cache.make_pair_sampler(config, "checkerboard", "cuda", torch.float32, training=True)
        source, target = sampler.sample(2, generator=torch.Generator(device="cuda").manual_seed(4))
        for constructor in (PointSetTransformer, train_horse.HorsePointSetTransformer):
            model = constructor(**config["model"]).cuda()
            loss = train.train_step(model, torch.optim.AdamW(model.parameters()), target,
                                    coupling=config["coupling"], paired_noise=source)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(source.device.type, "cuda")


class RunnerTests(unittest.TestCase):
    def test_four_way_plan_is_stream_not_bank_and_shared_geometry(self):
        jobs = runner.planned_jobs(runner.parser().parse_args([]))
        self.assertEqual(len(jobs), 8)
        for dataset in ("checkerboard", "horse"):
            selected = {job["variant"]: job["config"] for job in jobs if job["dataset"] == dataset}
            self.assertEqual(selected["stream"]["tg_cache"]["sampling"], "stream")
            self.assertIsNone(selected["stream"]["tg_cache"]["num_clouds"])
            for plain, hybrid, section in (("stream", "stream_hybrid", "tg_cache"),
                                            ("superset", "superset_hybrid", "tg_superset")):
                a, b = selected[plain], selected[hybrid]
                self.assertEqual(a[section]["path"], b[section]["path"])
                self.assertEqual(a[section]["beta"], 0)
                self.assertEqual(b[section]["beta"], .1)
                self.assertEqual(a["model"], b["model"])
                self.assertEqual(a["training"], b["training"])
            a, b = selected["stream"], selected["superset"]
            for key in ("model", "training", "data", "evaluation"):
                self.assertEqual(a[key], b[key])

    def test_dry_run_does_not_create_manifest_and_eval_requires_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "absent" / "manifest.json"
            args = runner.parser().parse_args(["--manifest", str(path), "--dry-run"])
            with contextlib.redirect_stdout(io.StringIO()):
                runner.run(args)
            self.assertFalse(path.parent.exists())
            with self.assertRaisesRegex(ValueError, "requires --manifest"):
                runner.run(runner.parser().parse_args(["--stage", "eval"]))


if __name__ == "__main__":
    unittest.main()
