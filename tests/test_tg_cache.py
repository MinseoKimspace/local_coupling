import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

import audit_generation
import coupling
from diagnostic_data import DiagnosticData
import eval as eval_checkerboard
import eval_horse
from experiment import load_model
from model import PointSetTransformer
import prepare_tg
import tg_cache
from tg_rounding import dependent_round, entropic_plan, feasible_plan, random_fine_permutation
import train
import train_horse


class RoundingTests(unittest.TestCase):
    def test_counts_and_monte_carlo_marginals_not_greedy(self):
        rng = np.random.default_rng(42)
        plans = [(np.array([[.8, .2], [.2, .8]]), np.array([1, 1])),
                 (np.full((6, 3), 1 / 3), np.array([2, 2, 2]))]
        for plan, counts in plans:
            accumulated = np.zeros_like(plan)
            for _ in range(3000):
                labels = dependent_round(plan, counts, rng)
                np.testing.assert_array_equal(np.bincount(labels, minlength=len(counts)), counts)
                accumulated += np.eye(len(counts))[labels]
            np.testing.assert_allclose(accumulated / 3000, plan, atol=.03, rtol=0)

    def test_extremes_zero_capacity_reproducibility_and_no_mutation(self):
        plan = np.array([[1., 0, 0], [0, 0, 1], [1, 0, 0]])
        original = plan.copy()
        a = dependent_round(plan, [2, 0, 1], np.random.default_rng(0))
        b = dependent_round(plan, [2, 0, 1], np.random.default_rng(0))
        np.testing.assert_array_equal(a, [0, 2, 0])
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(plan, original)
        for bad in (np.full((2, 2), .2), np.array([[1., -.1], [0, 1.1]])):
            with self.assertRaises(ValueError):
                dependent_round(bad, [1, 1], np.random.default_rng(0))
        with self.assertRaises(ValueError):
            dependent_round(np.full((2, 2), .5), [.5, 1.5], np.random.default_rng(0))

    def test_soft_cost_bias_and_temperature_extremes(self):
        cost = np.array([[0., 3.], [3., 0.]])
        small, before, after = entropic_plan(cost, [1, 1], epsilon=.1, iterations=3000)
        large, _, _ = entropic_plan(cost, [1, 1], epsilon=1e6, iterations=3000)
        self.assertGreater(small[0, 0], .999)
        self.assertLess(after, 1e-8)
        np.testing.assert_allclose(large, .5, atol=1e-6)
        repaired, _, _ = feasible_plan(np.full((2, 2), .5) * (1 + 1e-8), [1, 1])
        np.testing.assert_allclose(repaired.sum(0), 1, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "increase sinkhorn_iterations"):
            feasible_plan(np.full((2, 2), .4), [1, 1])

    def test_dual_refinement_recovers_small_epsilon_nonconvergence(self):
        rng = np.random.default_rng(4)
        cost = rng.random((16, 4)) * 20
        plan, _, after = entropic_plan(cost, [4, 4, 4, 4], epsilon=.1, iterations=100)
        np.testing.assert_allclose(plan.sum(0), 4, atol=1e-8)
        np.testing.assert_allclose(plan.sum(1), 1, atol=1e-8)

    def test_configured_size_counts_and_rounding_timing(self):
        rng = np.random.default_rng(7)
        source, centers = rng.normal(size=(256, 2)), rng.normal(size=(8, 2))
        cost = ((source[:, None] - centers[None]) ** 2).sum(-1)
        plan, _, _ = entropic_plan(cost, np.full(8, 32), epsilon=.1, iterations=3000)
        tick = perf_counter()
        for _ in range(3):
            labels = dependent_round(plan, np.full(8, 32), rng)
            np.testing.assert_array_equal(np.bincount(labels, minlength=8), np.full(8, 32))
        print(f"tg_rounding_N256_K8_seconds_per_cloud={(perf_counter() - tick) / 3:.6f}")

    def test_fine_bijection_preserves_membership_and_varies(self):
        source, target = np.array([0, 1, 0, 1, 0, 1]), np.array([1, 1, 0, 0, 1, 0])
        rng, seen = np.random.default_rng(3), set()
        for _ in range(20):
            permutation = random_fine_permutation(source, target, 2, rng)
            np.testing.assert_array_equal(np.sort(permutation), np.arange(6))
            np.testing.assert_array_equal(target[permutation], source)
            seen.add(tuple(permutation))
        self.assertGreater(len(seen), 1)


class TGCacheTests(unittest.TestCase):
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
    def config(method="target_guided_cached", dataset="checkerboard", sampling="bank"):
        result = {"seed": 0, "device": "cpu", "dtype": "float32", "coupling": method,
                  "num_regions": 2,
                  "tg_cache": {"path": f"{dataset}_{method}_{sampling}", "sampling": sampling,
                               "num_clouds": 8, "seed": 5, "prepare_batch_size": 4, "num_workers": 0},
                  "data": {"batch_size": 2, "n_points": 8},
                  "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                            "dim_feedforward": 16, "dropout": 0.0},
                  "training": {"num_steps": 2, "learning_rate": .001, "weight_decay": .01, "log_every": 1},
                  "evaluation": {"batch_size": 2, "histogram_bins": 8},
                  "checkpoint": f"{dataset}_{method}.pt"}
        if dataset == "checkerboard":
            result["data"]["grid_size"] = 4
        if method == "target_guided_soft_cached":
            result["tg_cache"].update(epsilon=.1, sinkhorn_iterations=3000)
        return result

    def prepare(self, config, dataset="checkerboard"):
        with contextlib.redirect_stdout(io.StringIO()):
            return tg_cache.prepare(config, dataset)

    def test_hard_soft_share_endpoints_partition_and_rng_isolation(self):
        hard, soft = self.config(), self.config("target_guided_soft_cached")
        torch.manual_seed(93)
        state = torch.get_rng_state()
        h_path, h_meta = self.prepare(hard)
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
        s_path, s_meta = self.prepare(soft)
        for name in ("source", "target", "target_labels", "capacities"):
            self.assertEqual(h_meta["array_sha256"][name], s_meta["array_sha256"][name])
        self.assertIn("plan", s_meta["array_sha256"])
        source = torch.from_numpy(np.load(h_path / "source.npy"))
        target = torch.from_numpy(np.load(h_path / "target.npy"))
        _, labels, centers, capacities = coupling.balanced_target_partition(target, 2, solver="exact_batched")
        expected = coupling.assign_regions(source, centers, capacities, solver="exact_batched").numpy()
        np.testing.assert_array_equal(expected, np.load(h_path / "source_labels.npy"))
        self.assertLess(s_meta["max_sinkhorn_residual_after_repair"], 1e-8)

    def test_actual_n256_k8_geometry_and_sampler_moment_identity(self):
        for dataset in ("checkerboard", "horse"):
            config = self.config("target_guided_soft_cached", dataset)
            config["num_regions"] = 8
            config["data"]["n_points"] = 256
            config["tg_cache"]["num_clouds"] = 32
            path, meta = self.prepare(config, dataset)
            sampler = tg_cache.TGCachedPairSampler(config, dataset, "cpu", torch.float32)
            try:
                rng = np.random.default_rng(32)
                for index in range(32):
                    source, target = sampler.dataset.draw(index, rng)
                    self.assertTrue(torch.isfinite(target).all())
                    np.testing.assert_array_equal(source.numpy(), np.load(path / "source.npy", mmap_mode="r")[index])
            finally:
                sampler.close()
        config = self.config("target_guided_soft_cached")
        config["tg_cache"]["path"] = "moment_identity"
        path, meta = self.prepare(config)
        sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32)
        target = np.load(path / "target.npy")[0].astype(float)
        labels = np.load(path / "target_labels.npy")[0]
        plan = np.load(path / "plan.npy")[0]
        centers = np.stack([target[labels == k].mean(0) for k in range(2)])
        mean = plan @ centers
        within = ((target - centers[labels]) ** 2).mean()
        between = (plan[:, :, None] * (centers[None] - mean[:, None]) ** 2).sum() / target.size
        rng = np.random.default_rng(12)
        endpoints = np.stack([sampler.dataset.draw(0, rng)[1].numpy() for _ in range(2000)])
        np.testing.assert_allclose(endpoints.mean(0), mean, atol=.035, rtol=0)
        self.assertAlmostEqual(endpoints.var(0).mean(), within + between, delta=.02)
        sampler.close()

    def test_cache_reuse_mismatch_integrity_and_incomplete_fail_loudly(self):
        config = self.config()
        with self.assertRaisesRegex(FileNotFoundError, "prepare_tg"):
            tg_cache.load_cache(config, "checkerboard")
        path, _ = self.prepare(config)
        with patch.object(tg_cache, "balanced_target_partition", side_effect=AssertionError("recomputed")):
            self.prepare(config)
        for section, key, value in (("tg_cache", "seed", 8), ("tg_cache", "num_clouds", 9),
                                    ("tg_cache", "cache_sha256", "bad"), ("data", "grid_size", 6)):
            changed = copy.deepcopy(config)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                tg_cache.load_cache(changed, "checkerboard")
        array = np.load(path / "source.npy")
        array[0, 0, 0] += 1
        np.save(path / "source.npy", array)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            tg_cache.load_cache(config, "checkerboard")
        changed = copy.deepcopy(config)
        changed["tg_cache"]["path"] = "incomplete"
        Path("incomplete").mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
            self.prepare(changed)

    def test_source_unchanged_target_full_set_fresh_pairing_no_online_ot(self):
        for method in tg_cache.METHODS:
            config = self.config(method)
            path, meta = self.prepare(config)
            sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32)
            original_source, original_target = np.load(path / "source.npy")[0], np.load(path / "target.npy")[0]
            seen = set()
            rng = np.random.default_rng(9)
            with patch.object(coupling.ot, "emd", side_effect=AssertionError("online OT")), \
                    patch.object(coupling.ot, "sinkhorn", side_effect=AssertionError("online Sinkhorn")):
                for _ in range(30):
                    source, target = sampler.dataset.draw(0, rng)
                    np.testing.assert_array_equal(source.numpy(), original_source)
                    self.assertEqual(sorted(map(tuple, target.numpy())), sorted(map(tuple, original_target)))
                    seen.add(tuple(target.flatten().tolist()))
            self.assertGreater(len(seen), 1)
            a = sampler.sample(3, generator=torch.Generator().manual_seed(7))
            b = sampler.sample(3, generator=torch.Generator().manual_seed(7))
            for left, right in zip(a, b):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_stream_single_use_no_online_rounding_and_steps_guard(self):
        for method in tg_cache.METHODS:
            config = self.config(method, sampling="stream")
            config["tg_cache"]["num_clouds"] = None
            path, meta = self.prepare(config)
            self.assertEqual(meta["num_clouds"], 4)
            self.assertNotIn("plan", meta["array_sha256"])
            sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32, training=True)
            expected = torch.from_numpy(np.load(path / "source.npy"))
            with patch.object(tg_cache, "dependent_round", side_effect=AssertionError("online rounding")):
                first, _ = sampler.sample(2)
                second, _ = sampler.sample(2)
            torch.testing.assert_close(torch.cat([first, second]), expected)
            with self.assertRaisesRegex(RuntimeError, "exhausted"):
                sampler.sample(2)
            sampler.close()
            changed = copy.deepcopy(config)
            changed["training"]["num_steps"] = 3
            with self.assertRaisesRegex(ValueError, "exceed"):
                tg_cache.load_cache(changed, "checkerboard")
            changed["training"]["num_steps"] = 1
            tg_cache.load_cache(changed, "checkerboard")

    def test_configs_keep_original_model_optimizer_steps_and_batch(self):
        root = Path(__file__).resolve().parents[1]
        for directory, prefix, baseline in (
                ("checkerboard_experiments", "", "target_guided.yaml"),
                ("horse_experiments", "horse_", "horse_target_guided_k8_n256_seed0.yaml")):
            original = yaml.safe_load((root / directory / baseline).read_text())
            for method in tg_cache.METHODS:
                config = yaml.safe_load((root / directory / f"{prefix}{method}_k8_n256_seed0.yaml").read_text())
                self.assertEqual({k: v for k, v in config.items() if k not in ("tg_cache", "coupling", "checkpoint")},
                                 {k: v for k, v in original.items() if k not in ("coupling", "checkpoint")})
                self.assertEqual(tg_cache.settings(config)["sampling"], "bank")

    def test_cli_stream_override_preserves_input_and_writes_trainable_config(self):
        for method in tg_cache.METHODS:
            config = self.config(method)
            original = Path(f"{method}.yaml")
            original.write_text(yaml.safe_dump(config), encoding="utf-8")
            before = original.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                path, meta = prepare_tg.main(original, "checkerboard", sampling="stream")
            self.assertEqual(original.read_bytes(), before)
            derived = yaml.safe_load((path / "config.yaml").read_text())
            self.assertEqual(derived["tg_cache"]["sampling"], "stream")
            self.assertIsNone(derived["tg_cache"]["num_clouds"])
            self.assertEqual(meta["num_clouds"], 4)
            with contextlib.redirect_stdout(io.StringIO()):
                train.main(path / "config.yaml", steps=1)

    def test_train_load_eval_and_generation_fm_audit_both_datasets_methods(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for dataset, trainer, evaluator, model_class in (
                ("checkerboard", train.main, eval_checkerboard.main, PointSetTransformer),
                ("horse", train_horse.main, eval_horse.main, train_horse.HorsePointSetTransformer)):
                for method in tg_cache.METHODS:
                    config = self.config(method, dataset)
                    path = Path(f"{dataset}_{method}.yaml")
                    path.write_text(yaml.safe_dump(config), encoding="utf-8")
                    prepare_tg.main(path, dataset)
                    with patch.object(train, "coupled_points", side_effect=AssertionError("online coupling")):
                        run = trainer(path, steps=1)
                    model, trained, checkpoint, metadata = load_model(run / "config.yaml", model_class, dataset)
                    self.assertTrue(metadata["training_config_verified"])
                    self.assertIn("cache_sha256", trained["tg_cache"])
                    self.assertNotIn("cache_sha256", config["tg_cache"])
                    self.assertIn("precompute_seconds", metadata["coupling_details"])
                    data = DiagnosticData(trained, dataset, "cpu", torch.float32, 2026)
                    fm = audit_generation.fm_summary(model, trained, data, 1, 2, 2026)
                    self.assertIn("fixed training cache", fm["sampling_scope"])
                    self.assertTrue(all(np.isfinite(r["mse"]["mean"]) for r in fm["time_errors"]))
                    # Generation metrics must not require or use training cache.
                    with patch.object(tg_cache, "load_cache", side_effect=AssertionError("cache used for generation")):
                        if dataset == "checkerboard":
                            output = evaluator(run / "config.yaml", 2, render=False)
                        else:
                            with patch.object(eval_horse, "render_comparison"), patch("horse_regions.HorseRegions.render"):
                                output = evaluator(run / "config.yaml", 2)
                    result = json.loads(output.with_suffix(".json").read_text())
                    self.assertEqual(result["coupling_details"], metadata["coupling_details"])
                    self.assertTrue(np.isfinite(result["chamfer"]))

    def test_windows_spawn_workers_hard_soft_and_stream(self):
        for method, sampling in (("target_guided_cached", "bank"),
                                 ("target_guided_soft_cached", "bank"),
                                 ("target_guided_soft_cached", "stream")):
            config = self.config(method, sampling=sampling)
            config["tg_cache"]["num_workers"] = 1
            self.prepare(config)
            sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32, training=True)
            try:
                for _ in range(2):
                    source, target = sampler.sample(2)
                    self.assertEqual(source.shape, (2, 8, 2))
                    self.assertTrue(torch.isfinite(target).all())
            finally:
                sampler.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_transfer_and_backward(self):
        for method in tg_cache.METHODS:
            config = self.config(method)
            self.prepare(config)
            sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cuda", torch.float32, training=True)
            try:
                source, target = sampler.sample(2)
                model = PointSetTransformer(**config["model"]).cuda()
                loss = train.train_step(model, torch.optim.AdamW(model.parameters()), target,
                                        coupling=method, paired_noise=source)
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(source.device.type, "cuda")
            finally:
                sampler.close()


if __name__ == "__main__":
    unittest.main()
