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

import audit_generation
import coupling
from diagnostic_data import DiagnosticData
import eval as eval_checkerboard
import eval_horse
from experiment import load_model
from model import PointSetTransformer
import prepare_tg
import tg_cache
from tg_cache import random_fine_permutation
import train
import train_horse


class FinePairingTests(unittest.TestCase):
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
        return result

    def prepare(self, config, dataset="checkerboard"):
        with contextlib.redirect_stdout(io.StringIO()):
            return tg_cache.prepare(config, dataset)

    def test_unknown_cache_options_are_rejected_without_preparation(self):
        config = self.config()
        config["tg_cache"]["obsolete_experiment"] = True
        with self.assertRaisesRegex(ValueError, "Unsupported tg_cache settings"):
            tg_cache.prepare(config, "checkerboard")
        self.assertFalse(Path(config["tg_cache"]["path"]).exists())

    def test_unsupported_cache_format_is_not_silently_reinterpreted(self):
        config = self.config()
        path, meta = self.prepare(config)
        meta["format_version"] = tg_cache.FORMAT_VERSION + 1
        (path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "format_version"):
            tg_cache.load_cache(config, "checkerboard")

    def test_hard_determinism_exact_assignment_and_rng_isolation(self):
        hard = self.config()
        repeated = copy.deepcopy(hard)
        repeated["tg_cache"]["path"] += "_repeat"
        torch.manual_seed(93)
        state = torch.get_rng_state()
        h_path, h_meta = self.prepare(hard)
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
        _, repeated_meta = self.prepare(repeated)
        self.assertEqual(h_meta["array_sha256"], repeated_meta["array_sha256"])
        source = torch.from_numpy(np.load(h_path / "source.npy"))
        target = torch.from_numpy(np.load(h_path / "target.npy"))
        _, labels, centers, capacities = coupling.balanced_target_partition(target, 2, solver="exact_batched")
        expected = coupling.assign_regions(source, centers, capacities, solver="exact_batched").numpy()
        np.testing.assert_array_equal(expected, np.load(h_path / "source_labels.npy"))

    def test_actual_n256_k8_geometry_and_full_target_set(self):
        for dataset in ("checkerboard", "horse"):
            config = self.config(dataset=dataset)
            config["num_regions"] = 8
            config["data"]["n_points"] = 256
            config["tg_cache"]["num_clouds"] = 32
            path, meta = self.prepare(config, dataset)
            sampler = tg_cache.TGCachedPairSampler(config, dataset, "cpu", torch.float32)
            try:
                rng = np.random.default_rng(32)
                original_source = np.load(path / "source.npy")
                original_target = np.load(path / "target.npy")
                for index in range(32):
                    source, target = sampler.dataset.draw(index, rng)
                    self.assertTrue(torch.isfinite(target).all())
                    np.testing.assert_array_equal(source.numpy(), original_source[index])
                    self.assertEqual(sorted(map(tuple, target.numpy())),
                                     sorted(map(tuple, original_target[index])))
            finally:
                sampler.close()

    def test_legacy_hard_metadata_and_checkpoint_fingerprint_remain_supported(self):
        config = self.config()
        path, meta = self.prepare(config)
        meta.update(epsilon=None, sinkhorn_iterations=None, rounding_numeric_tolerance=1e-8,
                    max_sinkhorn_residual_before_repair=0., max_sinkhorn_residual_after_repair=0.,
                    offline_rounding_seconds=0.)
        (path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        config["tg_cache"]["cache_sha256"] = tg_cache._fingerprint(meta)
        sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32)
        try:
            self.assertEqual(sampler.details()["cache_sha256"], config["tg_cache"]["cache_sha256"])
            self.assertEqual(sampler.sample(2)[0].shape, (2, 8, 2))
        finally:
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

    def test_stream_single_use_no_online_ot_and_steps_guard(self):
        for method in tg_cache.METHODS:
            config = self.config(method, sampling="stream")
            config["tg_cache"]["num_clouds"] = None
            path, meta = self.prepare(config)
            self.assertEqual(meta["num_clouds"], 4)
            self.assertNotIn("plan", meta["array_sha256"])
            sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32, training=True)
            expected = torch.from_numpy(np.load(path / "source.npy"))
            with patch.object(tg_cache, "assign_regions", side_effect=AssertionError("online OT")):
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

    def test_windows_spawn_workers_bank_and_stream(self):
        for method, sampling in (("target_guided_cached", "bank"),
                                 ("target_guided_cached", "stream")):
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
