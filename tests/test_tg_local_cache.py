"""Integration checks for cached path-affine coupling, including Windows workers."""

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

import coupling
import eval as eval_checkerboard
import eval_horse
from experiment import load_model
from model import PointSetTransformer
from nsot import file_sha256
import tg_cache
import tg_local
import train
import train_horse


MODES = ("path_affine", "path_affine_subpatch")
ORIGINAL_ARRAYS = ("source", "target", "target_labels", "capacities", "source_labels")


class FixedTableRng:
    """Pin the grouping table while retaining fresh fine-permutation randomness."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def integers(self, high):
        return 0

    def permutation(self, values):
        return self.rng.permutation(values)


class LocalCacheTests(unittest.TestCase):
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
    def config(mode="path_affine", dataset="checkerboard", sampling="bank"):
        result = {
            "seed": 0, "device": "cpu", "dtype": "float32",
            "coupling": "target_guided_cached", "num_regions": 2,
            "tg_cache": {"path": f"{dataset}_{mode}_{sampling}", "sampling": sampling,
                         "num_clouds": 4, "seed": 5, "prepare_batch_size": 2,
                         "num_workers": 0},
            "data": {"batch_size": 2, "n_points": 16},
            "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                      "dim_feedforward": 16, "dropout": 0.0},
            "training": {"num_steps": 2, "learning_rate": .001,
                         "weight_decay": .01, "log_every": 1},
            "evaluation": {"batch_size": 2, "histogram_bins": 8},
            "checkpoint": f"{dataset}_{mode}.pt",
        }
        if dataset == "checkerboard":
            result["data"]["grid_size"] = 4
        if mode != "baseline":
            result["tg_cache"].update(
                coarse_mode=mode,
                local={"subpatches": 2 if mode == "path_affine_subpatch" else 1,
                       "num_tables": 2, "score_permutations": 2,
                       "candidate_count": 2, "max_swaps": 1, "neighbors": 6})
        return result

    @staticmethod
    def prepare(config, dataset="checkerboard"):
        with contextlib.redirect_stdout(io.StringIO()):
            return tg_cache.prepare(config, dataset)

    def test_original_bank_arrays_identical_and_child_capacities_preserved(self):
        for dataset in ("checkerboard", "horse"):
            baseline_path, baseline = self.prepare(self.config("baseline", dataset), dataset)
            self.assertEqual(baseline["format_version"], 1)
            self.assertEqual(set(baseline["array_sha256"]), set(ORIGINAL_ARRAYS))
            for mode in MODES:
                config = self.config(mode, dataset)
                path, meta = self.prepare(config, dataset)
                self.assertEqual(meta["format_version"], 4)
                self.assertEqual(meta["implementation"], "tg_offline_local_v4")
                for name in ORIGINAL_ARRAYS:
                    self.assertEqual(meta["array_sha256"][name], baseline["array_sha256"][name])
                self.assertEqual(set(meta["array_sha256"]),
                                 set(ORIGINAL_ARRAYS) | {"pairing_tables", "fine_target_labels"})
                tables = np.load(path / "pairing_tables.npy")
                targets = np.load(path / "fine_target_labels.npy")
                source_parents = np.load(baseline_path / "source_labels.npy")
                target_parents = np.load(baseline_path / "target_labels.npy")
                capacities = np.load(path / "capacities.npy")
                children = config["tg_cache"]["local"]["subpatches"]
                groups = config["num_regions"] * children
                self.assertEqual(tables.shape, (4, 2, 16))
                self.assertEqual(tables.dtype, np.int32)
                self.assertEqual(targets.dtype, np.int32)
                np.testing.assert_array_equal(targets // children, target_parents)
                np.testing.assert_array_equal(tables[:, 0] // children, source_parents)
                rng = np.random.default_rng(13)
                for cloud in range(4):
                    counts = np.bincount(targets[cloud], minlength=groups)
                    self.assertTrue(np.all(counts > 0))
                    if children == 2:
                        self.assertTrue(np.all(np.abs(counts[::2] - counts[1::2]) <= 1))
                    for table in tables[cloud]:
                        np.testing.assert_array_equal(np.bincount(table, minlength=groups), counts)
                        np.testing.assert_array_equal(
                            np.bincount(table // children, minlength=2), capacities[cloud])
                        permutation = tg_cache.random_fine_permutation(table, targets[cloud], groups, rng)
                        np.testing.assert_array_equal(np.sort(permutation), np.arange(16))
                        np.testing.assert_array_equal(targets[cloud][permutation], table)
                self.assertEqual(meta["fine_num_regions"], groups)
                self.assertEqual(meta["local_summary"]["clouds"], 4)
                self.assertGreaterEqual(meta["local_guidance_seconds"], 0.)
                self.assertIn("NOT a neural Jacobian", meta["local_score_definition"])
                self.assertIn("NOT topology guarantees", meta["local_guard_definition"])
                self.assertIn("tg_local.py", meta["source_sha256"])

    def test_local_preparation_is_deterministic_and_does_not_change_torch_rng(self):
        for mode in MODES:
            config = self.config(mode)
            torch.manual_seed(93)
            before = torch.get_rng_state().clone()
            _, first = self.prepare(config)
            torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
            repeated = copy.deepcopy(config)
            repeated["tg_cache"]["path"] += "_repeat"
            _, second = self.prepare(repeated)
            self.assertEqual(first["array_sha256"], second["array_sha256"])
            self.assertEqual(first["local_summary"], second["local_summary"])

    def test_fresh_online_bijections_use_no_guidance_or_ot_and_preserve_full_cloud(self):
        for mode in MODES:
            config = self.config(mode)
            path, _ = self.prepare(config)
            original_source = np.load(path / "source.npy")[0]
            original_target = np.load(path / "target.npy")[0]
            sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32)
            try:
                seen = set()
                rng = FixedTableRng(9)
                with patch.object(tg_local, "build_tables", side_effect=AssertionError("online guidance")), \
                        patch.object(tg_cache, "assign_regions", side_effect=AssertionError("online assignment")), \
                        patch.object(coupling.ot, "emd", side_effect=AssertionError("online OT")):
                    for _ in range(30):
                        source, target = sampler.dataset.draw(0, rng)
                        np.testing.assert_array_equal(source.numpy(), original_source)
                        self.assertEqual(sorted(map(tuple, target.numpy())),
                                         sorted(map(tuple, original_target)))
                        seen.add(tuple(target.flatten().tolist()))
                    a = sampler.sample(3, generator=torch.Generator().manual_seed(7))
                    b = sampler.sample(3, generator=torch.Generator().manual_seed(7))
                self.assertGreater(len(seen), 1)
                for left, right in zip(a, b):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                # Returned tensors can be changed without writing through the cache.
                source.add_(10.)
                np.testing.assert_array_equal(np.load(path / "source.npy")[0], original_source)
            finally:
                sampler.close()

    def test_incompatible_local_settings_rejected_before_creation(self):
        cases = []
        for mode, children in (("path_affine", 2), ("path_affine_subpatch", 1)):
            config = self.config(mode)
            config["tg_cache"]["local"]["subpatches"] = children
            cases.append(config)
        baseline = self.config("baseline")
        baseline["tg_cache"]["local"] = {"subpatches": 1}
        cases.append(baseline)
        too_small = self.config("path_affine_subpatch")
        too_small["data"]["n_points"] = 2
        cases.append(too_small)
        unknown = self.config()
        unknown["tg_cache"]["local"]["unknown_option"] = 1
        cases.append(unknown)
        for config in cases:
            with self.assertRaises(ValueError):
                self.prepare(config)
            self.assertFalse(Path(config["tg_cache"]["path"]).exists())

    def test_local_reuse_config_fingerprint_array_shape_and_hash_guards(self):
        config = self.config()
        path, metadata = self.prepare(config)
        with patch.object(tg_local, "build_tables", side_effect=AssertionError("recomputed guidance")):
            self.prepare(config)
        for section, key, value in (
                ("tg_cache", "seed", 17), ("tg_cache", "cache_sha256", "bad"),
                ("tg_cache", "coarse_mode", "baseline"), ("data", "grid_size", 6)):
            changed = copy.deepcopy(config)
            changed[section][key] = value
            with self.assertRaises(ValueError):
                tg_cache.load_cache(changed, "checkerboard")
        changed = copy.deepcopy(config)
        changed["tg_cache"]["local"]["ridge"] = .02
        with self.assertRaisesRegex(ValueError, "local_options"):
            tg_cache.load_cache(changed, "checkerboard")
        changed_meta = copy.deepcopy(metadata)
        changed_meta["array_sha256"].pop("fine_target_labels")
        (path / "metadata.json").write_text(json.dumps(changed_meta), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing/extra arrays"):
            tg_cache.load_cache(config, "checkerboard")
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        tables = np.load(path / "pairing_tables.npy")
        tables[0, 0, 0] = (tables[0, 0, 0] + 1) % 2
        np.save(path / "pairing_tables.npy", tables)
        with self.assertRaisesRegex(ValueError, "SHA256.*pairing_tables"):
            tg_cache.load_cache(config, "checkerboard")
        np.save(path / "pairing_tables.npy", tables[:, :1])
        changed_meta = copy.deepcopy(metadata)
        changed_meta["array_sha256"]["pairing_tables"] = file_sha256(path / "pairing_tables.npy")
        (path / "metadata.json").write_text(json.dumps(changed_meta), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "shape/dtype.*pairing_tables"):
            tg_cache.load_cache(config, "checkerboard")

    def test_interrupted_local_preparation_is_incomplete_and_not_overwritten(self):
        config = self.config()
        path = Path(config["tg_cache"]["path"])
        with patch.object(tg_local, "build_tables", side_effect=RuntimeError("interrupted test")):
            with self.assertRaisesRegex(RuntimeError, "interrupted test"):
                self.prepare(config)
        self.assertTrue((path / "source.npy").exists())
        self.assertFalse((path / "metadata.json").exists())
        before = (path / "source.npy").read_bytes()
        with self.assertRaisesRegex(FileNotFoundError, "incomplete"):
            self.prepare(config)
        self.assertEqual((path / "source.npy").read_bytes(), before)

    def test_local_stream_single_use_and_prepared_steps_guard(self):
        for mode in MODES:
            config = self.config(mode, sampling="stream")
            config["tg_cache"]["num_clouds"] = None
            path, meta = self.prepare(config)
            self.assertEqual(meta["num_clouds"], 4)
            sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32, training=True)
            try:
                expected = torch.from_numpy(np.load(path / "source.npy"))
                with patch.object(tg_local, "build_tables", side_effect=AssertionError("online guidance")):
                    first, _ = sampler.sample(2)
                    second, _ = sampler.sample(2)
                torch.testing.assert_close(torch.cat([first, second]), expected, rtol=0, atol=0)
                with self.assertRaisesRegex(RuntimeError, "exhausted"):
                    sampler.sample(2)
            finally:
                sampler.close()
            changed = copy.deepcopy(config)
            changed["training"]["num_steps"] = 3
            with self.assertRaisesRegex(ValueError, "exceed"):
                tg_cache.load_cache(changed, "checkerboard")

    def test_one_step_train_load_and_generation_eval_keep_local_provenance(self):
        for dataset, trainer, evaluator, model_class in (
                ("checkerboard", train.main, eval_checkerboard.main, PointSetTransformer),
                ("horse", train_horse.main, eval_horse.main, train_horse.HorsePointSetTransformer)):
            for mode in MODES:
                config = self.config(mode, dataset)
                config_path = Path(f"{dataset}_{mode}.yaml")
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                self.prepare(config, dataset)
                with contextlib.redirect_stdout(io.StringIO()), \
                        patch.object(train, "coupled_points", side_effect=AssertionError("online coupling")):
                    run = trainer(config_path, steps=1)
                _, trained, _, metadata = load_model(run / "config.yaml", model_class, dataset)
                details = metadata["coupling_details"]
                self.assertTrue(metadata["training_config_verified"])
                self.assertEqual(details["coarse_mode"], mode)
                self.assertEqual(details["fine_num_regions"], 2 * config["tg_cache"]["local"]["subpatches"])
                for key in ("local_options", "local_summary", "local_guidance_seconds",
                            "local_score_definition", "local_guard_definition",
                            "cached_pairing_tables_sha256", "cached_fine_target_labels_sha256"):
                    self.assertIn(key, details)
                self.assertIn("cache_sha256", trained["tg_cache"])
                self.assertNotIn("cache_sha256", config["tg_cache"])
                with contextlib.redirect_stdout(io.StringIO()), \
                        patch.object(tg_cache, "load_cache", side_effect=AssertionError("generation used cache")):
                    if dataset == "checkerboard":
                        output = evaluator(run / "config.yaml", 2, render=False)
                    else:
                        with patch.object(eval_horse, "render_comparison"), \
                                patch("horse_regions.HorseRegions.render"):
                            output = evaluator(run / "config.yaml", 2)
                result = json.loads(output.with_suffix(".json").read_text())
                self.assertEqual(result["coupling_details"], details)
                self.assertTrue(np.isfinite(result["chamfer"]))

    def test_windows_spawn_workers_read_local_bank_and_stream(self):
        for mode, sampling in (("path_affine", "bank"), ("path_affine_subpatch", "stream")):
            config = self.config(mode, sampling=sampling)
            config["tg_cache"]["num_workers"] = 1
            self.prepare(config)
            sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32, training=True)
            try:
                for _ in range(2):
                    source, target = sampler.sample(2)
                    self.assertEqual(source.shape, (2, 16, 2))
                    self.assertTrue(torch.isfinite(target).all())
            finally:
                sampler.close()


if __name__ == "__main__":
    unittest.main()
