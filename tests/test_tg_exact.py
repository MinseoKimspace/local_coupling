"""Exact fine pairing changes only the within-patch cached correspondence."""

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
import scipy.optimize
import torch
import yaml

import coupling
import eval as eval_checkerboard
import eval_horse
from experiment import load_model
from model import PointSetTransformer
import prepare_tg
import tg_cache
import train
import train_horse


class ExactFinePermutationTests(unittest.TestCase):
    def test_exact_objective_matches_exhaustive_patch_bijections(self):
        source = np.array([[0., 0.], [4., 1.], [2., -1.], [8., 2.], [9., 0.]])
        target = np.array([[9., 1.], [2., 0.], [0., 1.], [8., 0.], [4., -1.]])
        source_labels = np.array([0, 0, 0, 2, 2])
        target_labels = np.array([2, 0, 0, 2, 0])
        permutation = tg_cache.exact_fine_permutation(source, target, source_labels, target_labels, 4)
        np.testing.assert_array_equal(np.sort(permutation), np.arange(5))
        np.testing.assert_array_equal(target_labels[permutation], source_labels)
        expected = 0.
        for region in (0, 2):
            src = np.flatnonzero(source_labels == region)
            dst = np.flatnonzero(target_labels == region)
            expected += min(float(np.square(source[src] - target[list(order)]).sum())
                            for order in itertools.permutations(dst))
        self.assertAlmostEqual(float(np.square(source - target[permutation]).sum()), expected, places=12)

    def test_unique_optimum_uses_geometry_not_input_target_order(self):
        source = np.array([[0., 0.], [1., 0.], [4., 0.], [5., 0.]], dtype=np.float32)
        target = source[[2, 1, 3, 0]].copy()
        labels = np.zeros(4, dtype=np.int32)
        before_source, before_target = source.copy(), target.copy()
        actual = tg_cache.exact_fine_permutation(source, target, labels, labels, 1)
        np.testing.assert_array_equal(actual, [3, 1, 0, 2])
        np.testing.assert_array_equal(source, before_source)
        np.testing.assert_array_equal(target, before_target)

    def test_duplicate_points_and_empty_regions_still_form_full_bijection(self):
        source, target = np.zeros((7, 2)), np.zeros((7, 2))
        source_labels = np.array([0, 0, 3, 3, 3, 3, 3])
        target_labels = np.array([3, 0, 3, 3, 0, 3, 3])
        permutation = tg_cache.exact_fine_permutation(source, target, source_labels, target_labels, 5)
        np.testing.assert_array_equal(np.sort(permutation), np.arange(7))
        np.testing.assert_array_equal(target_labels[permutation], source_labels)

    def test_mismatched_patch_capacities_are_rejected(self):
        with self.assertRaises(ValueError):
            tg_cache.exact_fine_permutation(np.zeros((3, 2)), np.ones((3, 2)),
                                            np.array([0, 0, 1]), np.array([0, 1, 1]), 2)


class ExactTGCacheTests(unittest.TestCase):
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
    def config(dataset="checkerboard", sampling="bank", fine_pairing="exact"):
        result = {
            "seed": 0, "device": "cpu", "dtype": "float32", "coupling": "target_guided_cached",
            "num_regions": 3,
            "tg_cache": {"path": f"{dataset}_{sampling}_{fine_pairing}", "sampling": sampling,
                         "fine_pairing": fine_pairing, "num_clouds": 6, "seed": 5,
                         "prepare_batch_size": 3, "num_workers": 0},
            "data": {"batch_size": 2, "n_points": 11},
            "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                      "dim_feedforward": 16, "dropout": 0.0},
            "training": {"num_steps": 2, "learning_rate": .001, "weight_decay": .01, "log_every": 1},
            "evaluation": {"batch_size": 2, "histogram_bins": 8},
            "checkpoint": f"{dataset}_exact.pt",
        }
        if dataset == "checkerboard":
            result["data"]["grid_size"] = 4
        return result

    @staticmethod
    def prepare(config, dataset="checkerboard"):
        with contextlib.redirect_stdout(io.StringIO()):
            return tg_cache.prepare(config, dataset)

    def test_invalid_fine_modes_fail_before_writing(self):
        for value in (None, "soft", "EXACT", True, 1, []):
            config = self.config(fine_pairing=value)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "fine_pairing"):
                tg_cache.prepare(config, "checkerboard")
            self.assertFalse(Path(config["tg_cache"]["path"]).exists())

    def test_random_and_exact_prepare_identical_base_clouds_and_coarse_assignments(self):
        for dataset in ("checkerboard", "horse"):
            random_config = self.config(dataset, fine_pairing="random")
            exact_config = self.config(dataset)
            torch.manual_seed(91)
            state = torch.get_rng_state()
            random_path, random_meta = self.prepare(random_config, dataset)
            exact_path, exact_meta = self.prepare(exact_config, dataset)
            torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
            self.assertEqual(random_meta["format_version"], 1)
            self.assertEqual(exact_meta["format_version"], 2)
            self.assertNotIn("fine_permutation", random_meta["array_sha256"])
            self.assertEqual(set(exact_meta["array_sha256"]), set(random_meta["array_sha256"]) | {"fine_permutation"})
            for name, digest in random_meta["array_sha256"].items():
                self.assertEqual(exact_meta["array_sha256"][name], digest)
                np.testing.assert_array_equal(np.load(random_path / f"{name}.npy"),
                                              np.load(exact_path / f"{name}.npy"))
            source, target = np.load(exact_path / "source.npy"), np.load(exact_path / "target.npy")
            source_labels = np.load(exact_path / "source_labels.npy")
            target_labels = np.load(exact_path / "target_labels.npy")
            permutations = np.load(exact_path / "fine_permutation.npy")
            self.assertEqual(permutations.dtype, np.int32)
            self.assertEqual(permutations.shape, (6, 11))
            for index, permutation in enumerate(permutations):
                np.testing.assert_array_equal(np.sort(permutation), np.arange(11))
                np.testing.assert_array_equal(target_labels[index, permutation], source_labels[index])
                expected = tg_cache.exact_fine_permutation(source[index], target[index], source_labels[index],
                                                          target_labels[index], 3)
                np.testing.assert_array_equal(permutation, expected)
            repeated = copy.deepcopy(exact_config)
            repeated["tg_cache"]["path"] += "_repeat"
            _, repeat_meta = self.prepare(repeated, dataset)
            self.assertEqual(repeat_meta["array_sha256"], exact_meta["array_sha256"])

    def test_exact_revisits_read_stored_pairing_without_solver_or_random_fine_work(self):
        config = self.config()
        path, _ = self.prepare(config)
        sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32)
        expected_source, target = np.load(path / "source.npy")[0], np.load(path / "target.npy")[0]
        expected_target = target[np.load(path / "fine_permutation.npy")[0]]
        try:
            with patch.object(tg_cache, "exact_fine_permutation", side_effect=AssertionError("online fine OT")), \
                    patch.object(tg_cache, "random_fine_permutation", side_effect=AssertionError("online shuffle")), \
                    patch.object(tg_cache, "assign_regions", side_effect=AssertionError("online coarse OT")), \
                    patch.object(scipy.optimize, "linear_sum_assignment", side_effect=AssertionError("online solve")), \
                    patch.object(coupling.ot, "emd", side_effect=AssertionError("online POT")):
                for seed in range(5):
                    source, paired_target = sampler.dataset.draw(0, np.random.default_rng(seed))
                    np.testing.assert_array_equal(source.numpy(), expected_source)
                    np.testing.assert_array_equal(paired_target.numpy(), expected_target)
                left = sampler.sample(4, generator=torch.Generator().manual_seed(9))
                right = sampler.sample(4, generator=torch.Generator().manual_seed(9))
                for a, b in zip(left, right):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        finally:
            sampler.close()

    def test_actual_n256_k8_stored_pairs_match_independent_local_ot_objectives(self):
        for dataset in ("checkerboard", "horse"):
            config = self.config(dataset)
            config["num_regions"] = 8
            config["data"]["n_points"] = 256
            config["tg_cache"]["num_clouds"] = 2
            path, _ = self.prepare(config, dataset)
            source, target = np.load(path / "source.npy"), np.load(path / "target.npy")
            source_labels = np.load(path / "source_labels.npy")
            target_labels = np.load(path / "target_labels.npy")
            stored = np.load(path / "fine_permutation.npy")
            np.testing.assert_array_equal(np.load(path / "capacities.npy"), np.full((2, 8), 32))
            rng = np.random.default_rng(19)
            for cloud in range(2):
                permutation = stored[cloud]
                np.testing.assert_array_equal(np.sort(permutation), np.arange(256))
                np.testing.assert_array_equal(target_labels[cloud, permutation], source_labels[cloud])
                independent_cost = 0.
                for region in range(8):
                    src = np.flatnonzero(source_labels[cloud] == region)
                    dst = np.flatnonzero(target_labels[cloud] == region)
                    self.assertEqual(len(src), 32)
                    self.assertEqual(len(dst), 32)
                    difference = source[cloud, src].astype(np.float64)[:, None] - target[cloud, dst].astype(np.float64)[None]
                    cost = np.square(difference).sum(axis=-1)
                    rows, columns = scipy.optimize.linear_sum_assignment(cost)
                    independent_cost += float(cost[rows, columns].sum())
                actual_cost = float(np.square(source[cloud].astype(np.float64) - target[cloud, permutation]).sum())
                random_permutation = tg_cache.random_fine_permutation(source_labels[cloud], target_labels[cloud], 8, rng)
                random_cost = float(np.square(source[cloud].astype(np.float64) - target[cloud, random_permutation]).sum())
                self.assertAlmostEqual(actual_cost, independent_cost, places=10)
                self.assertLessEqual(actual_cost, random_cost + 1e-10)

    def test_exact_permutation_hash_shape_and_dtype_are_verified(self):
        config = self.config()
        path, metadata = self.prepare(config)
        stored = np.load(path / "fine_permutation.npy")
        changed = stored.copy()
        changed[0, 0] = changed[0, 1]
        np.save(path / "fine_permutation.npy", changed)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            tg_cache.load_cache(config, "checkerboard")
        for value in (stored.astype(np.int64), stored[:, :-1]):
            np.save(path / "fine_permutation.npy", value)
            metadata["array_sha256"]["fine_permutation"] = tg_cache.file_sha256(path / "fine_permutation.npy")
            (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                tg_cache.load_cache(config, "checkerboard")

    def test_modes_cannot_silently_reuse_each_others_cache(self):
        for mode, incompatible in (("random", "exact"), ("exact", "random")):
            config = self.config(fine_pairing=mode)
            path, _ = self.prepare(config)
            before = (path / "metadata.json").read_bytes()
            changed = copy.deepcopy(config)
            changed["tg_cache"]["fine_pairing"] = incompatible
            with self.assertRaises(ValueError):
                tg_cache.load_cache(changed, "checkerboard")
            with self.assertRaises(ValueError):
                self.prepare(changed)
            self.assertEqual((path / "metadata.json").read_bytes(), before)

    def test_legacy_random_config_and_v1_fingerprint_remain_usable(self):
        config = self.config(fine_pairing="random")
        config["tg_cache"].pop("fine_pairing")
        _, metadata = self.prepare(config)
        self.assertEqual(metadata["format_version"], 1)
        config["tg_cache"]["cache_sha256"] = tg_cache._fingerprint(metadata)
        sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32)
        try:
            seen = {tuple(sampler.dataset.draw(0, np.random.default_rng(seed))[1].flatten().tolist())
                    for seed in range(20)}
            self.assertGreater(len(seen), 1)
            self.assertEqual(sampler.details()["cache_sha256"], config["tg_cache"]["cache_sha256"])
        finally:
            sampler.close()

    def test_stream_preserves_single_use_order_and_bank_keeps_fixed_cloud_pairs(self):
        for sampling in ("bank", "stream"):
            config = self.config(sampling=sampling)
            path, _ = self.prepare(config)
            original_source, original_target = np.load(path / "source.npy"), np.load(path / "target.npy")
            permutation = np.load(path / "fine_permutation.npy")
            expected_target = original_target[np.arange(len(original_target))[:, None], permutation]
            sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32, training=True)
            try:
                with patch.object(tg_cache, "exact_fine_permutation", side_effect=AssertionError("online solve")):
                    source, target = zip(*(sampler.sample(2) for _ in range(2)))
                source, target = torch.cat(source).numpy(), torch.cat(target).numpy()
                if sampling == "stream":
                    np.testing.assert_array_equal(source, original_source[:4])
                    np.testing.assert_array_equal(target, expected_target[:4])
                else:
                    for cloud_source, cloud_target in zip(source, target):
                        matches = np.flatnonzero((original_source == cloud_source).all(axis=(1, 2)))
                        self.assertEqual(len(matches), 1)
                        np.testing.assert_array_equal(cloud_target, expected_target[matches[0]])
                with self.assertRaisesRegex(RuntimeError, "exhausted"):
                    sampler.sample(2)
            finally:
                sampler.close()

    def test_cli_derives_new_cache_and_preserves_original_config(self):
        config = self.config(fine_pairing="random")
        config["tg_cache"].pop("fine_pairing")
        original = Path("original.yaml")
        original.write_text(yaml.safe_dump(config), encoding="utf-8")
        before = original.read_bytes()
        for sampling in (None, "stream"):
            with contextlib.redirect_stdout(io.StringIO()):
                path, metadata = prepare_tg.main(original, "checkerboard", sampling=sampling, fine_pairing="exact")
            self.assertEqual(original.read_bytes(), before)
            suffix = "_fine_exact_stream" if sampling == "stream" else "_fine_exact"
            self.assertTrue(str(path).endswith(suffix))
            derived = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(derived["tg_cache"]["fine_pairing"], "exact")
            self.assertEqual(derived["tg_cache"]["sampling"], sampling or "bank")
            self.assertEqual(metadata["format_version"], 2)
            self.assertNotEqual(derived["tg_cache"]["path"], config["tg_cache"]["path"])

    def test_exact_training_both_datasets_keeps_saved_cache_provenance(self):
        for dataset, trainer, model_class in (
                ("checkerboard", train.main, PointSetTransformer),
                ("horse", train_horse.main, train_horse.HorsePointSetTransformer)):
            config = self.config(dataset)
            path, metadata = self.prepare(config, dataset)
            config_path = Path(f"{dataset}.yaml")
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(tg_cache, "exact_fine_permutation", side_effect=AssertionError("online fine solve")):
                run = trainer(config_path, steps=1)
                _, trained, _, provenance = load_model(run / "config.yaml", model_class, dataset)
            self.assertTrue(provenance["training_config_verified"])
            self.assertEqual(trained["tg_cache"]["fine_pairing"], "exact")
            self.assertEqual(provenance["coupling_details"]["fine_pairing"], metadata["fine_pairing"])
            self.assertEqual(trained["tg_cache"]["cache_sha256"], tg_cache._fingerprint(metadata))
            self.assertIn("fine_exact", run.name)
            self.assertTrue((path / "fine_permutation.npy").is_file())
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(tg_cache, "load_cache", side_effect=AssertionError("training cache used for generation")), \
                    patch.object(tg_cache, "exact_fine_permutation", side_effect=AssertionError("OT used for generation")):
                if dataset == "checkerboard":
                    output = eval_checkerboard.main(run / "config.yaml", 2, render=False)
                else:
                    with patch.object(eval_horse, "render_comparison"), patch("horse_regions.HorseRegions.render"):
                        output = eval_horse.main(run / "config.yaml", 2)
            result = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertTrue(np.isfinite(result["chamfer"]))
            self.assertEqual(result["coupling_details"]["fine_pairing"], metadata["fine_pairing"])

    def test_exact_memmaps_are_usable_by_windows_spawn_workers(self):
        for sampling in ("bank", "stream"):
            config = self.config(sampling=sampling)
            config["tg_cache"]["num_workers"] = 1
            path, _ = self.prepare(config)
            source = np.load(path / "source.npy")
            target = np.load(path / "target.npy")
            permutation = np.load(path / "fine_permutation.npy")
            sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32, training=True)
            try:
                cloud_index = 0
                for _ in range(2):
                    actual_source, actual_target = sampler.sample(2)
                    for left, right in zip(actual_source.numpy(), actual_target.numpy()):
                        if sampling == "stream":
                            index = cloud_index
                        else:
                            matches = np.flatnonzero((source == left).all(axis=(1, 2)))
                            self.assertEqual(len(matches), 1)
                            index = matches[0]
                        np.testing.assert_array_equal(left, source[index])
                        np.testing.assert_array_equal(right, target[index, permutation[index]])
                        cloud_index += 1
            finally:
                sampler.close()


if __name__ == "__main__":
    unittest.main()
