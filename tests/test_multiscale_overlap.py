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

import multiscale_overlap as overlap
import multiscale_wasserstein as msw


class MultiscaleOverlapTests(unittest.TestCase):
    def setUp(self):
        self.previous = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        os.chdir(self.temp.name)

    def tearDown(self):
        os.chdir(self.previous)
        self.temp.cleanup()

    @staticmethod
    def config():
        return {"seed": 0, "coupling": overlap.METHOD,
                "multiscale_wasserstein": {"cache": "baseline.npz", "superset_size": 32,
                    "cache_seed": 5, "partition_counts": [2, 4], "scale_weights": [1., 1.],
                    "max_cost_mib": 512},
                "multiscale_overlap": {"cache": "overlap.npz", "window_sizes": [4, 8],
                    "max_windows_per_scale": 4, "sweeps": 2, "neighbors": 4,
                    "max_local_solves": 100},
                "data": {"batch_size": 2, "n_points": 4, "grid_size": 4},
                "model": {"point_dim": 2}}

    def prepare(self, config):
        with contextlib.redirect_stdout(io.StringIO()):
            msw.prepare(overlap.base_config(config), "checkerboard")
            return overlap.prepare(config, "checkerboard")

    def test_settings_work_bounds_include_spaces_scales_widths_and_sweeps(self):
        config = self.config()
        opts = overlap.settings(config)
        self.assertEqual(opts["local_solve_upper_bound"], 64)
        for key, value in (("window_sizes", []), ("window_sizes", [8, 4]),
                           ("window_sizes", [4, 4]), ("window_sizes", [True, 4]),
                           ("window_sizes", [4, 32]), ("sweeps", -1), ("sweeps", True),
                           ("neighbors", 0), ("neighbors", 32), ("max_windows_per_scale", 0),
                           ("max_local_solves", 63), ("cache", "baseline.npz"),
                           ("cache", "auto"), ("smoothing", 1.)):
            changed = copy.deepcopy(config)
            changed[overlap.METHOD][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                overlap.settings(changed)
        self.assertEqual(config["coupling"], overlap.METHOD)
        self.assertEqual(overlap.base_config(config)["coupling"], msw.METHOD)
        self.assertNotIn(overlap.METHOD, overlap.base_config(config))

    def test_windows_are_deterministic_fixed_bounded_and_cross_actual_labels(self):
        points = np.random.default_rng(19).normal(size=(48, 2)).astype(np.float32)
        labels = msw.balanced_partitions(points, [2, 4, 8])
        first, reports = overlap.boundary_windows(points, labels, [6, 12], 5, 8)
        second, second_reports = overlap.boundary_windows(points, labels, [6, 12], 5, 8)
        self.assertEqual(reports, second_reports)
        self.assertGreater(len(first), 0)
        self.assertLessEqual(len(first), 3 * 2 * 5)
        for window, again in zip(first, second):
            np.testing.assert_array_equal(window["indices"], again["indices"])
            self.assertEqual(len(window["indices"]), window["width"])
            self.assertGreaterEqual(len(np.unique(labels[window["level"]][window["indices"]])), 2)
            self.assertTrue(set(window["anchor"]).issubset(window["indices"]))
        self.assertTrue(any(len(set(a["indices"]) & set(b["indices"])) > 0
                            for i, a in enumerate(first) for b in first[i + 1:]))

    def test_cost_matches_original_baseline_and_decreases_without_endpoint_changes(self):
        rng = np.random.default_rng(32)
        source, target = rng.normal(size=(32, 2)), rng.normal(size=(32, 2))
        original = source.copy(), target.copy()
        base, baseline = msw.multiscale_permutation(source, target, [2, 4], [1., 2.])
        result, details = overlap.overlap_permutation(source, target, base, [2, 4], [1., 2.],
                        window_sizes=[4, 8], max_windows_per_scale=4, sweeps=2, neighbors=4)
        self.assertAlmostEqual(details["objective_before"], baseline["multiscale_objective"], places=12)
        self.assertLessEqual(details["objective_after"], details["objective_before"])
        np.testing.assert_array_equal(np.sort(result), np.arange(32))
        np.testing.assert_array_equal(source, original[0])
        np.testing.assert_array_equal(target, original[1])
        self.assertLessEqual(details["local_solves"], details["local_solve_upper_bound"])
        self.assertEqual(details["changed_fraction"], np.mean(base != result))
        json.dumps(details, allow_nan=False)

    def test_zero_sweeps_identity_and_repeatability(self):
        rng = np.random.default_rng(7)
        source, target = rng.normal(size=(32, 2)), rng.normal(size=(32, 2))
        base, _ = msw.multiscale_permutation(source, target, [2, 4], [1., 1.])
        kwargs = dict(window_sizes=[4, 8], max_windows_per_scale=4, neighbors=4)
        result, details = overlap.overlap_permutation(source, target, base, [2, 4], [1., 1.], sweeps=0, **kwargs)
        np.testing.assert_array_equal(result, base)
        self.assertEqual(details["local_solves"], 0)
        first, report = overlap.overlap_permutation(source, target, base, [2, 4], [1., 1.], **kwargs)
        second, again = overlap.overlap_permutation(source, target, base, [2, 4], [1., 1.], **kwargs)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(report["objective_after"], again["objective_after"])
        self.assertEqual(report["window_signature_sha256"], again["window_signature_sha256"])

    def test_overlapping_source_and_target_windows_use_current_partners(self):
        # Window 1 swaps 0/1. Window 2 overlaps at source 1 and must see target 0,
        # not its original target 1. Window 3 is target-anchored and must invert
        # the already updated permutation, not the initial one.
        class Cost:
            def __init__(self):
                self.calls = []
                self.table = np.array([[10., 0., 10.], [0., 10., -1.], [-1., 10., 10.]])

            def matrix(self, source_ids, target_ids):
                self.calls.append((source_ids.copy(), target_ids.copy()))
                return self.table[np.ix_(source_ids, target_ids)]

            def objective(self, order):
                return float(self.table[np.arange(3), order].mean())

        cost = Cost()
        windows = [("source", {"indices": np.array([0, 1])}),
                   ("source", {"indices": np.array([1, 2])}),
                   ("target", {"indices": np.array([1, 2])})]
        result, report = overlap._apply_windows(np.arange(3), windows, cost, 1)
        np.testing.assert_array_equal(cost.calls[1][1], [0, 2])
        np.testing.assert_array_equal(cost.calls[2][0], [0, 1])
        np.testing.assert_array_equal(result, [1, 2, 0])
        self.assertEqual(report["accepted_local_solves"], 2)
        np.testing.assert_array_equal(np.sort(result), np.arange(3))

    def test_never_allocates_global_point_cost_matrix(self):
        rng = np.random.default_rng(1)
        source, target = rng.normal(size=(32, 2)), rng.normal(size=(32, 2))
        base, _ = msw.multiscale_permutation(source, target, [2, 4], [1., 1.])
        real_local_cdist = overlap.cdist
        real_patch_cdist = msw.cdist
        calls = []

        def guarded(a, b, *args, **kwargs):
            calls.append((len(a), len(b)))
            self.assertNotEqual((len(a), len(b)), (32, 32))
            return real_local_cdist(a, b, *args, **kwargs)

        with patch.object(overlap, "cdist", side_effect=guarded), patch.object(msw, "cdist", side_effect=guarded):
            overlap.overlap_permutation(source, target, base, [2, 4], [1., 1.],
                window_sizes=[4, 8], max_windows_per_scale=4, sweeps=1, neighbors=4)
        self.assertTrue(calls)
        self.assertIs(real_patch_cdist, real_local_cdist)

    def test_cache_requires_baseline_and_does_not_modify_it(self):
        config = self.config()
        with self.assertRaises(FileNotFoundError):
            overlap.prepare(config, "checkerboard")
        with contextlib.redirect_stdout(io.StringIO()):
            baseline_path, _ = msw.prepare(overlap.base_config(config), "checkerboard")
        baseline_bytes = baseline_path.read_bytes()
        self.prepare(config)
        self.assertEqual(baseline_path.read_bytes(), baseline_bytes)
        source, target, result, metadata, _ = overlap.load_cache(config, "checkerboard")
        bs, bt, bp, bm, _ = msw.load_cache(overlap.base_config(config), "checkerboard")
        np.testing.assert_array_equal(source, bs)
        np.testing.assert_array_equal(target, bt)
        self.assertEqual(metadata["base_permutation_sha256"], msw._array_hash(bp))
        self.assertAlmostEqual(metadata["objective_before"], bm["multiscale_objective"])
        with np.load(config[overlap.METHOD]["cache"], allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["base_permutation"], bp)

    def test_cache_reuse_mismatch_checkpoint_hash_and_self_contained_loading(self):
        config = self.config()
        path, _ = self.prepare(config)
        original = path.read_bytes()
        with patch.object(overlap, "overlap_permutation", side_effect=AssertionError("recomputed")):
            self.prepare(config)
        self.assertEqual(path.read_bytes(), original)
        for key, value in (("neighbors", 8), ("window_sizes", [4, 12]), ("cache_sha256", "bad")):
            changed = copy.deepcopy(config)
            changed[overlap.METHOD][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                overlap.prepare(changed, "checkerboard")
            self.assertEqual(path.read_bytes(), original)
        changed = copy.deepcopy(config)
        changed[msw.METHOD]["cache_sha256"] = "bad"
        with self.assertRaises(ValueError):
            overlap.load_cache(changed, "checkerboard")
        Path(config[msw.METHOD]["cache"]).rename("baseline.moved")
        # The standalone overlap bank remains sufficient for pair sampling.
        overlap.load_cache(config, "checkerboard")

    def test_cache_rejects_tampered_endpoints_permutations_and_extra_arrays(self):
        config = self.config()
        path, _ = self.prepare(config)
        with np.load(path, allow_pickle=False) as archive:
            originals = {name: archive[name].copy() for name in archive.files}
        for corruption in ("source", "target", "permutation", "base_permutation", "extra"):
            arrays = {name: value.copy() for name, value in originals.items()}
            if corruption in ("source", "target"):
                arrays[corruption][0, 0] += .1
            elif corruption in ("permutation", "base_permutation"):
                arrays[corruption][[0, 1]] = arrays[corruption][[1, 0]]
            else:
                arrays["unexpected"] = np.zeros(1)
            np.savez_compressed(path, **arrays)
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):
                overlap.load_cache(config, "checkerboard")

    def test_sampler_uses_same_iid_indices_and_preserves_rng(self):
        config = self.config()
        self.prepare(config)
        source, target, permutation, _, _ = overlap.load_cache(config, "checkerboard")
        sampler = overlap.MultiscaleOverlapSampler(config, "checkerboard", "cpu", torch.float32)
        generator = torch.Generator().manual_seed(11)
        sx, ty = sampler.sample(20, generator=generator)
        reference = torch.Generator().manual_seed(11)
        indices = torch.randint(32, (20, 4), generator=reference)
        torch.testing.assert_close(sx, torch.from_numpy(source)[indices], rtol=0, atol=0)
        torch.testing.assert_close(ty, torch.from_numpy(target[permutation])[indices], rtol=0, atol=0)
        torch.testing.assert_close(generator.get_state(), reference.get_state(), rtol=0, atol=0)
        self.assertEqual(sampler.details()["beta"], 0.)


if __name__ == "__main__":
    unittest.main()
