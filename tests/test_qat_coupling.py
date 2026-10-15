import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import ndtri
import torch

import multiscale_wasserstein as msw
import qat_coupling as qat


class QATCouplingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        self.temp.cleanup()
        torch.set_num_threads(self.threads)

    def config(self, dataset="checkerboard", size=64):
        config = {
            "seed": 7, "coupling": "qat", "device": "cpu", "dtype": "float32",
            "qat": {"cache": str(Path(self.temp.name) / f"{dataset}.npz"),
                    "superset_size": size, "cache_seed": 11},
            "data": {"batch_size": 2, "n_points": min(16, size)},
            "model": {"point_dim": 2},
        }
        if dataset == "checkerboard":
            config["data"]["grid_size"] = 4
        return config

    def prepare(self, config, dataset="checkerboard"):
        with contextlib.redirect_stdout(io.StringIO()):
            return qat.prepare(config, dataset)

    def sampler(self, config=None, dataset="checkerboard", dtype=torch.float32, device="cpu"):
        config = config or self.config(dataset)
        self.prepare(config, dataset)
        return qat.QATPairSampler(config, dataset, device, dtype)

    def test_default_and_invalid_settings(self):
        config = self.config()
        del config["qat"]["superset_size"]
        del config["qat"]["cache_seed"]
        self.assertEqual(qat.settings(config)["superset_size"], 10000)
        self.assertEqual(qat.settings(config)["cache_seed"], 0)
        for changes in ({"superset_size": True}, {"superset_size": 10001}, {"superset_size": 0},
                        {"cache_seed": -1}, {"cache_seed": 1.2}, {"cache_seed": 2**63},
                        {"cache": ""}, {"cache": "auto"}, {"cache": None},
                        {"leaf_count": 100}, {"beta": .1}, {"cache_sha256": "wrong"}):
            other = self.config()
            other["qat"].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                qat.settings(other)
        for key, value in (("model", {"point_dim": 3}), ("data", {"n_points": 65}),
                           ("data", {"n_points": True}), ("coupling", "independent"),
                           ("qat", []), ("seed", -1)):
            other = self.config()
            other[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                qat.settings(other)

    def test_singletons_have_equal_gaussian_mass_without_moving_targets(self):
        points = np.random.default_rng(19).normal(size=(127, 2)).astype(np.float32)
        original = points.copy()
        lo, hi, labels, diagnostics = qat.build_quantile_tree(points)
        np.testing.assert_array_equal(points, original)
        self.assertEqual(diagnostics["leaf_count"], len(points))
        self.assertEqual(diagnostics["duplicate_group_count"], 0)
        np.testing.assert_array_equal(np.sort(labels), np.arange(len(points)))
        np.testing.assert_allclose(np.prod(hi - lo, axis=1), 1 / len(points), rtol=1e-12)
        lo2, hi2, labels2, diagnostics2 = qat.build_quantile_tree(points)
        for a, b in ((lo, lo2), (hi, hi2), (labels, labels2)):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(diagnostics, diagnostics2)

    def test_mean_split_uses_geometric_target_threshold_and_mass_ratio(self):
        target = np.array([[-4., 0.], [-1., -.5], [1., 0.], [9., 3.]])
        lo, hi, _, _ = qat.build_quantile_tree(target)
        # Root maximum-variance axis is x; mean is 1.25, not its median.
        # Three target points on the left require source-CDF split at 3/4.
        self.assertTrue(np.all(hi[:3, 0] <= .75))
        self.assertEqual(lo[3, 0], .75)
        self.assertEqual(hi[3, 0], 1.)
        source = ndtri((lo + hi) / 2)
        for t in np.linspace(0, 1, 11):
            interpolated = (1 - t) * source + t * target
            self.assertTrue(np.all(interpolated[:3, 0] < interpolated[3, 0]))

    def test_split_axis_is_max_variance_not_max_range(self):
        points = np.array([[0., -5.], [0., 5.], [0., -5.], [0., 5.], [10., -5.]])
        # x range=10 ties y range=10, but y variance (24) exceeds x (16).
        lo, hi, _, _ = qat.build_quantile_tree(points)
        left = points[:, 1] <= points[:, 1].mean()
        self.assertTrue(np.all(hi[left, 1] <= .6))
        self.assertTrue(np.all(lo[~left, 1] >= .6))

    def test_duplicate_coordinates_are_one_mass_weighted_terminal_group(self):
        points = np.array([[0., 0.], [0., 0.], [1., 0.], [2., 1.], [2., 1.]])
        lo, hi, labels, info = qat.build_quantile_tree(points)
        self.assertEqual(info["leaf_count"], 3)
        self.assertEqual(info["duplicate_group_count"], 2)
        self.assertEqual(info["max_leaf_size"], 2)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[3], labels[4])
        np.testing.assert_allclose(np.prod(hi - lo, 1), [.4, .4, .2, .4, .4])
        lo, hi, labels, info = qat.build_quantile_tree(np.zeros((200, 2)))
        np.testing.assert_array_equal(lo, np.zeros((200, 2)))
        np.testing.assert_array_equal(hi, np.ones((200, 2)))
        self.assertEqual(info["leaf_count"], 1)
        self.assertEqual(info["split_count"], 0)
        self.assertEqual(info["duplicate_group_count"], 1)

    def test_invalid_target_points_rejected(self):
        for points in (np.empty((0, 2)), np.zeros((4, 3)), [[float("nan"), 0]],
                       [[float("inf"), 0]], np.ones((2, 2), dtype=complex), [["a", "b"]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                qat.build_quantile_tree(points)

    def test_target_draw_matches_msw_byte_for_byte_and_preserves_global_rng(self):
        for dataset in ("checkerboard", "horse"):
            config = self.config(dataset)
            baseline = copy.deepcopy(config)
            baseline["coupling"] = msw.METHOD
            baseline[msw.METHOD] = {**config["qat"], "partition_counts": [2, 4],
                                    "scale_weights": [1, 1], "max_cost_mib": 512}
            before = torch.random.get_rng_state().clone()
            target = qat.draw_target(config, dataset)
            _, baseline_target = msw.draw_supersets(baseline, dataset)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            np.testing.assert_array_equal(target, baseline_target)
            self.prepare(config, dataset)
            stored, _, _, _, metadata, _ = qat.load_cache(config, dataset)
            np.testing.assert_array_equal(stored, target)
            self.assertEqual(metadata["endpoint_sha256"], {"target": qat._array_hash(target)})

    def test_source_is_continuous_gaussian_with_correct_unconditional_moments(self):
        config = self.config(size=128)
        sampler = self.sampler(config)
        source, target = sampler.sample(4096, generator=torch.Generator().manual_seed(71))
        self.assertEqual(source.shape, (4096, 16, 2))
        self.assertTrue(torch.isfinite(source).all())
        sample = source.reshape(-1, 2).double()
        self.assertLess(float(sample.mean(0).abs().max()), .025)
        covariance = (sample - sample.mean(0)).T @ (sample - sample.mean(0)) / len(sample)
        self.assertLess(float((covariance - torch.eye(2)).abs().max()), .04)
        self.assertGreater(len(torch.unique(source.reshape(-1, 2), dim=0)), 65000)
        self.assertLessEqual(len(torch.unique(target.reshape(-1, 2), dim=0)), 128)

    def test_sampling_reproducible_fresh_and_preserves_time_and_pair_index_rng(self):
        first, second = self.sampler(), self.sampler()
        generator = torch.Generator().manual_seed(31)
        clone = torch.Generator().manual_seed(31)
        global_state = torch.random.get_rng_state().clone()
        a, target_a = first.sample(3, generator=generator)
        b, target_b = second.sample(3, generator=clone)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(target_a, target_b))
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))
        expected = torch.Generator().manual_seed(31)
        torch.randint(len(first.target), (3, first.n_points), generator=expected)
        self.assertTrue(torch.equal(generator.get_state(), expected.get_state()))
        # Same target indices, new conditional Gaussian samples (no cached x0).
        c, target_c = first.sample(3, generator=torch.Generator().manual_seed(31))
        self.assertTrue(torch.equal(target_a, target_c))
        self.assertFalse(torch.equal(a, c))
        self.assertEqual(first.details()["source_sampling_seed"], 9)
        self.assertNotIn("source", vars(first))

    def test_source_stays_inside_its_cdf_rectangle_and_endpoint_guards_are_finite(self):
        sampler = self.sampler(dtype=torch.float64)
        for endpoint in (0., 1., None):
            reference = torch.Generator().manual_seed(7)
            indices = torch.randint(len(sampler.target), (2, sampler.n_points), generator=reference)
            if endpoint is None:
                source, _ = sampler.sample(2, generator=torch.Generator().manual_seed(7))
            else:
                with patch.object(qat.torch, "rand", return_value=torch.full((2, sampler.n_points, 2),
                                                                            endpoint, dtype=torch.float64)):
                    source, _ = sampler.sample(2, generator=torch.Generator().manual_seed(7))
            self.assertTrue(torch.isfinite(source).all())
            # ndtr can round an extreme lower-tail value to 0; coordinate bounds
            # still verify interval membership without a forward-CDF cancellation.
            self.assertTrue((source >= torch.special.ndtri(sampler.cdf_lower[indices])).all())
            self.assertTrue((source <= torch.special.ndtri(sampler.cdf_upper[indices])).all())

    def test_cache_reuse_no_overwrite_and_identity_checks(self):
        config = self.config()
        path, metadata = self.prepare(config)
        before = path.read_bytes()
        self.assertEqual(self.prepare(config)[1], metadata)
        self.assertEqual(path.read_bytes(), before)
        for changes in ({"cache_seed": 10}, {"superset_size": 32}, {"cache_sha256": "0" * 64}):
            changed = copy.deepcopy(config)
            changed["qat"].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "mismatch|SHA256"):
                self.prepare(changed)
            self.assertEqual(path.read_bytes(), before)
        changed = copy.deepcopy(config)
        changed["data"]["grid_size"] = 2
        with self.assertRaisesRegex(ValueError, "dataset_spec"):
            qat.load_cache(changed, "checkerboard")

    def test_cache_tampering_rejected_even_with_updated_array_hashes(self):
        config = self.config()
        path, _ = self.prepare(config)
        with np.load(path, allow_pickle=False) as archive:
            original = {key: archive[key] for key in archive.files}
        for kind in ("target", "cdf_lower", "labels", "source", "timing", "geometry"):
            arrays = {key: value.copy() for key, value in original.items()}
            metadata = json.loads(str(arrays["metadata"].item()))
            if kind == "target":
                arrays["target"][0, 0] += .01
            elif kind == "cdf_lower":
                arrays["cdf_lower"][0, 0] = -1
            elif kind == "labels":
                arrays["leaf_labels"][0] = -1
            elif kind == "source":
                arrays["source"] = np.zeros_like(arrays["target"])
            elif kind == "timing":
                metadata["precompute_seconds"] = -1
            else:
                arrays["cdf_lower"] = arrays["cdf_lower"][::-1].copy()
                arrays["cdf_upper"] = arrays["cdf_upper"][::-1].copy()
                metadata["tree_sha256"] = {key: qat._array_hash(arrays[key]) for key in
                                            ("cdf_lower", "cdf_upper", "leaf_labels")}
            arrays["metadata"] = np.array(json.dumps(metadata))
            np.savez_compressed(path, **arrays)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                qat.load_cache(config, "checkerboard")

    def test_invalid_sampling_inputs(self):
        sampler = self.sampler()
        for batch in (0, -1, True, 2.5):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                sampler.sample(batch)
        with self.assertRaisesRegex(ValueError, "float32 or float64"):
            qat.QATPairSampler(self.config(), "checkerboard", "cpu", torch.float16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_sampling(self):
        for dtype in (torch.float32, torch.float64):
            sampler = self.sampler(dtype=dtype, device="cuda")
            source, target = sampler.sample(2, generator=torch.Generator(device="cuda").manual_seed(41))
            self.assertEqual(source.device.type, "cuda")
            self.assertEqual(source.dtype, dtype)
            self.assertEqual(source.shape, target.shape)
            self.assertTrue(torch.isfinite(source).all())


if __name__ == "__main__":
    unittest.main()
