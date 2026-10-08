"""Independent numerical checks for frozen-teacher permutation ranking."""

import copy
import unittest

import numpy as np
import torch
from torch import nn

import tg_model_guidance as guidance


class AffineCloudTeacher(nn.Module):
    """Full-cloud affine map, including derivatives between distinct points."""

    def __init__(self, matrix, bias=None, time_bias=None):
        super().__init__()
        matrix = np.asarray(matrix, dtype=np.float64)
        self.matrix = nn.Parameter(torch.from_numpy(matrix.copy()), requires_grad=False)
        self.register_buffer("bias", torch.from_numpy(
            np.zeros(len(matrix)) if bias is None else np.asarray(bias, dtype=np.float64)))
        self.register_buffer("time_bias", torch.from_numpy(
            np.zeros(len(matrix)) if time_bias is None else np.asarray(time_bias, dtype=np.float64)))
        self.eval()

    def forward(self, positions, times):
        flat = positions.flatten(1)
        result = flat @ self.matrix.T + self.bias + times.reshape(-1, 1) * self.time_bias
        return result.reshape_as(positions)


class GuidanceScoreTests(unittest.TestCase):
    @staticmethod
    def fixture():
        source = np.array([[-1., 0.], [-.1, .1], [.2, -.1],
                           [1., 0.], [.1, .2], [-.2, -.2]])
        target = np.array([[-.7, .9], [.8, .7], [-.9, -.4],
                           [.9, -.8], [-.2, .8], [.4, -.9]])
        source_labels = np.array([0, 0, 0, 1, 1, 1])
        target_labels = np.array([0, 1, 0, 1, 0, 1])
        options = {"teacher_config": "runs/teacher/config.yaml", "neighbors": 2,
                   "times": [.2, .6], "candidates": 5, "keep": 2,
                   "probes": 3, "seed": 11, "inference_batch_size": 2}
        width = source.size
        matrix = .3 * np.eye(width) + np.arange(width * width).reshape(width, width) / 1000.
        teacher = AffineCloudTeacher(matrix, np.arange(width) / 30., -np.arange(width) / 50.)
        pool = guidance.generate_candidates(source_labels, target_labels, 2, options, cloud_index=9)
        return teacher, source, target, source_labels, target_labels, pool, options

    @staticmethod
    def manual_edges(source, neighbors):
        edges = set()
        for point in range(len(source)):
            others = [other for other in range(len(source)) if other != point]
            others.sort(key=lambda other: (sum((source[point] - source[other]) ** 2), other))
            for other in others[:neighbors]:
                edges.add(tuple(sorted((point, other))))
        return sorted(edges)

    @staticmethod
    def analytic_components(teacher, source, target, pool, options, cloud_index):
        matrix = teacher.matrix.detach().numpy()
        bias = teacher.bias.numpy()
        time_bias = teacher.time_bias.numpy()
        edges = GuidanceScoreTests.manual_edges(source, options["neighbors"])
        components = np.zeros((len(pool), 3))
        for time_index, time in enumerate(options["times"]):
            probe_energy = []
            for probe_index in range(options["probes"]):
                rng = np.random.default_rng(np.random.SeedSequence(
                    [options["seed"], cloud_index, 1, time_index, probe_index]))
                direction = (2. * rng.integers(0, 2, size=source.shape) - 1.).ravel()
                probe_energy.append(np.mean((matrix @ direction) ** 2))
            for index, permutation in enumerate(pool):
                velocity = target[permutation] - source
                position = source + time * velocity
                predicted = (matrix @ position.ravel() + bias + time * time_bias).reshape(source.shape)
                residual = predicted - velocity
                components[index, 0] += sum(residual.ravel() ** 2) / source.size
                components[index, 1] += sum(sum((residual[a] - residual[b]) ** 2)
                                            for a, b in edges) / (len(edges) * source.shape[1])
                components[index, 2] += sum(probe_energy) / len(probe_energy)
        return components / len(options["times"])

    def test_options_defaults_copy_partial_weights_and_disabled(self):
        self.assertFalse(guidance.normalize_options(None)["enabled"])
        self.assertFalse(guidance.normalize_options(False)["enabled"])
        self.assertFalse(guidance.normalize_options({"enabled": False})["enabled"])
        supplied = {"teacher_config": " teacher.yaml ", "times": [.4], "weights": {"local": 0.}}
        original = copy.deepcopy(supplied)
        options = guidance.normalize_options(supplied)
        self.assertEqual(supplied, original)
        self.assertTrue(options["enabled"])
        self.assertEqual(options["teacher_config"], "teacher.yaml")
        self.assertEqual(options["weights"], {"regression": 1., "local": 0., "jacobian": .1})
        options["times"].append(.8)
        options["weights"]["regression"] = 10.
        self.assertEqual(supplied, original)
        self.assertEqual(guidance.DEFAULTS["weights"]["regression"], 1.)

    def test_analytical_full_cloud_affine_components_and_score(self):
        teacher, source, target, _, _, pool, options = self.fixture()
        actual = guidance.score_candidates(teacher, source, target, pool, options, cloud_index=9)
        expected = self.analytic_components(teacher, source, target, pool, options, 9)
        np.testing.assert_allclose(actual["components"], expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(actual["normalizers"], expected.mean(axis=0), rtol=2e-12)
        weights = np.array([1., .25, .1])
        scores = np.sum(expected / expected.mean(axis=0) * weights, axis=1)
        np.testing.assert_allclose(actual["scores"], scores, rtol=2e-12)
        np.testing.assert_array_equal(actual["selected_indices"], np.argsort(scores, kind="stable")[:2])
        self.assertEqual(actual["components"].dtype, np.dtype(np.float64))
        self.assertEqual(actual["scores"].dtype, np.dtype(np.float64))
        self.assertEqual(actual["normalizers"].dtype, np.dtype(np.float64))
        self.assertEqual(actual["selected_indices"].dtype, np.dtype(np.int32))

    def test_diagonal_jacobian_is_frobenius_energy_per_cloud_coordinate(self):
        _, source, target, _, _, pool, options = self.fixture()
        diagonal = np.linspace(.2, 2.7, source.size)
        teacher = AffineCloudTeacher(np.diag(diagonal))
        options = {**options, "weights": {"regression": 0., "local": 0., "jacobian": 1.}}
        result = guidance.score_candidates(teacher, source, target, pool, options, cloud_index=9)
        np.testing.assert_allclose(result["components"][:, 2], np.mean(diagonal ** 2), rtol=1e-12)
        np.testing.assert_array_equal(result["components"][:, :2], 0.)
        np.testing.assert_allclose(result["scores"], 1., rtol=1e-12)
        # ||J||_F^2/(N D), not ||J||_F^2/N or max singular value squared.
        self.assertNotAlmostEqual(float(result["components"][0, 2]), float(np.max(diagonal) ** 2))

    def test_local_term_includes_cross_patch_edges(self):
        source = np.array([[0., 0.], [.01, 0.], [10., 0.], [10.01, 0.]])
        labels = np.array([0, 1, 0, 1])
        target = np.array([[1., 2.], [5., 6.], [3., 4.], [7., 8.]])
        pool = np.array([[0, 1, 2, 3], [2, 3, 0, 1]], dtype=np.int32)
        options = {"teacher_config": "teacher.yaml", "neighbors": 1, "candidates": 2,
                   "keep": 1, "times": [.5], "weights": {"regression": 0., "local": 1., "jacobian": 0.}}
        teacher = AffineCloudTeacher(np.zeros((source.size, source.size)))
        result = guidance.score_candidates(teacher, source, target, pool, options)
        self.assertEqual(result["edge_count"], 2)
        self.assertTrue(all(labels[a] != labels[b] for a, b in self.manual_edges(source, 1)))
        for index, permutation in enumerate(pool):
            velocity = target[permutation] - source
            expected = np.mean(np.array([velocity[0] - velocity[1], velocity[2] - velocity[3]]) ** 2)
            self.assertAlmostEqual(result["components"][index, 1], expected, places=12)

    def test_candidate_and_probe_random_streams_are_chunk_and_control_invariant(self):
        teacher, source, target, source_labels, target_labels, pool, options = self.fixture()
        baseline = guidance.score_candidates(teacher, source, target, pool, options, cloud_index=9)
        for chunk_size in (1, 3, 16):
            changed = guidance.score_candidates(teacher, source, target, pool,
                                                {**options, "inference_batch_size": chunk_size}, cloud_index=9)
            np.testing.assert_allclose(changed["components"], baseline["components"], rtol=3e-12, atol=1e-12)
            np.testing.assert_array_equal(changed["selected_indices"], baseline["selected_indices"])
        random_options = {**options, "selection": "random"}
        repeated_pool = guidance.generate_candidates(source_labels, target_labels, 2, random_options, cloud_index=9)
        np.testing.assert_array_equal(repeated_pool, pool)
        random = guidance.score_candidates(teacher, source, target, pool, random_options, cloud_index=9)
        np.testing.assert_array_equal(random["components"], baseline["components"])
        np.testing.assert_array_equal(random["normalizers"], baseline["normalizers"])
        np.testing.assert_array_equal(random["scores"], baseline["scores"])
        repeated = guidance.score_candidates(teacher, source, target, pool, random_options, cloud_index=9)
        np.testing.assert_array_equal(random["selected_indices"], repeated["selected_indices"])
        self.assertEqual(len(np.unique(random["selected_indices"])), options["keep"])
        larger = guidance.generate_candidates(source_labels, target_labels, 2,
                                              {**options, "candidates": 9}, cloud_index=9)
        np.testing.assert_array_equal(larger[:len(pool)], pool)
        prefix_score = guidance.score_candidates(teacher, source, target, larger,
                                                 {**options, "candidates": 9}, cloud_index=9)
        # Changing the final chunk's batch size can change floating BLAS
        # rounding; probes and candidate coordinates still remain identical.
        np.testing.assert_allclose(prefix_score["components"][:len(pool)], baseline["components"],
                                   rtol=3e-12, atol=1e-12)

    def test_probe_prefix_is_independent_of_probe_count(self):
        teacher, source, target, _, _, pool, options = self.fixture()
        one_options = {**options, "probes": 1}
        one = guidance.score_candidates(teacher, source, target, pool, one_options, cloud_index=9)
        expected = self.analytic_components(teacher, source, target, pool, one_options, 9)
        np.testing.assert_allclose(one["components"], expected, rtol=2e-12, atol=1e-12)
        three = guidance.score_candidates(teacher, source, target, pool, options, cloud_index=9)
        np.testing.assert_array_equal(one["components"][:, :2], three["components"][:, :2])

    def test_nonlinear_proxy_depends_on_candidate_not_a_crossing_guarantee(self):
        class QuadraticTeacher(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("dtype_reference", torch.zeros((), dtype=torch.float64))

            def forward(self, positions, times):
                return positions ** 2

        teacher = QuadraticTeacher().eval()
        source = np.array([[0.], [2.]])
        target = source.copy()
        pool = np.array([[0, 1], [1, 0]], dtype=np.int32)
        options = {"teacher_config": "teacher.yaml", "neighbors": 1, "times": [.25],
                   "candidates": 2, "keep": 1, "probes": 2,
                   "fd_epsilon": .01, "weights": {"regression": 0., "local": 0., "jacobian": 1.}}
        result = guidance.score_candidates(teacher, source, target, pool, options)
        positions = source[None, :, :] + .25 * (target[pool] - source[None, :, :])
        # Central FD of x^2 is exactly 2x before numerical rounding.
        np.testing.assert_allclose(result["components"][:, 2], np.mean(4. * positions ** 2, axis=(1, 2)), rtol=2e-12)
        self.assertEqual(result["selected_indices"].tolist(), [1])
        # A smaller Jacobian proxy may select a midpoint-crossing permutation.
        # It measures teacher sensitivity, not geometric untangling itself.

    def test_custom_stochastic_eval_teacher_does_not_advance_torch_global_rng(self):
        class RandomEvalTeacher(nn.Module):
            def forward(self, positions, times):
                return positions + torch.rand_like(positions)

        teacher, source, target, _, _, pool, options = self.fixture()
        random_teacher = RandomEvalTeacher().eval()
        before = torch.random.get_rng_state().clone()
        result = guidance.score_candidates(random_teacher, source, target, pool, options)
        self.assertTrue(np.isfinite(result["components"]).all())
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_real_transformer_cuda_finite_scores_and_state_preservation(self):
        from model import PointSetTransformer

        rng = np.random.default_rng(51)
        source = rng.normal(size=(16, 2)).astype(np.float32)
        target = rng.normal(size=(16, 2)).astype(np.float32)
        labels = np.repeat(np.arange(4), 4)
        options = {"teacher_config": "teacher.yaml", "neighbors": 3,
                   "times": [.25, .5], "candidates": 4, "keep": 2,
                   "probes": 1, "inference_batch_size": 2}
        teacher = PointSetTransformer(d_model=16, nhead=2, num_layers=1,
                                      dim_feedforward=32).cuda().eval()
        teacher.requires_grad_(False)
        pool = guidance.generate_candidates(labels, labels, 4, options)
        state = {name: value.detach().clone() for name, value in teacher.state_dict().items()}
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state().clone()
        result = guidance.score_candidates(teacher, source, target, pool, options)
        self.assertTrue(np.isfinite(result["components"]).all())
        self.assertTrue(np.isfinite(result["scores"]).all())
        self.assertEqual(result["selected_indices"].shape, (2,))
        for name, value in teacher.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))
        torch.testing.assert_close(torch.random.get_rng_state(), cpu_rng, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_realistic_n256_k8_cuda_checker_and_horse_teachers(self):
        from model import PointSetTransformer
        from train_horse import HorsePointSetTransformer

        rng = np.random.default_rng(25608)
        source = rng.normal(size=(256, 2)).astype(np.float32)
        target = rng.normal(size=(256, 2)).astype(np.float32)
        source_labels = np.repeat(np.arange(8), 32)
        target_labels = source_labels[::-1].copy()
        options = {"teacher_config": "runs/frozen_teacher/config.yaml"}
        normalized = guidance.normalize_options(options)
        self.assertEqual(normalized["candidates"], 8)
        self.assertEqual(normalized["keep"], 4)
        self.assertEqual(normalized["times"], [.25, .5, .75])
        self.assertEqual(normalized["probes"], 1)
        pool = guidance.generate_candidates(source_labels, target_labels, 8, options, cloud_index=3)
        self.assertEqual(pool.shape, (8, 256))
        for permutation in pool:
            np.testing.assert_array_equal(np.sort(permutation), np.arange(256))
            np.testing.assert_array_equal(target_labels[permutation], source_labels)
        original_source, original_target, original_pool = source.copy(), target.copy(), pool.copy()
        settings = [
            ("checkerboard", PointSetTransformer,
             {"point_dim": 2, "d_model": 128, "nhead": 4, "num_layers": 2,
              "dim_feedforward": 256, "dropout": 0.}),
            ("horse", HorsePointSetTransformer,
             {"point_dim": 2, "d_model": 256, "nhead": 8, "num_layers": 4,
              "dim_feedforward": 1024, "dropout": 0.}),
        ]
        for name, constructor, model_options in settings:
            with self.subTest(dataset=name):
                teacher = constructor(**model_options).cuda().eval()
                teacher.requires_grad_(False)
                checkpoint_state = {key: value.detach().clone()
                                    for key, value in teacher.state_dict().items()}
                cpu_rng = torch.random.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state().clone()
                result = guidance.score_candidates(teacher, source, target, pool, options, cloud_index=3)
                self.assertEqual(result["components"].shape, (8, 3))
                self.assertTrue(np.isfinite(result["components"]).all())
                self.assertTrue(np.isfinite(result["scores"]).all())
                self.assertTrue(np.isfinite(result["normalizers"]).all())
                self.assertTrue((result["normalizers"] > 0.).all())
                self.assertEqual(result["selected_indices"].shape, (4,))
                self.assertEqual(len(np.unique(result["selected_indices"])), 4)
                np.testing.assert_array_equal(result["selected_indices"],
                                              np.argsort(result["scores"], kind="stable")[:4])
                for permutation in pool[result["selected_indices"]]:
                    np.testing.assert_array_equal(target_labels[permutation], source_labels)
                    np.testing.assert_array_equal(np.sort(permutation), np.arange(256))
                for key, value in teacher.state_dict().items():
                    torch.testing.assert_close(value, checkpoint_state[key], rtol=0, atol=0)
                self.assertTrue(all(not parameter.requires_grad and parameter.grad is None
                                    for parameter in teacher.parameters()))
                self.assertTrue(all(not module.training for module in teacher.modules()))
                np.testing.assert_array_equal(source, original_source)
                np.testing.assert_array_equal(target, original_target)
                np.testing.assert_array_equal(pool, original_pool)
                torch.testing.assert_close(torch.random.get_rng_state(), cpu_rng, rtol=0, atol=0)
                torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng, rtol=0, atol=0)

    def test_generation_is_valid_deterministic_and_cloud_specific(self):
        _, _, _, source_labels, target_labels, pool, options = self.fixture()
        for permutation in pool:
            np.testing.assert_array_equal(np.sort(permutation), np.arange(len(source_labels)))
            np.testing.assert_array_equal(target_labels[permutation], source_labels)
        self.assertEqual(pool.dtype, np.dtype(np.int32))
        repeated = guidance.generate_candidates(source_labels, target_labels, 2, options, cloud_index=9)
        np.testing.assert_array_equal(repeated, pool)
        other = guidance.generate_candidates(source_labels, target_labels, 2, options, cloud_index=10)
        self.assertFalse(np.array_equal(other, pool))

    def test_preserves_inputs_teacher_gradients_modes_and_global_rng(self):
        teacher, source, target, source_labels, target_labels, pool, options = self.fixture()
        teacher.matrix.grad = torch.ones_like(teacher.matrix)
        arrays = [source, target, source_labels, target_labels, pool]
        original = [array.copy() for array in arrays]
        original_options = copy.deepcopy(options)
        state = {name: value.detach().clone() for name, value in teacher.state_dict().items()}
        original_gradient = teacher.matrix.grad.clone()
        numpy_rng, torch_rng = np.random.get_state(), torch.random.get_rng_state().clone()
        guidance.generate_candidates(source_labels, target_labels, 2, options, cloud_index=9)
        guidance.score_candidates(teacher, source, target, pool, options, cloud_index=9)
        for array, before in zip(arrays, original):
            np.testing.assert_array_equal(array, before)
        self.assertEqual(options, original_options)
        for name, value in teacher.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        torch.testing.assert_close(teacher.matrix.grad, original_gradient, rtol=0, atol=0)
        self.assertFalse(teacher.training)
        self.assertFalse(teacher.matrix.requires_grad)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], numpy_rng[0])
        np.testing.assert_array_equal(after_numpy[1], numpy_rng[1])
        self.assertEqual(after_numpy[2:], numpy_rng[2:])
        torch.testing.assert_close(torch.random.get_rng_state(), torch_rng, rtol=0, atol=0)

    def test_constant_teacher_zero_jacobian_and_stable_tie_selection(self):
        _, source, _, _, _, _, options = self.fixture()
        target = source + 1.
        pool = np.tile(np.arange(len(source), dtype=np.int32), (5, 1))
        teacher = AffineCloudTeacher(np.zeros((source.size, source.size)), np.ones(source.size))
        result = guidance.score_candidates(teacher, source, target, pool, options)
        np.testing.assert_array_equal(result["components"], 0.)
        np.testing.assert_array_equal(result["scores"], 0.)
        np.testing.assert_array_equal(result["normalizers"], np.full(3, 1e-12))
        np.testing.assert_array_equal(result["selected_indices"], np.array([0, 1]))

    def test_disabled_zero_weight_components_are_skipped(self):
        teacher, source, target, _, _, pool, options = self.fixture()
        calls = []
        handle = teacher.register_forward_hook(lambda module, args, result: calls.append(len(args[0])))
        try:
            result = guidance.score_candidates(teacher, source, target, pool,
                                                {**options, "weights": {"regression": 1., "local": 0., "jacobian": 0.}})
        finally:
            handle.remove()
        np.testing.assert_array_equal(result["components"][:, 1:], 0.)
        np.testing.assert_array_equal(result["normalizers"][1:], 1.)
        self.assertEqual(len(calls), len(options["times"]) * 3)
        self.assertLessEqual(max(calls), options["inference_batch_size"])

    def test_options_validation(self):
        enabled = {"teacher_config": "teacher.yaml"}
        bad = [True, 0, "true", {}, {"teacher_config": " "}, {"teacher_config": 1},
               {**enabled, "enabled": 1}, {**enabled, "unknown": 1},
               {**enabled, "neighbors": 0}, {**enabled, "candidates": True},
               {**enabled, "keep": 9}, {**enabled, "keep": 0},
               {**enabled, "probes": 0}, {**enabled, "seed": -1},
               {**enabled, "inference_batch_size": 0},
               {**enabled, "fd_epsilon": 0}, {**enabled, "fd_epsilon": float("inf")},
               {**enabled, "times": []}, {**enabled, "times": [0.]},
               {**enabled, "times": [1.]}, {**enabled, "times": [.2, .2]},
               {**enabled, "times": [float("nan")]}, {**enabled, "times": np.array(.5)},
               {**enabled, "weights": 1}, {**enabled, "weights": {"other": 1}},
               {**enabled, "weights": {"local": -1}},
               {**enabled, "weights": {"jacobian": float("nan")}},
               {**enabled, "weights": {"regression": 0, "local": 0, "jacobian": 0}},
               {**enabled, "normalization": "none"}, {**enabled, "selection": "best"}]
        for value in bad:
            with self.subTest(options=value), self.assertRaises(ValueError):
                guidance.normalize_options(value)

    def test_score_rejects_invalid_arrays_teacher_and_nonfinite_output(self):
        teacher, source, target, _, _, pool, options = self.fixture()
        invalid_source = source.copy()
        invalid_source[0, 0] = np.nan
        bad_inputs = [(invalid_source, target, pool), (source.astype(complex), target, pool),
                      (source, target[:-1], pool), (source, target, pool.astype(float)),
                      (source, target, np.zeros_like(pool)), (source, target, pool[:0]),
                      (source, target, pool[:, :-1])]
        for values in bad_inputs:
            with self.subTest(inputs=values), self.assertRaises(ValueError):
                guidance.score_candidates(teacher, *values, options)
        for changed in (False, {**options, "neighbors": len(source)}, {**options, "keep": 5}):
            with self.subTest(options=changed), self.assertRaises(ValueError):
                guidance.score_candidates(teacher, source, target, pool[:2], changed)
        teacher.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            guidance.score_candidates(teacher, source, target, pool, options)
        teacher.eval()
        teacher.matrix.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "frozen"):
            guidance.score_candidates(teacher, source, target, pool, options)
        teacher.matrix.requires_grad_(False)
        with torch.no_grad():
            teacher.bias[0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            guidance.score_candidates(teacher, source, target, pool, options)

    def test_candidate_generation_rejects_invalid_labels_cloud_and_disabled(self):
        _, _, _, source_labels, target_labels, _, options = self.fixture()
        bad_labels = [(source_labels.astype(float), target_labels, 2),
                      (source_labels, target_labels[:-1], 2),
                      (source_labels, np.zeros_like(target_labels), 2),
                      (source_labels, target_labels, 0)]
        for labels_a, labels_b, k in bad_labels:
            with self.subTest(labels=(labels_a, labels_b, k)), self.assertRaises(ValueError):
                guidance.generate_candidates(labels_a, labels_b, k, options)
        for index in (-1, True, .5):
            with self.subTest(cloud_index=index), self.assertRaises(ValueError):
                guidance.generate_candidates(source_labels, target_labels, 2, options, index)
        with self.assertRaises(ValueError):
            guidance.generate_candidates(source_labels, target_labels, 2, False)


if __name__ == "__main__":
    unittest.main()
