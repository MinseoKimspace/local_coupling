"""Mathematical and state-isolation checks for coupling model diagnostics."""

import hashlib
import unittest

import numpy as np
import torch

from msw2_coupling_model import fitting_audit, rollout_audit


class ConstantVelocity(torch.nn.Module):
    def __init__(self, velocity=(.25, -.5)):
        super().__init__()
        self.register_buffer("velocity", torch.tensor(velocity))
        self.dropout = torch.nn.Dropout(.5)
        self.forward_grad_modes = []

    def forward(self, x, t):
        self.forward_grad_modes.append(torch.is_grad_enabled())
        return self.velocity.expand_as(x)


class ContextMean(ConstantVelocity):
    def forward(self, x, t):
        self.forward_grad_modes.append(torch.is_grad_enabled())
        return x.mean(dim=1, keepdim=True).expand_as(x)


class LinearVelocity(ConstantVelocity):
    def forward(self, x, t):
        self.forward_grad_modes.append(torch.is_grad_enabled())
        return x


class MalformedVelocity(ConstantVelocity):
    def __init__(self, valid_calls=0):
        super().__init__()
        self.valid_calls, self.calls = valid_calls, 0

    def forward(self, x, t):
        self.calls += 1
        return self.velocity.expand_as(x) if self.calls <= self.valid_calls else x[:, :1, :]


class FakeData:
    def __init__(self, n, seed=97):
        self.n, self.seed = n, seed
        self.device, self.dtype = torch.device("cpu"), torch.float32

    def draw(self, index, purpose):
        raw = f"{self.seed}:{index}:{purpose}".encode()
        seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63-1)
        return torch.rand(self.n, 2, generator=torch.Generator().manual_seed(seed)) * 1.6 - .8

    def source(self, index, purpose):
        return self.draw(index, purpose + "source")

    def target(self, index, purpose):
        return self.draw(index, purpose + "target")


class FakeSpatial:
    region_names = ["left", "right"]
    expected_mass = np.array([.5, .5])
    grid_bins = 4

    def mass(self, points):
        points = np.asarray(points).reshape(-1, 2)
        inside = self.support(points)
        return np.array([(inside & (points[:, 0] < 0)).mean(),
                         (inside & (points[:, 0] >= 0)).mean()])

    @staticmethod
    def support(points):
        return (np.abs(np.asarray(points).reshape(-1, 2)) <= 1).all(axis=-1)

    def grid_hist(self, points):
        points = np.asarray(points).reshape(-1, 2)
        hist = np.histogram2d(points[:, 0], points[:, 1], bins=self.grid_bins,
                              range=[[-1, 1], [-1, 1]])[0]
        return np.append(hist.ravel(), len(points) - hist.sum()) / len(points)


class FakeWideSpatial(FakeSpatial):
    def __init__(self):
        self.path_hist_calls = 0

    def path_hist(self, points):
        self.path_hist_calls += 1
        points = np.asarray(points).reshape(-1, 2)
        hist = np.histogram2d(points[:, 0], points[:, 1], bins=2,
                              range=[[-4, 4], [-4, 4]])[0]
        return np.append(hist.ravel(), len(points) - hist.sum()) / len(points)


def quality(prediction, target):
    # Also asserts that the callback receives dtype-compatible CPU tensors.
    assert prediction.device.type == target.device.type == "cpu"
    distances = torch.cdist(prediction, target).square()
    return {"chamfer": float((distances.min(-1).values.mean(-1)
                               + distances.min(-2).values.mean(-1)).mean()),
            "leakage": float((prediction.abs() > 1).any(-1).double().mean())}


class CouplingModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    @staticmethod
    def banks():
        values = np.array([[-.75, -.25], [-.5, .25], [-.25, -.5],
                           [.25, .5], [.5, -.25], [.75, .25]], dtype=np.float32)
        return values, values + np.array([.25, -.5], dtype=np.float32)

    def test_constant_velocity_zero_fitting_and_exact_context_decomposition(self):
        source, target = self.banks()
        groups = {"left": source[:, 0] < 0, "right": source[:, 0] >= 0,
                  "empty": np.zeros(len(source), dtype=bool)}
        model = ConstantVelocity()
        summary, rows, arrays = fitting_audit(model, source, target, groups,
                                              n_points=4, clouds=5, batch_size=2,
                                              times=(0, .3, 1), anchors=4, contexts=3)
        self.assertFalse(summary["fresh_source_has_velocity_labels"])
        self.assertIn("not held-out bank", summary["scope"])
        self.assertTrue(all(row["mse_per_coordinate"] == 0
                            for row in rows["fitting_groups"] if row["sample_count"]))
        self.assertTrue(all(row["decomposition_absolute_error"] == 0
                            for row in rows["context_anchors"]))
        self.assertGreater(summary["context_test"]["group_coverage"]["left"], 0)
        self.assertGreater(summary["context_test"]["group_coverage"]["right"], 0)
        self.assertEqual(arrays["context_bank_indices"].shape, (4, 3, 4))
        np.testing.assert_array_equal(arrays["context_bank_indices"][:, :, 0],
                                      np.repeat(arrays["context_anchor_bank_indices"][:, None], 3, axis=1))
        self.assertFalse(any(model.forward_grad_modes))
        self.assertTrue(model.training)

    def test_context_variance_matches_direct_population_decomposition(self):
        source, target = self.banks()
        _, rows, arrays = fitting_audit(ContextMean(), source, target, {}, n_points=4,
                                        clouds=3, times=(.3,), anchors=3, contexts=7)
        predictions = arrays["context_anchor_predictions"][0]
        velocity = arrays["context_anchor_target_velocity"]
        for ai, row in enumerate(rows["context_anchors"]):
            direct = np.mean((predictions[ai] - velocity[ai]) ** 2)
            variance = np.var(predictions[ai], axis=0, ddof=0).mean()
            self.assertAlmostEqual(row["mse_per_coordinate"], direct, places=12)
            self.assertAlmostEqual(row["context_variance_per_coordinate"], variance, places=12)
            self.assertLess(row["decomposition_absolute_error"], 1e-12)
            self.assertGreater(row["context_variance_per_coordinate"], 0)

    def test_zero_energy_is_null_and_streams_do_not_depend_on_microbatch(self):
        source, _ = self.banks()
        results = [fitting_audit(ConstantVelocity((0, 0)), source, source, {},
                                n_points=4, clouds=5, times=(0, 1), anchors=4,
                                contexts=3, batch_size=batch) for batch in (1, 3)]
        self.assertTrue(all(row["normalized_mse"] is None
                            for row in results[0][1]["fitting_groups"]))
        for key in results[0][2]:
            np.testing.assert_array_equal(results[0][2][key], results[1][2][key])

    def test_preserve_mixed_modes_parameters_and_global_rng(self):
        source, target = self.banks()
        model = ConstantVelocity()
        model.eval()
        model.dropout.train()
        modes = [m.training for m in model.modules()]
        old = {key: value.clone() for key, value in model.state_dict().items()}
        torch_state, numpy_state = torch.get_rng_state().clone(), np.random.get_state()
        fitting_audit(model, source, target, {}, n_points=4, clouds=2,
                      times=(0, .3), anchors=3, contexts=2)
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        for left, right in zip(numpy_state, np.random.get_state()):
            np.testing.assert_equal(left, right)
        for key, value in old.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_rollout_constant_velocity_matches_teacher_and_shared_reference(self):
        source, target = self.banks()
        summary, rows, arrays = rollout_audit(
            ConstantVelocity(), source, target, FakeData(4), FakeSpatial(),
            {"data": {"n_points": 4}}, clouds=3, batch_size=2, nfes=(1, 4),
            path_steps=4, times=(0, .3, .75, 1), quality_score=quality)
        np.testing.assert_allclose(arrays["rollout_teacher_path"], arrays["rollout_bank_path"], atol=1e-7)
        for nfe in (1, 4):
            np.testing.assert_allclose(arrays[f"rollout_bank_prediction_nfe_{nfe}"],
                                       arrays["rollout_bank_target_control"], atol=1e-7)
            expected = arrays["rollout_fresh_source"] + np.array([.25, -.5])
            np.testing.assert_allclose(arrays[f"rollout_fresh_prediction_nfe_{nfe}"], expected, atol=1e-7)
        self.assertEqual(len(rows["rollout_quality"]), 6)
        self.assertEqual(len(rows["rollout_per_cloud"]), 12)
        self.assertEqual(len(summary["path_comparison"]), 4)
        self.assertTrue(all(row["bank_vs_paired_teacher_coordinate_mse"] < 1e-14
                            for row in rows["rollout_paths"]))
        self.assertTrue(all("leakage" not in row for row in rows["rollout_paths"]))
        for row in rows["rollout_quality"]:
            label, nfe = row["input"], row["nfe"]
            if nfe is None:
                continue
            direct = quality(torch.from_numpy(arrays[f"rollout_{label}_prediction_nfe_{nfe}"]).double(),
                             torch.from_numpy(arrays["rollout_target_reference"]).double())
            self.assertAlmostEqual(row["chamfer"], direct["chamfer"], places=12)

    def test_off_grid_snapshots_do_not_change_euler_trajectory(self):
        source, target = self.banks()
        model = LinearVelocity()
        _, rows, arrays = rollout_audit(model, source, target, FakeData(4), FakeSpatial(),
                                        {"data": {"n_points": 4}}, clouds=2, batch_size=1,
                                        nfes=(2,), path_steps=2, times=(.3, .75))
        self.assertEqual(len(model.forward_grad_modes), 8)
        self.assertFalse(any(model.forward_grad_modes))
        initial = arrays["rollout_bank_source"]
        for time, factor in ((0., 1.), (.3, 1.3), (.75, 1.875), (1., 2.25)):
            index = list(arrays["rollout_times"]).index(time)
            np.testing.assert_allclose(arrays["rollout_bank_path"][index], initial * factor, atol=1e-7)
        self.assertTrue(all("independent_teacher_histogram_tv" in row for row in rows["rollout_paths"]))
        self.assertFalse(np.array_equal(arrays["rollout_bank_indices"], arrays["rollout_teacher_control_indices"]))

    def test_rollout_draws_and_states_independent_of_microbatch_and_rng_isolated(self):
        source, target = self.banks()
        model = ConstantVelocity()
        model.eval()
        model.dropout.train()
        modes, rng = [m.training for m in model.modules()], torch.get_rng_state().clone()
        arguments = dict(clouds=3, nfes=(1, 2), path_steps=2, times=(0, .3, 1))
        results = [rollout_audit(model, source, target, FakeData(4), FakeSpatial(),
                                 {"data": {"n_points": 4}}, batch_size=batch, **arguments)
                   for batch in (1, 3)]
        for key in results[0][2]:
            np.testing.assert_array_equal(results[0][2][key], results[1][2][key])
        self.assertEqual(modes, [m.training for m in model.modules()])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_path_histogram_uses_optional_wider_common_grid(self):
        source, target = self.banks()
        spatial = FakeWideSpatial()
        summary, _, arrays = rollout_audit(
            ConstantVelocity(), source, target, FakeData(4), spatial,
            {"data": {"n_points": 4}}, clouds=1, nfes=(1,), path_steps=1, times=(0, 1))
        self.assertEqual(spatial.path_hist_calls, 8)
        self.assertIn("spatial.path_hist", summary["path_histogram_scope"])
        self.assertEqual(arrays["rollout_bank_path_histograms"].shape, (2, 5))
        self.assertEqual(spatial.grid_hist(arrays["rollout_bank_path"][-1]).shape, (17,))
        np.testing.assert_allclose(arrays["rollout_bank_path_histograms"].sum(axis=1), 1)

    def test_invalid_inputs_fail_without_model_calls(self):
        source, target = self.banks()
        model = ConstantVelocity()
        for overrides in ({"times": (0, 0)}, {"contexts": 1}, {"n_points": 1},
                          {"clouds": 0}, {"batch_size": 0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                fitting_audit(model, source, target, {}, **overrides)
        with self.assertRaises(ValueError):
            fitting_audit(model, source, target, {"bad": np.ones(3, dtype=bool)})
        self.assertFalse(model.forward_grad_modes)

    def test_broadcastable_malformed_velocity_fails_in_every_prediction_path(self):
        source, target = self.banks()
        # First fitting query or first context query (after one valid fitting query).
        for valid_calls in (0, 1):
            model = MalformedVelocity(valid_calls)
            model.dropout.eval()
            modes, rng = [m.training for m in model.modules()], torch.get_rng_state().clone()
            with self.subTest(path="fitting/context", valid_calls=valid_calls), \
                    self.assertRaisesRegex(ValueError, "does not match state shape"):
                fitting_audit(model, source, target, {}, n_points=4, clouds=1,
                              batch_size=1, times=(0,), anchors=1, contexts=2)
            self.assertEqual(modes, [m.training for m in model.modules()])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        # First path query, or endpoint query after the two one-step paths passed.
        for valid_calls in (0, 2):
            model = MalformedVelocity(valid_calls)
            model.dropout.eval()
            modes, rng = [m.training for m in model.modules()], torch.get_rng_state().clone()
            with self.subTest(path="snapshot/endpoint", valid_calls=valid_calls), \
                    self.assertRaisesRegex(ValueError, "does not match state shape"):
                rollout_audit(model, source, target, FakeData(4), FakeSpatial(),
                              {"data": {"n_points": 4}}, clouds=1, batch_size=1,
                              nfes=(2,), path_steps=1, times=(0, 1))
            self.assertEqual(modes, [m.training for m in model.modules()])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
