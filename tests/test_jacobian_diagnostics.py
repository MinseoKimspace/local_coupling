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

import torch
import yaml

from audit_jacobian import audit, cloud_jacobian, main, rollout_snapshots
from model import PointSetTransformer
from train_horse import HorsePointSetTransformer
import tg_cache
import train
import train_horse


class DenseVelocity(torch.nn.Module):
    def __init__(self, matrix):
        super().__init__()
        self.matrix = torch.nn.Parameter(matrix.clone())

    def forward(self, x, t):
        return (x.flatten(1) @ self.matrix.T).reshape_as(x) + t


class JacobianTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_exact_includes_cross_point_blocks_and_holds_time_fixed(self):
        matrix = torch.tensor([[1., 2., 3., 4.], [0., -1., 2., 3.],
                               [4., 0., 1., 2.], [1., 3., 0., -2.]], dtype=torch.float64)
        model = DenseVelocity(matrix)
        cloud = torch.randn(2, 2, dtype=torch.float64)
        for time in (0., .5, 1.):
            result = cloud_jacobian(model, cloud, time, method="exact")
            self.assertEqual(result["input_dimension"], 4)
            self.assertAlmostEqual(result["frobenius_squared"], matrix.square().sum().item())
            self.assertAlmostEqual(result["normalized_frobenius"], matrix.norm().item() / 2)
            self.assertEqual(result["mc_se_squared"], 0.)

    def test_hutchinson_is_exact_for_scaled_identity_and_preserves_rng(self):
        model = DenseVelocity(torch.eye(6, dtype=torch.float64) * 3)
        cloud = torch.randn(3, 2, dtype=torch.float64)
        state = torch.get_rng_state().clone()
        for probes in (1, 8):
            result = cloud_jacobian(model, cloud, .25, probes=probes)
            self.assertAlmostEqual(result["frobenius_squared"], 54.)
            self.assertAlmostEqual(result["normalized_frobenius"], 3.)
            self.assertEqual(result["mc_se_squared"], None if probes == 1 else 0.)
        torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)

    def test_dense_mc_estimate_matches_exact_with_reported_uncertainty(self):
        matrix = torch.tensor([[1., 2., 3., 4.], [0., -1., 2., 3.],
                               [4., 0., 1., 2.], [1., 3., 0., -2.]], dtype=torch.float64)
        model = DenseVelocity(matrix)
        cloud = torch.zeros(2, 2, dtype=torch.float64)
        value = cloud_jacobian(model, cloud, .5, probes=1024,
                               generator=torch.Generator().manual_seed(9))
        error = abs(value["frobenius_squared"] - matrix.square().sum().item())
        self.assertLess(error, 5 * value["mc_se_squared"])

    def test_constant_fields_have_zero_spatial_jacobian(self):
        class Constant(torch.nn.Module):
            def __init__(self, parameter):
                super().__init__()
                self.offset = torch.nn.Parameter(torch.tensor(2.)) if parameter else None

            def forward(self, x, t):
                return torch.ones_like(x) * (self.offset if self.offset is not None else 2.)
        for parameter in (True, False):
            for method in ("exact", "hutchinson"):
                result = cloud_jacobian(Constant(parameter), torch.zeros(2, 2), .5, method=method)
                self.assertEqual(result["frobenius_squared"], 0.)

    def test_model_modes_gradients_inputs_and_backend_are_restored(self):
        model = PointSetTransformer(point_dim=2, d_model=8, nhead=2, num_layers=1,
                                    dim_feedforward=16, dropout=.2)
        model.train()
        model.encoder.eval()  # Preserve mixed submodule modes, too.
        modes = [module.training for module in model.modules()]
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, .123)
        weights = copy.deepcopy(model.state_dict())
        gradients = [parameter.grad.clone() for parameter in model.parameters()]
        cloud = torch.randn(4, 2)
        original = cloud.clone()
        fastpath = torch.backends.mha.get_fastpath_enabled()
        with torch.inference_mode():
            result = cloud_jacobian(model, cloud, .5, probes=2)
        self.assertTrue(math.isfinite(result["frobenius"]))
        self.assertEqual(fastpath, torch.backends.mha.get_fastpath_enabled())
        self.assertEqual(modes, [module.training for module in model.modules()])
        for name, weight in model.state_dict().items():
            torch.testing.assert_close(weight, weights[name], rtol=0, atol=0)
        for parameter, gradient in zip(model.parameters(), gradients):
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
        torch.testing.assert_close(cloud, original, rtol=0, atol=0)
        self.assertIsNone(cloud.grad)

    def test_project_transformers_match_independent_full_jacobian(self):
        options = dict(point_dim=2, d_model=8, nhead=2, num_layers=1,
                       dim_feedforward=16, dropout=0.)
        for constructor in (PointSetTransformer, HorsePointSetTransformer):
            model = constructor(**options).double().eval()
            cloud = torch.randn(3, 2, dtype=torch.float64)
            value = cloud_jacobian(model, cloud, .25, method="exact")
            time = cloud.new_full((1, 1, 1), .25)
            full = torch.autograd.functional.jacobian(lambda x: model(x.unsqueeze(0), time)[0], cloud)
            self.assertAlmostEqual(value["frobenius_squared"], full.square().sum().item(), places=10)

    def test_invalid_arguments_and_exact_cost_guard_restore_modes(self):
        model = DenseVelocity(torch.eye(4))
        for options in ({"probes": 0}, {"probes": True}, {"method": "bad"},
                        {"method": "exact", "max_exact_dim": 3}):
            with self.assertRaises(ValueError):
                cloud_jacobian(model, torch.zeros(2, 2), .5, **options)
        for time in (-1., 2., float("nan"), True):
            with self.assertRaises(ValueError):
                cloud_jacobian(model, torch.zeros(2, 2), time)
        with self.assertRaises(ValueError):
            cloud_jacobian(model, torch.zeros(1, 2, 2), .5)
        model.train()
        fastpath = torch.backends.mha.get_fastpath_enabled()
        with torch.no_grad():
            model.matrix.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            cloud_jacobian(model, torch.zeros(2, 2), .5)
        self.assertTrue(model.training)
        self.assertEqual(torch.backends.mha.get_fastpath_enabled(), fastpath)

    def test_fixed_euler_grid_for_constant_field(self):
        class Constant(torch.nn.Module):
            def forward(self, x, t):
                return torch.ones_like(x) * 2
        model = Constant().train()
        source = torch.randn(2, 2)
        states = rollout_snapshots(model, source, (0., .5, 1.), 4)
        for time, state in states.items():
            torch.testing.assert_close(state, source + 2 * time)
        self.assertTrue(model.training)
        with self.assertRaisesRegex(ValueError, "Euler grid"):
            rollout_snapshots(model, source, (.3,), 4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_attention_backward_for_both_project_models(self):
        options = dict(point_dim=2, d_model=8, nhead=2, num_layers=1,
                       dim_feedforward=16, dropout=0.)
        for constructor in (PointSetTransformer, HorsePointSetTransformer):
            model = constructor(**options).cuda().eval()
            result = cloud_jacobian(model, torch.randn(4, 2, device="cuda"), .5, probes=3)
            self.assertTrue(math.isfinite(result["frobenius"]))
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_actual_n256_backbones_on_cuda(self):
        root = Path(__file__).resolve().parents[1]
        for name, constructor in (
                ("checkerboard_experiments/target_guided_cached_k8_n256_seed0.yaml", PointSetTransformer),
                ("horse_experiments/horse_target_guided_cached_k8_n256_seed0.yaml", HorsePointSetTransformer)):
            config = yaml.safe_load((root / name).read_text(encoding="utf-8"))
            model = constructor(**config["model"]).cuda().eval()
            cloud = torch.randn(config["data"]["n_points"], 2, device="cuda")
            result = cloud_jacobian(model, cloud, .5, probes=2)
            self.assertEqual(result["input_dimension"], 512)
            self.assertTrue(math.isfinite(result["frobenius"]))
            self.assertAlmostEqual(result["normalized_frobenius"] ** 2 * 512,
                                   result["frobenius_squared"], places=8)


class JacobianIntegrationTests(unittest.TestCase):
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
    def config(dataset):
        config = {"seed": 0, "device": "cpu", "dtype": "float32", "coupling": "target_guided_cached",
                  "num_regions": 2,
                  "tg_cache": {"path": dataset + "_cache", "sampling": "bank", "num_clouds": 4,
                               "seed": 5, "prepare_batch_size": 2, "num_workers": 0},
                  "data": {"batch_size": 2, "n_points": 8},
                  "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                            "dim_feedforward": 16, "dropout": 0.},
                  "training": {"num_steps": 1, "learning_rate": .001, "weight_decay": .01, "log_every": 1},
                  "evaluation": {"batch_size": 2, "histogram_bins": 8}, "checkpoint": dataset + ".pt"}
        if dataset == "checkerboard":
            config["data"]["grid_size"] = 4
        return config

    def test_both_datasets_full_report_and_probe_independent_sample_banks(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for dataset, trainer in (("horse", train_horse.main), ("checkerboard", train.main)):
                config = self.config(dataset)
                path = Path(dataset + ".yaml")
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                tg_cache.prepare(config, dataset)
                run = trainer(path)
                checkpoint = run / config["checkpoint"]
                before = checkpoint.read_bytes(), (run / "config.yaml").read_bytes()
                directory = audit(run / "config.yaml", dataset, scope="both", clouds=2,
                                  times=(0., .5, 1.), probes=3, rollout_steps=2)
                self.assertTrue((directory / "jacobian.png").is_file())
                result = json.loads((directory / "jacobian.json").read_text())
                self.assertEqual(result["jacobian_shape"], [16, 16])
                self.assertEqual(len(result["summary"]), 6)
                self.assertEqual(len(result["per_cloud"]), 12)
                self.assertTrue(all(row["frobenius"]["n"] == 2 for row in result["summary"]))
                self.assertEqual(before, (checkpoint.read_bytes(), (run / "config.yaml").read_bytes()))
                with patch("audit_jacobian.render"):
                    repeat = audit(run / "config.yaml", dataset, scope="both", clouds=2,
                                   times=(0., .5, 1.), probes=5, rollout_steps=2)
                other = json.loads((repeat / "jacobian.json").read_text())
                for scope in ("rollout", "coupling"):
                    self.assertEqual(result["banks"][scope]["source_sha256"], other["banks"][scope]["source_sha256"])
                self.assertEqual([row["state_sha256"] for row in result["per_cloud"]],
                                 [row["state_sha256"] for row in other["per_cloud"]])
                # Fresh rollout measurements do not load a training cache.
                with patch.object(tg_cache, "load_cache", side_effect=AssertionError("cache used")), \
                        patch("audit_jacobian.render"):
                    output = main([str(run / "config.yaml"), "--dataset", dataset, "--clouds", "1",
                                   "--times", "0", "--method", "exact"])
                exact = json.loads((output / "jacobian.json").read_text())
                self.assertIsNone(exact["probes"])
                self.assertEqual(exact["scope"], "rollout")
                self.assertEqual(exact["summary"][0]["mc_se_squared"]["mean"], 0.)

    def test_bad_audit_options_fail_before_checkpoint_loading(self):
        for options in ({"clouds": 0}, {"seed": -1}, {"times": []}, {"times": [0., 0.]},
                        {"times": [float("nan")]}, {"scope": "bad"}, {"method": "bad"},
                        {"times": [.1], "rollout_steps": 128}):
            with self.assertRaises(ValueError):
                audit("missing.yaml", "horse", **options)


if __name__ == "__main__":
    unittest.main()
