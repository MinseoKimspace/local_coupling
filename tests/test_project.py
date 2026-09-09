import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import yaml

import coupling
import eval as eval_checkerboard
import eval_horse
import train
import train_horse
from data import checkerboard_centers, sample_checkerboard
from experiment import load_model, save_evaluation, save_training
from metrics import chamfer_distance, checkerboard_metrics, horse_metrics
from model import PointSetTransformer
from sample import integrate_velocity
from train import train_step


class CouplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.source = torch.randn(2, 32, 2)
        self.target = sample_checkerboard(2, 32, "cpu", torch.float32)
        self.centers = checkerboard_centers(4, "cpu", torch.float32)

    def permutation(self, name, k=4, source=None, target=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return coupling.coupling_permutation(
                self.source if source is None else source, self.target if target is None else target,
                coupling=name, num_regions=k, target_centers=self.centers,
                generator=torch.Generator().manual_seed(8),
            )

    def test_all_methods_preserve_points(self):
        original = self.source.clone()
        for name in list(coupling.METHODS) + list(coupling.ALIASES):
            with self.subTest(name=name):
                result = self.permutation(name)
                if name == "independent":
                    self.assertIsNone(result)
                else:
                    torch.testing.assert_close(result.sort(1).values, torch.arange(32).expand(2, -1))
        torch.testing.assert_close(self.source, original)

    def test_exact_rectangular_cost_equals_scipy_slots(self):
        for capacity in ([4, 4, 4, 4], [5, 4, 4, 4], [0, 8, 0, 9]):
            counts = torch.tensor(capacity)
            cost = torch.rand(int(counts.sum()), 4, dtype=torch.float64)
            labels = coupling.exact_assignment(cost, counts)
            slots = np.repeat(cost.numpy(), capacity, axis=1)
            rows, cols = linear_sum_assignment(slots)
            self.assertAlmostEqual(cost[torch.arange(len(labels)), labels].sum().item(), slots[rows, cols].sum(), places=9)
            torch.testing.assert_close(torch.bincount(labels, minlength=4), counts)

    def test_fractional_plan_is_rejected(self):
        with patch.object(coupling.ot, "emd", return_value=(np.full((2, 2), 0.5), {"warning": None})):
            with self.assertRaises(RuntimeError):
                coupling.exact_assignment(torch.zeros(2, 2))

    def test_kn_equals_global(self):
        a = self.permutation("target_guided", 32)
        b = self.permutation("global_ot")
        cost = torch.cdist(self.source, self.target).square()
        torch.testing.assert_close(cost.gather(2, a.unsqueeze(-1)).sum(), cost.gather(2, b.unsqueeze(-1)).sum())

    def test_k1_remainders_and_3d(self):
        source, target = torch.randn(2, 17, 3), torch.rand(2, 17, 3)
        for name in ("target_guided", "target_guided_sinkhorn", "geometry_aware_ot", "geometry_aware_sinkhorn"):
            for k in (1, 4):
                with self.subTest(name=name, k=k):
                    result = self.permutation(name, k, source, target)
                    torch.testing.assert_close(result.sort(1).values, torch.arange(17).expand(2, -1))
        with self.assertRaises(ValueError):
            self.permutation("regional", 4, source, target)

    def test_duplicate_points_and_empty_patches(self):
        target = torch.zeros_like(self.target)
        indices = coupling.farthest_point_sample(target, 4)
        self.assertEqual(indices[0].unique().numel(), 4)
        for name in ("geometry_aware_ot", "geometry_aware_sinkhorn"):
            result = self.permutation(name, target=target)
            torch.testing.assert_close(result.sort(1).values, torch.arange(32).expand(2, -1))

    def test_sinkhorn_options_reach_pot(self):
        with patch.object(coupling.ot, "sinkhorn", wraps=coupling.ot.sinkhorn) as solve, warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coupling.coupling_permutation(self.source, self.target, coupling="target_guided_sinkhorn",
                                         num_regions=4, sinkhorn_epsilon=0.23, sinkhorn_iterations=7)
        self.assertEqual(solve.call_count, 4)
        for call in solve.call_args_list:
            self.assertEqual(call.args[3], 0.23)
            self.assertEqual(call.kwargs["numItermax"], 7)
            self.assertEqual(call.kwargs["method"], "sinkhorn_log")

    def test_invalid_capacities(self):
        for counts in (torch.tensor([1, 0]), torch.tensor([-1, 3]), torch.tensor([0.5, 1.5])):
            with self.assertRaises(ValueError):
                coupling.exact_assignment(torch.zeros(2, 2), counts)

    def test_configured_point_counts(self):
        for n, k in ((256, 8), (256, 32), (1024, 8)):
            source = torch.randn(2, n, 2)
            target = sample_checkerboard(2, n, "cpu", torch.float32)
            result = self.permutation("target_guided", k, source, target)
            torch.testing.assert_close(result.sort(1).values, torch.arange(n).expand(2, -1))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_couplings(self):
        source, target = self.source.cuda(), self.target.cuda()
        for name in ("target_guided", "target_guided_sinkhorn", "global_ot"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = coupling.coupling_permutation(source, target, coupling=name, num_regions=4,
                                                      generator=torch.Generator(device="cuda").manual_seed(0))
            torch.testing.assert_close(result.sort(1).values.cpu(), torch.arange(32).expand(2, -1))


class MetricModelTests(unittest.TestCase):
    def test_known_metrics(self):
        centers = checkerboard_centers(4, "cpu", torch.float32)
        points = torch.cat([centers, torch.full((72, 2), 10.)]).unsqueeze(0)
        leakage, mass, js = checkerboard_metrics(points, 4)
        self.assertAlmostEqual(leakage, 0.9, places=6)
        self.assertEqual(mass, 0.)
        self.assertTrue(0 <= js <= np.log(2))
        leakage, mass, js = checkerboard_metrics(torch.full((1, 10, 2), 10.), 4)
        self.assertEqual(leakage, 1.)
        self.assertTrue(np.isnan(mass))
        self.assertAlmostEqual(js, np.log(2), places=12)

    def test_perfect_histogram(self):
        centers = (torch.arange(64) + .5) / 32 - 1
        x, y = torch.meshgrid(centers, centers, indexing="ij")
        active = (((x + 1) * 2).floor() + ((1 - y) * 2).floor()) % 2 == 0
        points = torch.stack([x[active], y[active]], -1).unsqueeze(0)
        np.testing.assert_allclose(checkerboard_metrics(points, 4), (0, 0, 0), atol=1e-12)

    def test_horse_orientation(self):
        mask = torch.tensor([[1., 0.], [0., 0.]])
        leakage, js = horse_metrics(torch.tensor([[[-.5, .5]]]), mask, histogram_bins=2)
        self.assertEqual(leakage, 0.)
        self.assertAlmostEqual(js, 0., places=12)
        self.assertEqual(horse_metrics(torch.tensor([[[.5, -.5]]]), mask)[0], 1.)

    def test_chamfer_squared_sum(self):
        x = torch.tensor([[[0., 0.], [2., 0.]]])
        y = torch.tensor([[[1., 0.], [3., 0.]]])
        self.assertEqual(chamfer_distance(x, y).item(), 2.)

    def test_equivariance_and_training(self):
        for method in ("independent", "target_guided"):
            torch.manual_seed(0)
            model = PointSetTransformer(d_model=16, nhead=4, num_layers=2, dim_feedforward=32).eval()
            x, t = torch.randn(2, 8, 2), torch.rand(2, 1, 1)
            p = torch.randperm(8)
            with torch.no_grad():
                torch.testing.assert_close(model(x[:, p], t), model(x, t)[:, p], atol=1e-6, rtol=1e-5)
            optimizer = torch.optim.AdamW(model.parameters())
            loss = train_step(model, optimizer, x, coupling=method, num_regions=4)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(integrate_velocity(model, x, num_steps=2)).all())


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous = Path.cwd()
        os.chdir(self.temp.name)
        self.config = {"seed": 0, "device": "cpu", "dtype": "float32", "coupling": "target_guided",
                       "num_regions": 4, "data": {"batch_size": 2, "n_points": 16, "grid_size": 4},
                       "model": {"point_dim": 2, "d_model": 16, "nhead": 4, "num_layers": 2,
                                 "dim_feedforward": 32, "dropout": 0.0},
                       "training": {"num_steps": 1, "learning_rate": .001, "weight_decay": .01, "log_every": 1},
                       "checkpoint": "legacy.pt"}
        self.model = PointSetTransformer(**self.config["model"])
        Path("original.yaml").write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def tearDown(self):
        os.chdir(self.previous)
        self.temp.cleanup()

    def save(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return save_training(self.model, self.config, "checkerboard", "original.yaml", 1.23, .1)

    def test_preservation_metadata_and_mismatch(self):
        torch.save(self.model.state_dict(), "legacy.pt")
        original = Path("legacy.pt").read_bytes()
        first, second = self.save(), self.save()
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, Path("runs/checkerboard"))
        self.assertTrue((first / self.config["checkpoint"]).is_file())
        self.assertEqual(Path("legacy.pt").read_bytes(), original)
        _, config, _, metadata = load_model(first / "config.yaml", PointSetTransformer, "checkerboard")
        self.assertTrue(metadata["training_config_verified"])
        self.assertEqual(metadata["coupling_details"]["exact_solver"], "POT/network_simplex")
        for key in ("num_regions", "model"):
            changed = copy.deepcopy(config)
            if key == "num_regions":
                changed[key] = 8
            else:
                changed[key]["nhead"] = 8
            path = first / "changed.yaml"
            path.write_text(yaml.safe_dump(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, key):
                load_model(path, PointSetTransformer, "checkerboard")

    def test_legacy_warning_and_json(self):
        torch.save(self.model.state_dict(), "legacy.pt")
        with self.assertWarnsRegex(UserWarning, "UNVERIFIED"):
            _, config, checkpoint, metadata = load_model("original.yaml", PointSetTransformer, "checkerboard")
        self.assertFalse(metadata["training_config_verified"])
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(2):
                save_evaluation("original.yaml", config, checkpoint, metadata, "checkerboard", 100, .1,
                                {"cell_mass_error": float("nan"), "leakage": 1.}, render=False)
        records = list(Path("eval_results/checkerboard").glob("*.json"))
        self.assertEqual(len(records), 2)
        result = json.loads(records[0].read_text())
        self.assertIsNone(result["cell_mass_error"])
        self.assertEqual(result["coupling_details"]["implementation"], "legacy_unverified")

    def test_training_and_evaluation_entrypoints(self):
        with contextlib.redirect_stdout(io.StringIO()):
            run = train.main("original.yaml")
            picture = eval_checkerboard.main(run / "config.yaml")
            self.assertTrue(picture.is_file())
            self.assertEqual(picture.parent, Path("eval_results/checkerboard"))
            record = json.loads(picture.with_suffix(".json").read_text())
            self.assertTrue(record["training_config_verified"])
            self.assertEqual(record["total_points"], 128)
            config = copy.deepcopy(self.config)
            config["data"].pop("grid_size")
            Path("horse.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            run = train_horse.main("horse.yaml")
            self.assertEqual(run.parent, Path("runs/horse"))
            picture = eval_horse.main(run / "config.yaml", 2)
            self.assertTrue(picture.is_file())
            self.assertEqual(picture.parent, Path("eval_results/horse"))
            record = json.loads(picture.with_suffix(".json").read_text())
            self.assertEqual(record["dataset"], "horse")
            self.assertEqual(record["euler_steps"], 2)


if __name__ == "__main__":
    unittest.main()
