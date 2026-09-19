import contextlib
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import yaml

import coupling
import eval as eval_checkerboard
import eval_horse
import train
import train_horse
from model import PointSetTransformer


METHOD = "target_guided_mahalanobis"


class MahalanobisTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)

    def reference_case(self, dim=3, device="cpu"):
        source = torch.randn(2, 17, dim, device=device)
        target = torch.randn(2, 17, dim, device=device)
        labels = torch.tensor([[0] * 2 + [1] * 3 + [2] * 5 + [3] * 7], device=device).expand(2, -1)
        centers, counts = coupling.region_centroids(target, labels, 4)
        return source, target, labels, centers, counts

    def check_reference(self, dim, device="cpu"):
        source, target, labels, centers, counts = self.reference_case(dim, device)
        ridge = 0.037
        costs = coupling.mahalanobis_cost(source, target, labels, centers, counts, ridge=ridge)
        x, y, c = (value.cpu().double().numpy() for value in (source, target, centers))
        expected = np.empty((2, 17, 4))
        for b in range(2):
            for k in range(4):
                centered = y[b, labels[b].cpu().numpy() == k] - c[b, k]
                covariance = centered.T @ centered / len(centered)
                delta = x[b] - c[b, k]
                expected[b, :, k] = np.einsum("ni,ij,nj->n", delta,
                    np.linalg.inv(covariance + ridge * np.eye(dim)), delta)
        self.assertEqual(costs.dtype, torch.float64)
        self.assertEqual(costs.device, source.device)
        np.testing.assert_allclose(costs.cpu().numpy(), expected, atol=1e-9, rtol=1e-9)
        assigned = coupling.exact_assignment_batched(costs, counts)
        for b in range(2):
            capacity = counts[b].cpu().numpy()
            slots = np.repeat(expected[b], capacity, axis=1)
            rows, columns = linear_sum_assignment(slots)
            actual = costs[b, torch.arange(17, device=device), assigned[b]].sum().item()
            self.assertAlmostEqual(actual, slots[rows, columns].sum(), places=8)

    def test_population_covariance_and_exact_cost(self):
        for dim in (2, 3):
            with self.subTest(dim=dim):
                self.check_reference(dim)

    def test_singular_covariance_and_singletons(self):
        source = torch.randn(1, 4, 3)
        target = torch.tensor([[[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.], [3., 2., 1.]]])
        labels = torch.tensor([[0, 0, 0, 1]])
        centers, counts = coupling.region_centroids(target, labels, 2)
        ridge = 0.02
        costs = coupling.mahalanobis_cost(source, target, labels, centers, counts, ridge=ridge)
        self.assertTrue(torch.isfinite(costs).all())
        expected_singleton = (source.double() - target[:, 3:4].double()).square().sum(-1) / ridge
        torch.testing.assert_close(costs[:, :, 1], expected_singleton)
        for invalid in (0., -1., float("nan"), float("inf")):
            with self.subTest(ridge=invalid), self.assertRaises(ValueError):
                coupling.mahalanobis_cost(source, target, labels, centers, counts, ridge=invalid)

    def test_partition_capacities_local_pairing_and_immutability(self):
        source, target = torch.randn(2, 17, 3), torch.rand(2, 17, 3)
        original_source, original_target = source.clone(), target.clone()
        for k in (1, 4, 17):
            captured = []
            original = coupling.pair_within_regions

            def record(*args, **kwargs):
                captured.append((args[2].clone(), args[3].clone(), kwargs["local"]))
                return original(*args, **kwargs)

            with patch.object(coupling, "pair_within_regions", side_effect=record):
                for method in ("target_guided_exact_optimized", METHOD):
                    permutation = coupling.coupling_permutation(source, target, coupling=method,
                        num_regions=k, generator=torch.Generator().manual_seed(11))
                    torch.testing.assert_close(permutation.sort(1).values, torch.arange(17).expand(2, -1))
            torch.testing.assert_close(captured[0][1], captured[1][1], atol=0, rtol=0)
            self.assertEqual(captured[1][2], "random")
            for source_labels, target_labels in zip(captured[1][0], captured[1][1]):
                torch.testing.assert_close(torch.bincount(source_labels, minlength=k),
                                           torch.bincount(target_labels, minlength=k))
        torch.testing.assert_close(source, original_source, atol=0, rtol=0)
        torch.testing.assert_close(target, original_target, atol=0, rtol=0)

    def test_configs_and_metadata(self):
        root = Path(__file__).resolve().parents[1]
        for folder, prefix, suffix in (("checkerboard_experiments", "", ""),
                                      ("horse_experiments", "horse_", "_k8_n256_seed0")):
            baseline = yaml.safe_load((root / folder / f"{prefix}target_guided_exact_optimized{suffix}.yaml").read_text())
            config = yaml.safe_load((root / folder / f"{prefix}{METHOD}{suffix}.yaml").read_text())
            self.assertEqual(config["coupling"], METHOD)
            self.assertGreater(config["mahalanobis_ridge"], 0)
            self.assertNotEqual(config["checkpoint"], baseline["checkpoint"])
            for key in ("seed", "data", "model", "training", "evaluation", "num_regions"):
                self.assertEqual(config[key], baseline[key])
        info = coupling.coupling_info(METHOD)
        self.assertEqual(info["target_partition"], "balanced")
        self.assertEqual(info["target_partition_solver"], "POT/network_simplex")
        self.assertEqual(info["source_assignment"], "exact")
        self.assertEqual(info["local_pairing"], "random")
        self.assertEqual(info["exact_solver"], "POT/network_simplex")
        self.assertIn("mahalanobis", info["source_cost"])

    def test_nondefault_ridge_reaches_training_coupling(self):
        model = PointSetTransformer(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
        data = torch.rand(2, 12, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        helper = coupling.mahalanobis_cost
        with patch.object(coupling, "mahalanobis_cost", wraps=helper) as cost:
            loss = train.train_step(model, optimizer, data, coupling=METHOD, num_regions=3,
                                    mahalanobis_ridge=0.07)
        self.assertTrue(torch.isfinite(loss))
        bound = inspect.signature(helper).bind(*cost.call_args.args, **cost.call_args.kwargs)
        self.assertEqual(bound.arguments["ridge"], 0.07)
        config = {"seed": 0, "coupling": METHOD, "num_regions": 3, "mahalanobis_ridge": 0.09,
                  "training": {"num_steps": 1, "log_every": 1, "learning_rate": 0.001, "weight_decay": 0.01}}
        with patch.object(train, "train_step", return_value=torch.tensor(0.1)) as step, \
                patch.object(train, "save_training"), contextlib.redirect_stdout(io.StringIO()):
            train.train_model(model, config, lambda: data, dataset="checkerboard", config_path="unused.yaml")
        self.assertEqual(step.call_args.kwargs["mahalanobis_ridge"], 0.09)

    def test_train_eval_artifacts_both_datasets(self):
        root = Path(__file__).resolve().parents[1]
        previous = Path.cwd()
        helper = coupling.mahalanobis_cost
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            try:
                os.chdir(temporary)
                for dataset in ("checkerboard", "horse"):
                    filename = (f"horse_{METHOD}_k8_n256_seed0.yaml" if dataset == "horse"
                                else f"{METHOD}.yaml")
                    config = yaml.safe_load((root / f"{dataset}_experiments" / filename).read_text())
                    config.update(device="cpu", num_regions=3, mahalanobis_ridge=0.043)
                    config["data"].update(batch_size=2, n_points=17)
                    config["evaluation"]["batch_size"] = 2
                    config["model"].update(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
                    config["training"]["num_steps"] = 1
                    Path("config.yaml").write_text(yaml.safe_dump(config))
                    trainer = train_horse if dataset == "horse" else train
                    evaluator = eval_horse if dataset == "horse" else eval_checkerboard
                    with patch.object(coupling, "mahalanobis_cost", wraps=helper) as cost:
                        run = trainer.main("config.yaml")
                    bound = inspect.signature(helper).bind(*cost.call_args.args, **cost.call_args.kwargs)
                    self.assertEqual(bound.arguments["ridge"], config["mahalanobis_ridge"])
                    artifact = evaluator.main(run / "config.yaml", 2)
                    self.assertTrue(artifact.is_file())
                    record = json.loads(artifact.with_suffix(".json").read_text())
                    self.assertTrue(record["training_config_verified"])
                    self.assertEqual(record["config"]["mahalanobis_ridge"], config["mahalanobis_ridge"])
                    self.assertEqual(record["coupling_details"], coupling.coupling_info(METHOD))
                    snapshot = yaml.safe_load((run / "config.yaml").read_text())
                    self.assertEqual(snapshot["mahalanobis_ridge"], config["mahalanobis_ridge"])
            finally:
                os.chdir(previous)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_cost_and_permutation(self):
        self.check_reference(3, "cuda")
        source, target, _, _, _ = self.reference_case(device="cuda")
        permutation = coupling.coupling_permutation(source, target, coupling=METHOD, num_regions=4,
            generator=torch.Generator(device="cuda").manual_seed(5))
        torch.testing.assert_close(permutation.sort(1).values.cpu(), torch.arange(17).expand(2, -1))


if __name__ == "__main__":
    unittest.main()
