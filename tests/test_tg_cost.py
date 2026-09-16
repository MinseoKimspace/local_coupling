import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

import torch
import yaml

import coupling
import eval as eval_checkerboard
import eval_horse
import train
import train_horse


VARIANTS = ("target_guided_exact_optimized", "target_guided_source_greedy", "target_guided_source_sinkhorn")


class TGCostTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(21)
        self.source = torch.randn(3, 17, 3)
        self.target = torch.rand(3, 17, 3)

    def run_coupling(self, method, source=None, target=None, k=4):
        source = self.source if source is None else source
        target = self.target if target is None else target
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return coupling.coupling_permutation(source, target, coupling=method, num_regions=k,
                generator=torch.Generator(device=source.device).manual_seed(42))

    def test_optimized_exact_is_identical_to_baseline(self):
        for source, target in ((self.source, self.target), (self.source, torch.zeros_like(self.target))):
            for k in (1, 4, 17):
                with self.subTest(k=k):
                    expected = self.run_coupling("target_guided", source, target, k)
                    actual = self.run_coupling(VARIANTS[0], source, target, k)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_only_source_solver_changes(self):
        original = coupling.assign_regions
        for method, solver in zip(VARIANTS[1:], ("greedy", "sinkhorn")):
            calls = []
            def capture(*args, **kwargs):
                result = original(*args, **kwargs)
                calls.append((kwargs["solver"], result.clone(), args[2].clone()))
                return result
            with patch.object(coupling, "assign_regions", side_effect=capture):
                self.run_coupling("target_guided")
                self.run_coupling(method)
            self.assertEqual([c[0] for c in calls], ["exact", "exact", "exact", solver])
            torch.testing.assert_close(calls[0][1], calls[2][1], rtol=0, atol=0)
            torch.testing.assert_close(calls[1][2], calls[3][2], rtol=0, atol=0)

    def test_capacity_and_bijection(self):
        capacities = torch.tensor([[5, 4, 4, 4], [0, 8, 0, 9], [17, 0, 0, 0]])
        centers = torch.rand(3, 4, 3)
        for solver in ("exact_batched", "greedy", "sinkhorn"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = coupling.assign_regions(self.source, centers, capacities, solver=solver)
            for labels_i, counts in zip(labels, capacities):
                torch.testing.assert_close(torch.bincount(labels_i, minlength=4), counts)
        source, target = self.source.clone(), self.target.clone()
        for method in VARIANTS:
            for k in (1, 4, 17):
                permutation = self.run_coupling(method, k=k)
                torch.testing.assert_close(permutation.sort(1).values, torch.arange(17).expand(3, -1))
        torch.testing.assert_close(self.source, source)
        torch.testing.assert_close(self.target, target)

    def test_sinkhorn_called_only_for_source(self):
        with patch.object(coupling.ot, "emd", wraps=coupling.ot.emd) as exact, \
                patch.object(coupling.ot, "sinkhorn", wraps=coupling.ot.sinkhorn) as sinkhorn, \
                patch.object(coupling, "pair_within_regions", wraps=coupling.pair_within_regions) as local, \
                warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coupling.coupling_permutation(self.source, self.target, coupling=VARIANTS[2], num_regions=4,
                                         sinkhorn_epsilon=0.23, sinkhorn_iterations=7)
        self.assertEqual(exact.call_count, 3)
        self.assertEqual(sinkhorn.call_count, 3)
        self.assertEqual(local.call_args.kwargs["local"], "random")
        for call in sinkhorn.call_args_list:
            self.assertEqual(call.args[3], 0.23)
            self.assertEqual(call.kwargs["numItermax"], 7)
            self.assertEqual(call.kwargs["method"], "sinkhorn_log")

    def test_configs_and_metadata(self):
        root = Path(__file__).resolve().parents[1]
        for folder, baseline_name in (("checkerboard_experiments", "target_guided.yaml"),
                                      ("horse_experiments", "horse_target_guided_k8_n256_seed0.yaml")):
            baseline = yaml.safe_load((root / folder / baseline_name).read_text())
            for method in VARIANTS:
                filename = f"{method}.yaml" if folder.startswith("checkerboard") else f"horse_{method}_k8_n256_seed0.yaml"
                config = yaml.safe_load((root / folder / filename).read_text())
                for key in ("seed", "data", "model", "training", "evaluation"):
                    self.assertEqual(config[key], baseline[key])
                self.assertEqual(config["coupling"], method)
                self.assertNotEqual(config["checkpoint"], baseline["checkpoint"])
                info = coupling.coupling_info(method)
                self.assertEqual(info["target_partition_solver"], "POT/network_simplex")
                self.assertEqual(info["local_pairing"], "random")
                self.assertEqual(info["exact_solver"], "POT/network_simplex")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda(self):
        source, target = self.source.cuda(), self.target.cuda()
        expected = self.run_coupling("target_guided", source, target)
        actual = self.run_coupling(VARIANTS[0], source, target)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for method in VARIANTS[1:]:
            actual = self.run_coupling(method, source, target)
            torch.testing.assert_close(actual.sort(1).values.cpu(), torch.arange(17).expand(3, -1))

    def test_train_and_eval_both_datasets(self):
        root = Path(__file__).resolve().parents[1]
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            try:
                os.chdir(temp)
                for dataset in ("checkerboard", "horse"):
                    for method in VARIANTS:
                        config = yaml.safe_load((root / "checkerboard_experiments" / f"{method}.yaml").read_text())
                        config["device"] = "cpu"
                        config["data"].update(batch_size=2, n_points=16)
                        config["evaluation"]["batch_size"] = 2
                        config["num_regions"] = 4
                        config["model"].update(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
                        config["training"]["num_steps"] = 1
                        if dataset == "horse":
                            config["data"].pop("grid_size")
                        Path("config.yaml").write_text(yaml.safe_dump(config))
                        trainer = train_horse if dataset == "horse" else train
                        evaluator = eval_horse if dataset == "horse" else eval_checkerboard
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            run = trainer.main("config.yaml")
                        artifact = evaluator.main(run / "config.yaml", 2)
                        self.assertTrue(artifact.is_file())
                        record = json.loads(artifact.with_suffix(".json").read_text())
                        self.assertEqual(record["coupling"], method)
                        self.assertEqual(record["coupling_details"]["target_partition_solver"], "POT/network_simplex")
                        self.assertTrue(record["training_config_verified"])
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
