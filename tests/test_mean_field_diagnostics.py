import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

from audit_mean_field import MomentAccumulator, audit, conditional_moments
from coupling import balanced_target_partition, coupled_points
from diagnostic_data import DiagnosticData, tensor_sha256
from experiment import save_training
from model import PointSetTransformer
from train_horse import HorsePointSetTransformer


def tiny_config(dataset, coupling="target_guided"):
    config = {
        "seed": 0, "device": "cpu", "dtype": "float32", "coupling": coupling,
        "num_regions": 2, "data": {"batch_size": 2, "n_points": 8},
        "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                  "dim_feedforward": 16, "dropout": 0.0},
        "training": {"num_steps": 1, "learning_rate": .001, "weight_decay": .01, "log_every": 1},
        "evaluation": {"batch_size": 2, "histogram_bins": 8}, "checkpoint": "fixture.pt",
    }
    if dataset == "checkerboard":
        config["data"]["grid_size"] = 4
    return config


class MeanFieldTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    def test_tg_moments_equal_all_legal_random_bijections(self):
        source = torch.tensor([[[-2., 0], [-1., 0], [1., 0], [2., 0]]], dtype=torch.float64)
        target = torch.tensor([[[-2., -1], [-2., 1], [2., -2], [2., 2]]], dtype=torch.float64)
        config = {"coupling": "target_guided", "num_regions": 2}
        mean, fine, source_labels = conditional_moments(source, target, config)
        _, target_labels, _, _ = balanced_target_partition(target, 2)
        members = [torch.where(source_labels[0] == k)[0] for k in range(2)]
        orders = [list(itertools.permutations(torch.where(target_labels[0] == k)[0].tolist()))
                  for k in range(2)]
        velocities = []
        for choices in itertools.product(*orders):
            permutation = torch.empty(4, dtype=torch.long)
            for src, dst in zip(members, choices):
                permutation[src] = torch.tensor(dst)
            velocities.append(target[0, permutation] - source[0])
        stacked = torch.stack(velocities)
        torch.testing.assert_close(mean[0], stacked.mean(0))
        self.assertAlmostEqual(fine.item(), (stacked - mean).square().mean().item())
        prediction = source[0] * .17
        risk = (stacked - prediction).square().mean()
        self.assertAlmostEqual(risk.item(), (mean[0] - prediction).square().mean().item() + fine.item())
        for seed in range(8):
            paired_source, paired_target = coupled_points(
                source, target, coupling="target_guided", num_regions=2,
                generator=torch.Generator().manual_seed(seed))
            torch.testing.assert_close(paired_source, source)
            self.assertTrue(any(torch.equal(paired_target[0] - source[0], v) for v in velocities))

    def test_optimized_and_independent_and_global_moments(self):
        source = torch.randn(2, 8, 2, dtype=torch.float64)
        target = torch.randn_like(source)
        a = conditional_moments(source, target, {"coupling": "target_guided", "num_regions": 2})
        b = conditional_moments(source, target, {"coupling": "target_guided_exact_optimized", "num_regions": 2})
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right)
        mean, fine, _ = conditional_moments(source, target, {"coupling": "independent"})
        expected = target.mean(1, keepdim=True)
        torch.testing.assert_close(mean, expected - source)
        torch.testing.assert_close(fine, (target - expected).square().mean((1, 2)))
        mean, fine, _ = conditional_moments(source, target, {"coupling": "global_ot"})
        _, paired = coupled_points(source, target, coupling="global_ot")
        torch.testing.assert_close(mean, paired - source)
        self.assertEqual(fine.sum().item(), 0)

    def test_finite_sample_mc_correction_is_unbiased_and_not_clipped(self):
        corrected = []
        for sequence in itertools.product((-1., 1.), repeat=2):
            accumulator = MomentAccumulator(torch.full((2, 2), 2., dtype=torch.float64))
            for value in sequence:
                accumulator.update(torch.full((2, 2), value, dtype=torch.float64), .3)
            scores = accumulator.scores()
            corrected.append(scores["mean_field_mse_corrected"])
            self.assertAlmostEqual(scores["decomposition_residual"], 0)
            self.assertAlmostEqual(scores["expected_fm_mse"],
                                   np.mean([(2 - value)**2 for value in sequence]) + .3)
        self.assertAlmostEqual(np.mean(corrected), 4.)
        accumulator = MomentAccumulator(torch.zeros(2, 2))
        for value in (-1., 1.):
            accumulator.update(torch.full((2, 2), value), 0)
        self.assertLess(accumulator.scores()["mean_field_mse_corrected"], 0)

    def test_shared_draws_ignore_global_rng_and_restore_cpu_rng(self):
        for dataset in ("checkerboard", "horse"):
            config = tiny_config(dataset)
            data = DiagnosticData(config, dataset, "cpu", torch.float32, 41)
            before = torch.get_rng_state().clone()
            source, target = data.bank(3)
            torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
            torch.randn(300)
            np.random.rand(100)
            other = DiagnosticData({**config, "coupling": "minibatch_ot"}, dataset,
                                   "cpu", torch.float32, 41)
            source2, target2 = other.bank(3)
            torch.testing.assert_close(source, source2, rtol=0, atol=0)
            torch.testing.assert_close(target, target2, rtol=0, atol=0)
            self.assertEqual(tensor_sha256(target), tensor_sha256(target2))

    def test_saved_checkpoint_both_datasets_and_microbatch_invariance(self):
        import os
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            try:
                os.chdir(temp)
                for dataset in ("checkerboard", "horse"):
                    config = tiny_config(dataset)
                    model_class = HorsePointSetTransformer if dataset == "horse" else PointSetTransformer
                    model = model_class(**config["model"])
                    run = save_training(model, config, dataset, "fixture.yaml", 0., 0.)
                    results = []
                    for batch_size in (1, 3):
                        directory = audit(run / "config.yaml", dataset, clouds=2, targets=4,
                                          prefixes=[2, 4], batch_size=batch_size, output=temp)
                        self.assertTrue((directory / "mean_field.png").is_file())
                        results.append(json.loads((directory / "mean_field.json").read_text()))
                    for key in ("source_sha256", "target_draws_sha256", "records", "summary_by_targets"):
                        self.assertEqual(results[0][key], results[1][key], key)
                    self.assertTrue(results[0]["training_config_verified"])
                    self.assertIn("t=0 ONLY", results[0]["definitions"]["scope"])
            finally:
                os.chdir(previous)

    def test_unsupported_law_and_bad_inputs_rejected(self):
        source = torch.randn(1, 8, 2)
        for method in ("minibatch_ot", "equivariant_ot_permutation"):
            with self.assertRaisesRegex(ValueError, "different conditional estimator"):
                conditional_moments(source, source, {"coupling": method})
        with self.assertRaises(ValueError):
            MomentAccumulator(torch.zeros(2, 2)).scores()
        for prediction in (torch.empty(0, 2), torch.full((2, 2), float("nan"))):
            with self.assertRaisesRegex(ValueError, "nonempty and finite"):
                MomentAccumulator(prediction)
        with self.assertRaises(ValueError):
            conditional_moments(source[:, :0], source[:, :0], {"coupling": "independent"})
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, "config.yaml")
            path.write_text(yaml.safe_dump(tiny_config("checkerboard", "minibatch_ot")), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "supports"):
                audit(path, "checkerboard")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_sampler_is_isolated(self):
        data = DiagnosticData(tiny_config("checkerboard"), "checkerboard", "cuda", torch.float32, 41)
        before = torch.cuda.get_rng_state().clone()
        _, target = data.bank(2)
        torch.testing.assert_close(before, torch.cuda.get_rng_state(), rtol=0, atol=0)
        _, again = data.bank(2)
        torch.testing.assert_close(target, again, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
