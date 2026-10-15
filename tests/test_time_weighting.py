import unittest

import torch
import torch.nn.functional as F

from time_weighting import settings, validate_training_options, weights
from train import coupled_flow_matching_loss, flow_matching_loss


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25))
        self.times = []

    def forward(self, x, t):
        self.times.append(t.detach().clone())
        return self.scale * x


class TimeWeightingTests(unittest.TestCase):
    def test_default_is_bitwise_original_mse(self):
        model = RecordingModel()
        source, target = torch.randn(4, 5, 2), torch.randn(4, 5, 2)
        t = torch.rand(4, 1, 1)
        expected = F.mse_loss(model((1 - t) * source + t * target, t), target - source)
        self.assertTrue(torch.equal(flow_matching_loss(model, target, source, t), expected))
        self.assertTrue(torch.equal(flow_matching_loss(model, target, source, t,
                                                       time_weighting={"mode": "uniform"}), expected))

    def test_normalization_and_threshold(self):
        config = {"mode": "late", "threshold": .75, "multiplier": 2}
        t = torch.tensor([0., .749, .75, 1.], dtype=torch.float64)
        self.assertTrue(torch.equal(weights(t, config), t.new_tensor([.8, .8, 1.6, 1.6])))
        self.assertAlmostEqual(.75 * .8 + .25 * 1.6, 1.)
        grid = (torch.arange(10000, dtype=torch.float64) + .5) / 10000
        self.assertAlmostEqual(weights(grid, config).mean().item(), 1.)

    def test_per_cloud_weighting_and_gradients(self):
        model = RecordingModel()
        source = torch.zeros(2, 3, 2)
        target = torch.stack([torch.ones(3, 2), 2 * torch.ones(3, 2)])
        t = torch.tensor([.25, .9]).reshape(2, 1, 1)
        config = {"mode": "late"}
        predicted = model((1 - t) * source + t * target, t)
        expected = ((predicted - target).square().mean((1, 2)) * torch.tensor([.8, 1.6])).mean()
        actual = flow_matching_loss(model, target, source, t, time_weighting=config)
        self.assertTrue(torch.allclose(actual, expected))
        actual.backward()
        self.assertTrue(torch.isfinite(model.scale.grad))

    def test_weight_does_not_change_uniform_time_draws(self):
        target, source = torch.randn(4, 5, 2), torch.randn(4, 5, 2)
        first, second = RecordingModel(), RecordingModel()
        generators = [torch.Generator().manual_seed(12) for _ in range(2)]
        for _ in range(3):
            coupled_flow_matching_loss(first, target, coupling="multiscale_wasserstein",
                                       paired_noise=source, time_generator=generators[0])
            torch.rand(100)  # Global RNG consumption must not alter a dedicated time stream.
            coupled_flow_matching_loss(second, target, coupling="multiscale_overlap",
                                       paired_noise=source, time_generator=generators[1],
                                       time_weighting={"mode": "late"})
        for a, b in zip(first.times, second.times):
            self.assertTrue(torch.equal(a, b))

    def test_settings_reject_invalid_options(self):
        invalid = [False, [], {"mode": "typo"}, {"mode": "uniform", "multiplier": 2},
                   {"mode": "late", "threshold": 0}, {"mode": "late", "threshold": 1},
                   {"mode": "late", "threshold": float("nan")},
                   {"mode": "late", "multiplier": float("inf")},
                   {"mode": "late", "multiplier": .5}, {"mode": "late", "multiplier": True},
                   {"mode": "late", "typo": 2}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                settings(value)
        for seed in (-1, True, 1.2, 2 ** 63, "2"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                validate_training_options({"time_seed": seed})
        validate_training_options({"time_seed": 2, "time_weighting": {"mode": "late"}})


if __name__ == "__main__":
    unittest.main()
