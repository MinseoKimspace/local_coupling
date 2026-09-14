import json
from pathlib import Path
import tempfile
import unittest

import torch

from diagnose import fm_errors, rollout
from summarize_results import collect_results, statistics


class AnalysisTests(unittest.TestCase):
    def test_statistics_are_sample_sd(self):
        self.assertEqual(statistics([1, 2, 3]), {"mean": 2, "std": 1, "n": 3})
        self.assertEqual(statistics([None]), {"mean": None, "std": None, "n": 0})
        self.assertIsNone(statistics([1])["std"])

    def records(self, directory):
        for seed in range(3):
            for nfe in (1, 2):
                record = {"dataset": "horse", "config": {"seed": seed, "coupling": "independent",
                          "data": {"n_points": 256}}, "training_config_verified": True,
                          "checkpoint_sha256": str(seed), "evaluated_at": "2026-01-01",
                          "euler_steps": nfe, "chamfer": seed + nfe}
                (directory / f"{seed}_{nfe}.json").write_text(json.dumps(record), encoding="utf-8")

    def test_seed_grouping_deduplication_and_missing_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.records(directory)
            duplicate = json.loads((directory / "0_1.json").read_text())
            duplicate["evaluated_at"] = "2026-02-01"
            duplicate["chamfer"] = 4
            (directory / "duplicate.json").write_text(json.dumps(duplicate))
            groups, skipped = collect_results(directory, [0, 1, 2])
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]["rows"][0]["metrics"]["chamfer"], statistics([4, 2, 3]))
            self.assertEqual(len(skipped), 1)
            (directory / "2_2.json").unlink()
            groups, _ = collect_results(directory, [0, 1, 2])
            self.assertEqual(len(groups[0]["rows"]), 1)

    def test_retrainings_not_cherry_picked(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.records(directory)
            record = json.loads((directory / "0_1.json").read_text())
            record["checkpoint_sha256"] = "another-training"
            (directory / "retrained.json").write_text(json.dumps(record))
            groups, skipped = collect_results(directory, [0, 1, 2])
            self.assertEqual(groups, [])
            self.assertIn("multiple checkpoints", skipped[0]["reason"])

    def test_different_settings_never_pooled(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.records(directory)
            path = directory / "2_1.json"
            record = json.loads(path.read_text())
            record["config"]["data"]["n_points"] = 512
            path.write_text(json.dumps(record))
            groups, _ = collect_results(directory, [0, 1, 2])
            self.assertEqual([r["nfe"] for r in groups[0]["rows"]], [2])

    def test_constant_velocity_exact_and_stationary_paths(self):
        class Constant(torch.nn.Module):
            def forward(self, x, t):
                return torch.ones_like(x)
        source = torch.zeros(2, 5, 2, dtype=torch.float64)
        error, energy = fm_errors(Constant(), source, source + 1, source.new_full((2, 1, 1), 0.3))
        torch.testing.assert_close(error, torch.zeros_like(error))
        torch.testing.assert_close(energy, torch.ones_like(energy))
        end, ratio, path = rollout(Constant(), source, 8, True)
        torch.testing.assert_close(end, source + 1)
        torch.testing.assert_close(ratio, torch.ones_like(ratio))
        self.assertEqual(path.shape, (9, 5, 2))
        class Stationary(torch.nn.Module):
            def forward(self, x, t):
                return torch.zeros_like(x)
        self.assertEqual(rollout(Stationary(), source, 4)[1].numel(), 0)

    def test_rotation_has_nonstraight_path_and_euler_converges(self):
        class Rotation(torch.nn.Module):
            def forward(self, x, t):
                return torch.stack([-x[..., 1], x[..., 0]], dim=-1)
        source = torch.tensor([[[1., 0.]]], dtype=torch.float64)
        reference, ratio, _ = rollout(Rotation(), source, 256)
        self.assertGreater(ratio.item(), 1.02)
        rough = (rollout(Rotation(), source, 4)[0] - reference).square().mean()
        fine = (rollout(Rotation(), source, 64)[0] - reference).square().mean()
        self.assertLess(fine, rough)
        # The input noise must never be mutated by a rollout.
        torch.testing.assert_close(source, torch.tensor([[[1., 0.]]], dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
