import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from audit_k import audit, partition_scores
from coupling import balanced_target_partition
from experiment import read_config


ROOT = Path(__file__).resolve().parents[1]


class KAuditTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.points = np.array([[0., 0.], [0., 2.], [4., 0.], [4., 2.]])
        self.labels = np.array([0, 0, 1, 1])

    def test_hand_calculated_scores(self):
        scores = partition_scores(self.points, self.labels)
        for name, value in (("wcss", 4), ("wcss_per_point", 1), ("total_scatter", 20),
                            ("between_scatter", 16), ("normalized_wcss", .2), ("ch", 8), ("db", .5)):
            self.assertAlmostEqual(scores[name], value)
        self.assertEqual(scores["capacities"], [2, 2])

    def test_scale_translation_and_order_invariance_in_3d(self):
        points = np.column_stack((self.points, np.zeros(4)))
        original = partition_scores(points, self.labels)
        transformed = partition_scores((points * 3 + [2, -4, 8])[::-1], self.labels[::-1] + 9)
        for name in ("ch", "db", "normalized_wcss"):
            self.assertAlmostEqual(original[name], transformed[name])
        self.assertAlmostEqual(transformed["wcss"], original["wcss"] * 9)
        self.assertAlmostEqual(transformed["wcss"] + transformed["between_scatter"],
                               transformed["total_scatter"])

    def test_degenerate_scores_are_not_false_optima(self):
        for labels, normalized in ((np.zeros(4, dtype=int), 1), (np.arange(4), 0)):
            scores = partition_scores(self.points, labels)
            self.assertIsNone(scores["ch"])
            self.assertIsNone(scores["db"])
            self.assertEqual(scores["normalized_wcss"], normalized)
        scores = partition_scores(np.zeros((4, 2)), self.labels)
        for name in ("ch", "db", "normalized_wcss"):
            self.assertIsNone(scores[name])
        scores = partition_scores([[-1, 0], [1, 0], [0, -1], [0, 1]], self.labels)
        self.assertEqual(scores["ch"], 0)
        self.assertIsNone(scores["db"])  # Distinct patches with the same centroid.

    def test_invalid_points_and_labels(self):
        for points, labels in (([1, 2], self.labels), ([], []),
                               (self.points, self.labels[:-1]),
                               (self.points, self.labels.astype(float)),
                               (self.points * np.nan, self.labels)):
            with self.subTest(points=points), self.assertRaises(ValueError):
                partition_scores(points, labels)

    def test_scores_against_sklearn_if_available(self):
        # Optional independent reference; sklearn is NOT a project dependency.
        try:
            from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
        except ImportError:
            self.skipTest("Optional sklearn reference unavailable")
        rng = np.random.default_rng(17)
        for d in (2, 3):
            points = rng.normal(size=(32, d))
            labels = np.arange(32) % 4
            scores = partition_scores(points, labels)
            self.assertAlmostEqual(scores["ch"], calinski_harabasz_score(points, labels), places=11)
            self.assertAlmostEqual(scores["db"], davies_bouldin_score(points, labels), places=11)

    def test_new_configs_change_only_k_and_checkpoint(self):
        for dataset, baseline in (
                ("checkerboard", "checkerboard_experiments/target_guided.yaml"),
                ("horse", "horse_experiments/horse_target_guided_k8_n256_seed0.yaml")):
            original = read_config(ROOT / baseline)
            for k in (4, 32):
                prefix = "horse_" if dataset == "horse" else ""
                path = ROOT / f"{dataset}_experiments/{prefix}target_guided_k{k}_n256_seed0.yaml"
                config = read_config(path)
                self.assertEqual(config["num_regions"], k)
                self.assertEqual(config["seed"], 0)
                self.assertEqual(config["checkpoint"], f"{prefix}target_guided_k{k}_n256_seed0.pt")
                excluded = ("num_regions", "checkpoint")
                self.assertEqual({key: value for key, value in original.items() if key not in excluded},
                                 {key: value for key, value in config.items() if key not in excluded})

    def test_new_configs_training_and_evaluation_smoke(self):
        import eval as eval_checkerboard
        import eval_horse
        import train
        import train_horse
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            try:
                os.chdir(tmp)
                for dataset in ("checkerboard", "horse"):
                    prefix = "horse_" if dataset == "horse" else ""
                    trainer = train if dataset == "checkerboard" else train_horse
                    evaluator = eval_checkerboard if dataset == "checkerboard" else eval_horse
                    for k in (4, 32):
                        path = ROOT / f"{dataset}_experiments/{prefix}target_guided_k{k}_n256_seed0.yaml"
                        config = read_config(path)
                        config["device"] = "cpu"
                        config["data"].update(batch_size=2, n_points=64)
                        config["evaluation"]["batch_size"] = 2
                        config["model"].update(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
                        config["training"]["num_steps"] = 1
                        local = Path("config.yaml")
                        local.write_text(yaml.safe_dump(config), encoding="utf-8")
                        run = trainer.main(local)
                        picture = evaluator.main(run / "config.yaml", 2)
                        self.assertTrue(picture.is_file())
                        record = json.loads(picture.with_suffix(".json").read_text())
                        self.assertEqual(record["config"]["num_regions"], k)
                        self.assertEqual(record["config"]["seed"], 0)
                        self.assertTrue(record["training_config_verified"])
            finally:
                os.chdir(previous)

    def test_both_datasets_shared_clouds_baseline_partition_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for dataset, config_path in (
                    ("checkerboard", "checkerboard_experiments/target_guided.yaml"),
                    ("horse", "horse_experiments/horse_target_guided_k8_n256_seed0.yaml")):
                calls = []
                def capture(target, k, **kwargs):
                    calls.append((target.clone(), k, kwargs))
                    return balanced_target_partition(target, k, **kwargs)
                with patch("audit_k.balanced_target_partition", side_effect=capture):
                    directory = audit(ROOT / config_path, dataset, clouds=2, device="cpu", output=tmp)
                payload = json.loads((directory / "k_audit.json").read_text())
                self.assertEqual(payload["candidate_ks"], [4, 8, 16, 32])
                self.assertEqual(len(calls), 8)
                self.assertGreater((directory / "scores.png").stat().st_size, 1000)
                for cloud in range(2):
                    rows = payload["records"][cloud]["scores"]
                    for j, k in enumerate(payload["candidate_ks"]):
                        target, actual_k, kwargs = calls[cloud * 4 + j]
                        torch.testing.assert_close(target, calls[cloud * 4][0], rtol=0, atol=0)
                        self.assertEqual((actual_k, kwargs), (k, {"solver": "exact"}))
                        _, labels, _, _ = balanced_target_partition(target, k)
                        expected = partition_scores(target[0].numpy(), labels[0].numpy())
                        self.assertEqual(rows[j]["capacities"], [256 // k] * k)
                        for name in ("ch", "db", "normalized_wcss"):
                            self.assertAlmostEqual(rows[j][name], expected[name])
                        self.assertAlmostEqual(rows[j]["wcss"] + rows[j]["between_scatter"],
                                               rows[j]["total_scatter"])
                for metric, select in (("ch", max), ("db", min)):
                    expected = select(payload["summary"], key=lambda row: row[metric]["mean"])["k"]
                    self.assertEqual(payload["selected_k"][metric], expected)
                repeated = audit(ROOT / config_path, dataset, ks=(32, 8, 4, 16), clouds=2,
                                 device="cpu", output=tmp)
                self.assertNotEqual(directory, repeated)
                other = json.loads((repeated / "k_audit.json").read_text())
                self.assertEqual(payload["records"], other["records"])
                self.assertEqual(payload["summary"], other["summary"])

    def test_invalid_audit_inputs_fail_before_sampling(self):
        path = ROOT / "checkerboard_experiments/target_guided.yaml"
        for options in ({"clouds": 0}, {"ks": ()}, {"ks": (1,)}, {"ks": (256,)},
                        {"ks": (4.5,)}, {"ks": (True,)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                audit(path, "checkerboard", **options)
        with tempfile.TemporaryDirectory() as tmp:
            config = read_config(path)
            config["coupling"] = "geometry_aware_ot"
            other = Path(tmp, "config.yaml")
            other.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact TG"):
                audit(other, "checkerboard")


if __name__ == "__main__":
    unittest.main()
