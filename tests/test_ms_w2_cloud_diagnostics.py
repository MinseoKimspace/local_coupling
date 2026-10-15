import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from audit_jacobian import audit as audit_jacobian
from audit_ms_w2_clouds import audit, correlations, jacobian_features, main, quartiles
from diagnostic_data import tensor_sha256
from experiment import save_training
from model import PointSetTransformer
from train_horse import HorsePointSetTransformer
import test_multiscale_wasserstein as ms_tests


class SameCloudTests(unittest.TestCase):
    def setUp(self):
        self.previous = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        os.chdir(self.temp.name)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()

    def tearDown(self):
        self.stdout.__exit__(None, None, None)
        torch.set_num_threads(self.threads)
        os.chdir(self.previous)
        self.temp.cleanup()

    def fixture(self, dataset="checkerboard", zero_velocity=True):
        torch.manual_seed(7)
        config = ms_tests.MultiscaleWassersteinTests.config(dataset)
        config["data"]["n_points"] = 16
        cls = HorsePointSetTransformer if dataset == "horse" else PointSetTransformer
        model = cls(**config["model"])
        if zero_velocity:
            with torch.no_grad():
                model.output_proj.weight.zero_()
                model.output_proj.bias.zero_()
        run = save_training(model, config, dataset, "fixture.yaml", 0., 0.)
        directory = audit_jacobian(run / "config.yaml", dataset, clouds=4, probes=2,
                                   times=[0, .25, .5, .75, 1], rollout_steps=4)
        return run, directory / "jacobian.json"

    @staticmethod
    def options():
        return dict(nfes=[1, 2], reference_nfe=2, max_reference_nfe=8, control_clouds=4)

    def test_complete_pipeline_both_datasets_without_training_cache(self):
        for dataset in ("checkerboard", "horse"):
            with self.subTest(dataset=dataset):
                run, path = self.fixture(dataset)
                old = json.loads(path.read_text())
                original_checkpoint_hash = hashlib.sha256(Path(old["checkpoint"]).read_bytes()).hexdigest()
                with patch("multiscale_wasserstein.MultiscaleWassersteinSampler",
                           side_effect=AssertionError("No training cache may be opened")):
                    directory = audit(path, examples_per_group=1, **self.options())
                report = json.loads((directory / "same_cloud.json").read_text())
                self.assertTrue(report["all_original_rollout_state_hashes_verified"])
                self.assertEqual(report["source_sha256"], old["banks"]["rollout"]["source_sha256"])
                self.assertEqual(report["nfes"], [1, 2, 4, 8])
                self.assertEqual(report["ranking_nfe"], 4)
                self.assertEqual(len(report["per_cloud"]), 16)
                self.assertEqual(report["reference_levels"], [2, 4, 8])
                self.assertEqual(len(report["target_controls"]), 4)
                self.assertTrue(all(r["finite_refinement_checks_passed"] for r in report["per_cloud"]))
                self.assertTrue(all(r["paired_mse_vs_finest"] == 0 for r in report["per_cloud"]))
                self.assertNotIn("histogram_js", report["per_cloud"][0])
                self.assertTrue(all(c["spearman_rho"] is None for c in report["correlations_at_original_rollout_nfe"]))
                self.assertEqual(hashlib.sha256(Path(old["checkpoint"]).read_bytes()).hexdigest(), original_checkpoint_hash)
                self.assertFalse(Path(f"{dataset}.npz").exists())
                for name in ("per_cloud.csv", "target_controls.csv", "clouds.npz", "quality_vs_jacobian.png",
                             "quality_by_cloud.png", "gallery_chamfer.png", "jacobian_groups_chamfer.png"):
                    self.assertTrue((directory / name).is_file(), name)
                with np.load(directory / "clouds.npz") as arrays:
                    self.assertEqual(tensor_sha256(torch.from_numpy(arrays["source"])), report["source_sha256"])
                    self.assertEqual(arrays["original_rollout_states"].shape, (4, 5, 16, 2))
                    np.testing.assert_array_equal(arrays["prediction_nfe_4"], arrays["original_rollout_states"][:, -1])
                    self.assertFalse(np.array_equal(arrays["target"], arrays["target_control"]))
                if dataset == "horse":
                    self.assertIn("front_legs_count", report["per_cloud"][0])
                    self.assertTrue((directory / "horse_rois.png").is_file())
                    mass = report["per_cloud"][0]["roi_front_legs_mass"]
                    self.assertEqual(report["per_cloud"][0]["front_legs_count"], round(mass * 16))

    def test_mismatches_fail_before_quality_join(self):
        run, path = self.fixture(zero_velocity=False)
        old = json.loads(path.read_text())
        for kind in ("checkpoint", "source", "state", "other_coupling", "missing", "duplicate"):
            value = json.loads(json.dumps(old))
            if kind == "checkpoint":
                value["checkpoint_sha256"] = "bad"
            elif kind == "source":
                value["banks"]["rollout"]["source_sha256"] = "bad"
            elif kind == "state":
                value["per_cloud"][1]["state_sha256"] = "bad"
            elif kind == "other_coupling":
                value["config"]["coupling"] = "independent"
            elif kind == "missing":
                value["per_cloud"].pop()
            else:
                value["per_cloud"].append(value["per_cloud"][0])
            changed = Path(kind + ".json")
            changed.write_text(json.dumps(value), encoding="utf-8")
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                audit(changed, skip_render=True, **self.options())
        self.assertEqual(list(Path("analysis_results").rglob("same_cloud.json")), [])

    def test_cli_config_override_reproducible_and_coupling_records_not_joined(self):
        run, path = self.fixture(zero_velocity=False)
        old = json.loads(path.read_text())
        old["config_path"] = "missing/config.yaml"
        old["per_cloud"].append({"scope": "coupling", "cloud_index": 0, "time": 0,
                                 "normalized_frobenius": 12345})
        moved = Path("moved.json")
        moved.write_text(json.dumps(old), encoding="utf-8")
        with self.assertRaisesRegex(FileNotFoundError, "--config"):
            audit(moved, skip_render=True, **self.options())
        args = [str(moved), "--config", str(run / "config.yaml"), "--nfes", "1", "2",
                "--reference-nfe", "2", "--max-reference-nfe", "8", "--control-clouds", "4", "--skip-render"]
        directories = [main(args), main(args)]
        first, second = [json.loads((d / "same_cloud.json").read_text()) for d in directories]
        self.assertNotEqual(*directories)
        for key in ("source_sha256", "target_sha256", "target_control_sha256", "per_cloud", "quality_groups"):
            self.assertEqual(first[key], second[key], key)
        self.assertEqual(len(first["original_rollout_jacobian_records"]), 20)
        self.assertFalse(list(directories[0].glob("*.png")))

    def test_rankings_ties_nulls_and_correlations(self):
        rows = [{"cloud_index": i, "chamfer": i, "jacobian": 8 - i} for i in range(8)]
        groups = quartiles(rows, "chamfer")
        self.assertEqual(groups["low_error_ids"], [0, 1])
        self.assertEqual(groups["high_error_ids"], [7, 6])
        self.assertFalse(groups["boundary_ties"])
        self.assertAlmostEqual(correlations(rows, ["chamfer"], ["jacobian"])[0]["spearman_rho"], -1.)
        for row in rows:
            row["jacobian"] = row["cloud_index"] * 1e10
        self.assertEqual(quartiles(rows, "chamfer"), groups)  # NO Jacobian selection.
        self.assertEqual(quartiles([{"cloud_index": i, "x": 1.} for i in range(8)], "x")["low_error_ids"], [])
        self.assertEqual(quartiles([{"cloud_index": i, "x": None} for i in range(8)], "x")["n"], 0)
        tied = [{"cloud_index": i, "x": 0 if i < 4 else 1} for i in range(8)]
        self.assertTrue(quartiles(tied, "x")["boundary_ties"])

    def test_energy_uses_squared_norm_and_actual_time_grid(self):
        times = [0., .75, 1.]
        records = {(0, t): {"normalized_frobenius": 2., "frobenius_squared": 40.} for t in times}
        features = jacobian_features(records, 0, times, 10)
        self.assertEqual(features["jacobian_energy_trapezoid"], 2.)
        self.assertEqual(features["jacobian_late_mean"], 2.)


if __name__ == "__main__":
    unittest.main()
