import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from audit_generation import audit, convergence_checks, reference_levels
from experiment import save_training
from horse_regions import HorseRegions
from model import PointSetTransformer
from train_horse import HorsePointSetTransformer, load_horse_mask
from test_mean_field_diagnostics import tiny_config


class GenerationDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_num_threads(self.threads)

    @staticmethod
    def roi_file(directory):
        path = Path(directory, "rois.json")
        path.write_text(json.dumps({"version": 1, "regions": [
            {"name": "thin", "kind": "foreground", "box": [-1, -.5, -1, 1]},
            {"name": "gap_a", "kind": "background", "box": [0, 1, -1, 1]},
            {"name": "gap_b", "kind": "background", "box": [0, 1, -1, 1]},
        ]}), encoding="utf-8")
        return path

    def test_roi_exact_partial_pixel_mass_and_all_point_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            regions = HorseRegions(torch.tensor([[1, 0], [1, 0]]), self.roi_file(temp))
            self.assertAlmostEqual(regions.regions[0]["expected_mass"], .5)
            points = torch.tensor([[[-.75, 0], [-.25, 0], [.5, 0], [1.1, 0]]])
            scores = regions.score(points)
            self.assertEqual(scores["roi_thin_mass"], .25)
            self.assertEqual(scores["roi_thin_mass_deficit"], .25)
            self.assertEqual(scores["thin_region_mass_mae"], .25)
            self.assertEqual(scores["roi_gap_a_leakage"], .25)
            self.assertEqual(scores["roi_gap_b_leakage"], .25)
            self.assertEqual(scores["gap_region_leakage"], .25)  # union, NOT .5
            regions.render(Path(temp, "rois.png"))
            self.assertTrue(Path(temp, "rois.png").is_file())

    def test_default_horse_rois_are_nonempty_and_fixed(self):
        mask = load_horse_mask("cpu", torch.float32)
        regions = HorseRegions(mask)
        foreground = [r for r in regions.regions if r["kind"] == "foreground"]
        self.assertTrue(all(0 < r["expected_mass"] < 1 for r in foreground))
        self.assertEqual(regions.definition(), HorseRegions(mask).definition())

    def test_invalid_roi_definition_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.roi_file(temp)
            spec = json.loads(path.read_text())
            spec["regions"][0]["box"] = [0, 1, -1, 1]
            path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no target foreground"):
                HorseRegions(torch.tensor([[1, 0], [1, 0]]), path)

    def test_refinement_checks_include_quality_and_p95(self):
        points = torch.zeros(20, 2, 2)
        predictions = {2: points.clone(), 4: points.clone(), 8: points.clone()}
        quality = {n: {"chamfer": .1, "leakage": .1, "cell_mass_error": None} for n in predictions}
        checks = convergence_checks(predictions, quality, [2, 4, 8], 1e-5, 1e-3)
        self.assertTrue(all(r["checks_passed"] for r in checks))
        self.assertEqual(checks[0]["undefined_quality_metrics"], ["cell_mass_error"])
        quality[8]["leakage"] = .2
        checks = convergence_checks(predictions, quality, [2, 4, 8], 1e-5, 1e-3)
        self.assertFalse(checks[-1]["quality_check_passed"])
        predictions[8][:2] = .01
        checks = convergence_checks(predictions, quality, [2, 4, 8], 2e-5, 1e-3)
        self.assertLess(checks[-1]["endpoint_mse"]["mean"], 2e-5)
        self.assertFalse(checks[-1]["endpoint_check_passed"])
        self.assertEqual(reference_levels(2, 8), [2, 4, 8])
        for start, maximum in ((2, 2), (2, 7), (0, 8), (True, 8), (2., 8)):
            with self.assertRaises(ValueError):
                reference_levels(start, maximum)

    def test_cli_mean_field_and_quality_routing(self):
        root = Path(__file__).resolve().parents[1]
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            try:
                os.chdir(temp)
                config = tiny_config("checkerboard")
                run = save_training(PointSetTransformer(**config["model"]), config,
                                    "checkerboard", "fixture.yaml", 0., 0.)
                common = [str(run / "config.yaml"), "--dataset", "checkerboard", "--output", temp]
                env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
                commands = [
                    [str(root / "audit_mean_field.py"), *common, "--clouds", "1",
                     "--targets", "2", "--batch-size", "2"],
                    [str(root / "diagnose.py"), *common, "--quality", "--clouds", "2",
                     "--eval-batch-size", "2", "--nfes", "1", "2", "--reference-nfe", "2",
                     "--max-reference-nfe", "8", "--skip-fm"],
                ]
                for command in commands:
                    completed = subprocess.run([sys.executable, *command], env=env, text=True,
                                               capture_output=True, timeout=60)
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    self.assertIn("saved=", completed.stdout)
                self.assertEqual(len(list(Path(temp).rglob("mean_field.json"))), 1)
                reports = list(Path(temp).rglob("diagnostics.json"))
                self.assertEqual(len(reports), 1)
                self.assertIsNone(json.loads(reports[0].read_text())["fm"])
            finally:
                os.chdir(previous)

    def test_joint_report_both_datasets_and_shared_banks_across_couplings(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            try:
                os.chdir(temp)
                for dataset in ("checkerboard", "horse"):
                    model_class = HorsePointSetTransformer if dataset == "horse" else PointSetTransformer
                    model = model_class(**tiny_config(dataset)["model"])
                    results = []
                    for method in ("target_guided", "minibatch_ot"):
                        config = tiny_config(dataset, method)
                        run = save_training(model, config, dataset, "fixture.yaml", 0., 0.)
                        directory = audit(run / "config.yaml", dataset, clouds=3, batch_size=2,
                                          nfes=[1, 2], reference_nfe=2, max_reference_nfe=8,
                                          fm_batches=1, matching_batch_size=2, output=temp)
                        result = json.loads((directory / "diagnostics.json").read_text())
                        results.append(result)
                        self.assertEqual(result["reference_levels"], [2, 4, 8])
                        self.assertEqual(result["reference_nfe"], 8)
                        self.assertEqual(result["endpoint_errors"][-1]["mse"]["mean"], 0)
                        self.assertEqual(result["fm"]["matching_batch_size"], 2)
                        self.assertTrue((directory / "quality.png").is_file())
                        self.assertTrue((directory / "integration.png").is_file())
                        if dataset == "horse":
                            self.assertTrue((directory / "horse_rois.png").is_file())
                            self.assertIn("thin_region_mass_mae", result["quality_by_nfe"][0]["metrics"])
                        if method == "minibatch_ot":
                            with self.assertRaisesRegex(ValueError, "training data.batch_size"):
                                audit(run / "config.yaml", dataset, clouds=2, nfes=[1],
                                      reference_nfe=2, max_reference_nfe=8,
                                      matching_batch_size=3, fm_batches=1, output=temp)
                    for key in ("noise_sha256", "target_sha256", "endpoint_errors"):
                        self.assertEqual(results[0][key], results[1][key], key)
                    self.assertEqual([r["metrics"] for r in results[0]["quality_by_nfe"]],
                                     [r["metrics"] for r in results[1]["quality_by_nfe"]])
            finally:
                os.chdir(previous)

    def test_evaluation_bank_not_replaced_by_fm_resampling(self):
        import audit_generation
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            try:
                os.chdir(temp)
                config = tiny_config("checkerboard", "minibatch_ot")
                model = PointSetTransformer(**config["model"])
                run = save_training(model, config, "checkerboard", "fixture.yaml", 0., 0.)
                def resample(source, target, **kwargs):
                    torch.randn(100)
                    return source[:1].expand_as(source).clone(), target
                with patch.object(audit_generation, "coupled_points", side_effect=resample):
                    first = audit(run / "config.yaml", "checkerboard", clouds=3, nfes=[1],
                                  reference_nfe=2, max_reference_nfe=8, fm_batches=1, output=temp)
                second = audit(run / "config.yaml", "checkerboard", clouds=3, nfes=[1],
                               reference_nfe=2, max_reference_nfe=8, fm_batches=0, output=temp)
                a = json.loads((first / "diagnostics.json").read_text())
                b = json.loads((second / "diagnostics.json").read_text())
                for key in ("noise_sha256", "target_sha256", "endpoint_errors"):
                    self.assertEqual(a[key], b[key])
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
