import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from wasserstein_study_report import integrated_validation, rank_correlation, summarize_study


class WassersteinStudyReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def curve(value):
        return [{"time": 0.0, "mse_per_coordinate": value},
                {"time": 0.1, "mse_per_coordinate": value + .1},
                {"time": 1.0, "mse_per_coordinate": value + 1}]

    def fixture(self, *, early=True, smoke=False, pilot=False):
        root = {"children": [], "datasets": ["checkerboard"], "bank_seeds": [0, 1, 2],
                "training_seeds": [0, 1], "planned_candidate_ids": [0, 1, 2, 3],
                "primary_nfe": 128, "early_fraction": .05, "smoke": smoke,
                "study_mode": "pilot" if pilot else "all_candidates"}
        values = [4., 1., 3., 2.]
        for bank_seed in root["bank_seeds"]:
            directory = self.directory / f"bank{bank_seed}"
            rows = [{"candidate": c, "transport": float(c), "displacement": value,
                     "joint": value, "multiscale": value, "variance": value, "path_joint": value}
                    for c, value in enumerate(values)]
            self.write(directory / "bank_scores.json", {"reports": {"selection": {"candidates": rows}},
                                                        "selected": {"random": 2}})
            child = {"dataset": "checkerboard", "bank_seed": bank_seed, "jobs": [], "training_steps": 100}
            for candidate, value in enumerate(values):
                for seed in root["training_seeds"]:
                    run = directory / f"candidate{candidate}_seed{seed}"
                    run.mkdir()
                    (run / "config.yaml").write_text("training:\n  num_steps: 100\n", encoding="utf-8")
                    history = []
                    if early:
                        history.append({"step": 5, "elapsed_seconds": 10., "validation_by_time": self.curve(value + .2)})
                    history.extend([
                        {"step": 6, "elapsed_seconds": 12., "validation_by_time": self.curve(.001 if candidate == 0 else 9.)},
                        {"step": 100, "elapsed_seconds": 200., "validation_by_time": self.curve(value + seed / 10)},
                    ])
                    self.write(run / "learning_curve.json", {"history": history})
                    self.write(run / "eval.json", {"chamfer": value + seed / 10, "leakage": value / 10,
                                                   "histogram_js": .25, "cell_mass_error": .1,
                                                   "training_seconds": 200., "inference_seconds": .4,
                                                   "coupling_details": {"validation_seconds": 20.,
                                                                        "training_seconds_excluding_validation": 180.},
                                                   "evaluation_seed": 1, "training_config_verified": True})
                    child["jobs"].append({"candidate_id": candidate, "seed": seed,
                                          "config": str(run / "config.yaml"), "run_config": str(run / "config.yaml"),
                                          "evaluations": {"128": str(run / "eval.json")}})
            self.write(directory / "manifest.json", child)
            root["children"].append({"dataset": "checkerboard", "bank_seed": bank_seed,
                                      "manifest": str(directory / "manifest.json")})
        path = self.directory / "manifest.json"
        self.write(path, root)
        return path, root

    def test_time_integral_not_nonuniform_sample_mean(self):
        value = integrated_validation([{"time": 0., "mse_per_coordinate": 0.},
                                       {"time": .1, "mse_per_coordinate": 1.},
                                       {"time": 1., "mse_per_coordinate": 1.}])
        self.assertAlmostEqual(value, .95)
        self.assertIsNone(integrated_validation([]))
        with self.assertRaisesRegex(ValueError, "distinct"):
            integrated_validation([{"time": 0., "mse_per_coordinate": 0.}] * 2)

    def test_rank_correlation_handles_ties_and_collinearity(self):
        self.assertAlmostEqual(rank_correlation([0, 1, 1, 2], [2, 1, 1, 0]), -1.)
        self.assertIsNone(rank_correlation([1, 1, 1], [0, 1, 2]))
        self.assertIsNone(rank_correlation([0, 1, 2, 3], [1, 3, 2, 0], [0, 1, 2, 3]))
        self.assertAlmostEqual(rank_correlation([4, 1, 3, 2], [4, 1, 3, 2], [0, 1, 2, 3]), 1.)

    def test_complete_study_paired_regret_early_cost_and_secondary_metrics(self):
        path, root = self.fixture()
        report = summarize_study(path, root)
        self.assertTrue(all(group["complete"] for group in report["groups"]))
        self.assertTrue(all(decision["status"] == "passed_provisional_gate" for decision in report["decisions"]
                            if decision["role"] == "primary"))
        self.assertTrue(all(decision["status"] == "secondary_only" for decision in report["decisions"]
                            if decision["role"] == "secondary"))
        self.assertTrue((path.parent / "predictivity.json").is_file())
        with (path.parent / "selector_regret.csv").open(newline="", encoding="utf-8") as stream:
            regrets = list(csv.DictReader(stream))
        early = [row for row in regrets if row["selector"] == "early_validation"]
        self.assertEqual(len(early), 12)
        self.assertTrue(all(int(row["candidate_id"]) == 1 for row in early))
        self.assertTrue(all(float(row["early_total_online_seconds"]) == 40. for row in early))
        self.assertTrue(all(int(row["early_total_optimizer_updates"]) == 20 for row in early))
        transport = [row for row in regrets if row["selector"] == "transport"]
        self.assertTrue(all(abs(float(row["regret"]) - 3.) < 1e-12 for row in transport))
        with (path.parent / "candidate_outcomes.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 24)
        self.assertEqual(float(rows[0]["validation_seconds"]), 20.)
        self.assertEqual(float(rows[0]["training_seconds_excluding_validation"]), 180.)
        self.assertIn("primary_quality_cell_mass_error", rows[0])
        self.assertNotIn("evaluation_seed", json.loads(rows[0]["quality_metrics_by_nfe_json"])["128"])

    def test_later_than_five_percent_is_never_used_for_early_selector(self):
        path, root = self.fixture(early=False)
        summarize_study(path, root)
        with (path.parent / "selector_regret.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertNotIn("early_validation", {row["selector"] for row in rows})

    def test_missing_jobs_blocks_claim_and_is_reported(self):
        path, root = self.fixture()
        child_path = Path(root["children"][0]["manifest"])
        child = json.loads(child_path.read_text(encoding="utf-8"))
        child["jobs"].pop()
        self.write(child_path, child)
        report = summarize_study(path, root)
        self.assertEqual(report["groups"][0]["missing_jobs"], [{"candidate_id": 3, "training_seed": 1}])
        self.assertTrue(all(decision["status"] == "incomplete" for decision in report["decisions"]
                            if decision["role"] == "primary"))

    def test_missing_bank_seed_blocks_claim(self):
        path, root = self.fixture()
        root["children"].pop()
        report = summarize_study(path, root)
        self.assertTrue(all(decision["status"] == "incomplete" for decision in report["decisions"]
                            if decision["role"] == "primary"))

    def test_pilot_and_smoke_cannot_pass_gate(self):
        path, root = self.fixture(pilot=True)
        report = summarize_study(path, root)
        self.assertTrue(all(decision["status"] == "pilot_only" for decision in report["decisions"]
                            if decision["role"] == "primary"))
        root["study_mode"], root["smoke"] = "all_candidates", True
        report = summarize_study(path, root)
        self.assertTrue(all(decision["status"] == "pilot_only" for decision in report["decisions"]
                            if decision["role"] == "primary"))
        root["study_mode"], root["smoke"] = "smoke", False
        report = summarize_study(path, root)
        self.assertTrue(all(decision["status"] == "pilot_only" for decision in report["decisions"]
                            if decision["role"] == "primary"))

    def test_missing_or_partial_final_curve_blocks_completion(self):
        path, root = self.fixture()
        child = json.loads(Path(root["children"][0]["manifest"]).read_text(encoding="utf-8"))
        history_path = Path(child["jobs"][0]["run_config"]).parent / "learning_curve.json"
        history = json.loads(history_path.read_text(encoding="utf-8"))
        history["history"][-1]["validation_by_time"].pop()
        self.write(history_path, history)
        report = summarize_study(path, root)
        self.assertFalse(report["groups"][0]["complete"])
        self.assertTrue(all(decision["status"] == "incomplete" for decision in report["decisions"]
                            if decision["role"] == "primary"))

    def test_partial_preparation_and_predeclared_primary_scores(self):
        path, root = self.fixture()
        root["children"][0]["manifest"] = None
        root["claim"] = {"primary_scores": ["path_joint"]}
        report = summarize_study(path, root)
        self.assertEqual(len(report["preparation_failures"]), 1)
        self.assertEqual(report["primary_scores"], ["path_joint"])
        self.assertEqual([d["score"] for d in report["decisions"] if d["role"] == "primary"], ["path_joint"])
        self.assertEqual(next(d for d in report["decisions"] if d["role"] == "primary")["status"], "incomplete")

    def test_gate_requires_same_banks_to_improve_both_endpoints(self):
        path, root = self.fixture()
        # CD succeeds at banks 0,1; held-out MSE succeeds at banks 1,2.
        # Each endpoint independently wins 2/3, but only bank 1 wins both.
        for bank_seed, metric in ((0, "mse"), (2, "cd")):
            child = json.loads(Path(root["children"][bank_seed]["manifest"]).read_text(encoding="utf-8"))
            for job in child["jobs"]:
                run = Path(job["run_config"]).parent
                value = float(job["candidate_id"] + 1)
                if metric == "cd":
                    output = json.loads((run / "eval.json").read_text(encoding="utf-8"))
                    output["chamfer"] = value
                    self.write(run / "eval.json", output)
                else:
                    output = json.loads((run / "learning_curve.json").read_text(encoding="utf-8"))
                    output["history"][-1]["validation_by_time"] = self.curve(value)
                    self.write(run / "learning_curve.json", output)
        report = summarize_study(path, root)
        for decision in report["decisions"]:
            if decision["role"] == "primary":
                self.assertEqual(decision["qualifying_bank_seeds"], [1])
                self.assertEqual(decision["status"], "not_supported")

    def test_duplicate_candidate_jobs_are_rejected(self):
        path, root = self.fixture()
        child_path = Path(root["children"][0]["manifest"])
        child = json.loads(child_path.read_text(encoding="utf-8"))
        child["jobs"].append(child["jobs"][0])
        self.write(child_path, child)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize_study(path, root)

    def test_saved_config_hash_is_checked_even_with_child_training_steps(self):
        path, root = self.fixture()
        child_path = Path(root["children"][0]["manifest"])
        child = json.loads(child_path.read_text(encoding="utf-8"))
        job = child["jobs"][0]
        config = Path(job["config"])
        job["config_sha256"] = hashlib.sha256(config.read_bytes()).hexdigest()
        self.write(child_path, child)
        summarize_study(path, root)
        config.write_text("training:\n  num_steps: 200\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "config hash changed"):
            summarize_study(path, root)

    def test_training_timing_survives_missing_primary_eval(self):
        path, root = self.fixture()
        child_path = Path(root["children"][0]["manifest"])
        child = json.loads(child_path.read_text(encoding="utf-8"))
        job = child["jobs"][0]
        run = Path(job["run_config"]).parent
        self.write(run / "training.json", {"training_seconds": 321.,
                                          "coupling_details": {"validation_seconds": 21.,
                                                               "training_seconds_excluding_validation": 300.}})
        self.write(run / "nfe1.json", {"chamfer": 1., "inference_seconds": .001})
        self.write(run / "nfe2.json", {"chamfer": .8, "inference_seconds": .002})
        job["evaluations"] = {"1": str(run / "nfe1.json"), "2": str(run / "nfe2.json")}
        self.write(child_path, child)
        summarize_study(path, root)
        with (path.parent / "candidate_outcomes.csv").open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(float(row["training_seconds"]), 321.)
        self.assertEqual(float(row["validation_seconds"]), 21.)
        self.assertEqual(float(row["training_seconds_excluding_validation"]), 300.)
        self.assertEqual(row["inference_seconds"], "")
        self.assertEqual(json.loads(row["inference_seconds_by_nfe_json"]), {"1": .001, "2": .002})
        job["evaluations"] = {}
        self.write(child_path, child)
        summarize_study(path, root)
        with (path.parent / "candidate_outcomes.csv").open(newline="", encoding="utf-8") as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(float(row["training_seconds"]), 321.)
        self.assertEqual(json.loads(row["inference_seconds_by_nfe_json"]), {})


if __name__ == "__main__":
    unittest.main()
