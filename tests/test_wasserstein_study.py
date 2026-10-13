import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

import run_wasserstein_bank as runner
import run_wasserstein_study as study
from wasserstein_bank import WassersteinBankSampler, cache_spec
from wasserstein_study_report import summarize_study


class WassersteinStudyTests(unittest.TestCase):
    def setUp(self):
        self.previous = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        os.chdir(self.temp.name)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        os.chdir(self.previous)
        self.temp.cleanup()
        torch.set_num_threads(self.threads)

    def test_templates_match_original_tg_and_nsot_protocol(self):
        for dataset in ("checkerboard", "horse"):
            config = yaml.safe_load((runner.ROOT / "wasserstein_experiments" / f"{dataset}.yaml").read_text())
            names = (["checkerboard_experiments/nsot.yaml", "checkerboard_experiments/target_guided_cached_k8_n256_seed0.yaml"]
                     if dataset == "checkerboard" else
                     ["horse_experiments/horse_nsot_n256_seed0.yaml", "horse_experiments/horse_target_guided_cached_k8_n256_seed0.yaml"])
            for name in names:
                reference = yaml.safe_load((runner.ROOT / name).read_text())
                for key in ("data", "model", "evaluation"):
                    self.assertEqual(config[key], reference[key])
                for key, value in reference["training"].items():
                    self.assertEqual(config["training"][key], value)

    def test_full_axes_and_ordered_family_are_explicit(self):
        with contextlib.redirect_stdout(io.StringIO()):
            path, manifest = study.create_study()
        self.assertEqual(manifest["datasets"], ["checkerboard", "horse"])
        self.assertEqual(len(manifest["children"]) * len(manifest["planned_candidate_ids"]) * len(manifest["training_seeds"]), 144)
        self.assertTrue(all(c["manifest"] is None for c in manifest["children"]))
        self.assertTrue(all(r["original_model_data_training_evaluation_match"] for r in manifest["protocol_reference"].values()))
        report = summarize_study(path, manifest)
        self.assertTrue(all(d["status"] != "passed_provisional_gate" for d in report["decisions"]))
        levels = [c["noise_std"] for c in study.candidate_family("cost_noise") if c["kind"] == "perturbed_exact"]
        self.assertEqual(levels, sorted(levels))
        self.assertEqual(len(levels), 8)

    def test_smoke_candidate_jobs_use_original_evaluators_and_resume(self):
        with contextlib.redirect_stdout(io.StringIO()):
            path, manifest = study.create_study(smoke=True)
            study.prepare_children(path, manifest)
            before = copy.deepcopy(manifest)
            study.prepare_children(path, manifest)
            self.assertEqual(before, manifest)
            for child in manifest["children"]:
                child_path, child_manifest = study.load_child(child)
                self.assertEqual(len(child_manifest["jobs"]), 3)
                for job in child_manifest["jobs"]:
                    config = yaml.safe_load(Path(job["config"]).read_text())
                    sampler = WassersteinBankSampler(config, child["dataset"], "cpu", torch.float32)
                    self.assertEqual(sampler.selected, job["candidate_id"])
                    changed = copy.deepcopy(config)
                    changed["wasserstein_bank"]["candidate_id"] = 1
                    self.assertEqual(cache_spec(config, child["dataset"]), cache_spec(changed, child["dataset"]))
                runner.train_jobs(child_path, child_manifest)
                runner.evaluate_jobs(child_path, child_manifest, [1])
                for job in child_manifest["jobs"]:
                    result = json.loads(Path(job["evaluations"]["1"]).read_text())
                    self.assertEqual(result["coupling_details"]["selected_candidate"], job["candidate_id"])
                    self.assertIn("validation_seconds", result["coupling_details"])
                    self.assertTrue(Path(result["image"]).is_file())
                    history = json.loads((Path(job["run_config"]).parent / "learning_curve.json").read_text())["history"]
                    self.assertEqual(history[-1]["step"], 2)
            report = summarize_study(path, manifest)
        self.assertTrue(all(d["status"] != "passed_provisional_gate" for d in report["decisions"]))
        self.assertTrue((path.parent / "candidate_outcomes.csv").is_file())
        child_path, child_manifest = study.load_child(manifest["children"][0])
        # Prepared job configs are immutable, including before reusing completed jobs.
        job = child_manifest["jobs"][0]
        with Path(job["config"]).open("a") as stream:
            stream.write("\n# unexpected change\n")
        with self.assertRaisesRegex(ValueError, "config changed"):
            runner.train_jobs(child_path, child_manifest)

    def test_invalid_axes_rejected_before_preparing(self):
        for kwargs in ({"bank_seeds": [0, 0]}, {"training_seeds": [2**32]}, {"datasets": []},
                       {"nfe": [0]}, {"steps": 0}, {"smoke": True, "device": "cuda"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                study.create_study(**kwargs)


if __name__ == "__main__":
    unittest.main()
