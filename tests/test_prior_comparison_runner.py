"""Orchestration tests: no expensive OT preparation or convergence claim."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

import run_prior_comparison as runner


PROJECT_ROOT = runner.ROOT


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ComparisonRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name).resolve()
        self.original_cwd = Path.cwd()
        os.chdir(self.root)
        self.root_patch = patch.object(runner, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        os.chdir(self.original_cwd)
        self.directory.cleanup()

    def save_run(self, job, steps=10000):
        # Deliberately unlike template names, to catch guessed run paths.
        run = self.root / "unpredictable_actual_runs" / f"{job['dataset']}_{job['prior']}_7f41"
        run.mkdir(parents=True)
        (run / "config.yaml").write_text("saved: true\n", encoding="utf-8")
        dump_json(run / "training.json", {
            "training_seconds": 10.0, "config": {"training": {"num_steps": steps}},
            "coupling_details": {"precompute_seconds": 3.0, "prior_fit_seconds": .5},
        })
        return run

    def prepare_fake(self, path, payload):
        for job in payload["jobs"]:
            job["preparation"] = {"original_precompute_seconds": 3.0,
                                  "original_prior_fit_seconds": .5,
                                  "prepare_invocation_seconds": .1, "cache_reused": True}
        runner.save_json(path, payload)

    def train_fake(self, path, payload, steps):
        for job in payload["jobs"]:
            run = self.save_run(job, steps)
            job.update(saved_config=str(run / "config.yaml"), training_json=str(run / "training.json"))
            runner.save_json(path, payload)

    def eval_fake(self, config, nfe, **kwargs):
        self.assertTrue(Path(config).is_file())
        self.assertIn("unpredictable_actual_runs", config)
        result = Path(config).parent / f"eval_{nfe}_{len(self.eval_calls)}.json"
        self.eval_calls.append((config, nfe, kwargs))
        dump_json(result, {
            "chamfer": .01 / nfe, "leakage": .02, "histogram_js": .03,
            "source_chamfer": .4, "source_leakage": .5, "source_histogram_js": .6,
            "inference_seconds": .001 * nfe, "evaluation_batch_size": 64, "total_points": 16384,
        })
        return result.with_suffix(".png")

    def evaluation_patches(self):
        self.eval_calls = []
        return (patch("eval.main", side_effect=self.eval_fake),
                patch("eval_horse.main", side_effect=self.eval_fake),
                patch.object(runner, "release_memory"))

    def test_default_is_four_new_trainings_then_all_evaluations_and_summary(self):
        p1, p2, p3 = self.evaluation_patches()
        with patch.object(runner, "prepare_jobs", side_effect=self.prepare_fake) as prepare, \
             patch.object(runner, "train_jobs", side_effect=self.train_fake) as train, p1, p2, p3:
            path = runner.main([])
        manifest = runner.read_json(path)
        prepare.assert_called_once()
        train.assert_called_once()
        self.assertEqual(train.call_args.args[2], 10000)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(manifest["jobs"]), 4)
        self.assertEqual(len(self.eval_calls), 4 * len(runner.DEFAULT_NFES))
        self.assertTrue(all(call[2]["target_seed"] == 2026 for call in self.eval_calls))
        summary = runner.read_json(manifest["summary_json"])
        for job in summary["jobs"]:
            self.assertEqual(job["source_baseline"]["nfe"], 0)
            self.assertEqual(job["source_baseline"]["chamfer"], .4)
            self.assertEqual(job["training_seconds"], 10.0)
            self.assertEqual(job["precompute_seconds"], 3.0)
            self.assertEqual(job["prior_fit_seconds"], .5)
            self.assertEqual(job["precompute_plus_training_seconds"], 13.0)
            self.assertEqual(job["evaluations"][-1]["nfe"], 128)

    def test_evaluation_recovery_uses_only_completed_saved_runs_and_preserves_parent(self):
        jobs = [{"dataset": dataset, "prior": prior, "template_config": template}
                for dataset, prior, template in runner.JOBS]
        run = self.save_run(jobs[0], 7)
        jobs[0].update(saved_config=str(run / "config.yaml"), training_json=str(run / "training.json"))
        parent, payload = runner.new_manifest("all", 7, jobs, target_seed=145)
        payload["status"] = "failed"
        runner.save_json(parent, payload)
        unchanged = parent.read_bytes()
        p1, p2, p3 = self.evaluation_patches()
        with patch.object(runner, "prepare_jobs") as prepare, patch.object(runner, "train_jobs") as train, p1, p2, p3:
            child = runner.main(["--stage", "eval", "--manifest", str(parent), "--nfes", "1", "2"])
        prepare.assert_not_called()
        train.assert_not_called()
        self.assertNotEqual(child, parent)
        self.assertEqual(parent.read_bytes(), unchanged)
        recovered = runner.read_json(child)
        self.assertEqual(recovered["training_steps"], 7)
        self.assertEqual(recovered["target_seed"], 145)
        self.assertEqual(len(recovered["jobs"]), 1)
        self.assertEqual([nfe for _, nfe, _ in self.eval_calls], [1, 2])
        self.assertTrue(all(kwargs["target_seed"] == 145 for _, _, kwargs in self.eval_calls))

    def test_failure_records_status_and_completed_run_before_next_job(self):
        def failing_train(path, payload, steps):
            job = payload["jobs"][0]
            run = self.save_run(job, steps)
            job.update(saved_config=str(run / "config.yaml"), training_json=str(run / "training.json"))
            runner.save_json(path, payload)
            raise RuntimeError("second training failed")

        with patch.object(runner, "prepare_jobs", side_effect=self.prepare_fake), \
             patch.object(runner, "train_jobs", side_effect=failing_train):
            with self.assertRaisesRegex(RuntimeError, "second training failed"):
                runner.main(["--stage", "train"])
        path, = self.root.glob("comparison_results/*/manifest.json")
        payload = runner.read_json(path)
        self.assertEqual(payload["status"], "failed")
        self.assertIn("second training failed", payload["error"])
        self.assertEqual(sum("saved_config" in job for job in payload["jobs"]), 1)
        self.assertTrue(Path(payload["summary_json"]).is_file())

    def test_training_dispatch_captures_the_trainer_return_value(self):
        jobs = [{"dataset": dataset, "prior": prior, "template_config": str(self.root / template)}
                for dataset, prior, template in runner.JOBS]
        returned = {job["template_config"]: self.save_run(job) for job in jobs}
        path, payload = runner.new_manifest("train", 9, jobs)
        def trainer(template, *, steps):
            self.assertEqual(steps, 9)
            return returned[template]
        with patch("train.main", side_effect=trainer) as checker, \
             patch("train_horse.main", side_effect=trainer) as horse, \
             patch.object(runner, "release_memory"):
            runner.train_jobs(path, payload, 9)
        self.assertEqual(checker.call_count, 2)
        self.assertEqual(horse.call_count, 2)
        for job in runner.read_json(path)["jobs"]:
            self.assertEqual(job["saved_config"], str(returned[job["template_config"]] / "config.yaml"))

    def test_preparation_preserves_original_compute_time_when_cache_reused(self):
        cache = self.root / "existing_cache.npz"
        cache.write_bytes(b"existing cache must survive")
        before = cache.read_bytes()
        job = {"dataset": "horse", "prior": "gmm_full", "template_config": "not-a-guessed-run.yaml"}
        path, payload = runner.new_manifest("prepare", 10000, [job])
        config = {"nsot": {"cache": str(cache)}}
        with patch("experiment.read_config", return_value=config), \
             patch("nsot.prepare", return_value=(cache, {
                 "precompute_seconds": 40.0, "prior_fit_seconds": 2.0,
                 "precompute_timing_scope": "original sampling, fit and OT",
             })) as prepare:
            runner.prepare_jobs(path, payload)
        prepare.assert_called_once_with(config, "horse")
        preparation = runner.read_json(path)["jobs"][0]["preparation"]
        self.assertTrue(preparation["cache_reused"])
        self.assertEqual(preparation["original_precompute_seconds"], 40.0)
        self.assertEqual(preparation["original_prior_fit_seconds"], 2.0)
        self.assertEqual(before, cache.read_bytes())

    def test_dataset_and_prior_filters_do_not_start_other_jobs(self):
        with patch.object(runner, "prepare_jobs", side_effect=self.prepare_fake), \
             patch.object(runner, "train_jobs", side_effect=self.train_fake):
            path = runner.main(["--stage", "train", "--datasets", "horse", "--priors", "anchor_prior", "--steps", "8"])
        payload = runner.read_json(path)
        self.assertEqual([(job["dataset"], job["prior"]) for job in payload["jobs"]], [("horse", "anchor_prior")])
        self.assertEqual(payload["training_steps"], 8)

    def test_smoke_writes_new_isolated_small_templates(self):
        for _, _, template in runner.JOBS:
            dest = self.root / template
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text((PROJECT_ROOT / template).read_text(encoding="utf-8"), encoding="utf-8")
        with patch.object(runner, "prepare_jobs", side_effect=self.prepare_fake), \
             patch.object(runner, "train_jobs", side_effect=self.train_fake), \
             patch.object(runner, "evaluate_jobs") as evaluate:
            path = runner.main(["--smoke"])
        payload = runner.read_json(path)
        self.assertTrue(payload["smoke"])
        self.assertEqual(payload["training_steps"], 4)
        self.assertEqual(evaluate.call_args.args[2], [1, 2])
        for job in payload["jobs"]:
            config = yaml.safe_load(Path(job["template_config"]).read_text(encoding="utf-8"))
            original = yaml.safe_load(Path(job["production_template_config"]).read_text(encoding="utf-8"))
            self.assertEqual(config["device"], "cpu")
            self.assertEqual(config["data"]["n_points"], 32)
            self.assertEqual(config["nsot"]["superset_size"], 128)
            self.assertTrue(Path(config["nsot"]["cache"]).is_relative_to(path.parent))
            self.assertEqual(original["data"]["n_points"], 256)
            self.assertEqual(original["nsot"]["superset_size"], 10000)

    def test_invalid_cli_and_legacy_manifests_fail_before_training(self):
        legacy = self.root / "old.json"
        dump_json(legacy, {"format_version": 1, "jobs": []})
        for arguments in (["--stage", "eval"], ["--manifest", str(legacy)],
                          ["--stage", "eval", "--manifest", str(legacy)],
                          ["--nfes", "1", "1"], ["--smoke", "--steps", "11"]):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                runner.main(arguments)

    def test_saved_config_validation_rejects_templates_and_relative_paths(self):
        job = {"dataset": "horse", "prior": "gmm_full", "saved_config": "config.yaml"}
        with self.assertRaisesRegex(ValueError, "absolute"):
            runner.validate_saved_config(job)
        job["saved_config"] = str(self.root / "template.yaml")
        with self.assertRaises(FileNotFoundError):
            runner.validate_saved_config(job)


class MatchedConfigurationTests(unittest.TestCase):
    def test_full_gmm_changes_only_prior_and_separate_artifact_names(self):
        for dataset in ("checkerboard", "horse"):
            jobs = [job for job in runner.JOBS if job[0] == dataset]
            configs = [yaml.safe_load((PROJECT_ROOT / template).read_text(encoding="utf-8")) for _, _, template in jobs]
            anchor, gmm = configs
            self.assertEqual(anchor["anchor_flow"]["mode"], "anchor_prior")
            self.assertEqual(gmm["anchor_flow"], {
                "mode": "gmm_prior", "reference_points": 4096, "seed": 0, "n_init": 5,
                "max_iter": 300, "tol": .0001, "reg_covar": .000001,
            })
            self.assertNotEqual(anchor["checkpoint"], gmm["checkpoint"])
            self.assertNotEqual(anchor["nsot"]["cache"], gmm["nsot"]["cache"])
            left, right = copy.deepcopy(anchor), copy.deepcopy(gmm)
            for config in (left, right):
                config.pop("anchor_flow")
                config.pop("checkpoint")
                config["nsot"].pop("cache")
            self.assertEqual(left, right)


if __name__ == "__main__":
    unittest.main()
