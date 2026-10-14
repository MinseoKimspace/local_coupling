import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

import multiscale_wasserstein as msw
import run_multiscale as base
import run_multiscale_ablation as runner


class MultiscaleAblationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.cache_patch = patch.object(base, "_cache_path", side_effect=self.cache_path)
        self.cache_patch.start()
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)
        self.cache_patch.stop()
        torch.set_num_threads(self.threads)
        self.temp.cleanup()

    def cache_path(self, config, dataset, method):
        options = {key: value for key, value in config[method].items()
                   if key not in ("cache", "cache_sha256")}
        digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[:16]
        return self.directory / "cache" / f"{dataset}_{digest}.npz"

    def create(self, **kwargs):
        kwargs.setdefault("datasets", ["checkerboard"])
        kwargs.setdefault("smoke", True)
        kwargs.setdefault("output_root", self.directory / "experiments")
        kwargs.setdefault("results_root", self.directory / "eval_results")
        return runner.create_experiment(**kwargs)

    @staticmethod
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else base.ROOT / path

    def fake_train(self, config_path):
        config = base._read_yaml(config_path)
        directory = self.directory / "runs" / Path(config["checkpoint"]).stem
        directory.mkdir(parents=True)
        (directory / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        (directory / config["checkpoint"]).write_bytes(b"fake checkpoint for orchestration test")
        (directory / "training.json").write_text(
            json.dumps({"training_seconds": .3}), encoding="utf-8")
        return directory

    def fake_evaluate(self, config_path, nfe):
        config = base._read_yaml(config_path)
        stem = Path(config["checkpoint"]).stem
        image = self.directory / "original_evaluator" / f"{stem}_nfe{nfe}.png"
        image.parent.mkdir(exist_ok=True)
        image.write_bytes(b"original evaluator PNG placeholder")
        result = {"chamfer": .1 if "baseline" in stem else .07,
                  "leakage": .2 if "baseline" in stem else .15,
                  "histogram_js": .05, "cell_mass_error": .01,
                  "roi_front_legs_mass_error": .02,
                  "inference_seconds": .005,
                  "evaluation_seed": 1, "evaluation_batch_size": 2,
                  "n_points": 8, "euler_steps": nfe, "histogram_bins": 8}
        image.with_suffix(".json").write_text(json.dumps(result), encoding="utf-8")
        return image

    def test_default_protocol_has_exact_five_variants_and_paired_settings(self):
        _, manifest = self.create(smoke=False, datasets=["checkerboard", "horse"])
        expected = {
            "baseline": ([4, 20, 100], [1, 1, 1]),
            "fine_only": ([4, 20, 100], [0, 0, 1]),
            "fine_emphasis": ([4, 20, 100], [1, 1, 2]),
            "larger_leaf": ([4, 20, 40], [1, 1, 1]),
            "smaller_leaf": ([4, 20, 200], [1, 1, 1]),
        }
        self.assertEqual(len(manifest["jobs"]), 10)
        self.assertEqual(manifest["evaluation_nfes"], base.NFES)
        for dataset in manifest["datasets"]:
            reference = base._read_yaml(base.ROOT / "multiscale_experiments" / f"{dataset}.yaml")
            jobs = [job for job in manifest["jobs"] if job["dataset"] == dataset]
            self.assertEqual({job["label"] for job in jobs}, set(expected))
            for job in jobs:
                config = base.verify_job(job)
                counts, weights = expected[job["label"]]
                self.assertEqual(config[msw.METHOD]["superset_size"], 10000)
                self.assertEqual(config[msw.METHOD]["partition_counts"], counts)
                np.testing.assert_allclose(msw.settings(config)["scale_weights"],
                                           np.asarray(weights) / sum(weights))
                for key in ("model", "data", "training", "evaluation", "seed", "dtype", "device"):
                    self.assertEqual(config[key], reference[key])
                self.assertEqual(config["coupling"], msw.METHOD)
                self.assertIsNone(job["bank"])
                self.assertIsNone(job["run"])

    def test_default_cli_only_plans_and_never_prepares_trains_or_evaluates(self):
        original_create = runner.create_experiment

        def isolated_create(**kwargs):
            return original_create(**kwargs, output_root=self.directory / "experiments",
                                   results_root=self.directory / "eval_results")

        with patch.object(runner, "create_experiment", side_effect=isolated_create), \
                patch.object(msw, "prepare", side_effect=AssertionError("unexpected OT preparation")), \
                patch.object(base, "train_jobs", side_effect=AssertionError("unexpected training")), \
                patch.object(base, "evaluate_jobs", side_effect=AssertionError("unexpected evaluation")):
            runner.main(["--smoke", "--datasets", "checkerboard"])
        manifests = list((self.directory / "experiments").glob("*/manifest.json"))
        self.assertEqual(len(manifests), 1)
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["jobs"]), 5)
        self.assertTrue(all(job["bank"] is None and job["run"] is None
                            for job in manifest["jobs"]))

    def test_project_local_plan_uses_portable_relative_paths(self):
        with tempfile.TemporaryDirectory(dir=base.ROOT, prefix=".test_ablation_") as temporary:
            directory = Path(temporary)
            with patch.object(base, "_cache_path", side_effect=lambda config, dataset, method:
                              directory / "cache" / self.cache_path(config, dataset, method).name):
                _, manifest = self.create(output_root=directory / "experiments",
                                          results_root=directory / "eval_results")
            self.assertFalse(Path(manifest["results_directory"]).is_absolute())
            for job in manifest["jobs"]:
                self.assertFalse(Path(job["config"]).is_absolute())
                config = base.verify_job(job)
                self.assertFalse(Path(config[msw.METHOD]["cache"]).is_absolute())
                self.assertFalse(Path(config["checkpoint"]).is_absolute())

    def test_tiny_preparation_preserves_identical_endpoints_and_reuses_artifacts(self):
        path, manifest = self.create()
        original_configs = {job["label"]: self.resolve(job["config"]).read_bytes()
                            for job in manifest["jobs"]}
        original_paths = {job["label"]: self.resolve(job["config"])
                          for job in manifest["jobs"]}
        runner.run_stage(path, manifest, "prepare")
        self.assertEqual(len({job["bank"]["path"] for job in manifest["jobs"]}), 5)
        first_source = first_target = None
        for job in manifest["jobs"]:
            config = base.verify_job(job)
            self.assertEqual(config[msw.METHOD]["superset_size"], 80)
            self.assertEqual(original_paths[job["label"]].read_bytes(), original_configs[job["label"]])
            with np.load(self.resolve(job["bank"]["path"]), allow_pickle=False) as bank:
                if first_source is None:
                    first_source, first_target = bank["source"].copy(), bank["target"].copy()
                np.testing.assert_array_equal(bank["source"], first_source)
                np.testing.assert_array_equal(bank["target"], first_target)
                np.testing.assert_array_equal(np.sort(bank["permutation"]), np.arange(80))
        runner.verify_manifest(path, manifest)
        saved_jobs = copy.deepcopy(manifest["jobs"])
        with patch.object(msw, "prepare", side_effect=AssertionError("prepared twice")):
            runner.run_stage(path, manifest, "prepare")
        self.assertEqual(saved_jobs, manifest["jobs"])

    def test_semantic_tampering_rejected_even_when_yaml_hash_is_updated(self):
        path, manifest = self.create()
        job = next(job for job in manifest["jobs"] if job["label"] == "fine_emphasis")
        config_path = self.resolve(job["config"])
        original = config_path.read_text(encoding="utf-8")
        config = yaml.safe_load(original)
        config["training"]["learning_rate"] *= 2
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        job["config_sha256"] = base.file_sha256(config_path)
        with self.assertRaises(ValueError):
            runner.verify_manifest(path, manifest)
        config_path.write_text(original, encoding="utf-8")
        job["config_sha256"] = base.file_sha256(config_path)
        config = yaml.safe_load(original)
        config[msw.METHOD]["scale_weights"] = [1, 1, 1]
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        job["config_sha256"] = base.file_sha256(config_path)
        with self.assertRaises(ValueError):
            runner.verify_manifest(path, manifest)

    def test_manifest_requires_exact_variant_set_and_refuses_other_protocol(self):
        path, manifest = self.create()
        incomplete = copy.deepcopy(manifest)
        incomplete["jobs"].pop()
        with self.assertRaises(ValueError):
            runner.verify_manifest(path, incomplete)
        duplicate = copy.deepcopy(manifest)
        duplicate["jobs"][-1] = copy.deepcopy(duplicate["jobs"][0])
        with self.assertRaises(ValueError):
            runner.verify_manifest(path, duplicate)
        wrong_format = copy.deepcopy(manifest)
        wrong_format["format"] = "unrelated_experiment"
        with self.assertRaises(ValueError):
            runner.verify_manifest(path, wrong_format)
        wrong_kind = copy.deepcopy(manifest)
        wrong_kind.pop("experiment_type")
        with self.assertRaises(ValueError):
            runner.verify_manifest(path, wrong_kind)

    def test_resume_extends_nfes_without_replacing_saved_configs(self):
        path, manifest = self.create()
        before = [job["config"] for job in manifest["jobs"]]
        with patch.object(runner, "run_stage") as run:
            runner.main(["--manifest", str(path), "--stage", "eval", "--nfe", "4"])
        self.assertEqual(run.call_args.args[1]["evaluation_nfes"], [1, 2, 4])
        resumed = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual([job["config"] for job in resumed["jobs"]], before)
        for change in (["--steps", "3"], ["--bank-seed", "2"], ["--smoke"]):
            with self.subTest(change=change), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                runner.main(["--manifest", str(path), *change])

    def test_mocked_pipeline_uses_existing_evaluators_and_reuses_completed_work(self):
        path, manifest = self.create()
        import train
        import eval as evaluator
        with patch.object(train, "main", side_effect=self.fake_train) as train_call, \
                patch.object(evaluator, "main", side_effect=self.fake_evaluate) as eval_call:
            runner.run_stage(path, manifest, "all")
            runner.run_stage(path, manifest, "all")
        self.assertEqual(train_call.call_count, 5)
        self.assertEqual(eval_call.call_count, 10)
        for job in manifest["jobs"]:
            self.assertEqual(set(job["evaluations"]), {"1", "2"})
            self.assertIsNotNone(job["run"])
            for result in job["evaluations"].values():
                self.assertEqual(self.resolve(result["png"]["path"]).parent,
                                 self.directory / "original_evaluator")
        summary = runner.write_summary(path, manifest)
        self.assertEqual(len(summary["paired_comparisons"]), 8)
        for comparison in summary["paired_comparisons"]:
            self.assertTrue(comparison["evaluation_settings_match"])
            deltas = comparison["metrics_variant_minus_baseline"]
            self.assertAlmostEqual(deltas["chamfer"]["delta"], -.03)
            self.assertAlmostEqual(deltas["leakage"]["delta"], -.05)
            self.assertIn("roi_front_legs_mass_error", deltas)
        results = self.resolve(manifest["results_directory"])
        self.assertTrue((results / "summary.json").is_file())
        with (results / "comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        chamfer = [row for row in rows if row["metric"] == "chamfer"]
        self.assertEqual(len(chamfer), 8)
        for row in chamfer:
            self.assertAlmostEqual(float(row["baseline"]), .1)
            self.assertAlmostEqual(float(row["variant_value"]), .07)
            self.assertAlmostEqual(float(row["delta"]), -.03)
        image = self.resolve(manifest["jobs"][0]["evaluations"]["1"]["png"]["path"])
        image.write_bytes(b"modified evaluator image")
        with self.assertRaisesRegex(ValueError, "Evaluation artifact changed"):
            runner.write_summary(path, manifest)

    def test_report_declines_quality_deltas_for_mismatched_evaluation_settings(self):
        path, manifest = self.create()
        import train
        import eval as evaluator
        with patch.object(train, "main", side_effect=self.fake_train), \
                patch.object(evaluator, "main", side_effect=self.fake_evaluate):
            runner.run_stage(path, manifest, "all")
        job = next(job for job in manifest["jobs"] if job["label"] == "fine_only")
        artifact = job["evaluations"]["1"]["json"]
        result_path = self.resolve(artifact["path"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["evaluation_seed"] = 99
        result_path.write_text(json.dumps(result), encoding="utf-8")
        artifact["sha256"] = base.file_sha256(result_path)
        summary = runner.write_summary(path, manifest)
        comparison = next(row for row in summary["paired_comparisons"]
                          if row["variant"] == "fine_only" and row["nfe"] == 1)
        self.assertFalse(comparison["evaluation_settings_match"])
        self.assertEqual(comparison["metrics_variant_minus_baseline"], {})


if __name__ == "__main__":
    unittest.main()
