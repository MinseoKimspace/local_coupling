import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

import multiscale_overlap as overlap
import multiscale_wasserstein as msw
import run_msw2_boundary_study as runner


class BoundaryStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()

    def tearDown(self):
        self.stdout.__exit__(None, None, None)
        torch.set_num_threads(self.threads)
        self.temp.cleanup()

    def create(self, **kwargs):
        kwargs.setdefault("smoke", True)
        kwargs.setdefault("datasets", ["checkerboard"])
        return runner.create_experiment(output_root=self.directory / "study", **kwargs)

    def test_full_defaults_preserve_templates_and_factorial_axes(self):
        path, manifest = self.create(smoke=False, datasets=["checkerboard", "horse"], seeds=[0, 7])
        self.assertTrue(path.is_file())
        self.assertEqual(len(manifest["jobs"]), 16)
        self.assertEqual(manifest["evaluation_nfes"], [1, 2, 4, 8, 16, 32, 128])
        runner.verify_manifest(manifest)
        for job in manifest["jobs"]:
            config = runner.verify_job(job)
            template = runner.read_yaml(f"multiscale_experiments/{job['dataset']}.yaml")
            for field in ("model", "data", "evaluation", "dtype", "device"):
                self.assertEqual(config[field], template[field])
            training = copy.deepcopy(config["training"])
            self.assertEqual(training.pop("time_seed"), job["seed"] + 2)
            self.assertEqual(training.pop("time_weighting", None), runner.LATE if job["arm"] in "BD" else None)
            self.assertEqual(training, template["training"])
            self.assertEqual(config[runner.BASE]["partition_counts"], [4, 20, 100])
            self.assertEqual(config[runner.BASE]["superset_size"], 10000)
            self.assertFalse(Path(job["config"]["path"]).is_absolute())
            self.assertFalse(Path(config[runner.BASE]["cache"]).is_absolute())
            if job["arm"] in "CD":
                self.assertEqual(config[runner.OVERLAP]["window_sizes"], [64, 128])
            self.assertIsNone(job["run"])

    def test_prepare_shares_two_banks_across_all_arms_and_seeds(self):
        path, manifest = self.create(seeds=[0, 1])
        initial = {job["config"]["path"]: runner.absolute(job["config"]["path"]).read_bytes()
                   for job in manifest["jobs"]}
        with patch.object(msw, "multiscale_permutation", wraps=msw.multiscale_permutation) as base, \
                patch.object(overlap, "overlap_permutation", wraps=overlap.overlap_permutation) as improve:
            runner.prepare_jobs(path, manifest)
        self.assertEqual(base.call_count, 1)
        self.assertEqual(improve.call_count, 1)
        self.assertEqual(len({job["base_bank"]["path"] for job in manifest["jobs"]}), 1)
        self.assertEqual(len({job["bank"]["path"] for job in manifest["jobs"]}), 2)
        self.assertTrue(manifest["endpoint_checks"]["checkerboard"]["same_source"])
        self.assertTrue(manifest["endpoint_checks"]["checkerboard"]["same_target"])
        for filename, content in initial.items():
            self.assertEqual(runner.absolute(filename).read_bytes(), content)
        saved = path.read_bytes()
        with patch.object(msw, "prepare", side_effect=AssertionError("base prepared again")), \
                patch.object(overlap, "prepare", side_effect=AssertionError("overlap prepared again")):
            runner.prepare_jobs(path, manifest)
        self.assertEqual(path.read_bytes(), saved)

    def test_default_stage_does_not_train_or_eval(self):
        original = runner.create_experiment
        with patch.object(runner, "create_experiment", side_effect=lambda **kw:
                          original(output_root=self.directory, **kw)), \
                patch.object(runner, "train_jobs", side_effect=AssertionError("unexpected training")), \
                patch.object(runner, "evaluate_jobs", side_effect=AssertionError("unexpected eval")):
            path = runner.main(["--smoke", "--datasets", "checkerboard"])
        manifest = runner.read_json(path)
        self.assertTrue(all(job["bank"] is not None and job["run"] is None for job in manifest["jobs"]))
        self.assertTrue(path.with_name("summary.json").is_file())
        self.assertTrue(path.with_name("comparison.csv").is_file())

    def test_artifact_tampering_rejected_before_resume_manifest_mutation(self):
        path, manifest = self.create()
        runner.prepare_jobs(path, manifest)
        last = manifest["jobs"][-1]
        filename = runner.absolute(last["config"]["path"])
        filename.write_bytes(filename.read_bytes() + b"\n# changed\n")
        before = path.read_bytes()
        with patch.object(runner, "prepare_jobs") as prepare, \
                self.assertRaisesRegex(ValueError, "Prepared YAML changed"):
            runner.main(["--manifest", str(path), "--nfe", "4"])
        prepare.assert_not_called()
        self.assertEqual(path.read_bytes(), before)

    def test_bank_tampering_is_detected_for_every_arm_before_training(self):
        path, manifest = self.create()
        runner.prepare_jobs(path, manifest)
        filename = runner.absolute(manifest["jobs"][-1]["bank"]["path"])
        filename.write_bytes(filename.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Prepared bank changed"):
            runner.train_jobs(path, manifest)

    def test_arm_mismatch_is_rejected_even_with_updated_config_hash(self):
        path, manifest = self.create()
        job = manifest["jobs"][-1]
        config = runner.verify_job(job)
        config["evaluation"]["batch_size"] += 1
        filename = runner.absolute(job["config"]["path"])
        filename.write_text(yaml.safe_dump(config), encoding="utf-8")
        job["config"] = runner.artifact(filename)
        with self.assertRaisesRegex(ValueError, "matched model/data/training/evaluation"):
            runner.verify_manifest(manifest)

    def test_interrupted_preparation_adopts_immutable_prepared_yaml(self):
        path, manifest = self.create()
        saved = copy.deepcopy(manifest)
        original_save = runner.save_manifest
        with patch.object(runner, "save_manifest", side_effect=RuntimeError("interrupted")), \
                self.assertRaisesRegex(RuntimeError, "interrupted"):
            runner.prepare_jobs(path, manifest)
        # Disk still points at initial YAML; the bank and prepared YAML already exist.
        with patch.object(runner, "save_manifest", wraps=original_save):
            runner.prepare_jobs(path, saved)
        runner.verify_manifest(saved)
        self.assertTrue(all(job["bank"] for job in saved["jobs"]))

    def _fake_train(self, config_path):
        config = runner.read_yaml(config_path)
        directory = self.directory / "runs" / Path(config["checkpoint"]).stem
        directory.mkdir(parents=True)
        (directory / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        (directory / config["checkpoint"]).write_bytes(b"mock checkpoint")
        (directory / "training.json").write_text(json.dumps({"training_seconds": .3}), encoding="utf-8")
        return directory

    def _fake_eval(self, config_path, nfe):
        config = runner.read_yaml(config_path)
        arm = Path(config["checkpoint"]).stem.split("_")[-2]
        image = self.directory / f"{arm}_{nfe}.png"
        image.write_bytes(b"mock image")
        result = {"dataset": "checkerboard", "evaluation_seed": 1, "evaluation_batch_size": 6,
                  "n_points": 16, "total_points": 96, "euler_steps": nfe, "histogram_bins": 16,
                  "metric_definitions": {"chamfer": "squared CD"}, "metric_metadata": None,
                  "chamfer": {"A": .4, "B": .3, "C": .2, "D": .05}[arm],
                  "leakage": .1, "histogram_js": .01, "cell_mass_error": None,
                  "training_seconds": .3, "inference_seconds": .012}
        image.with_suffix(".json").write_text(json.dumps(result), encoding="utf-8")
        return image

    def test_train_eval_resume_and_metric_contrasts(self):
        import train
        import eval as evaluator
        path, manifest = self.create()
        runner.prepare_jobs(path, manifest)
        with patch.object(train, "main", side_effect=self._fake_train) as training:
            runner.train_jobs(path, manifest)
            runner.train_jobs(path, manifest)
        self.assertEqual(training.call_count, 4)
        with patch.object(evaluator, "main", side_effect=self._fake_eval) as evaluation:
            runner.evaluate_jobs(path, manifest)
            runner.evaluate_jobs(path, manifest)
        self.assertEqual(evaluation.call_count, 8)
        runner.write_summary(path, manifest)
        summary = runner.read_json(path.with_name("summary.json"))
        self.assertEqual(len(summary["rows"]), 8)
        self.assertEqual(len(summary["contrasts"]), 10)
        row = next(row for row in summary["rows"] if row["arm"] == "C")
        self.assertGreater(row["overlap_incremental_seconds"], 0)
        self.assertEqual(row["inference_ms_per_cloud"], 2.)
        interaction = next(row for row in summary["contrasts"] if row["contrast"] == "interaction_D-C-B+A")
        self.assertAlmostEqual(interaction["metrics"]["chamfer"], -.05)
        self.assertIsNone(interaction["metrics"]["cell_mass_error"])
        # New NFEs are additive; no settings can be changed on resume.
        runner.main(["--manifest", str(path), "--nfe", "4"])
        self.assertEqual(runner.read_json(path)["evaluation_nfes"], [1, 2, 4])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(["--manifest", str(path), "--steps", "3"])

    def test_contrasts_reject_unmatched_evaluation_protocol(self):
        import train
        import eval as evaluator
        path, manifest = self.create()
        runner.prepare_jobs(path, manifest)
        with patch.object(train, "main", side_effect=self._fake_train):
            runner.train_jobs(path, manifest)
        with patch.object(evaluator, "main", side_effect=self._fake_eval):
            runner.evaluate_jobs(path, manifest)
        item = manifest["jobs"][-1]["evaluations"]["1"]["json"]
        result = runner.read_json(item["path"])
        result["histogram_bins"] = 32
        runner.absolute(item["path"]).write_text(json.dumps(result), encoding="utf-8")
        manifest["jobs"][-1]["evaluations"]["1"]["json"] = runner.artifact(item["path"])
        with self.assertRaisesRegex(ValueError, "evaluation protocols differ"):
            runner.write_summary(path, manifest)

    def test_existing_base_config_is_read_only_and_stored_as_relative_reference(self):
        config = runner.read_yaml("multiscale_experiments/checkerboard.yaml")
        bank = self.directory / "existing_base.npz"
        bank.write_bytes(b"mock bank; only config path handling is under test")
        config[runner.BASE]["cache"] = str(bank)
        base_config = self.directory / "base.yaml"
        base_config.write_text(yaml.safe_dump(config), encoding="utf-8")
        before = base_config.read_bytes(), bank.read_bytes()
        with patch.object(msw, "load_cache", return_value=(None, None, None, {}, runner.file_sha256(bank))) as load:
            path, manifest = self.create(smoke=False, base_checkerboard_config=base_config)
        self.assertEqual(load.call_count, 1)
        self.assertEqual((base_config.read_bytes(), bank.read_bytes()), before)
        origin = manifest["origins"]["checkerboard"]
        self.assertFalse(Path(origin["cache"]["path"]).is_absolute())
        for job in manifest["jobs"]:
            generated = runner.verify_job(job)
            self.assertEqual(runner.absolute(generated[runner.BASE]["cache"]), bank.resolve())
            self.assertEqual(generated[runner.BASE]["cache_sha256"], runner.file_sha256(bank))
        runner.verify_manifest(manifest)

    def test_bad_base_paths_and_settings_fail_before_creating_output(self):
        config = runner.read_yaml("multiscale_experiments/checkerboard.yaml")
        config[runner.BASE]["cache"] = "this_bank_does_not_exist.npz"
        base_config = self.directory / "base.yaml"
        base_config.write_text(yaml.safe_dump(config), encoding="utf-8")
        with self.assertRaisesRegex(FileNotFoundError, "Base cache not found"):
            self.create(smoke=False, base_checkerboard_config=base_config)
        self.assertFalse((self.directory / "study").exists())
        with self.assertRaisesRegex(ValueError, "selected --datasets"):
            self.create(smoke=False, base_horse_config=base_config)
        with self.assertRaisesRegex(ValueError, "cannot reuse"):
            self.create(base_checkerboard_config=base_config)
        with self.assertRaisesRegex(ValueError, "distinct integers"):
            self.create(seeds=[1, 1])


if __name__ == "__main__":
    unittest.main()
