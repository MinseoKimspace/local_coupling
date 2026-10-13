import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

import run_multiscale as runner


class MultiscaleRefinementRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.cache_patch = patch.object(runner, "_cache_path", side_effect=lambda config, dataset, method:
                                       self.directory / "cache" / method / f"{dataset}.npz")
        self.ref_cache_patch = patch.object(runner, "_refinement_cache_path", side_effect=lambda config, dataset:
                                           self.directory / "cache" / runner.REFINED / f"{dataset}_k{config[runner.REFINED]['neighbors']}.npz")
        self.cache_patch.start()
        self.ref_cache_patch.start()
        self.stdout_patch = contextlib.redirect_stdout(io.StringIO())
        self.stdout_patch.__enter__()

    def tearDown(self):
        self.stdout_patch.__exit__(None, None, None)
        self.cache_patch.stop()
        self.ref_cache_patch.stop()
        torch.set_num_threads(self.threads)
        self.temp.cleanup()

    def create(self, **kwargs):
        kwargs.setdefault("datasets", ["checkerboard"])
        kwargs.setdefault("smoke", True)
        kwargs.setdefault("with_refinement", True)
        return runner.create_experiment(**kwargs, output_root=self.directory / "experiments")

    def fake_run(self, config_path):
        config = runner._read_yaml(config_path)
        directory = self.directory / ("run_" + config["coupling"])
        directory.mkdir()
        with (directory / "config.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream)
        with (directory / config["checkpoint"]).open("xb") as stream:
            stream.write(b"fake model for runner-only test")
        with (directory / "training.json").open("x", encoding="utf-8") as stream:
            json.dump({"training_seconds": .5}, stream)
        return directory

    def test_flag_adds_paired_jobs_without_changing_baseline_protocol(self):
        _, manifest = self.create(datasets=["checkerboard", "horse"], smoke=False)
        self.assertEqual(len(manifest["jobs"]), 4)
        for dataset in manifest["datasets"]:
            jobs = [job for job in manifest["jobs"] if job["dataset"] == dataset]
            baseline, refined = [runner.verify_job(job) for job in jobs]
            for key in ("data", "model", "training", "evaluation", "seed", "device", "dtype", "multiscale_wasserstein"):
                self.assertEqual(baseline[key], refined[key])
            self.assertEqual(refined["coupling"], runner.REFINED)
            self.assertEqual(refined[runner.REFINED]["neighbors"], 8)
        _, controls = self.create(datasets=["checkerboard", "horse"], with_nsot=True)
        self.assertEqual(len(controls["jobs"]), 8)

    def test_prepare_uses_exact_base_bank_and_identity_control(self):
        path, manifest = self.create(refinement_neighbors=0)
        runner.prepare_jobs(path, manifest)
        baseline, refined = manifest["jobs"]
        config = runner.verify_job(refined)
        self.assertEqual(config["multiscale_wasserstein"]["cache_sha256"], baseline["bank"]["sha256"])
        self.assertTrue(manifest["refinement_checks"]["checkerboard:seed0"]["same_base_permutation"])
        with np.load(baseline["bank"]["path"], allow_pickle=False) as first, np.load(refined["bank"]["path"], allow_pickle=False) as second:
            for key in ("source", "target", "permutation"):
                np.testing.assert_array_equal(first[key], second[key])
            np.testing.assert_array_equal(first["permutation"], second["base_permutation"])
        saved = path.read_bytes()
        with patch.object(runner, "_refinement_cache_path", side_effect=AssertionError("should reuse saved path")):
            runner.prepare_jobs(path, manifest)
        self.assertEqual(saved, path.read_bytes())

    def test_clone_preserves_old_manifest_and_reuses_completed_baseline(self):
        path, original = self.create(with_refinement=False)
        runner.prepare_jobs(path, original)
        import train
        with patch.object(train, "main", side_effect=self.fake_run):
            runner.train_jobs(path, original)
        previous = path.read_bytes()
        old_config = Path(original["jobs"][0]["config"]).read_bytes()
        clone_path, clone = runner.add_refinement(path, original, neighbors=0)
        self.assertNotEqual(path, clone_path)
        self.assertEqual(path.read_bytes(), previous)
        self.assertEqual(Path(original["jobs"][0]["config"]).read_bytes(), old_config)
        self.assertEqual(clone["jobs"][0], original["jobs"][0])
        self.assertEqual(clone["parent_manifest"]["sha256"], runner.file_sha256(path))
        runner.prepare_jobs(clone_path, clone)
        with patch.object(train, "main", side_effect=self.fake_run) as train_mock:
            runner.train_jobs(clone_path, clone)
        self.assertEqual(train_mock.call_count, 1)
        self.assertEqual(train_mock.call_args.args[0], clone["jobs"][1]["config"])
        self.assertEqual(path.read_bytes(), previous)
        with self.assertRaisesRegex(ValueError, "already contains"):
            runner.add_refinement(clone_path, clone)

    def test_refined_cache_key_uses_base_bytes_options_and_refinement_dependencies(self):
        path, manifest = self.create(with_refinement=False)
        runner.prepare_jobs(path, manifest)
        config = runner._refined_config(runner.verify_job(manifest["jobs"][0]), 8)
        base_path = Path(config["multiscale_wasserstein"]["cache"])
        base_hash = config["multiscale_wasserstein"]["cache_sha256"]
        self.ref_cache_patch.stop()
        try:
            def hash_value(path):
                return base_hash if Path(path) == base_path else "refinement-code"
            with patch.object(runner, "file_sha256", side_effect=hash_value) as hashes:
                key = runner._refinement_cache_path(config, "checkerboard")
            self.assertEqual({Path(call.args[0]).name for call in hashes.call_args_list},
                             {base_path.name, "multiscale_refinement.py", "multiscale_wasserstein.py"})
            changed = copy.deepcopy(config)
            changed[runner.REFINED]["neighbors"] = 4
            with patch.object(runner, "file_sha256", side_effect=hash_value):
                self.assertNotEqual(key, runner._refinement_cache_path(changed, "checkerboard"))
            changed["multiscale_wasserstein"]["cache_sha256"] = "new-base"
            with patch.object(runner, "file_sha256", side_effect=lambda path: "new-base" if Path(path) == base_path else "refinement-code"):
                self.assertNotEqual(key, runner._refinement_cache_path(changed, "checkerboard"))
            with patch.object(runner, "file_sha256", side_effect=lambda path: base_hash if Path(path) == base_path else "new-code"):
                self.assertNotEqual(key, runner._refinement_cache_path(config, "checkerboard"))
            with patch.object(runner, "file_sha256", side_effect=lambda path:
                              "changed-cost-code" if Path(path).name == "multiscale_wasserstein.py" else hash_value(path)):
                self.assertNotEqual(key, runner._refinement_cache_path(config, "checkerboard"))
        finally:
            self.ref_cache_patch.start()

    def test_summary_reports_paired_quality_and_deduplicated_incremental_costs(self):
        path, manifest = self.create(refinement_neighbors=0, seeds=[0, 1])
        runner.prepare_jobs(path, manifest)
        for index, job in enumerate(manifest["jobs"]):
            image = self.directory / f"eval{index}.png"
            result = {"chamfer": .1 if job["label"] == "multiscale" else .07,
                      "leakage": .2 if job["label"] == "multiscale" else .15,
                      "histogram_js": .05, "roi_front_legs_mass_error": .01,
                      "inference_seconds": .005, "evaluation_seed": 1, "euler_steps": 1}
            with image.open("xb") as stream:
                stream.write(b"test image")
            with image.with_suffix(".json").open("x", encoding="utf-8") as stream:
                json.dump(result, stream)
            job["evaluations"]["1"] = {"json": runner._artifact(image.with_suffix(".json")), "png": runner._artifact(image)}
        runner.write_summary(path, manifest)
        summary = json.loads(path.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(summary["banks"]), 2)
        self.assertEqual(len(summary["paired_refinement"]), 2)
        expected_cost = sum(bank["precompute_seconds"] for bank in summary["banks"].values())
        self.assertEqual(summary["unique_precompute_seconds"], expected_cost)
        for comparison in summary["paired_refinement"]:
            self.assertAlmostEqual(comparison["metrics_refined_minus_baseline"]["chamfer"]["delta"], -.03)
            self.assertAlmostEqual(comparison["metrics_refined_minus_baseline"]["leakage"]["delta"], -.05)
            self.assertIn("roi_front_legs_mass_error", comparison["metrics_refined_minus_baseline"])
        self.assertEqual(sum(bank["kind"] == "incremental_refinement" for bank in summary["banks"].values()), 1)

    def test_cli_rejects_accidental_refinement_resume_overrides(self):
        for arguments in (["--add-refinement"], ["--refinement-neighbors", "8"],
                          ["--with-refinement", "--add-refinement"],
                          ["--manifest", "unused", "--with-refinement"],
                          ["--manifest", "unused", "--add-refinement", "--stage", "eval"]):
            with self.subTest(arguments=arguments), patch.object(sys, "argv", ["run_multiscale.py", *arguments]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main()

    def test_invalid_refinement_settings_fail_before_creating_experiment_files(self):
        destination = self.directory / "experiments"
        self.assertFalse(destination.exists())
        with self.assertRaisesRegex(ValueError, "neighbors"):
            self.create(refinement_neighbors=64)
        self.assertFalse(destination.exists())
        with self.assertRaisesRegex(ValueError, "edge upper bound"):
            self.create(smoke=False, refinement_neighbors=1000)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
