"""Controlled MS-W2 / leaf-only / QAT study, including native rendering."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np
from PIL import Image
import yaml

import multiscale_wasserstein as msw
import run_multiscale as runner


class CouplingStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.original_root = runner.ROOT
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

        def isolated_cache(config, dataset, method):
            options = {key: value for key, value in config[method].items()
                       if key not in ("cache", "cache_sha256")}
            digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()[:16]
            return self.directory / "cache" / method / f"{dataset}_{digest}.npz"

        self.cache_patch = patch.object(runner, "_cache_path", side_effect=isolated_cache)
        self.cache_patch.start()
        self.stdout_patch = contextlib.redirect_stdout(io.StringIO())
        self.stdout_patch.__enter__()

    def tearDown(self):
        self.stdout_patch.__exit__(None, None, None)
        self.cache_patch.stop()
        torch.set_num_threads(self.threads)
        self.temp.cleanup()

    def resolve(self, path):
        return runner.project_path(path)

    def create(self, **kwargs):
        kwargs.setdefault("smoke", True)
        kwargs.setdefault("datasets", ["checkerboard"])
        kwargs.setdefault("coupling_study", True)
        kwargs["output_root"] = self.directory / "experiments"
        return runner.create_experiment(**kwargs)

    def test_full_protocol_only_changes_coupling(self):
        _, manifest = self.create(smoke=False, datasets=["checkerboard", "horse"], seeds=[0, 1, 2])
        self.assertTrue(manifest["coupling_study"])
        self.assertEqual(len(manifest["jobs"]), 18)
        self.assertEqual(manifest["evaluation_nfes"], [1, 2, 4, 8, 16, 32, 64, 128])
        for dataset in manifest["datasets"]:
            for seed in manifest["training_seeds"]:
                jobs = [job for job in manifest["jobs"] if job["dataset"] == dataset and job["seed"] == seed]
                self.assertEqual([job["label"] for job in jobs], ["multiscale", "leaf_only", "qat"])
                configs = [runner.verify_job(job) for job in jobs]
                for key in ("model", "data", "training", "evaluation", "seed", "dtype", "device"):
                    self.assertTrue(all(config[key] == configs[0][key] for config in configs), key)
                self.assertEqual(configs[0]["data"]["n_points"], 256)
                self.assertEqual(configs[0]["training"]["num_steps"], 10000)
                for config, weights in zip(configs[:2], ([1., 1., 1.], [0., 0., 1.])):
                    self.assertEqual(config[msw.METHOD]["superset_size"], 10000)
                    self.assertEqual(config[msw.METHOD]["partition_counts"], [4, 20, 100])
                    self.assertEqual(config[msw.METHOD]["scale_weights"], weights)
                self.assertNotEqual(configs[0][msw.METHOD]["cache"], configs[1][msw.METHOD]["cache"])
                self.assertEqual(configs[2]["coupling"], "qat")
                self.assertNotIn(msw.METHOD, configs[2])
                self.assertEqual(configs[2]["qat"]["superset_size"], 10000)
                self.assertEqual(configs[2]["qat"]["cache_seed"], manifest["bank_seed"])

    def test_incompatible_studies_and_overrides_are_rejected(self):
        for options in ({"partition_study": True}, {"with_nsot": True},
                        {"source_partition": "axis"}, {"target_partition": "pca"},
                        {"partition_counts": [4, 20, 200]}, {"scale_weights": [1, 1, 2]}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.create(smoke=False, **options)
        self.assertFalse((self.directory / "experiments").exists())

    def test_preparation_uses_identical_targets_without_qat_source_bank(self):
        path, manifest = self.create(datasets=["checkerboard", "horse"], seeds=[0, 1])
        runner.prepare_jobs(path, manifest)
        for dataset in manifest["datasets"]:
            jobs = {job["label"]: job for job in manifest["jobs"] if job["dataset"] == dataset}
            with np.load(self.resolve(jobs["multiscale"]["bank"]["path"]), allow_pickle=False) as baseline, \
                    np.load(self.resolve(jobs["leaf_only"]["bank"]["path"]), allow_pickle=False) as leaf, \
                    np.load(self.resolve(jobs["qat"]["bank"]["path"]), allow_pickle=False) as qat:
                np.testing.assert_array_equal(baseline["source"], leaf["source"])
                np.testing.assert_array_equal(baseline["target"], leaf["target"])
                np.testing.assert_array_equal(baseline["target"], qat["target"])
                self.assertNotIn("source", qat.files)
            self.assertEqual(len({job["bank"]["path"] for job in manifest["jobs"]
                                  if job["dataset"] == dataset}), 3)
        runner.write_summary(path, manifest)
        self.assertTrue(path.with_name("bank_comparison.csv").is_file())
        self.assertTrue(path.with_name("comparison.csv").is_file())
        saved = path.read_bytes()
        with patch.object(msw, "prepare", side_effect=AssertionError("prepared again")), \
                patch("qat_coupling.prepare", side_effect=AssertionError("prepared again")):
            runner.prepare_jobs(path, manifest)
        self.assertEqual(path.read_bytes(), saved)

    def test_default_cli_is_prepare_only_and_resume_preserves_protocol(self):
        original = runner.create_experiment
        with patch.object(sys, "argv", ["run_multiscale.py", "--coupling-study", "--smoke",
                                         "--datasets", "checkerboard"]), \
                patch.object(runner, "create_experiment", side_effect=lambda **kw:
                             original(**kw, output_root=self.directory / "experiments")), \
                patch.object(runner, "train_jobs", side_effect=AssertionError("unexpected training")), \
                patch.object(runner, "evaluate_jobs", side_effect=AssertionError("unexpected inference")):
            runner.main()
        path = next((self.directory / "experiments").glob("*/manifest.json"))
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["coupling_study"])
        self.assertEqual(len(manifest["jobs"]), 3)
        for job in manifest["jobs"]:
            self.assertIsNotNone(job["bank"])
            self.assertIsNone(job["run"])
            self.assertEqual(job["evaluations"], {})
        with patch.object(sys, "argv", ["run_multiscale.py", "--manifest", str(path)]):
            runner.main()
        for flags in (["--coupling-study"], ["--partition-study"], ["--with-nsot"], ["--steps", "3"]):
            with self.subTest(flags=flags), \
                    patch.object(sys, "argv", ["run_multiscale.py", "--manifest", str(path), *flags]), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main()

    def test_updated_config_hash_cannot_hide_uncontrolled_protocol(self):
        for changed in ("data", "training", "scale_weights", "partition_counts", "cache_seed"):
            with self.subTest(changed=changed):
                path, manifest = self.create()
                job = next(job for job in manifest["jobs"] if job["label"] == "leaf_only")
                config = runner.verify_job(job)
                if changed == "data":
                    config["data"]["batch_size"] += 1
                elif changed == "training":
                    config["training"]["learning_rate"] *= 2
                elif changed == "scale_weights":
                    config[msw.METHOD][changed] = [1, 1, 1]
                elif changed == "partition_counts":
                    config[msw.METHOD][changed] = [8]
                else:
                    config[msw.METHOD][changed] += 1
                config_path = self.resolve(job["config"])
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                job["config_sha256"] = runner.file_sha256(config_path)
                with patch.object(msw, "prepare", side_effect=AssertionError("validation was too late")), \
                        self.assertRaises(ValueError):
                    runner.prepare_jobs(path, manifest)
                self.assertTrue(all(item["bank"] is None for item in manifest["jobs"]))

    def test_study_paths_are_relative_and_prepared_manifest_can_relocate(self):
        first_root, second_root = self.directory / "first", self.directory / "second"
        templates = first_root / "multiscale_experiments"
        templates.mkdir(parents=True)
        shutil.copy2(self.original_root / "multiscale_experiments" / "checkerboard.yaml", templates)

        def local_cache(config, dataset, method):
            suffix = json.dumps(config[method].get("scale_weights", []))
            digest = hashlib.sha256(suffix.encode()).hexdigest()[:12]
            return first_root / "coupling_cache" / method / f"{dataset}_{digest}.npz"

        with patch.object(runner, "ROOT", first_root), patch.object(runner, "_cache_path", side_effect=local_cache):
            path, manifest = runner.create_experiment(coupling_study=True, smoke=True, datasets=["checkerboard"])
            runner.prepare_jobs(path, manifest)
            for job in manifest["jobs"]:
                self.assertFalse(Path(job["config"]).is_absolute())
                self.assertFalse(Path(job["bank"]["path"]).is_absolute())
                config = runner.verify_job(job)
                self.assertFalse(Path(config[config["coupling"]]["cache"]).is_absolute())
            relative_manifest = path.relative_to(first_root)
        shutil.copytree(first_root, second_root)
        with patch.object(runner, "ROOT", second_root):
            relocated = second_root / relative_manifest
            manifest = json.loads(relocated.read_text(encoding="utf-8"))
            with patch.object(msw, "prepare", side_effect=AssertionError("relocation rebuilt cache")), \
                    patch("qat_coupling.prepare", side_effect=AssertionError("relocation rebuilt cache")):
                runner.prepare_jobs(relocated, manifest)
            runner.write_summary(relocated, manifest)
            for job in manifest["jobs"]:
                runner.verify_job(job)
            self.assertTrue(relocated.with_name("summary.json").is_file())

    def test_real_tiny_train_eval_same_fresh_sources_native_panels_and_resume(self):
        import experiment
        from experiment import evaluation_title

        path, manifest = self.create(datasets=["checkerboard", "horse"])
        runner.prepare_jobs(path, manifest)
        inputs = []
        integrate = experiment.integrate_velocity

        def capture(model, noise, **kwargs):
            inputs.append((kwargs["num_steps"], noise.detach().cpu().clone()))
            return integrate(model, noise, **kwargs)

        with patch.object(runner, "ROOT", self.directory):
            runner.train_jobs(path, manifest)
            with patch.object(experiment, "integrate_velocity", side_effect=capture):
                runner.evaluate_jobs(path, manifest)
            self.assertEqual(len(inputs), len(manifest["jobs"]) * 2)
            grouped = {}
            for index, job in enumerate(manifest["jobs"]):
                self.assertIsNotNone(job["run"])
                self.assertEqual(set(job["evaluations"]), {"1", "2"})
                config = runner.verify_job(job)
                title = evaluation_title(config).lower()
                if job["label"] == "leaf_only":
                    self.assertIn("leaf", title)
                if job["label"] == "qat":
                    self.assertIn("qat", title)
                for nfe, noise in inputs[index * 2:index * 2 + 2]:
                    key = (job["dataset"], nfe)
                    if key in grouped:
                        torch.testing.assert_close(noise, grouped[key], rtol=0, atol=0)
                    else:
                        grouped[key] = noise
            runner.write_summary(path, manifest)
            self.assertTrue(path.with_name("comparison_by_seed.csv").is_file())
            for dataset in manifest["datasets"]:
                with Image.open(path.with_name(f"{dataset}_seed0_nfe_001_comparison.png")) as combined:
                    width = 0
                    for job in [job for job in manifest["jobs"] if job["dataset"] == dataset]:
                        with Image.open(self.resolve(job["evaluations"]["1"]["png"]["path"])) as original:
                            np.testing.assert_array_equal(
                                np.asarray(combined.crop((width, 0, width + original.width, original.height))),
                                np.asarray(original.convert("RGB")))
                            width += original.width
                    self.assertEqual(combined.width, width)
            with patch("train.main", side_effect=AssertionError("trained again")), \
                    patch("train_horse.main", side_effect=AssertionError("trained again")), \
                    patch("eval.main", side_effect=AssertionError("evaluated again")), \
                    patch("eval_horse.main", side_effect=AssertionError("evaluated again")):
                runner.train_jobs(path, manifest)
                runner.evaluate_jobs(path, manifest)


if __name__ == "__main__":
    unittest.main()
