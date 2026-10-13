import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

import multiscale_wasserstein as msw
import nsot
import run_multiscale as runner


class MultiscaleRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.cache_patch = patch.object(
            runner, "_cache_path", side_effect=lambda config, dataset, method:
            self.directory / "cache" / method / f"{dataset}_{config[method]['superset_size']}.npz")
        self.cache_patch.start()
        self.stdout_patch = contextlib.redirect_stdout(io.StringIO())
        self.stdout_patch.__enter__()

    def tearDown(self):
        self.stdout_patch.__exit__(None, None, None)
        self.cache_patch.stop()
        torch.set_num_threads(self.threads)
        self.temp.cleanup()

    def create(self, **kwargs):
        kwargs.setdefault("smoke", True)
        kwargs.setdefault("datasets", ["checkerboard"])
        kwargs["output_root"] = self.directory / "experiments"
        return runner.create_experiment(**kwargs)

    def test_full_default_protocol_matches_existing_nsot_templates(self):
        path, manifest = self.create(smoke=False, datasets=["checkerboard", "horse"])
        self.assertEqual(len(manifest["jobs"]), 2)
        self.assertEqual(manifest["evaluation_nfes"], [1, 2, 4, 8, 16, 32, 64, 128])
        for job in manifest["jobs"]:
            config = runner.verify_job(job)
            baseline_path = runner.ROOT / ("checkerboard_experiments/nsot.yaml" if job["dataset"] == "checkerboard"
                                          else "horse_experiments/horse_nsot_n256_seed0.yaml")
            baseline = runner._read_yaml(baseline_path)
            for key in ("model", "data", "training", "evaluation", "seed", "dtype", "device"):
                self.assertEqual(config[key], baseline[key])
            self.assertEqual(config[msw.METHOD]["superset_size"], 10000)
            self.assertEqual(config[msw.METHOD]["partition_counts"], [4, 20, 100])
            self.assertIsNone(job["bank"])
            self.assertIsNone(job["run"])
        self.assertTrue(path.is_file())

    def test_cache_key_tracks_every_preparation_dependency_but_not_nsot_beta(self):
        self.cache_patch.stop()
        try:
            multiscale = runner._read_yaml(runner.ROOT / "multiscale_experiments/checkerboard.yaml")
            baseline = runner._read_yaml(runner.ROOT / "checkerboard_experiments/nsot.yaml")
            for method, config, dependencies in (
                    (msw.METHOD, multiscale, ["multiscale_wasserstein.py", "nsot.py", "data.py", "train_horse.py"]),
                    ("nsot", baseline, ["nsot.py", "data.py", "train_horse.py"])):
                with self.subTest(method=method):
                    with patch.object(runner, "file_sha256", return_value="original") as hashes:
                        original = runner._cache_path(config, "checkerboard", method)
                    self.assertEqual({Path(call.args[0]).name for call in hashes.call_args_list}, set(dependencies))
                    for changed in dependencies:
                        with self.subTest(dependency=changed), patch.object(
                                runner, "file_sha256", side_effect=lambda path, changed=changed:
                                "changed" if Path(path).name == changed else "original"):
                            self.assertNotEqual(original, runner._cache_path(config, "checkerboard", method))
            with patch.object(runner, "file_sha256", return_value="original"):
                baseline["nsot"]["beta"] = 0.
                beta0 = runner._cache_path(baseline, "checkerboard", "nsot")
                baseline["nsot"]["beta"] = .2
                self.assertEqual(beta0, runner._cache_path(baseline, "checkerboard", "nsot"))
        finally:
            self.cache_patch.start()

    def test_default_cli_prepares_without_training_or_evaluating(self):
        original = runner.create_experiment

        def isolated_create(**kwargs):
            return original(**kwargs, output_root=self.directory / "experiments")

        with patch.object(sys, "argv", ["run_multiscale.py", "--smoke", "--datasets", "checkerboard"]), \
                patch.object(runner, "create_experiment", side_effect=isolated_create), \
                patch.object(runner, "train_jobs", side_effect=AssertionError("unexpected training")), \
                patch.object(runner, "evaluate_jobs", side_effect=AssertionError("unexpected evaluation")):
            runner.main()
        manifests = list((self.directory / "experiments").glob("multiscale_*/manifest.json"))
        self.assertEqual(len(manifests), 1)
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        job = manifest["jobs"][0]
        config = runner.verify_job(job)
        self.assertEqual(config[msw.METHOD]["superset_size"], 64)
        self.assertEqual(config["training"]["num_steps"], 2)
        self.assertEqual(manifest["evaluation_nfes"], [1, 2])
        self.assertIsNotNone(job["bank"])
        self.assertIsNone(job["run"])
        self.assertEqual(job["evaluations"], {})
        self.assertTrue(manifests[0].with_name("summary.json").is_file())

    def test_with_nsot_prepares_shared_cache_and_verifies_identical_endpoints(self):
        path, manifest = self.create(with_nsot=True)
        with patch.object(nsot, "exact_superset_permutation", wraps=nsot.exact_superset_permutation) as exact:
            runner.prepare_jobs(path, manifest)
        self.assertEqual(exact.call_count, 1)
        self.assertEqual(len(manifest["jobs"]), 3)
        self.assertTrue(manifest["endpoint_checks"]["checkerboard"]["same_source"])
        self.assertTrue(manifest["endpoint_checks"]["checkerboard"]["same_target"])
        jobs = {job["label"]: job for job in manifest["jobs"]}
        self.assertEqual(jobs["nsot_beta0"]["bank"]["path"], jobs["nsot_beta02"]["bank"]["path"])
        self.assertEqual(jobs["nsot_beta0"]["bank"]["sha256"], jobs["nsot_beta02"]["bank"]["sha256"])
        self.assertEqual(runner.verify_job(jobs["nsot_beta0"])["nsot"]["beta"], 0.)
        self.assertEqual(runner.verify_job(jobs["nsot_beta02"])["nsot"]["beta"], .2)
        # Prepared jobs are skipped, so this patch reaches only the endpoint verification.
        with patch.object(runner.np, "array_equal", return_value=False), \
                self.assertRaisesRegex(ValueError, "endpoints differ"):
            runner.prepare_jobs(path, manifest)

    def test_preparation_reuses_bank_and_keeps_input_yaml_immutable(self):
        path, manifest = self.create()
        initial = Path(manifest["jobs"][0]["config"])
        before = initial.read_bytes()
        runner.prepare_jobs(path, manifest)
        prepared = Path(manifest["jobs"][0]["config"])
        self.assertNotEqual(initial, prepared)
        self.assertEqual(initial.read_bytes(), before)
        saved = path.read_bytes()
        with patch.object(msw, "prepare", side_effect=AssertionError("prepared again")):
            runner.prepare_jobs(path, manifest)
        self.assertEqual(path.read_bytes(), saved)
        runner.verify_job(manifest["jobs"][0])

    def test_config_and_bank_hash_tampering_fail_fast(self):
        path, manifest = self.create()
        runner.prepare_jobs(path, manifest)
        job = manifest["jobs"][0]
        config_path = Path(job["config"])
        original = config_path.read_bytes()
        with config_path.open("ab") as stream:
            stream.write(b"\n# altered\n")
        with self.assertRaisesRegex(ValueError, "YAML changed"):
            runner.verify_job(job)
        with config_path.open("wb") as stream:
            stream.write(original)
        with Path(job["bank"]["path"]).open("ab") as stream:
            stream.write(b"altered")
        with self.assertRaisesRegex(ValueError, "bank changed"):
            runner.prepare_jobs(path, manifest)

    def test_nfe_resume_extends_axes_and_disallows_changed_training_settings(self):
        path, manifest = self.create()
        runner.prepare_jobs(path, manifest)
        config_path = manifest["jobs"][0]["config"]
        with patch.object(sys, "argv", ["run_multiscale.py", "--manifest", str(path), "--stage", "eval", "--nfe", "4"]), \
                patch.object(runner, "_cache_path", side_effect=AssertionError("resume must use saved cache paths")), \
                patch.object(runner, "evaluate_jobs") as evaluate:
            runner.main()
        self.assertEqual(evaluate.call_args.args[1]["evaluation_nfes"], [1, 2, 4])
        resumed = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(resumed["jobs"][0]["config"], config_path)
        with patch.object(sys, "argv", ["run_multiscale.py", "--manifest", str(path), "--steps", "3"]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main()
        with patch.object(sys, "argv", ["run_multiscale.py", "--stage", "eval"]), \
                patch.object(runner, "create_experiment", side_effect=AssertionError("should fail before preparation")), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main()

    def test_completed_training_and_evaluations_are_reused_and_summary_checks_hashes(self):
        path, manifest = self.create()
        runner.prepare_jobs(path, manifest)
        job = manifest["jobs"][0]
        run = self.directory / "fake_run"
        run.mkdir()
        config = runner.verify_job(job)
        with (run / "config.yaml").open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream)
        with (run / config["checkpoint"]).open("xb") as stream:
            stream.write(b"test checkpoint, not model weights")
        with (run / "training.json").open("x", encoding="utf-8") as stream:
            json.dump({"training_seconds": .3}, stream)
        import train
        with patch.object(train, "main", return_value=run) as train_mock:
            runner.train_jobs(path, manifest)
            runner.train_jobs(path, manifest)
        self.assertEqual(train_mock.call_count, 1)

        def fake_evaluate(config_path, nfe):
            image = self.directory / f"nfe_{nfe}.png"
            with image.open("xb") as stream:
                stream.write(b"test PNG placeholder")
            with image.with_suffix(".json").open("x", encoding="utf-8") as stream:
                json.dump({"chamfer": .1, "inference_seconds": .01, "nfe": nfe}, stream)
            return image

        import eval as evaluator
        with patch.object(evaluator, "main", side_effect=fake_evaluate) as evaluate:
            runner.evaluate_jobs(path, manifest)
            runner.evaluate_jobs(path, manifest)
        self.assertEqual(evaluate.call_count, 2)
        runner.write_summary(path, manifest)
        summary = json.loads(path.with_name("summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["jobs"][0]["training_seconds"], .3)
        self.assertEqual(len(summary["jobs"][0]["evaluations"]), 2)
        image = Path(job["evaluations"]["1"]["png"]["path"])
        with image.open("ab") as stream:
            stream.write(b"modified")
        with self.assertRaisesRegex(ValueError, "Evaluation artifact changed"):
            runner.write_summary(path, manifest)


if __name__ == "__main__":
    unittest.main()
