"""Runner orchestration tests: no real prior fitting, OT, training or CUDA work."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_prior_comparison as runner


class PriorComparisonRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.previous_cwd = Path.cwd()
        os.chdir(self.root)
        self.calls = {"prepare": [], "train": [], "eval": []}
        self.training_failure_at = None
        for _, _, template in runner.JOBS:
            path = self.root / template
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("template: true\n", encoding="utf-8")

        def read_config(template):
            path = Path(template)
            self.assertTrue(path.is_file())
            return {"nsot": {"cache": str(self.root / "caches" / f"{path.stem}.npz")}}

        def prepare(config, dataset):
            self.calls["prepare"].append((config, dataset))
            cache = Path(config["nsot"]["cache"])
            if not cache.exists():
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(b"mock cache")
            return cache, {"precompute_seconds": 12.5, "precompute_timing_scope": "mock sampling and OT"}

        def train(dataset, template, *, steps):
            self.calls["train"].append((dataset, template, steps))
            index = len(self.calls["train"])
            if self.training_failure_at == index:
                raise RuntimeError("mock training failure")
            # Deliberately not derivable from an experiment template prefix.
            directory = Path("runs") / dataset / f"returned_actual_run_{index}"
            directory.mkdir(parents=True)
            (directory / "config.yaml").write_text("saved: true\n", encoding="utf-8")
            (directory / "training.json").write_text("{}", encoding="utf-8")
            return directory

        def evaluate(dataset, config, nfe, **kwargs):
            self.calls["eval"].append((dataset, config, nfe, kwargs))
            self.assertTrue(Path(config).is_file())
            self.assertIn("returned_actual_run_", config)
            directory = self.root / "eval_results" / dataset
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / f"result_{len(self.calls['eval'])}_nfe_{nfe}.png"
            output.write_bytes(b"mock PNG")
            output.with_suffix(".json").write_text("{}", encoding="utf-8")
            return output

        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.object(runner, "ROOT", self.root))
        self.stack.enter_context(patch.object(runner, "release_memory"))
        self.stack.enter_context(patch.dict(sys.modules, {
            "experiment": SimpleNamespace(read_config=read_config),
            "nsot": SimpleNamespace(prepare=prepare),
            "train": SimpleNamespace(main=lambda path, **kw: train("checkerboard", path, **kw)),
            "train_horse": SimpleNamespace(main=lambda path, **kw: train("horse", path, **kw)),
            "eval": SimpleNamespace(main=lambda path, nfe, **kw: evaluate("checkerboard", path, nfe, **kw)),
            "eval_horse": SimpleNamespace(main=lambda path, nfe, **kw: evaluate("horse", path, nfe, **kw)),
        }))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def tearDown(self):
        self.stack.close()
        os.chdir(self.previous_cwd)
        self.temp.cleanup()

    def read_manifest(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def manifests(self):
        return list((self.root / "comparison_results").glob("*/manifest.json"))

    def test_all_captures_actual_returned_paths_steps_and_evaluations(self):
        path = runner.main(["--steps", "37", "--nfes", "1", "4"])
        payload = self.read_manifest(path)
        self.assertEqual(payload["training_steps"], 37)
        self.assertEqual(len(payload["jobs"]), 4)
        self.assertEqual(len(self.calls["prepare"]), 4)
        self.assertEqual(len(self.calls["train"]), 4)
        self.assertEqual(len(self.calls["eval"]), 8)
        self.assertIn("completed_at", payload)
        for index, job in enumerate(payload["jobs"], start=1):
            expected = self.root / "runs" / job["dataset"] / f"returned_actual_run_{index}" / "config.yaml"
            self.assertEqual(Path(job["saved_config"]), expected)
            self.assertTrue(Path(job["training_json"]).is_file())
            self.assertEqual([row["nfe"] for row in job["evaluations"]], [1, 4])
            self.assertTrue(all(Path(row["json"]).is_file() for row in job["evaluations"]))
            self.assertEqual(job["preparation"]["original_precompute_seconds"], 12.5)
            self.assertFalse(job["preparation"]["cache_reused"])
        self.assertEqual([call[2] for call in self.calls["train"]], [37] * 4)
        self.assertEqual([call[0] for call in self.calls["train"]],
                         ["checkerboard", "checkerboard", "horse", "horse"])
        self.assertEqual([call[1] for call in self.calls["eval"]],
                         [job["saved_config"] for job in payload["jobs"] for _ in (1, 4)])

    def test_prepare_only_does_not_train_or_evaluate_and_reports_cache_reuse(self):
        first = self.read_manifest(runner.main(["--stage", "prepare"]))
        self.assertEqual(self.calls["train"], [])
        self.assertEqual(self.calls["eval"], [])
        self.assertTrue(all("saved_config" not in job for job in first["jobs"]))
        second = self.read_manifest(runner.main(["--stage", "prepare"]))
        self.assertTrue(all(job["preparation"]["cache_reused"] for job in second["jobs"]))
        self.assertEqual(len(self.manifests()), 2)

    def test_train_stage_also_prepares_but_does_not_evaluate(self):
        payload = self.read_manifest(runner.main(["--stage", "train", "--steps", "5"]))
        self.assertEqual(len(self.calls["prepare"]), 4)
        self.assertEqual(len(self.calls["train"]), 4)
        self.assertEqual(self.calls["eval"], [])
        self.assertTrue(all(Path(job["saved_config"]).is_file() for job in payload["jobs"]))
        self.assertTrue(all("evaluations" not in job for job in payload["jobs"]))

    def test_eval_only_reuses_exact_configs_and_preserves_parent_bytes(self):
        parent = runner.main(["--stage", "train", "--steps", "5"])
        original = parent.read_bytes()
        saved_configs = [job["saved_config"] for job in self.read_manifest(parent)["jobs"]]
        roi = self.root / "fixed_rois.json"
        roi.write_text("{}", encoding="utf-8")
        path = runner.main(["--stage", "eval", "--manifest", str(parent),
                            "--steps", "999", "--nfes", "2", "8", "--roi-file", str(roi)])
        self.assertNotEqual(path, parent)
        self.assertEqual(parent.read_bytes(), original)
        payload = self.read_manifest(path)
        self.assertEqual(payload["parent_manifest"], str(parent))
        self.assertEqual(payload["training_steps"], 5)
        self.assertEqual([job["saved_config"] for job in payload["jobs"]], saved_configs)
        self.assertEqual(len(self.calls["prepare"]), 4)  # Only the earlier train stage.
        self.assertEqual(len(self.calls["train"]), 4)
        self.assertEqual([call[1] for call in self.calls["eval"]],
                         [config for config in saved_configs for _ in (2, 8)])
        horse_calls = [call for call in self.calls["eval"] if call[0] == "horse"]
        self.assertTrue(all(call[3]["roi_file"] == str(roi) for call in horse_calls))

    def test_partial_training_failure_retains_completed_config_for_eval(self):
        self.training_failure_at = 2
        with self.assertRaisesRegex(RuntimeError, "mock training failure"):
            runner.main(["--steps", "3", "--nfes", "1"])
        parent, = self.manifests()
        payload = self.read_manifest(parent)
        self.assertNotIn("completed_at", payload)
        self.assertTrue(Path(payload["jobs"][0]["saved_config"]).is_file())
        self.assertTrue(all("saved_config" not in job for job in payload["jobs"][1:]))
        self.assertEqual(self.calls["eval"], [])
        original = parent.read_bytes()
        path = runner.main(["--stage", "eval", "--manifest", str(parent), "--nfes", "1"])
        result = self.read_manifest(path)
        self.assertEqual(len(result["jobs"]), 1)
        self.assertEqual(result["jobs"][0]["saved_config"], payload["jobs"][0]["saved_config"])
        self.assertEqual(len(self.calls["eval"]), 1)
        self.assertEqual(parent.read_bytes(), original)

    def test_rejects_missing_manifest_flag_without_creating_outputs(self):
        with self.assertRaises(SystemExit):
            runner.main(["--stage", "eval"])
        self.assertEqual(self.manifests(), [])

    def test_rejects_nonexistent_manifest_before_creating_outputs(self):
        with self.assertRaises(FileNotFoundError):
            runner.main(["--stage", "eval", "--manifest", "not_present.json"])
        self.assertEqual(self.manifests(), [])

    def test_rejects_manifest_for_non_eval_stage(self):
        with self.assertRaises(SystemExit):
            runner.main(["--stage", "train", "--manifest", "unused.json"])
        self.assertEqual(self.manifests(), [])

    def test_rejects_wrong_cwd_without_starting_any_job(self):
        wrong = self.root / "different_directory"
        wrong.mkdir()
        os.chdir(wrong)
        with self.assertRaises(SystemExit):
            runner.main([])
        self.assertTrue(all(not calls for calls in self.calls.values()))
        self.assertEqual(self.manifests(), [])

    def test_rejects_duplicate_nfes_without_starting_any_job(self):
        with self.assertRaises(SystemExit):
            runner.main(["--nfes", "1", "1"])
        self.assertTrue(all(not calls for calls in self.calls.values()))
        self.assertEqual(self.manifests(), [])

    def test_rejects_eval_manifest_with_no_completed_training(self):
        parent = runner.main(["--stage", "prepare"])
        original = parent.read_bytes()
        with self.assertRaises(SystemExit):
            runner.main(["--stage", "eval", "--manifest", str(parent)])
        self.assertEqual(parent.read_bytes(), original)
        self.assertEqual(len(self.manifests()), 1)

    def test_failed_manifest_serialization_preserves_previous_recovery_record(self):
        path = runner.main(["--stage", "prepare"])
        original = path.read_bytes()
        payload = self.read_manifest(path)
        payload["not_valid_json"] = float("nan")
        with self.assertRaises(ValueError):
            runner.save_manifest(path, payload)
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.read_manifest(path)["format_version"], 1)


if __name__ == "__main__":
    unittest.main()
