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

import multiscale_wasserstein as msw
import run_bank_mass_control as runner
import run_multiscale as base
from target_mass_control import TargetStrata, largest_remainder, sampling_settings


class TargetMassControlTests(unittest.TestCase):
    def config(self, dataset, size=64):
        config = base._read_yaml(base.ROOT / "multiscale_experiments" / f"{dataset}.yaml")
        config[msw.METHOD].update(superset_size=size, partition_counts=[2, 4, 8], cache_seed=13)
        config["data"]["n_points"] = 16
        return config

    def test_allocation_preserves_size_and_has_less_than_one_point_error(self):
        for size in (1, 7, 64, 10000):
            for p in ([0., .125, .125, .75], [.1] * 10, [.001, .399, .6]):
                a = largest_remainder(p, size, np.random.default_rng(10))
                b = largest_remainder(p, size, np.random.default_rng(10))
                np.testing.assert_array_equal(a, b)
                self.assertEqual(a.sum(), size)
                self.assertTrue((np.abs(a - size * np.array(p)) < 1 + 1e-12).all())
                self.assertTrue((a[np.array(p) == 0] == 0).all())
        for p in ([0., 0.], [-.1, 1.1], [np.nan, 1.], [[1.]], []):
            with self.assertRaises(ValueError):
                largest_remainder(p, 8, np.random.default_rng(0))

    def test_full_size_banks_are_reproducible_and_preserve_source_and_rng(self):
        for dataset in ("checkerboard", "horse"):
            with self.subTest(dataset=dataset):
                config = self.config(dataset, 10000)
                state = torch.get_rng_state().clone()
                iid_source, iid_target = msw.draw_supersets(config, dataset)
                config[msw.METHOD]["target_sampling"] = {"mode": "stratified", "bins": 16}
                source, target = msw.draw_supersets(config, dataset)
                repeated_source, repeated_target = msw.draw_supersets(config, dataset)
                np.testing.assert_array_equal(source, iid_source)
                np.testing.assert_array_equal(source, repeated_source)
                np.testing.assert_array_equal(target, repeated_target)
                torch.testing.assert_close(torch.get_rng_state(), state, rtol=0, atol=0)
                self.assertFalse(np.array_equal(target, iid_target))
                self.assertEqual(target.shape, (10000, 2))
                self.assertEqual(target.dtype, np.float32)
                strata = TargetStrata(config, dataset, 16)
                report = strata.report(target)
                self.assertEqual(report["leakage"], 0)
                self.assertLess(report["max_abs_count_error"], 1 + 1e-9)
                self.assertLess(report["region_mass_tv"], strata.report(iid_target)["region_mass_tv"])
                self.assertEqual(sum(report["counts"]), 10000)
                from metrics import checkerboard_metrics, horse_metrics
                points = torch.from_numpy(target)[None]
                if dataset == "horse":
                    from train_horse import load_horse_mask
                    self.assertEqual(horse_metrics(points, load_horse_mask("cpu", torch.float32), 16)[0], 0)
                    areas = np.array([len(p) for p in strata.pixels])
                    np.testing.assert_array_equal(areas.sum(), strata.mask.sum())
                    np.testing.assert_allclose(strata.probabilities, areas / areas.sum(), rtol=0, atol=0)
                else:
                    self.assertEqual(checkerboard_metrics(points, 4, 16)[0], 0)
                    self.assertEqual((strata.probabilities > 0).sum(), 128)
                    np.testing.assert_array_equal(strata.probabilities[strata.probabilities > 0], np.full(128, 1 / 128))

    def test_invalid_settings_fail_and_empty_regions_never_receive_points(self):
        for raw in (False, [], {"mode": "noise"}, {"mode": "iid", "bins": 4},
                    {"mode": "stratified", "bins": True}, {"mode": "stratified", "bins": 0},
                    {"mode": "stratified", "bins": 2.5}, {"mode": "stratified", "beta": .1}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                sampling_settings(raw)
        with self.assertRaises(ValueError):
            TargetStrata(self.config("checkerboard"), "checkerboard", 5)
        for dataset in ("checkerboard", "horse"):
            strata = TargetStrata(self.config(dataset), dataset, 16)
            report = strata.report(strata.sample(3, 42))
            self.assertEqual(sum(report["counts"]), 3)
            self.assertTrue((np.array(report["counts"])[strata.probabilities == 0] == 0).all())

    def test_cache_modes_cannot_be_confused_and_sampler_keeps_same_source_draws(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for dataset in ("checkerboard", "horse"):
                iid = self.config(dataset)
                iid[msw.METHOD]["cache"] = str(Path(tmp) / f"{dataset}_iid.npz")
                controlled = copy.deepcopy(iid)
                controlled[msw.METHOD].update(cache=str(Path(tmp) / f"{dataset}_stratified.npz"),
                                            target_sampling={"mode": "stratified", "bins": 16})
                msw.prepare(iid, dataset)
                path, meta = msw.prepare(controlled, dataset)
                self.assertIn("target_sampling", meta)
                self.assertIn("target_mass_control.py", meta["source_sha256"])
                before = path.read_bytes()
                with patch.object(msw, "multiscale_permutation", side_effect=AssertionError("recomputed")):
                    msw.prepare(controlled, dataset)
                self.assertEqual(before, path.read_bytes())
                for original, wrong in ((iid, controlled), (controlled, iid)):
                    changed = copy.deepcopy(original)
                    changed[msw.METHOD]["cache"] = wrong[msw.METHOD]["cache"]
                    with self.assertRaisesRegex(ValueError, "target_sampling"):
                        msw.load_cache(changed, dataset)
                samplers = [msw.MultiscaleWassersteinSampler(c, dataset, "cpu", torch.float32) for c in (iid, controlled)]
                samples = [s.sample(8, generator=torch.Generator().manual_seed(99)) for s in samplers]
                torch.testing.assert_close(samples[0][0], samples[1][0], rtol=0, atol=0)
                self.assertEqual(samplers[1].details()["beta"], 0.)
                self.assertFalse(torch.equal(samples[0][1], samples[1][1]))

    def test_cache_key_is_separate_and_tracks_sampling_code(self):
        config = self.config("horse")
        original = base._cache_path(config, "horse", msw.METHOD)
        config[msw.METHOD]["target_sampling"] = {"mode": "stratified", "bins": 16}
        self.assertNotEqual(base._cache_path(config, "horse", msw.METHOD), original)
        with patch.object(base, "file_sha256", return_value="old"):
            before = base._cache_path(config, "horse", msw.METHOD)
        with patch.object(base, "file_sha256", side_effect=lambda p: "new" if Path(p).name == "target_mass_control.py" else "old"):
            self.assertNotEqual(before, base._cache_path(config, "horse", msw.METHOD))


class MassControlRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.cache_patch = patch.object(base, "_cache_path", side_effect=lambda config, dataset, method:
            self.directory / f"{dataset}_{config[method].get('target_sampling', {}).get('mode', 'iid')}.npz")
        self.cache_patch.start()
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()

    def tearDown(self):
        self.output.__exit__(None, None, None)
        self.cache_patch.stop()
        torch.set_num_threads(self.threads)
        self.temp.cleanup()

    def create(self, **kwargs):
        kwargs.setdefault("smoke", True)
        path, manifest = runner.create_experiment(output_root=self.directory / "plans", **kwargs)
        manifest["results_directory"] = str(self.directory / "results" / path.parent.name)
        base.save_manifest(path, manifest)
        return path, manifest

    def test_pairing_protocol_and_audit_do_not_train_or_match(self):
        path, manifest = self.create(smoke=False)
        self.assertEqual(len(manifest["jobs"]), 4)
        for pair, configs in runner.paired_jobs(manifest):
            self.assertEqual(configs[0][msw.METHOD]["superset_size"], 10000)
            self.assertEqual(configs[0][msw.METHOD]["partition_counts"], [4, 20, 100])
            self.assertIsNone(pair[0]["run"])
        with patch.object(msw, "multiscale_permutation", side_effect=AssertionError("unexpected matching")), \
                patch.object(base, "train_jobs", side_effect=AssertionError("unexpected training")):
            audit = runner.audit_banks(path, manifest)
            before = Path(manifest["bank_audit"]["path"]).read_bytes()
            self.assertEqual(audit, runner.audit_banks(path, manifest))
            self.assertEqual(before, Path(manifest["bank_audit"]["path"]).read_bytes())
        self.assertTrue(all(v["same_source"] for v in audit["datasets"].values()))
        self.assertTrue(all(job["bank"] is None for job in manifest["jobs"]))

    def test_prepare_resume_preserves_configs_and_checks_audited_endpoints(self):
        path, manifest = self.create(seeds=[0, 1])
        audit = runner.audit_banks(path, manifest)
        originals = {j["config"]: Path(j["config"]).read_bytes() for j in manifest["jobs"]}
        with patch.object(msw, "multiscale_permutation", wraps=msw.multiscale_permutation) as matching:
            base.prepare_jobs(path, manifest)
        self.assertEqual(matching.call_count, 4)  # Shared across training seeds.
        runner.verify_prepared_pairs(path, manifest, audit)
        self.assertEqual(len(manifest["endpoint_checks"]), 4)
        for config, content in originals.items():
            self.assertEqual(Path(config).read_bytes(), content)
        with patch.object(msw, "prepare", side_effect=AssertionError("recomputed")):
            base.prepare_jobs(path, manifest)
        corrupted = copy.deepcopy(audit)
        corrupted["datasets"]["horse"]["conditions"][runner.LABELS[1]]["endpoint_sha256"]["target"] = "bad"
        with self.assertRaisesRegex(ValueError, "audited endpoints"):
            runner.verify_prepared_pairs(path, manifest, corrupted)

    def test_small_pipeline_uses_original_evaluators_and_can_resume(self):
        path, manifest = self.create()
        # Keep real smoke training/rendering inside the test directory.
        import os
        @contextlib.contextmanager
        def isolated_directory():
            previous = Path.cwd()
            try:
                os.chdir(self.directory)
                yield
            finally:
                os.chdir(previous)
        with patch.object(base, "project_directory", isolated_directory):
            runner.main(["--manifest", str(path), "--stage", "all"])
            with patch.object(msw, "prepare", side_effect=AssertionError("recomputed")), \
                    patch("train.main", side_effect=AssertionError("retrained")), \
                    patch("train_horse.main", side_effect=AssertionError("retrained")), \
                    patch("eval.main", side_effect=AssertionError("reevaluated")), \
                    patch("eval_horse.main", side_effect=AssertionError("reevaluated")):
                runner.main(["--manifest", str(path), "--stage", "all"])
        completed = json.loads(path.read_text(encoding="utf-8"))
        for job in completed["jobs"]:
            self.assertIsNotNone(job["run"])
            self.assertEqual(set(job["evaluations"]), {"1", "2"})
            for result in job["evaluations"].values():
                self.assertTrue(Path(result["png"]["path"]).is_file())
        summary = json.loads((Path(completed["results_directory"]) / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["paired_mass_control"])
        self.assertTrue((Path(completed["results_directory"]) / "comparison.csv").is_file())
        rows = summary["paired_mass_control"]
        for row in rows:
            self.assertAlmostEqual(row["delta"], row["stratified"] - row["iid"])

    def test_resuming_rejects_setting_changes_and_unrelated_manifests(self):
        path, manifest = self.create()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(["--manifest", str(path), "--bins", "8"])
        manifest["experiment_type"] = "wrong"
        base.save_manifest(path, manifest)
        with self.assertRaisesRegex(ValueError, "Not a mass-control"):
            runner.main(["--manifest", str(path)])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(["--stage", "eval"])


if __name__ == "__main__":
    unittest.main()
