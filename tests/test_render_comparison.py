import contextlib
import copy
import io
import json
import os
from pathlib import Path, PureWindowsPath
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from experiment import save_training
from model import PointSetTransformer
from render_comparison import (DEFAULT_NFES, compare, main, point_density, predict_bank,
                               preflight, read_manifest)
from train_horse import HorsePointSetTransformer


class ComparisonTests(unittest.TestCase):
    def test_shipped_manifest_is_portable_and_complete(self):
        path = Path(__file__).resolve().parents[1] / "comparison_runs.yaml"
        manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(set(manifest["datasets"]), {"checkerboard", "horse"})
        for dataset, entries in manifest["datasets"].items():
            self.assertEqual([e["id"] for e in entries],
                             ["independent", "efm", "minibatch_ot", "nsot", "tg", "ms_w2"])
            for entry in entries:
                value = entry["config"]
                self.assertTrue(value.startswith(f"./runs/{dataset}/"))
                self.assertTrue(value.endswith("/config.yaml"))
                self.assertFalse(Path(value).is_absolute())
                self.assertFalse(PureWindowsPath(value).drive)
                self.assertNotIn("REPLACE_", value)
                self.assertIn("_n256_seed0_", value)

    def setUp(self):
        self.previous = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        os.chdir(self.temp.name)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()

    def tearDown(self):
        self.stdout.__exit__(None, None, None)
        torch.set_num_threads(self.threads)
        os.chdir(self.previous)
        self.temp.cleanup()

    def fixture(self, dataset, method, **overrides):
        torch.manual_seed(7)
        config = {
            "seed": 0, "device": "cpu", "dtype": "float32", "coupling": method,
            "data": {"batch_size": 2, "n_points": 8},
            "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                      "dim_feedforward": 16, "dropout": 0.},
            "training": {"num_steps": 2}, "checkpoint": "model.pt",
        }
        if dataset == "checkerboard":
            config["data"]["grid_size"] = 4
        if method == "nsot":
            config["nsot"] = {"cache": "absent_nsot.npz", "superset_size": 16, "cache_seed": 0, "beta": .2}
        if method == "multiscale_wasserstein":
            config["multiscale_wasserstein"] = {"cache": "absent_ms.npz", "superset_size": 16,
                "cache_seed": 0, "partition_counts": [2, 4], "scale_weights": [1., 1.], "max_cost_mib": 512}
        if method == "target_guided_cached":
            config["num_regions"] = 2
        config.update(overrides)
        cls = HorsePointSetTransformer if dataset == "horse" else PointSetTransformer
        model = cls(**config["model"]).to(dtype=getattr(torch, config["dtype"]))
        run = save_training(model, config, dataset, "fixture.yaml", 0., 0.)
        return {"id": method, "label": method, "coupling": method, "config": str(run / "config.yaml")}

    def manifest(self, groups, name="comparison.yaml"):
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"version": 1, "datasets": groups}), encoding="utf-8")
        return path

    def test_pipeline_six_methods_both_datasets_without_training_banks(self):
        methods = ["independent", "equivariant_ot_permutation", "minibatch_ot", "nsot",
                   "target_guided_cached", "multiscale_wasserstein"]
        groups = {d: [self.fixture(d, m) for m in methods] for d in ("checkerboard", "horse")}
        manifest = self.manifest(groups)
        before = {str(p): p.read_bytes() for p in Path("runs").rglob("*") if p.is_file()}
        with patch("nsot.load_cache", side_effect=AssertionError("No NSOT bank access")), \
                patch("multiscale_wasserstein.load_cache", side_effect=AssertionError("No MS bank access")), \
                patch("tg_cache.TGCachedPairSampler", side_effect=AssertionError("No TG bank access")):
            directory = compare(manifest, nfes=[1, 2], clouds=2, batch_size=1, bins=16,
                                histogram_bins=8, dpi=40, view="both", device="cpu")
        root = json.loads((directory / "comparison.json").read_text())
        self.assertEqual(root["status"], "complete")
        sources = []
        for dataset in groups:
            dest = directory / dataset
            report = json.loads((dest / "comparison.json").read_text())
            self.assertEqual([e["id"] for e in report["models"]], methods)
            self.assertEqual(len(report["results"]), 12)
            self.assertEqual(len(report["rendering"]["images"]), 6)
            self.assertTrue((dest / "metrics.csv").is_file())
            for name in report["rendering"]["images"]:
                self.assertGreater((dest / name).stat().st_size, 100)
            with np.load(dest / "clouds.npz", allow_pickle=False) as arrays:
                self.assertEqual(arrays["source"].shape, (2, 8, 2))
                self.assertEqual(len(arrays.files), 14)
                sources.append(arrays["source"].copy())
                # Same architecture and deliberately identical fixture weights:
                # coupling must NOT be reapplied or cached points used at inference.
                for nfe in [1, 2]:
                    for method in methods[1:]:
                        np.testing.assert_array_equal(arrays[f"prediction_independent_nfe_{nfe}"],
                                                      arrays[f"prediction_{method}_nfe_{nfe}"])
                self.assertFalse(np.array_equal(arrays["source"], arrays["target"]))
        np.testing.assert_array_equal(*sources)
        self.assertEqual(before, {str(p): p.read_bytes() for p in Path("runs").rglob("*") if p.is_file()})
        self.assertFalse(Path("absent_ms.npz").exists())

    def test_same_draws_under_reordering_and_default_nfes_and_check_only(self):
        entries = [self.fixture("checkerboard", m) for m in ("independent", "minibatch_ot")]
        manifest = self.manifest({"checkerboard": entries})
        self.assertIsNone(main([str(manifest), "--check-only"]))
        self.assertFalse(Path("analysis_results").exists())
        options = dict(clouds=2, batch_size=2, bins=8, histogram_bins=8, dpi=30, view="scatter", device="cpu")
        # Rendering is covered above; avoid generating all 8 images twice here.
        with patch("render_comparison.render_dataset", return_value={}):
            first = compare(manifest, **options)
            reversed_path = self.manifest({"checkerboard": list(reversed(entries))}, "reversed.yaml")
            second = compare(reversed_path, **options)
        reports = [json.loads((d / "checkerboard/comparison.json").read_text()) for d in (first, second)]
        self.assertEqual(reports[0]["nfes"], [1, 2, 4, 8, 16, 32, 128])
        self.assertEqual(reports[0]["nfes"], DEFAULT_NFES)
        for key in ("source_sha256", "target_sha256"):
            self.assertEqual(reports[0][key], reports[1][key])
        values = [{(r["method_id"], r["nfe"]): (r["prediction_sha256"], r["metrics"]) for r in p["results"]}
                  for p in reports]
        self.assertEqual(*values)

    def test_each_nfe_restarts_at_source_and_reaches_t_one(self):
        class LinearVelocity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
                self.calls = []

            def forward(self, x, t):
                self.calls.append(t.detach().clone())
                return x * self.weight

        model = LinearVelocity()
        source = torch.ones(3, 4, 2)
        one, _ = predict_bank(model, source, 1, 2)
        two, _ = predict_bank(model, source, 2, 1)
        torch.testing.assert_close(one, source * 2.)
        torch.testing.assert_close(two, source * 2.25)
        torch.testing.assert_close(source, torch.ones_like(source))
        self.assertEqual(sum(t.numel() for t in model.calls), 3 * (1 + 2))
        self.assertEqual(sorted({float(t.flatten()[0]) for t in model.calls}), [0., .5])
        with torch.no_grad():
            model.weight.fill_(float("nan"))
        with self.assertRaises(FloatingPointError):
            predict_bank(model, source, 1, 1)

    def test_density_uses_all_points_and_does_not_renormalize_outliers(self):
        one = np.array([[0., 0.]])
        first, outside = point_density(one, 4, [-1., 1.], 0.)
        second, fraction = point_density(np.array([[0., 0.], [10., 10.]]), 4, [-1., 1.], 0.)
        self.assertEqual(outside, 0.)
        self.assertEqual(fraction, .5)
        np.testing.assert_array_equal(second, first / 2)
        self.assertAlmostEqual(first.sum() * .25, 1.)

    def test_manifest_resolution_and_rejections(self):
        entry = self.fixture("checkerboard", "independent")
        nested = copy.deepcopy(entry)
        nested["config"] = "../" + nested["config"]
        manifest = self.manifest({"checkerboard": [nested]}, "nested/list.yaml")
        self.assertEqual(read_manifest(manifest)["checkerboard"][0]["config_path"], str(Path(entry["config"]).resolve()))
        bad_cases = [dict(entry, config="missing.yaml"), dict(entry, id="../escape"),
                     dict(entry, coupling="not_a_method"), dict(entry, unknown=1)]
        for bad in bad_cases:
            with self.subTest(bad=bad), self.assertRaises((ValueError, FileNotFoundError)):
                read_manifest(self.manifest({"checkerboard": [bad]}))
        with self.assertRaisesRegex(ValueError, "unique"):
            read_manifest(self.manifest({"checkerboard": [entry, entry]}))
        with self.assertRaisesRegex(ValueError, "duplicate config"):
            read_manifest(self.manifest({"checkerboard": [entry, dict(entry, id="other")]}))

    def test_preflight_rejects_mismatch_before_inference(self):
        first = self.fixture("checkerboard", "independent")
        wrong_method = dict(first, coupling="multiscale_wasserstein")
        with self.assertRaisesRegex(ValueError, "expected coupling"):
            compare(self.manifest({"checkerboard": [wrong_method]}), check_only=True)
        for data in ({"batch_size": 2, "n_points": 16, "grid_size": 4},
                     {"batch_size": 2, "n_points": 8, "grid_size": 8}):
            second = self.fixture("checkerboard", "minibatch_ot", data=data)
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "matching n_points"):
                compare(self.manifest({"checkerboard": [first, second]}), check_only=True)
        with self.assertRaisesRegex(ValueError, "dataset"):
            compare(self.manifest({"horse": [first]}), check_only=True)
        # Editing a saved YAML without retraining must not pass provenance checks.
        path = Path(first["config"])
        config = yaml.safe_load(path.read_text())
        config["seed"] = 111
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "YAML differs"):
            compare(self.manifest({"checkerboard": [first]}), check_only=True)
        self.assertFalse(Path("analysis_results").exists())

    def test_legacy_opt_in_and_training_confounds_visible(self):
        first = self.fixture("checkerboard", "independent")
        second = self.fixture("checkerboard", "minibatch_ot", seed=1, training={"num_steps": 10})
        path = self.manifest({"checkerboard": [first, second]})
        groups, warnings = preflight(read_manifest(path))
        self.assertTrue(any("seed differs" in w for w in warnings))
        self.assertTrue(any("training differs" in w for w in warnings))
        checkpoint = Path(groups["checkerboard"][0]["checkpoint"])
        payload = torch.load(checkpoint, weights_only=True)
        torch.save(payload["model_state_dict"], checkpoint)
        with self.assertWarns(UserWarning), self.assertRaisesRegex(ValueError, "unverified legacy"):
            preflight(read_manifest(path))
        with self.assertWarns(UserWarning):
            allowed, warnings = preflight(read_manifest(path), allow_legacy=True)
        self.assertFalse(allowed["checkerboard"][0]["training_config_verified"])
        self.assertTrue(any("LEGACY UNVERIFIED" in w for w in warnings))

    def test_argument_validation_before_opening_files(self):
        for options in ({"nfes": [1, 1]}, {"nfes": [0]}, {"clouds": 0}, {"batch_size": 0},
                        {"cloud_index": 64}, {"sigma": -1}, {"gamma": float("nan")}, {"seed": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                compare("does_not_exist.yaml", **options)


if __name__ == "__main__":
    unittest.main()
