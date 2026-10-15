"""Tiny real-cache/checkpoint integration checks for the four-way audit."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from audit_ms_w2_coupling import audit
from experiment import save_training
from model import PointSetTransformer
import multiscale_wasserstein as msw
from train_horse import HorsePointSetTransformer


class CouplingAuditIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.previous = Path.cwd()
        self.temporary = tempfile.TemporaryDirectory()
        os.chdir(self.temporary.name)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()

    def tearDown(self):
        self.stdout.__exit__(None, None, None)
        torch.set_num_threads(self.threads)
        os.chdir(self.previous)
        self.temporary.cleanup()

    @staticmethod
    def config(dataset="checkerboard", cache_seed=5):
        config = {
            "seed": 0, "device": "cpu", "dtype": "float32", "coupling": msw.METHOD,
            "multiscale_wasserstein": {
                "cache": f"{dataset}_{cache_seed}.npz", "superset_size": 16,
                "cache_seed": cache_seed, "partition_counts": [2, 4],
                "scale_weights": [1., 1.], "max_cost_mib": 512,
            },
            "data": {"batch_size": 2, "n_points": 4},
            "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                      "dim_feedforward": 16, "dropout": 0.0},
            "training": {"num_steps": 2, "learning_rate": .001,
                         "weight_decay": .01, "log_every": 1},
            "evaluation": {"batch_size": 2, "histogram_bins": 8},
            "checkpoint": f"{dataset}_multiscale.pt",
        }
        if dataset == "checkerboard":
            config["data"]["grid_size"] = 4
        return config

    def fixture(self, dataset="checkerboard", *, source_partition=None, target_partition=None):
        torch.manual_seed(7)
        config = self.config(dataset)
        if source_partition is not None:
            config[msw.METHOD]["source_partition"] = source_partition
        if target_partition is not None:
            config[msw.METHOD]["target_partition"] = target_partition
        bank, metadata = msw.prepare(config, dataset)
        digest = self.sha256(bank)
        cls = HorsePointSetTransformer if dataset == "horse" else PointSetTransformer
        model = cls(**config["model"])
        run = save_training(model, config, dataset, "fixture.yaml", 0., 0.,
                            coupling_metadata={**metadata, "cache_sha256": digest})
        return run / "config.yaml", run / config["checkpoint"], bank

    def test_pca_checkpoint_audit_uses_the_actual_source_partition(self):
        config, _, _ = self.fixture(source_partition="pca", target_partition="pca")
        with patch.object(msw, "balanced_partitions", wraps=msw.balanced_partitions) as partitions:
            directory = audit(config, "checkerboard", **self.options())
        partitions.assert_called_once()
        self.assertEqual(partitions.call_args.kwargs["strategy"], "pca")
        report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["bank_metadata"]["partition_settings"]["source_partition"], "pca")

    @staticmethod
    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def options():
        return dict(clouds=2, batch_size=1, neighbors=2, distance_bins=3,
                    routing_draws=32, routing_neighbors=(1, 2), routing_chunk_size=7,
                    anchors=4, contexts=2, nfes=(1, 2), path_steps=2,
                    times=(0., .25, .5, .9, 1.), grid_bins=4, seed=2026,
                    skip_render=True)

    def assert_finite_json(self, item):
        if isinstance(item, dict):
            for value in item.values():
                self.assert_finite_json(value)
        elif isinstance(item, list):
            for value in item:
                self.assert_finite_json(value)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            self.assertTrue(np.isfinite(item), item)

    def test_all_four_diagnostics_both_datasets_and_inputs_immutable(self):
        for dataset in ("checkerboard", "horse"):
            with self.subTest(dataset=dataset):
                config, checkpoint, bank = self.fixture(dataset)
                before = {path: self.sha256(path) for path in (config, checkpoint, bank)}
                with patch("multiscale_wasserstein.prepare", side_effect=AssertionError("No bank rebuilding")), \
                        patch("train.train_model", side_effect=AssertionError("No training")):
                    directory = audit(config, dataset, **self.options())
                report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(report["format_version"], 1)
                self.assertEqual(set(report["diagnostics"]), {
                    "1_geometry", "2_gaussian_routing", "3_fitting", "4_generation"})
                self.assertEqual(report["bank_sha256"], before[bank])
                self.assertTrue(report["input_hashes"])
                self.assertTrue(report["spatial_definition"])
                self.assert_finite_json(report)
                for prefix in ("geometry_", "routing_", "fitting_", "generation_"):
                    self.assertTrue(list(directory.glob(prefix + "*.csv")), prefix)
                with np.load(directory / "diagnostics.npz", allow_pickle=False) as arrays:
                    self.assertTrue(arrays.files)
                    for key in arrays.files:
                        if arrays[key].dtype.kind in "biufc":
                            self.assertTrue(np.isfinite(arrays[key]).all(), key)
                self.assertEqual(before, {path: self.sha256(path) for path in before})
                self.assertFalse(list(directory.glob("*.png")))

    def test_check_only_validates_without_creating_output(self):
        config, checkpoint, bank = self.fixture()
        before = {path: self.sha256(path) for path in (config, checkpoint, bank)}
        with patch("multiscale_wasserstein.prepare", side_effect=AssertionError("No bank rebuilding")), \
                patch("train.train_model", side_effect=AssertionError("No training")):
            result = audit(config, "checkerboard", check_only=True,
                           output="check_only_output", **self.options())
        self.assertIsNone(result)
        self.assertFalse(Path("check_only_output").exists())
        self.assertEqual(before, {path: self.sha256(path) for path in before})

    def test_relocated_identical_bank_works_and_different_bank_is_rejected(self):
        config, _, original = self.fixture()
        relocated = Path("relocated") / original.name
        relocated.parent.mkdir()
        shutil.copy2(original, relocated)
        original_digest = self.sha256(original)
        original.unlink()  # proves the override is used, not the stale config path
        with patch("multiscale_wasserstein.prepare", side_effect=AssertionError("No bank rebuilding")):
            directory = audit(config, "checkerboard", bank=relocated, **self.options())
        report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(report["bank_sha256"], original_digest)
        self.assertEqual(self.sha256(relocated), original_digest)
        wrong, _ = msw.prepare(self.config(cache_seed=6), "checkerboard")
        with patch("multiscale_wasserstein.prepare", side_effect=AssertionError("No bank rebuilding")), \
                self.assertRaises(ValueError):
            audit(config, "checkerboard", bank=wrong,
                  output="rejected_output", **self.options())
        self.assertFalse(Path("rejected_output").exists())

    def test_fixed_seed_reproduces_all_saved_numeric_arrays(self):
        config, _, _ = self.fixture()
        outputs = [audit(config, "checkerboard", **self.options()) for _ in range(2)]
        self.assertNotEqual(outputs[0], outputs[1])
        first = json.loads((outputs[0] / "summary.json").read_text(encoding="utf-8"))
        second = json.loads((outputs[1] / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(first["input_hashes"], second["input_hashes"])
        with np.load(outputs[0] / "diagnostics.npz", allow_pickle=False) as a, \
                np.load(outputs[1] / "diagnostics.npz", allow_pickle=False) as b:
            self.assertEqual(set(a.files), set(b.files))
            for name in a.files:
                np.testing.assert_array_equal(a[name], b[name], err_msg=name)


if __name__ == "__main__":
    unittest.main()
