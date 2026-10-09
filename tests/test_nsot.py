import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import cdist
import torch
import yaml

import audit_generation
import coupling
from diagnostic_data import DiagnosticData, tensor_sha256
import eval as eval_checkerboard
import eval_horse
from experiment import load_model
from model import PointSetTransformer
import nsot
import prepare_nsot
import train
import train_horse


class NSOTTests(unittest.TestCase):
    def setUp(self):
        self.previous = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        os.chdir(self.temp.name)
        self.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        os.chdir(self.previous)
        self.temp.cleanup()
        torch.set_num_threads(self.threads)

    @staticmethod
    def config(dataset="checkerboard"):
        config = {
            "seed": 0, "device": "cpu", "dtype": "float32", "coupling": "nsot",
            "nsot": {"cache": f"{dataset}.npz", "superset_size": 64, "cache_seed": 5,
                     "beta": .2, "solver": nsot.SOLVER},
            "data": {"batch_size": 2, "n_points": 8},
            "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                      "dim_feedforward": 16, "dropout": 0.0},
            "training": {"num_steps": 2, "learning_rate": .001, "weight_decay": .01, "log_every": 1},
            "evaluation": {"batch_size": 2, "histogram_bins": 8},
            "checkpoint": f"{dataset}_nsot.pt",
        }
        if dataset == "checkerboard":
            config["data"]["grid_size"] = 4
        return config

    def prepare(self, config, dataset="checkerboard"):
        with contextlib.redirect_stdout(io.StringIO()):
            return nsot.prepare(config, dataset)

    def test_exact_bijection_optimality_and_no_coordinate_changes(self):
        config = self.config()
        source, target = nsot.draw_supersets(config, "checkerboard")
        before = source.copy(), target.copy()
        permutation, cost = nsot.exact_superset_permutation(source, target)
        np.testing.assert_array_equal(np.sort(permutation), np.arange(64))
        np.testing.assert_array_equal(source, before[0])
        np.testing.assert_array_equal(target, before[1])
        self.assertLessEqual(cost, cdist(source, target, "sqeuclidean").diagonal().mean())
        # Small independent brute-force optimality check.
        from itertools import permutations
        costs = cdist(source[:4], target[:4], "sqeuclidean")
        _, small_cost = nsot.exact_superset_permutation(source[:4], target[:4])
        expected = min(costs[np.arange(4), list(p)].mean() for p in permutations(range(4)))
        self.assertAlmostEqual(small_cost, expected, places=12)

    def test_preparation_rng_isolation_reproducibility_and_reuse(self):
        config = self.config()
        torch.manual_seed(93)
        state = torch.get_rng_state()
        source, target = nsot.draw_supersets(config, "checkerboard")
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
        again = nsot.draw_supersets(config, "checkerboard")
        for left, right in zip((source, target), again):
            np.testing.assert_array_equal(left, right)
        path, _ = self.prepare(config)
        original = path.read_bytes()
        with patch.object(nsot, "exact_superset_permutation", side_effect=AssertionError("recomputed")):
            self.prepare(config)
        self.assertEqual(path.read_bytes(), original)

    def test_sampling_uses_same_index_and_correct_hybrid_coefficients(self):
        config = self.config()
        self.prepare(config)
        sampler = nsot.NSOTPairSampler(config, "checkerboard", "cpu", torch.float32)
        actual = sampler.sample(3, generator=torch.Generator().manual_seed(42))
        generator = torch.Generator().manual_seed(42)
        index = torch.randint(64, (3, 8), generator=generator)
        noise = torch.randn(3, 8, 2, generator=generator)
        expected = (.8 ** .5 * sampler.source[index] + .2 ** .5 * noise, sampler.target[index])
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for beta in (0., 1.):
            config["nsot"]["beta"] = beta
            sampler = nsot.NSOTPairSampler(config, "checkerboard", "cpu", torch.float32)
            source, target = sampler.sample(3, generator=torch.Generator().manual_seed(42))
            torch.testing.assert_close(source, sampler.source[index] if beta == 0 else noise, rtol=0, atol=0)
            torch.testing.assert_close(target, sampler.target[index], rtol=0, atol=0)

    def test_config_cache_integrity_and_missing_cache_fail_loudly(self):
        config = self.config()
        with self.assertRaisesRegex(FileNotFoundError, "prepare_nsot"):
            nsot.NSOTPairSampler(config, "checkerboard", "cpu", torch.float32)
        self.prepare(config)
        for section, key, value in (("nsot", "cache_seed", 9), ("nsot", "superset_size", 65),
                                    ("data", "grid_size", 6), ("nsot", "cache_sha256", "bad")):
            changed = copy.deepcopy(config)
            changed[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                nsot.load_cache(changed, "checkerboard")
        for value in (-.1, 1.1, float("nan"), True):
            changed = copy.deepcopy(config)
            changed["nsot"]["beta"] = value
            with self.assertRaises(ValueError):
                nsot.settings(changed)
        for function in (coupling.coupled_points, coupling.coupling_permutation):
            with self.assertRaisesRegex(ValueError, "NSOTPairSampler"):
                function(torch.zeros(1, 8, 2), torch.zeros(1, 8, 2), coupling="nsot")

    def test_removed_source_settings_rejected_before_cache_creation(self):
        config = self.config()
        config["anchor_flow"] = {"mode": "removed_experiment"}
        for function in (nsot.settings, lambda value: nsot.draw_supersets(value, "checkerboard"),
                         lambda value: nsot.prepare(value, "checkerboard")):
            with self.assertRaisesRegex(ValueError, "removed"):
                function(config)
        self.assertFalse(Path(config["nsot"]["cache"]).exists())

    def test_unsupported_cache_variants_are_not_reinterpreted(self):
        config = self.config()
        path, metadata = self.prepare(config)
        source, target, permutation, _, _ = nsot.load_cache(config, "checkerboard")
        original = path.read_bytes()
        for version in (2, 3):
            changed = {**metadata, "format_version": version}
            np.savez_compressed(path, source=source, target=target, permutation=permutation,
                                metadata=np.array(json.dumps(changed)))
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "format_version"):
                nsot.prepare(config, "checkerboard")
            self.assertEqual(path.read_bytes(), before)
        changed = {**metadata, "implementation": "unsupported_source_variant"}
        np.savez_compressed(path, source=source, target=target, permutation=permutation,
                            metadata=np.array(json.dumps(changed)))
        with self.assertRaisesRegex(ValueError, "implementation"):
            nsot.load_cache(config, "checkerboard")
        np.savez_compressed(path, source=source, target=target, permutation=permutation,
                            metadata=np.array(json.dumps(metadata)), source_components=np.zeros(64, dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "missing/extra arrays"):
            nsot.load_cache(config, "checkerboard")
        path.write_bytes(original)
        nsot.load_cache(config, "checkerboard")

    def test_superset_gaussian_and_target_draw_order_unchanged(self):
        from data import sample_checkerboard
        config = self.config()
        actual = nsot.draw_supersets(config, "checkerboard")
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(config["nsot"]["cache_seed"]).get_state())
            source = torch.randn(64, 2)
            target = sample_checkerboard(1, 64, "cpu", torch.float32, 4)[0]
        for left, right in zip(actual, (source.numpy(), target.numpy())):
            np.testing.assert_array_equal(left, right)

    def test_nsot_yaml_matches_existing_comparison_hyperparameters(self):
        root = Path(__file__).resolve().parents[1]
        for directory, name, baseline in (
            ("checkerboard_experiments", "nsot.yaml", "minibatch_ot.yaml"),
            ("horse_experiments", "horse_nsot_n256_seed0.yaml", "horse_minibatch_ot_n256_seed0.yaml"),
        ):
            config = yaml.safe_load((root / directory / name).read_text())
            reference = yaml.safe_load((root / directory / baseline).read_text())
            self.assertEqual({k: v for k, v in config.items() if k not in ("nsot", "coupling", "checkpoint")},
                             {k: v for k, v in reference.items() if k not in ("coupling", "checkpoint")})
            self.assertEqual(nsot.settings(config)["beta"], .2)
            self.assertEqual(nsot.settings(config)["superset_size"], 10000)

    def test_both_datasets_prepare_train_save_load_eval_and_fm_audit(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for dataset, trainer, evaluator, model_class in (
                ("checkerboard", train.main, eval_checkerboard.main, PointSetTransformer),
                ("horse", train_horse.main, eval_horse.main, train_horse.HorsePointSetTransformer),
            ):
                config = self.config(dataset)
                path = Path(f"{dataset}.yaml")
                path.write_text(yaml.safe_dump(config), encoding="utf-8")
                prepare_nsot.main(path, dataset)
                with patch.object(train, "coupled_points", side_effect=AssertionError("online OT called")):
                    run = trainer(path, steps=1)
                model, trained, checkpoint, metadata = load_model(run / "config.yaml", model_class, dataset)
                self.assertTrue(metadata["training_config_verified"])
                self.assertEqual(trained["nsot"]["cache_sha256"], nsot.file_sha256(config["nsot"]["cache"]))
                self.assertNotIn("cache_sha256", yaml.safe_load(path.read_text())["nsot"])
                self.assertFalse(metadata["coupling_details"]["author_code"])
                data = DiagnosticData(trained, dataset, "cpu", torch.float32, 2026)
                fm = audit_generation.fm_summary(model, trained, data, 1, 2, 2026)
                self.assertIn("fixed training cache", fm["sampling_scope"])
                self.assertTrue(all(np.isfinite(r["mse"]["mean"]) for r in fm["time_errors"]))
                with patch.object(audit_generation, "render"):
                    directory = audit_generation.audit(
                        run / "config.yaml", dataset, clouds=2, batch_size=2, nfes=[1, 2],
                        reference_nfe=2, max_reference_nfe=8, fm_batches=1, matching_batch_size=2)
                diagnostics = json.loads((directory / "diagnostics.json").read_text())
                self.assertEqual(diagnostics["noise_sha256"], tensor_sha256(data.bank(2)[0]))
                self.assertEqual(diagnostics["target_sha256"], tensor_sha256(data.bank(2)[1]))
                self.assertIn("fixed training cache", diagnostics["fm"]["sampling_scope"])
                # Generation evaluation must work WITHOUT its training cache.
                cache = Path(config["nsot"]["cache"])
                cache.rename(cache.with_suffix(".backup"))
                if dataset == "checkerboard":
                    output = evaluator(run / "config.yaml", 2, render=False)
                else:
                    with patch.object(eval_horse, "render_comparison"), patch("horse_regions.HorseRegions.render"):
                        output = evaluator(run / "config.yaml", 2)
                result = json.loads(output.with_suffix(".json").read_text())
                self.assertEqual(result["coupling_details"], metadata["coupling_details"])
                self.assertTrue(np.isfinite(result["chamfer"]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_sampler_and_backward(self):
        config = self.config()
        self.prepare(config)
        sampler = nsot.NSOTPairSampler(config, "checkerboard", "cuda", torch.float32)
        noise, target = sampler.sample(2, generator=torch.Generator(device="cuda").manual_seed(0))
        model = PointSetTransformer(**config["model"]).cuda()
        optimizer = torch.optim.AdamW(model.parameters())
        loss = train.train_step(model, optimizer, target, coupling="nsot", paired_noise=noise)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(noise.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
