import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
import yaml

import coupling
from experiment import load_model
from model import PointSetTransformer
import run_wasserstein_bank as runner
import wasserstein_bank as bank


class WassersteinBankTests(unittest.TestCase):
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
    def config(dataset="mixture2d"):
        return {
            "dataset": dataset, "seed": 0, "device": "cpu", "dtype": "float32", "coupling": bank.METHOD,
            "wasserstein_bank": {"cache": "bank.npz", "cache_seed": 9, "superset_size": 24,
                                 "pool_per_shape": 8, "subset_size": 4, "scales": [2, 4, 8],
                                 "repetitions": 2, "projections": 8, "variance_times": [0.0, 0.5, 1.0],
                                 "bandwidths": [0.1, 0.2], "variance_bandwidth": 0.1,
                                 "candidates": [{"kind": "random"}, {"kind": "exact"},
                                                {"kind": "projection", "angle": 0.5}], "selector": "joint"},
            "data": {"shapes": ["disk", "ring", "two_disks"], "n_points": 8, "batch_size": 2, "grid_size": 4},
            "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1, "dim_feedforward": 16, "dropout": 0.0},
            "training": {"num_steps": 2, "learning_rate": .001, "weight_decay": .01, "log_every": 1, "validation_batch_size": 2},
            "evaluation": {"batch_size": 6, "seed": 1, "histogram_bins": 8}, "checkpoint": "bank.pt",
        }

    def prepare(self, config):
        with contextlib.redirect_stdout(io.StringIO()):
            bank.prepare(config, config["dataset"])
        return bank.load_cache(config, config["dataset"])

    def test_candidate_bijections_preserve_original_coordinates(self):
        rng = np.random.default_rng(3)
        source, target = rng.normal(size=(16, 2)).astype(np.float32), rng.normal(size=(16, 2)).astype(np.float32)
        before = source.copy(), target.copy()
        for candidate in bank.DEFAULT_CANDIDATES:
            mapping = bank.candidate_permutation(source, target, candidate, rng)
            np.testing.assert_array_equal(np.sort(mapping), np.arange(16))
        np.testing.assert_array_equal(source, before[0])
        np.testing.assert_array_equal(target, before[1])

    def test_exact_and_sliced_wasserstein_and_singleton_identity(self):
        values = np.array([[0., 0.], [1., 2.], [3., 1.]])
        self.assertEqual(bank.wasserstein_squared(values, values), 0.)
        measured = np.mean([bank.wasserstein_squared(v[None], values) for v in values])
        expected = 2 * np.square(values - values.mean(0)).sum(1).mean()
        self.assertAlmostEqual(measured, expected)
        directions = np.eye(2)
        self.assertAlmostEqual(bank.wasserstein_squared(values, values, "sliced", directions), 0.)
        self.assertTrue(np.isfinite(bank.wasserstein_squared(values[:2], values, "sliced", directions)))

    def test_preparation_rng_isolation_bijections_holdout_and_reuse(self):
        config = self.config()
        torch.manual_seed(42)
        state = torch.get_rng_state().clone()
        arrays, metadata, digest = self.prepare(config)
        torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(len(metadata["selected"]), len(bank.SELECTORS))
        self.assertFalse(np.array_equal(arrays["source"], arrays["validation_source"]))
        self.assertFalse(np.array_equal(arrays["targets"], arrays["validation_targets"]))
        for mapping in arrays["permutations"]:
            for shape, order in enumerate(mapping):
                np.testing.assert_array_equal(arrays["targets"][shape, np.sort(order)], arrays["targets"][shape])
        original = Path("bank.npz").read_bytes()
        self.prepare(config)
        self.assertEqual(Path("bank.npz").read_bytes(), original)
        config["wasserstein_bank"]["cache_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "differs"):
            bank.load_cache(config, "mixture2d")

    def test_sampler_pair_indices_and_shape_frequencies_are_preserved(self):
        config = self.config()
        self.prepare(config)
        sampler = bank.WassersteinBankSampler(config, "mixture2d", "cpu", torch.float32)
        source, target = sampler.sample(600, generator=torch.Generator().manual_seed(14))
        generator = torch.Generator().manual_seed(14)
        shapes = torch.randint(3, (600,), generator=generator)
        indices = torch.randint(24, (600, 8), generator=generator)
        expected_source = sampler.arrays["source"][indices]
        order = sampler.arrays["permutations"][sampler.selected, shapes[:, None], indices]
        expected_target = sampler.arrays["targets"][shapes[:, None], order]
        torch.testing.assert_close(source, expected_source, rtol=0, atol=0)
        torch.testing.assert_close(target, expected_target, rtol=0, atol=0)
        self.assertTrue((torch.bincount(shapes) > 150).all())

    def test_variogram_excludes_self_and_is_not_called_exact_variance(self):
        position = np.array([[0., 0.], [1., 0.]])
        velocity = np.array([[1., 0.], [-1., 0.]])
        result = bank.velocity_variogram(position, velocity, .1)
        self.assertAlmostEqual(result["value"], 1.)
        self.assertAlmostEqual(result["effective_neighbors"], 1.)
        self.assertEqual(bank.velocity_variogram(position, np.ones_like(velocity), .1)["value"], 0.)

    def test_invalid_config_and_cache_are_rejected(self):
        for key, value in (("backend", "unknown"), ("variance_times", [0.5]),
                           ("pool_per_shape", 25), ("scales", [1000]), ("max_cost_mib", 0.0001)):
            config = self.config()
            config["wasserstein_bank"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bank.settings(config)
        config = self.config()
        arrays, metadata, _ = self.prepare(config)
        arrays["permutations"][0, 0, :] = 0
        np.savez_compressed("bank.npz", **arrays, metadata=np.array(json.dumps(metadata)))
        with self.assertRaisesRegex(ValueError, "non-bijective"):
            bank.load_cache(config, "mixture2d")
        with self.assertRaisesRegex(ValueError, "WassersteinBankSampler"):
            coupling.coupling_permutation(torch.zeros(1, 8, 2), torch.zeros(1, 8, 2), coupling=bank.METHOD)

    def test_end_to_end_prepare_train_evaluate_and_validation_rng(self):
        config = self.config()
        with Path("template.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream)
        with contextlib.redirect_stdout(io.StringIO()):
            path, manifest = runner.create_experiment("template.yaml", selectors=["displacement", "joint", "multiscale"])
            runner.train_jobs(path, manifest)
            runner.evaluate_jobs(path, manifest, [1, 2])
        self.assertEqual(len(manifest["jobs"]), 3)
        self.assertTrue((path.parent / "summary.csv").is_file())
        for job in manifest["jobs"]:
            self.assertEqual(len(job["evaluations"]), 2)
            with Path(job["evaluations"]["2"]).open(encoding="utf-8") as stream:
                result = json.load(stream)
            self.assertTrue(result["training_config_verified"])
            self.assertIn("one_nna_cd", result)
            model, saved_config, _, _ = load_model(job["run_config"], PointSetTransformer, "mixture2d")
            sampler = bank.WassersteinBankSampler(saved_config, "mixture2d", "cpu", torch.float32)
            model.train()
            state = torch.get_rng_state().clone()
            curve = sampler.validation_curve(model, 2, 100)
            self.assertTrue(model.training)
            torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)
            self.assertTrue(np.isfinite([p["mse_per_coordinate"] for p in curve]).all())

    def test_original_checkerboard_and_horse_evaluators_work(self):
        import eval as checker_evaluator
        import eval_horse
        from train_horse import HorsePointSetTransformer
        for dataset, evaluator, model_class in (("checkerboard", checker_evaluator.main, PointSetTransformer),
                                                ("horse", eval_horse.main, HorsePointSetTransformer)):
            with self.subTest(dataset=dataset), contextlib.redirect_stdout(io.StringIO()):
                config = self.config(dataset)
                config["wasserstein_bank"]["pool_per_shape"] = 16
                config["wasserstein_bank"]["cache"] = f"{dataset}.npz"
                with Path(f"{dataset}.yaml").open("w", encoding="utf-8") as stream:
                    yaml.safe_dump(config, stream)
                path, manifest = runner.create_experiment(f"{dataset}.yaml", selectors=["joint"])
                runner.train_jobs(path, manifest)
                output = evaluator(manifest["jobs"][0]["run_config"], 2)
                self.assertTrue(Path(output).with_suffix(".json").is_file())


if __name__ == "__main__":
    unittest.main()
