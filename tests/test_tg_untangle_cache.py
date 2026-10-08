"""Integration guarantees for opt-in offline conflict-guided Hard banks."""

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
import torch
import yaml

import coupling
import eval as eval_checkerboard
import eval_horse
from experiment import load_model
from model import PointSetTransformer
import prepare_tg_untangle
import tg_cache
import train
import train_horse


class TGUntangleCacheTests(unittest.TestCase):
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
    def config(dataset="checkerboard", suffix="", workers=0):
        config = {
            "seed": 0, "device": "cpu", "dtype": "float32",
            "coupling": "target_guided_cached", "num_regions": 2,
            "tg_cache": {
                "base_path": f"{dataset}_base{suffix}",
                "path": f"{dataset}_guided{suffix}", "sampling": "bank",
                "num_clouds": 4, "seed": 5, "prepare_batch_size": 2,
                "num_workers": workers,
                "untangle": {
                    "enabled": True, "neighbors": 2, "times": [.05, .15, .3],
                    "lipschitz": 1.0, "permutations": 3, "swap_steps": 16,
                    "seed": 0,
                },
            },
            "data": {"batch_size": 2, "n_points": 8},
            "model": {"point_dim": 2, "d_model": 8, "nhead": 2,
                      "num_layers": 1, "dim_feedforward": 16, "dropout": 0.0},
            "training": {"num_steps": 2, "learning_rate": .001,
                         "weight_decay": .01, "log_every": 1},
            "evaluation": {"batch_size": 2, "histogram_bins": 8},
            "checkpoint": f"{dataset}_guided{suffix}.pt",
        }
        if dataset == "checkerboard":
            config["data"]["grid_size"] = 4
        return config

    @staticmethod
    def baseline(config):
        result = copy.deepcopy(config)
        result["tg_cache"]["path"] = result["tg_cache"].pop("base_path")
        result["tg_cache"].pop("untangle")
        return result

    @staticmethod
    def prepare(config, dataset="checkerboard"):
        with contextlib.redirect_stdout(io.StringIO()):
            return tg_cache.prepare(config, dataset)

    @staticmethod
    def write_config(config, name="input.yaml"):
        path = Path(name)
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def test_both_datasets_preserve_base_arrays_bijections_and_monotonic_scores(self):
        for dataset in ("checkerboard", "horse"):
            config = self.config(dataset)
            baseline = self.baseline(config)
            base_path, base_meta = self.prepare(baseline, dataset)
            original_metadata = (base_path / "metadata.json").read_bytes()
            fingerprint = tg_cache._fingerprint(base_meta)
            path, meta = self.prepare(config, dataset)

            self.assertEqual(tg_cache.FORMAT_VERSION, 1)
            self.assertEqual(base_meta["format_version"], 1)
            self.assertEqual(meta["format_version"], 2)
            self.assertEqual((base_path / "metadata.json").read_bytes(), original_metadata)
            self.assertEqual(meta["parent_cache"]["sha256"], fingerprint)
            self.assertEqual(meta["parent_cache"]["metadata"], base_meta)
            self.assertIn("untangle_summary", meta)
            self.assertGreaterEqual(meta["untangle_optimization_seconds"], 0.)
            self.assertEqual(tg_cache.load_cache(baseline, dataset)[2], fingerprint)

            for name, digest in base_meta["array_sha256"].items():
                self.assertEqual(meta["array_sha256"][name], digest)
                self.assertEqual((base_path / f"{name}.npy").read_bytes(),
                                 (path / f"{name}.npy").read_bytes())

            permutations = np.load(path / "fine_permutations.npy", allow_pickle=False)
            scores = np.load(path / "untangle_scores.npy", allow_pickle=False)
            accepted = np.load(path / "untangle_accepted_swaps.npy", allow_pickle=False)
            source_labels = np.load(path / "source_labels.npy", allow_pickle=False)
            target_labels = np.load(path / "target_labels.npy", allow_pickle=False)
            self.assertEqual(permutations.shape, (4, 3, 8))
            self.assertEqual(permutations.dtype, np.dtype("int32"))
            self.assertEqual(scores.shape, (4, 3, 2))
            self.assertEqual(scores.dtype, np.dtype("float64"))
            self.assertEqual(accepted.shape, (4, 3))
            self.assertEqual(accepted.dtype, np.dtype("int32"))
            self.assertTrue(np.isfinite(scores).all())
            self.assertTrue((scores >= 0).all())
            self.assertTrue((scores[..., 1] <= scores[..., 0] + 1e-12).all())
            self.assertTrue(((accepted >= 0) & (accepted <= 16)).all())
            for cloud, pool in enumerate(permutations):
                for permutation in pool:
                    np.testing.assert_array_equal(np.sort(permutation), np.arange(8))
                    np.testing.assert_array_equal(target_labels[cloud][permutation],
                                                  source_labels[cloud])

    def test_deterministic_pools_and_prepare_does_not_advance_global_rng(self):
        config = self.config()
        repeated = self.config(suffix="_repeat")
        torch.manual_seed(73)
        before = torch.get_rng_state()
        _, first = self.prepare(config)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        _, second = self.prepare(repeated)
        self.assertEqual(first["array_sha256"], second["array_sha256"])

    def test_cli_control_is_matched_initial_pool_and_preserves_input(self):
        config = self.config()
        original = self.write_config(config)
        before = original.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            guided_path, _ = prepare_tg_untangle.main(original, "checkerboard")
            control_path, _ = prepare_tg_untangle.main(original, "checkerboard", control=True)
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual(control_path, Path(config["tg_cache"]["path"] + "_control"))
        guided_config = yaml.safe_load((guided_path / "config.yaml").read_text(encoding="utf-8"))
        control_config = yaml.safe_load((control_path / "config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(control_config["tg_cache"]["untangle"]["swap_steps"], 0)
        self.assertEqual(control_config["checkpoint"], "checkerboard_guided_control.pt")
        self.assertEqual(guided_config["model"], control_config["model"])
        self.assertEqual(guided_config["training"], control_config["training"])
        self.assertEqual(guided_config["data"], control_config["data"])
        guided_scores = np.load(guided_path / "untangle_scores.npy")
        control_scores = np.load(control_path / "untangle_scores.npy")
        np.testing.assert_array_equal(guided_scores[..., 0], control_scores[..., 0])
        np.testing.assert_array_equal(control_scores[..., 0], control_scores[..., 1])
        np.testing.assert_array_equal(np.load(control_path / "untangle_accepted_swaps.npy"),
                                      np.zeros((4, 3), dtype=np.int32))
        for name in ("source", "target", "source_labels", "target_labels", "capacities"):
            self.assertEqual((guided_path / f"{name}.npy").read_bytes(),
                             (control_path / f"{name}.npy").read_bytes())

    def test_sampler_uses_only_pool_without_online_ot_and_is_generator_deterministic(self):
        for dataset in ("checkerboard", "horse"):
            config = self.config(dataset)
            path, _ = self.prepare(config, dataset)
            sampler = tg_cache.TGCachedPairSampler(config, dataset, "cpu", torch.float32)
            original_source = np.load(path / "source.npy")
            original_target = np.load(path / "target.npy")
            permutations = np.load(path / "fine_permutations.npy")
            try:
                with patch.object(coupling.ot, "emd", side_effect=AssertionError("online OT")), \
                        patch.object(coupling.ot, "sinkhorn", side_effect=AssertionError("online Sinkhorn")), \
                        patch.object(tg_cache, "random_fine_permutation",
                                     side_effect=AssertionError("fresh pairing instead of pool")):
                    for cloud in range(4):
                        for _ in range(10):
                            source, target = sampler.dataset.draw(cloud, np.random.default_rng(14))
                            np.testing.assert_array_equal(source.numpy(), original_source[cloud])
                            self.assertTrue(any(np.array_equal(target.numpy(), original_target[cloud][p])
                                                for p in permutations[cloud]))
                    a = sampler.sample(3, generator=torch.Generator().manual_seed(19))
                    b = sampler.sample(3, generator=torch.Generator().manual_seed(19))
                for left, right in zip(a, b):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
            finally:
                sampler.close()

    def test_prepared_derivative_does_not_need_live_parent(self):
        config = self.config()
        self.prepare(config)
        base = Path(config["tg_cache"]["base_path"])
        base.rename(base.with_name(base.name + "_detached"))
        sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32)
        try:
            source, target = sampler.sample(2)
            self.assertEqual(source.shape, (2, 8, 2))
            self.assertTrue(torch.isfinite(target).all())
        finally:
            sampler.close()

    def test_pool_hash_and_internal_parent_lineage_are_checked(self):
        config = self.config()
        path, _ = self.prepare(config)
        permutations = np.load(path / "fine_permutations.npy")
        permutations[0, 0, [0, 1]] = permutations[0, 0, [1, 0]]
        np.save(path / "fine_permutations.npy", permutations)
        with self.assertRaisesRegex(ValueError, "SHA256|hash|integrity"):
            tg_cache.load_cache(config, "checkerboard")

        lineage_config = self.config(suffix="_lineage")
        lineage_path, meta = self.prepare(lineage_config)
        meta["parent_cache"]["metadata"]["cache_seed"] += 1
        (lineage_path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "parent|lineage|SHA256|hash"):
            tg_cache.load_cache(lineage_config, "checkerboard")

    def test_existing_mismatched_or_incomplete_derivative_is_not_overwritten(self):
        config = self.config()
        path, _ = self.prepare(config)
        before = (path / "metadata.json").read_bytes()
        pool_before = (path / "fine_permutations.npy").read_bytes()
        changed = copy.deepcopy(config)
        changed["tg_cache"]["untangle"]["lipschitz"] = .5
        with self.assertRaises(ValueError):
            self.prepare(changed)
        self.assertEqual((path / "metadata.json").read_bytes(), before)
        self.assertEqual((path / "fine_permutations.npy").read_bytes(), pool_before)
        incomplete = self.config(suffix="_incomplete")
        Path(incomplete["tg_cache"]["path"]).mkdir()
        with self.assertRaises((FileNotFoundError, ValueError)):
            self.prepare(incomplete)

    def test_one_step_train_checkpoint_load_and_cache_independent_generation(self):
        for dataset, trainer, evaluator, model_class in (
                ("checkerboard", train.main, eval_checkerboard.main, PointSetTransformer),
                ("horse", train_horse.main, eval_horse.main, train_horse.HorsePointSetTransformer)):
            config = self.config(dataset)
            path = self.write_config(config, f"{dataset}.yaml")
            with contextlib.redirect_stdout(io.StringIO()):
                prepared, _ = prepare_tg_untangle.main(path, dataset)
                with patch.object(train, "coupled_points", side_effect=AssertionError("online coupling")):
                    run = trainer(prepared / "config.yaml", steps=1)
                _, trained, _, metadata = load_model(run / "config.yaml", model_class, dataset)
                self.assertTrue(metadata["training_config_verified"])
                self.assertIn("cache_sha256", trained["tg_cache"])
                self.assertTrue(trained["tg_cache"]["untangle"]["enabled"])
                with patch.object(tg_cache, "load_cache", side_effect=AssertionError("generation cache access")):
                    if dataset == "checkerboard":
                        output = evaluator(run / "config.yaml", 2, render=False)
                    else:
                        with patch.object(eval_horse, "render_comparison"), \
                                patch("horse_regions.HorseRegions.render"):
                            output = evaluator(run / "config.yaml", 2)
            result = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(result["coupling_details"], metadata["coupling_details"])
            self.assertTrue(np.isfinite(result["chamfer"]))

    def test_spawn_worker_reads_optimized_pool(self):
        config = self.config(workers=1)
        self.prepare(config)
        sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32,
                                                training=True)
        try:
            for _ in range(2):
                source, target = sampler.sample(2)
                self.assertEqual(source.shape, (2, 8, 2))
                self.assertTrue(torch.isfinite(target).all())
        finally:
            sampler.close()

    def test_supplied_yaml_keeps_hard_baseline_training_and_evaluation(self):
        root = Path(__file__).resolve().parents[1]
        for directory, prefix in (("checkerboard_experiments", ""),
                                  ("horse_experiments", "horse_")):
            baseline = yaml.safe_load((root / directory /
                                       f"{prefix}target_guided_cached_k8_n256_seed0.yaml").read_text())
            guided = yaml.safe_load((root / directory /
                                     f"{prefix}target_guided_cached_untangle_k8_n256_seed0.yaml").read_text())
            for key in baseline.keys() - {"tg_cache", "checkpoint"}:
                self.assertEqual(guided[key], baseline[key])
            self.assertEqual(guided["tg_cache"]["base_path"], baseline["tg_cache"]["path"])
            for key in baseline["tg_cache"].keys() - {"path"}:
                self.assertEqual(guided["tg_cache"][key], baseline["tg_cache"][key])
            self.assertNotEqual(guided["tg_cache"]["path"], baseline["tg_cache"]["path"])
            self.assertTrue(tg_cache.settings(guided)["untangle"]["enabled"])

    def test_real_n256_k8_preparation_and_optional_cuda_backward(self):
        for dataset in ("checkerboard", "horse"):
            config = self.config(dataset, suffix="_n256")
            config["num_regions"] = 8
            config["data"]["n_points"] = 256
            config["tg_cache"]["num_clouds"] = 2
            config["tg_cache"]["untangle"].update(neighbors=8, swap_steps=64, permutations=4)
            path, meta = self.prepare(config, dataset)
            self.assertLessEqual(meta["untangle_summary"]["final_score_mean"],
                                 meta["untangle_summary"]["initial_score_mean"])
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sampler = tg_cache.TGCachedPairSampler(config, dataset, device, torch.float32)
            try:
                source, target = sampler.sample(2)
                self.assertEqual(source.shape, (2, 256, 2))
                model = PointSetTransformer(**config["model"]).to(device)
                loss = train.train_step(model, torch.optim.AdamW(model.parameters()), target,
                                        coupling=config["coupling"], paired_noise=source)
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(len(np.load(path / "fine_permutations.npy")[0]), 4)
            finally:
                sampler.close()


if __name__ == "__main__":
    unittest.main()
