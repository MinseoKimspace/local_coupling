"""Integration guarantees for offline frozen-teacher TG permutation banks."""

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

import coupling
import eval as eval_checkerboard
import eval_horse
from experiment import evaluation_title, load_model
from model import PointSetTransformer
import prepare_tg_model_guidance
import tg_cache
import tg_model_guidance_cache
import train
import train_horse


class TGModelGuidanceCacheTests(unittest.TestCase):
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
                "path": f"{dataset}_model_guided{suffix}", "sampling": "bank",
                "num_clouds": 4, "seed": 5, "prepare_batch_size": 2,
                "num_workers": workers,
                "model_guidance": {
                    "enabled": True, "teacher_config": None,
                    "neighbors": 2, "times": [.25, .5, .75],
                    "candidates": 4, "keep": 2, "probes": 1,
                    "fd_epsilon": 1e-3, "inference_batch_size": 3,
                    "weights": {"regression": 1., "local": .25, "jacobian": .1},
                    "selection": "score", "seed": 0,
                },
            },
            "data": {"batch_size": 2, "n_points": 8},
            "model": {"point_dim": 2, "d_model": 8, "nhead": 2,
                      "num_layers": 1, "dim_feedforward": 16, "dropout": 0.0},
            "training": {"num_steps": 2, "learning_rate": .001,
                         "weight_decay": .01, "log_every": 1},
            "evaluation": {"batch_size": 2, "histogram_bins": 8},
            "checkpoint": f"{dataset}_model_guided{suffix}.pt",
        }
        if dataset == "checkerboard":
            config["data"]["grid_size"] = 4
        return config

    @staticmethod
    def baseline(config):
        result = copy.deepcopy(config)
        result["tg_cache"]["path"] = result["tg_cache"].pop("base_path")
        result["tg_cache"].pop("model_guidance")
        result["checkpoint"] = "teacher.pt"
        return result

    @staticmethod
    def write_config(config, name="input.yaml"):
        path = Path(name)
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    @staticmethod
    def prepare(config, dataset="checkerboard"):
        with contextlib.redirect_stdout(io.StringIO()):
            return tg_cache.prepare(config, dataset)

    def teacher(self, config, dataset="checkerboard"):
        """Use actual one-update trainer output, never synthetic verified metadata."""
        baseline = self.baseline(config)
        self.prepare(baseline, dataset)
        baseline_path = self.write_config(baseline, f"{dataset}_teacher_input.yaml")
        trainer = train_horse.main if dataset == "horse" else train.main
        with contextlib.redirect_stdout(io.StringIO()):
            run = trainer(baseline_path, steps=1)
        teacher_config = (run / "config.yaml").resolve()
        config["tg_cache"]["model_guidance"]["teacher_config"] = str(teacher_config)
        snapshot = yaml.safe_load(teacher_config.read_text(encoding="utf-8"))
        checkpoint = teacher_config.parent / snapshot["checkpoint"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.assertEqual(payload["format_version"], 2)
        self.assertEqual(payload["dataset"], dataset)
        self.assertEqual(payload["config"]["training"]["num_steps"], 1)
        return teacher_config, checkpoint

    def test_both_datasets_preserve_originals_and_select_valid_low_score_pools(self):
        for dataset in ("checkerboard", "horse"):
            with self.subTest(dataset=dataset):
                config = self.config(dataset)
                _, checkpoint = self.teacher(config, dataset)
                teacher_bytes = checkpoint.read_bytes()
                original_path = Path(config["tg_cache"]["base_path"])
                original_metadata = (original_path / "metadata.json").read_bytes()
                original = json.loads(original_metadata)
                path, meta = self.prepare(config, dataset)
                self.assertEqual(tg_cache.FORMAT_VERSION, 1)
                self.assertEqual(meta["format_version"], 3)
                self.assertEqual(meta["parent_cache"]["sha256"], tg_cache._fingerprint(original))
                self.assertEqual(meta["parent_cache"]["metadata"], original)
                self.assertEqual(meta["teacher"]["checkpoint_sha256"],
                                 hashlib.sha256(teacher_bytes).hexdigest())
                self.assertTrue(meta["teacher"]["training_config_verified"])
                self.assertTrue(meta["model_guidance_summary"]["teacher_frozen"])
                self.assertFalse(meta["model_guidance_summary"]["one_lipschitz_guaranteed"])
                self.assertFalse(meta["teacher_training_cost_included"])
                self.assertEqual(checkpoint.read_bytes(), teacher_bytes)
                self.assertEqual((original_path / "metadata.json").read_bytes(), original_metadata)
                for name, digest in original["array_sha256"].items():
                    self.assertEqual(meta["array_sha256"][name], digest)
                    self.assertEqual((path / f"{name}.npy").read_bytes(),
                                     (original_path / f"{name}.npy").read_bytes())

                candidates = np.load(path / "guidance_candidates.npy")
                pool = np.load(path / "fine_permutations.npy")
                components = np.load(path / "guidance_components.npy")
                scores = np.load(path / "guidance_scores.npy")
                normalizers = np.load(path / "guidance_normalizers.npy")
                selected = np.load(path / "guidance_selected_indices.npy")
                source_labels = np.load(path / "source_labels.npy")
                target_labels = np.load(path / "target_labels.npy")
                self.assertEqual(candidates.shape, (4, 4, 8))
                self.assertEqual(pool.shape, (4, 2, 8))
                self.assertEqual(components.shape, (4, 4, 3))
                self.assertEqual(scores.shape, (4, 4))
                self.assertEqual(normalizers.shape, (4, 3))
                self.assertEqual(selected.shape, (4, 2))
                self.assertEqual(candidates.dtype, np.dtype("int32"))
                self.assertEqual(pool.dtype, np.dtype("int32"))
                self.assertEqual(components.dtype, np.dtype("float64"))
                for values in (components, scores, normalizers):
                    self.assertTrue(np.isfinite(values).all())
                self.assertTrue((components >= 0).all())
                self.assertTrue((normalizers > 0).all())
                np.testing.assert_allclose(normalizers,
                                           np.maximum(components.mean(axis=1), 1e-12))
                expected_scores = np.sum(components / normalizers[:, None, :] *
                                         np.array([1., .25, .1])[None, None, :], axis=2)
                np.testing.assert_allclose(scores, expected_scores, rtol=1e-14, atol=1e-14)
                np.testing.assert_array_equal(selected, np.argsort(scores, axis=1, kind="stable")[:, :2])
                for cloud in range(4):
                    np.testing.assert_array_equal(pool[cloud], candidates[cloud][selected[cloud]])
                    for permutation in candidates[cloud]:
                        np.testing.assert_array_equal(np.sort(permutation), np.arange(8))
                        np.testing.assert_array_equal(target_labels[cloud][permutation], source_labels[cloud])
                summary = meta["model_guidance_summary"]
                self.assertLessEqual(summary["selected_score_mean"], summary["candidate_score_mean"])
                report = json.loads((path / "model_guidance_report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["summary"], summary)
                self.assertEqual(report["teacher"], meta["teacher"])

    def test_cli_control_matches_candidates_components_probes_and_preserves_input(self):
        config = self.config()
        teacher, _ = self.teacher(config)
        original = self.write_config(config)
        before = original.read_bytes()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            guided_path, guided = prepare_tg_model_guidance.main(
                original, "checkerboard", teacher_config=teacher)
            control_path, control = prepare_tg_model_guidance.main(
                original, "checkerboard", teacher_config=teacher, control=True)
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual(control_path, Path(config["tg_cache"]["path"] + "_control"))
        self.assertIn(f'train_command=python train.py "{guided_path / "config.yaml"}"', output.getvalue())
        guided_config = yaml.safe_load((guided_path / "config.yaml").read_text(encoding="utf-8"))
        control_config = yaml.safe_load((control_path / "config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(guided_config["tg_cache"]["teacher_checkpoint_sha256"],
                         guided["teacher"]["checkpoint_sha256"])
        self.assertEqual(control_config["tg_cache"]["teacher_checkpoint_sha256"],
                         guided["teacher"]["checkpoint_sha256"])
        self.assertEqual(control_config["tg_cache"]["model_guidance"]["selection"], "random")
        self.assertEqual(control_config["checkpoint"], "checkerboard_model_guided_control.pt")
        for key in ("model", "training", "data", "evaluation"):
            self.assertEqual(guided_config[key], control_config[key])
        for name in ("source", "target", "source_labels", "target_labels", "capacities",
                     "guidance_candidates", "guidance_components", "guidance_scores", "guidance_normalizers"):
            self.assertEqual((guided_path / f"{name}.npy").read_bytes(),
                             (control_path / f"{name}.npy").read_bytes())
        guided_selected = np.load(guided_path / "guidance_selected_indices.npy")
        control_selected = np.load(control_path / "guidance_selected_indices.npy")
        self.assertFalse(np.array_equal(guided_selected, control_selected))
        candidates = np.load(control_path / "guidance_candidates.npy")
        for cloud in range(4):
            self.assertEqual(len(np.unique(control_selected[cloud])), 2)
            np.testing.assert_array_equal(np.load(control_path / "fine_permutations.npy")[cloud],
                                          candidates[cloud][control_selected[cloud]])
        self.assertEqual(guided["fine_pairing_variant"], "model_guided_pool")
        self.assertEqual(control["fine_pairing_variant"], "model_guided_random_control")
        self.assertIn("model_guided_pool", evaluation_title(guided_config))
        self.assertIn("model_guided_random_control", evaluation_title(control_config))
        with contextlib.redirect_stdout(io.StringIO()):
            control_run = train.main(control_path / "config.yaml", steps=1)
            _, _, _, trained_control = load_model(
                control_run / "config.yaml", PointSetTransformer, "checkerboard")
        self.assertIn("model_guided_random_control", control_run.name)
        self.assertEqual(trained_control["coupling_details"]["fine_pairing_variant"],
                         "model_guided_random_control")

    def test_prepare_restores_rng_and_is_deterministic(self):
        config = self.config()
        self.teacher(config)
        random.seed(19)
        np.random.seed(23)
        torch.manual_seed(73)
        before_python, before_numpy = random.getstate(), np.random.get_state()
        before_torch = torch.get_rng_state().clone()
        before_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        _, first = self.prepare(config)
        self.assertEqual(random.getstate(), before_python)
        after_numpy = np.random.get_state()
        self.assertEqual(after_numpy[0], before_numpy[0])
        np.testing.assert_array_equal(after_numpy[1], before_numpy[1])
        self.assertEqual(after_numpy[2:], before_numpy[2:])
        torch.testing.assert_close(torch.get_rng_state(), before_torch, rtol=0, atol=0)
        for before, after in zip(before_cuda, torch.cuda.get_rng_state_all() if before_cuda else []):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
        repeated = copy.deepcopy(config)
        repeated["tg_cache"]["path"] += "_repeat"
        _, second = self.prepare(repeated)
        self.assertEqual(first["array_sha256"], second["array_sha256"])

    def test_sampling_training_and_generation_need_neither_live_teacher_nor_parent(self):
        for dataset, trainer, evaluator, model_class in (
                ("checkerboard", train.main, eval_checkerboard.main, PointSetTransformer),
                ("horse", train_horse.main, eval_horse.main, train_horse.HorsePointSetTransformer)):
            with self.subTest(dataset=dataset):
                config = self.config(dataset)
                teacher, _ = self.teacher(config, dataset)
                original = self.write_config(config, f"{dataset}_student_input.yaml")
                with contextlib.redirect_stdout(io.StringIO()):
                    prepared, _ = prepare_tg_model_guidance.main(
                        original, dataset, teacher_config=teacher)
                parent = Path(config["tg_cache"]["base_path"])
                parent.rename(parent.with_name(parent.name + "_detached"))
                teacher.parent.rename(teacher.parent.with_name(teacher.parent.name + "_detached"))
                bound = yaml.safe_load((prepared / "config.yaml").read_text(encoding="utf-8"))
                with patch.object(tg_model_guidance_cache, "_load_teacher",
                                  side_effect=AssertionError("online teacher load")), \
                        patch.object(coupling.ot, "emd", side_effect=AssertionError("online OT")), \
                        patch.object(coupling.ot, "sinkhorn", side_effect=AssertionError("online Sinkhorn")), \
                        patch.object(tg_cache, "random_fine_permutation",
                                     side_effect=AssertionError("fresh fine pairing")):
                    sampler = tg_cache.TGCachedPairSampler(bound, dataset, "cpu", torch.float32)
                    try:
                        target = np.load(prepared / "target.npy")
                        pool = np.load(prepared / "fine_permutations.npy")
                        for cloud in range(4):
                            for _ in range(3):
                                _, paired = sampler.dataset.draw(cloud, np.random.default_rng(14))
                                self.assertTrue(any(np.array_equal(paired.numpy(), target[cloud][p])
                                                    for p in pool[cloud]))
                        a = sampler.sample(3, generator=torch.Generator().manual_seed(19))
                        b = sampler.sample(3, generator=torch.Generator().manual_seed(19))
                        for left, right in zip(a, b):
                            torch.testing.assert_close(left, right, rtol=0, atol=0)
                    finally:
                        sampler.close()
                    with contextlib.redirect_stdout(io.StringIO()), \
                            patch.object(train, "coupled_points", side_effect=AssertionError("online coupling")):
                        run = trainer(prepared / "config.yaml", steps=1)
                self.assertIn("model_guided_pool", run.name)
                with contextlib.redirect_stdout(io.StringIO()):
                    _, trained, _, metadata = load_model(run / "config.yaml", model_class, dataset)
                self.assertTrue(metadata["training_config_verified"])
                self.assertIn("cache_sha256", trained["tg_cache"])
                self.assertEqual(metadata["coupling_details"]["fine_pairing_variant"], "model_guided_pool")
                self.assertEqual(metadata["coupling_details"]["teacher"]["checkpoint_sha256"],
                                 bound["tg_cache"]["teacher_checkpoint_sha256"])
                with contextlib.redirect_stdout(io.StringIO()), \
                        patch.object(tg_cache, "load_cache", side_effect=AssertionError("generation cache access")), \
                        patch.object(tg_model_guidance_cache, "_load_teacher",
                                     side_effect=AssertionError("generation teacher load")):
                    if dataset == "checkerboard":
                        output = evaluator(run / "config.yaml", 2, render=False)
                    else:
                        with patch.object(eval_horse, "render_comparison"), \
                                patch("horse_regions.HorseRegions.render"):
                            output = evaluator(run / "config.yaml", 2)
                result = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
                self.assertTrue(np.isfinite(result["chamfer"]))
                self.assertEqual(result["coupling_details"], metadata["coupling_details"])

    def test_cache_array_parent_teacher_lineage_and_bound_sha_are_checked(self):
        config = self.config()
        self.teacher(config)
        for kind in ("array", "parent", "teacher_signature", "teacher_config", "teacher_binding"):
            with self.subTest(kind=kind):
                current = copy.deepcopy(config)
                current["tg_cache"]["path"] += "_" + kind
                path, meta = self.prepare(current)
                current["tg_cache"]["teacher_checkpoint_sha256"] = meta["teacher"]["checkpoint_sha256"]
                if kind == "array":
                    scores = np.load(path / "guidance_scores.npy")
                    scores[0, 0] += 1.
                    np.save(path / "guidance_scores.npy", scores)
                elif kind == "parent":
                    meta["parent_cache"]["metadata"]["cache_seed"] += 1
                    (path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
                elif kind == "teacher_signature":
                    meta["teacher"]["training_signature"]["seed"] += 1
                    (path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
                elif kind == "teacher_config":
                    meta["teacher"]["config"]["seed"] += 1
                    (path / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
                else:
                    current["tg_cache"]["teacher_checkpoint_sha256"] = "0" * 64
                with self.assertRaisesRegex(ValueError, "SHA256|hash|integrity|parent|teacher|fingerprint|signature"):
                    tg_cache.load_cache(current, "checkerboard")

    def test_repreparation_refuses_changed_teacher_and_never_overwrites(self):
        config = self.config()
        _, checkpoint = self.teacher(config)
        path, _ = self.prepare(config)
        before = (path / "metadata.json").read_bytes()
        pool_before = (path / "fine_permutations.npy").read_bytes()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        weight = next(iter(payload["model_state_dict"].values()))
        weight.add_(.125)
        torch.save(payload, checkpoint)
        with self.assertRaisesRegex(ValueError, "teacher|Teacher|SHA256|changed"):
            self.prepare(config)
        self.assertEqual((path / "metadata.json").read_bytes(), before)
        self.assertEqual((path / "fine_permutations.npy").read_bytes(), pool_before)

    def test_prepare_rejects_mismatched_teacher_model_points_dataset_and_legacy(self):
        config = self.config()
        teacher, checkpoint = self.teacher(config)
        for kind in ("model", "points", "dataset", "legacy"):
            with self.subTest(kind=kind):
                current = copy.deepcopy(config)
                current["tg_cache"]["path"] += "_bad_" + kind
                dataset = "checkerboard"
                if kind == "model":
                    current["model"]["d_model"] = 16
                elif kind == "points":
                    current["data"]["n_points"] = 16
                elif kind == "dataset":
                    dataset = "horse"
                else:
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    legacy = Path("legacy.pt").resolve()
                    torch.save(payload["model_state_dict"], legacy)
                    legacy_config = yaml.safe_load(teacher.read_text(encoding="utf-8"))
                    legacy_config["checkpoint"] = str(legacy)
                    legacy_config_path = self.write_config(legacy_config, "legacy.yaml").resolve()
                    current["tg_cache"]["model_guidance"]["teacher_config"] = str(legacy_config_path)
                with self.assertRaises(ValueError):
                    self.prepare(current, dataset)
                self.assertFalse(Path(current["tg_cache"]["path"]).exists())

    def test_incomplete_or_differently_scored_cache_is_not_overwritten(self):
        config = self.config()
        self.teacher(config)
        path, _ = self.prepare(config)
        before = (path / "metadata.json").read_bytes()
        changed = copy.deepcopy(config)
        changed["tg_cache"]["model_guidance"]["weights"]["regression"] = .5
        with self.assertRaises(ValueError):
            self.prepare(changed)
        self.assertEqual((path / "metadata.json").read_bytes(), before)
        incomplete = copy.deepcopy(config)
        incomplete["tg_cache"]["path"] += "_incomplete"
        Path(incomplete["tg_cache"]["path"]).mkdir()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.prepare(incomplete)

    def test_windows_spawn_worker_reads_guided_pool(self):
        config = self.config(workers=1)
        self.teacher(config)
        self.prepare(config)
        sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cpu", torch.float32, training=True)
        try:
            for _ in range(2):
                source, target = sampler.sample(2)
                self.assertEqual(source.shape, (2, 8, 2))
                self.assertTrue(torch.isfinite(target).all())
        finally:
            sampler.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_guidance_then_student_backward(self):
        config = self.config()
        self.teacher(config)
        config["device"] = "cuda"
        self.prepare(config)
        sampler = tg_cache.TGCachedPairSampler(config, "checkerboard", "cuda", torch.float32)
        try:
            source, target = sampler.sample(2)
            model = PointSetTransformer(**config["model"]).cuda()
            loss = train.train_step(model, torch.optim.AdamW(model.parameters()), target,
                                    coupling=config["coupling"], paired_noise=source)
            self.assertTrue(torch.isfinite(loss))
        finally:
            sampler.close()

    def test_supplied_yaml_preserves_hard_training_and_rejects_incompatible_modes(self):
        project = Path(__file__).resolve().parents[1]
        for directory, prefix in (("checkerboard_experiments", "target_guided"),
                                  ("horse_experiments", "horse_target_guided")):
            with self.subTest(dataset=directory):
                baseline = yaml.safe_load((project / directory / f"{prefix}_cached_k8_n256_seed0.yaml").read_text(encoding="utf-8"))
                guided = yaml.safe_load((project / directory / f"{prefix}_cached_model_guided_k8_n256_seed0.yaml").read_text(encoding="utf-8"))
                self.assertEqual({key: value for key, value in baseline.items() if key not in ("tg_cache", "checkpoint")},
                                 {key: value for key, value in guided.items() if key not in ("tg_cache", "checkpoint")})
                self.assertEqual(guided["tg_cache"]["base_path"], baseline["tg_cache"]["path"])
                options = tg_cache.settings(guided)["model_guidance"]
                self.assertEqual(options["candidates"], 8)
                self.assertEqual(options["keep"], 4)
                self.assertEqual(options["times"], [.25, .5, .75])
                changed = copy.deepcopy(guided)
                changed["tg_cache"]["sampling"] = "stream"
                with self.assertRaisesRegex(ValueError, "bank"):
                    tg_cache.settings(changed)
                changed = copy.deepcopy(guided)
                changed["tg_cache"]["obsolete_experiment"] = True
                with self.assertRaisesRegex(ValueError, "Unsupported|Unknown"):
                    tg_cache.settings(changed)

    def test_guided_pool_preserves_independent_original_hard_fresh_pairing(self):
        config = self.config()
        self.teacher(config)
        original = self.baseline(config)
        parent_path, parent_before, digest_before = tg_cache.load_cache(original, "checkerboard")
        arrays_before = {name: (parent_path / f"{name}.npy").read_bytes()
                         for name in parent_before["array_sha256"]}
        guided_path, guided = self.prepare(config)
        self.assertEqual(parent_before["format_version"], 1)
        self.assertNotIn("fine_permutations", parent_before["array_sha256"])
        self.assertEqual(guided["format_version"], 3)
        self.assertIn("fine_permutations", guided["array_sha256"])
        sampler = tg_cache.TGCachedPairSampler(original, "checkerboard", "cpu", torch.float32)
        try:
            target = np.load(parent_path / "target.npy")[0]
            source_labels = np.load(parent_path / "source_labels.npy")[0]
            target_labels = np.load(parent_path / "target_labels.npy")[0]
            first = tg_cache.random_fine_permutation(
                source_labels, target_labels, 2, np.random.default_rng(3))
            second = first.copy()
            patch_indices = np.flatnonzero(source_labels == 0)[:2]
            second[patch_indices] = second[patch_indices[::-1]]
            with patch.object(tg_cache, "random_fine_permutation", side_effect=[first, second]) as fresh:
                source_a, target_a = sampler.dataset.draw(0, np.random.default_rng(14))
                source_b, target_b = sampler.dataset.draw(0, np.random.default_rng(14))
            self.assertEqual(fresh.call_count, 2)
            torch.testing.assert_close(source_a, source_b, rtol=0, atol=0)
            np.testing.assert_array_equal(target_a.numpy(), target[first])
            np.testing.assert_array_equal(target_b.numpy(), target[second])
            self.assertFalse(torch.equal(target_a, target_b))
        finally:
            sampler.close()
        _, parent_after, digest_after = tg_cache.load_cache(original, "checkerboard")
        self.assertEqual(parent_after, parent_before)
        self.assertEqual(digest_after, digest_before)
        for name, before in arrays_before.items():
            self.assertEqual((parent_path / f"{name}.npy").read_bytes(), before)
        self.assertTrue((guided_path / "metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
