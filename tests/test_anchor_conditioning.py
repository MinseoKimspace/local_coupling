"""Source-component conditioning is an isolated model-input ablation.

The same anchor prior, cached OT pairs, hybrid beta and FM loss are retained.
Labels come from the original Gaussian draw and remain fixed along rollouts.
"""

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
from torch import nn
import yaml

import anchor_conditioning
import anchor_flow
import audit_generation
import audit_mean_field
from diagnostic_data import DiagnosticData
from diagnostic_data import tensor_sha256
import diagnose
import eval as eval_checkerboard
import eval_horse
import experiment
from model import PointSetTransformer
import nsot
import sample
import train
import train_horse


def tiny_config(dataset="checkerboard", *, conditioned=True):
    config = {
        "seed": 0, "device": "cpu", "dtype": "float32", "coupling": "nsot",
        "num_regions": 2,
        "anchor_flow": {"mode": "anchor_prior", "sigma": .1,
                        "reference_points": 64, "seed": 11},
        "nsot": {"cache": f"{dataset}_anchor_pairs.npz", "superset_size": 64,
                 "cache_seed": 5, "beta": .2, "solver": nsot.SOLVER},
        "data": {"batch_size": 2, "n_points": 8},
        "model": {"point_dim": 2, "d_model": 8, "nhead": 2, "num_layers": 1,
                  "dim_feedforward": 16, "dropout": 0.0},
        "training": {"num_steps": 1, "learning_rate": .001, "weight_decay": .01,
                     "log_every": 1},
        "evaluation": {"batch_size": 2, "histogram_bins": 8},
        "checkpoint": f"{dataset}_anchor_id.pt",
    }
    if conditioned:
        config["model"]["anchor_id_count"] = 2
    if dataset == "checkerboard":
        config["data"]["grid_size"] = 4
    return config


class LabelVelocity(nn.Module):
    """A transparent conditioned field to detect lost/reassigned labels."""

    def __init__(self):
        super().__init__()
        self.anchor_id_count = 2
        self.scale = nn.Parameter(torch.ones(()))
        self.seen = []

    def forward(self, x_t, t, component_ids=None):
        if component_ids is None:
            raise AssertionError("Source component IDs were dropped")
        self.seen.append((x_t.detach().clone(), t.detach().clone(), component_ids))
        return self.scale * component_ids.to(x_t.dtype).unsqueeze(-1).expand_as(x_t)


class AnchorConditioningUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model_pairs(self, *, count=2):
        options = tiny_config(conditioned=False)["model"]
        for model_class in (PointSetTransformer, train_horse.HorsePointSetTransformer):
            torch.manual_seed(73)
            baseline = model_class(**options)
            baseline_rng = torch.get_rng_state().clone()
            torch.manual_seed(73)
            conditioned = model_class(**options, anchor_id_count=count)
            yield model_class, baseline, conditioned, baseline_rng

    def test_opt_in_settings_validate_without_resolving_centers(self):
        config = tiny_config()
        before = copy.deepcopy(config)
        self.assertIsNotNone(anchor_conditioning.settings(config))
        self.assertEqual(config, before)
        self.assertNotIn("centers", config["anchor_flow"])
        disabled = tiny_config(conditioned=False)
        self.assertIsNone(anchor_conditioning.settings(disabled))
        disabled["model"]["anchor_id_count"] = 0
        self.assertIsNone(anchor_conditioning.settings(disabled))
        disabled.pop("anchor_flow")
        disabled["coupling"] = "independent"
        self.assertIsNone(anchor_conditioning.settings(disabled))

    def test_invalid_counts_and_incompatible_experiments_are_rejected(self):
        for count in (-1, True, 1.5, "2", None, 3):
            config = tiny_config()
            config["model"]["anchor_id_count"] = count
            with self.subTest(count=count), self.assertRaises(ValueError):
                anchor_conditioning.settings(config)
        for change in ("coupling", "prior", "dimension", "dtype", "directional"):
            config = tiny_config()
            if change == "coupling":
                config["coupling"] = "target_guided_cached"
            elif change == "prior":
                config.pop("anchor_flow")
            elif change == "dimension":
                config["model"]["point_dim"] = 3
            elif change == "dtype":
                config["dtype"] = "float64"
            else:
                config["nsot"]["directional_hybrid"] = {"artifact": "directional.json"}
            with self.subTest(change=change), self.assertRaises(ValueError):
                anchor_conditioning.settings(config)

    def test_zero_embedding_preserves_backbone_weights_output_and_initialization_rng(self):
        for model_class, baseline, conditioned, baseline_rng in self.model_pairs():
            with self.subTest(model=model_class.__name__):
                torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
                self.assertIsNone(baseline.anchor_embedding)
                self.assertEqual(conditioned.anchor_id_count, 2)
                self.assertIsInstance(conditioned.anchor_embedding, nn.Embedding)
                torch.testing.assert_close(conditioned.anchor_embedding.weight,
                                           torch.zeros(2, 8), rtol=0, atol=0)
                conditioned_state = conditioned.state_dict()
                self.assertEqual(set(conditioned_state) - set(baseline.state_dict()),
                                 {"anchor_embedding.weight"})
                for name, value in baseline.state_dict().items():
                    torch.testing.assert_close(value, conditioned_state[name], rtol=0, atol=0)
                x = torch.randn(2, 8, 2)
                time = torch.rand(2, 1, 1)
                labels = torch.randint(2, (2, 8))
                torch.testing.assert_close(conditioned(x, time, component_ids=labels),
                                           baseline(x, time), rtol=0, atol=0)

    def test_explicit_zero_has_identical_state_and_draws_to_absent_option(self):
        for model_class, baseline, disabled, baseline_rng in self.model_pairs(count=0):
            with self.subTest(model=model_class.__name__):
                torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
                self.assertIsNone(disabled.anchor_embedding)
                self.assertEqual(set(baseline.state_dict()), set(disabled.state_dict()))
                for name, value in baseline.state_dict().items():
                    torch.testing.assert_close(value, disabled.state_dict()[name], rtol=0, atol=0)

    def test_component_labels_are_required_and_strictly_typed_when_enabled(self):
        for model_class, baseline, conditioned, _ in self.model_pairs():
            x, time = torch.randn(2, 8, 2), torch.rand(2, 1, 1)
            for labels in (None, torch.zeros(2, 8), torch.zeros(2, 8, dtype=torch.int32),
                           torch.zeros(8, dtype=torch.long), torch.zeros(2, 8, 1, dtype=torch.long),
                           torch.zeros(2, 7, dtype=torch.long)):
                with self.subTest(model=model_class.__name__, shape=getattr(labels, "shape", None)), \
                        self.assertRaises(ValueError):
                    conditioned(x, time, component_ids=labels)
            with self.assertRaises(ValueError):
                baseline(x, time, component_ids=torch.zeros(2, 8, dtype=torch.long))
            for bad in (-1, 2):
                with self.assertRaises((ValueError, IndexError)):
                    conditioned(x, time, component_ids=torch.full((2, 8), bad, dtype=torch.long))

    def test_conditioning_remains_jointly_permutation_equivariant(self):
        permutation = torch.tensor([3, 1, 7, 0, 2, 5, 4, 6])
        for model_class, _, model, _ in self.model_pairs():
            with torch.no_grad():
                model.anchor_embedding.weight.normal_(std=.2)
            model.eval()
            x, time = torch.randn(2, 8, 2), torch.rand(2, 1, 1)
            labels = torch.randint(2, (2, 8))
            expected = model(x, time, component_ids=labels)[:, permutation]
            actual = model(x[:, permutation], time, component_ids=labels[:, permutation])
            with self.subTest(model=model_class.__name__):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_used_embedding_rows_receive_gradients_and_unused_rows_do_not(self):
        for model_class, _, model, _ in self.model_pairs(count=4):
            x, time = torch.randn(2, 8, 2), torch.rand(2, 1, 1)
            labels = torch.tensor([[0, 2, 0, 2, 0, 2, 0, 2]]).expand(2, -1)
            loss = train.flow_matching_loss(model, x + .4, x, time, component_ids=labels)
            loss.backward()
            gradient = model.anchor_embedding.weight.grad
            with self.subTest(model=model_class.__name__):
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertGreater(gradient[0].abs().sum().item(), 0)
                self.assertGreater(gradient[2].abs().sum().item(), 0)
                torch.testing.assert_close(gradient[[1, 3]], torch.zeros(2, 8), rtol=0, atol=0)

    def test_helper_preserves_source_coordinates_rng_and_original_component_ids(self):
        config = tiny_config()
        # Coincident centers make nearest-center recovery incapable of retaining IDs.
        anchor_flow.bind_prior_centers(config, [[0., 0.], [0., 0.]])
        generator = torch.Generator().manual_seed(38)
        actual, labels = anchor_conditioning.sample_source_for_model(
            config, 4, device="cpu", dtype=torch.float32, generator=generator)
        replay = torch.Generator().manual_seed(38)
        expected, expected_labels = anchor_flow.sample_source_with_components(
            config, 4, device="cpu", dtype=torch.float32, generator=replay)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(labels, expected_labels, rtol=0, atol=0)
        torch.testing.assert_close(generator.get_state(), replay.get_state(), rtol=0, atol=0)
        self.assertEqual(set(labels.flatten().tolist()), {0, 1})
        disabled = copy.deepcopy(config)
        disabled["model"].pop("anchor_id_count")
        replay = torch.Generator().manual_seed(38)
        actual, ids = anchor_conditioning.sample_source_for_model(
            disabled, 4, device="cpu", dtype=torch.float32, generator=replay)
        self.assertIsNone(ids)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(generator.get_state(), replay.get_state(), rtol=0, atol=0)

    def test_unconditioned_helper_keeps_original_two_argument_model_api(self):
        class OldVelocity(nn.Module):
            def forward(self, x, t):
                return x + t
        x, time = torch.randn(2, 8, 2), torch.rand(2, 1, 1)
        expected = OldVelocity()(x, time)
        actual = anchor_conditioning.velocity(OldVelocity(), x, time)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rollout_and_snapshot_paths_keep_original_ids_at_every_step(self):
        model = LabelVelocity()
        model.train()
        source = torch.randn(2, 8, 2)
        labels = torch.randint(2, (2, 8))
        expected_velocity = labels.float().unsqueeze(-1).expand_as(source)
        actual = sample.integrate_velocity(model, source, num_steps=4, component_ids=labels)
        torch.testing.assert_close(actual, source + expected_velocity, rtol=1e-6, atol=1e-6)
        self.assertTrue(model.training)
        self.assertEqual(len(model.seen), 4)
        self.assertTrue(all(row[2] is labels for row in model.seen))
        model.seen.clear()
        snapshots = eval_checkerboard.sample_snapshots(model, source, 4, (1, 2, 4), component_ids=labels)
        for step, points in snapshots.items():
            torch.testing.assert_close(points, source + (step / 4) * expected_velocity,
                                       rtol=1e-6, atol=1e-6)
        self.assertTrue(all(row[2] is labels for row in model.seen))
        self.assertTrue(model.training)

    def test_fm_losses_use_same_linear_path_target_and_component_labels(self):
        model = LabelVelocity()
        source, target, time = torch.randn(2, 8, 2), torch.randn(2, 8, 2), torch.rand(2, 1, 1)
        labels = torch.randint(2, (2, 8))
        expected = ((labels.float().unsqueeze(-1).expand_as(source) - (target - source))**2).mean()
        actual = train.flow_matching_loss(model, target, source, time, component_ids=labels)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(model.seen[-1][0], train.linear_path(target, source, time), rtol=0, atol=0)
        self.assertIs(model.seen[-1][2], labels)
        config = tiny_config()
        with patch.object(train, "sample_time", return_value=time), \
                patch.object(train, "coupled_points", side_effect=AssertionError("Online coupling")):
            actual = train.coupled_flow_matching_loss(
                model, target, coupling="nsot", paired_noise=source,
                anchor_config=config, component_ids=labels)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_evaluation_warmup_and_integration_share_sampling_labels(self):
        config = tiny_config()
        anchor_flow.bind_prior_centers(config, [[-.8, -.2], [.7, .3]])
        model = LabelVelocity()
        torch.manual_seed(151)
        expected_source, expected_labels = anchor_flow.sample_source_with_components(
            config, 2, device="cpu", dtype=torch.float32)
        torch.manual_seed(151)
        source, prediction, seconds, labels = experiment.sample_for_evaluation(
            model, config, 2, return_components=True)
        torch.testing.assert_close(source, expected_source, rtol=0, atol=0)
        torch.testing.assert_close(labels, expected_labels, rtol=0, atol=0)
        torch.testing.assert_close(prediction, source + labels.float().unsqueeze(-1), rtol=1e-6, atol=1e-6)
        self.assertGreaterEqual(seconds, 0)
        self.assertEqual(len(model.seen), 12)
        self.assertTrue(all(row[2] is labels for row in model.seen))
        torch.manual_seed(151)
        default = experiment.sample_for_evaluation(model, config, 2)
        self.assertEqual(len(default), 3)
        torch.testing.assert_close(default[0], source, rtol=0, atol=0)
        torch.testing.assert_close(default[1], prediction, rtol=0, atol=0)

    def test_diagnostic_banks_retain_original_ids_without_changing_indexed_draws(self):
        config = tiny_config()
        anchor_flow.bind_prior_centers(config, [[0., 0.], [0., 0.]])
        data = DiagnosticData(config, "checkerboard", "cpu", torch.float32, 2026)
        source, target, labels = data.bank_with_components(3)
        original_source, original_target = data.bank(3)
        torch.testing.assert_close(source, original_source, rtol=0, atol=0)
        torch.testing.assert_close(target, original_target, rtol=0, atol=0)
        for index in range(3):
            generator = torch.Generator().manual_seed(data.draw_seed("evaluation:source", index))
            expected_source, expected_ids = anchor_flow.sample_source_with_components(
                config, 1, device="cpu", dtype=torch.float32, generator=generator)
            torch.testing.assert_close(source[index], expected_source[0], rtol=0, atol=0)
            torch.testing.assert_close(labels[index], expected_ids[0], rtol=0, atol=0)

    def test_diagnostic_rollouts_and_fm_errors_use_fixed_source_ids(self):
        model = LabelVelocity()
        source, target = torch.randn(3, 8, 2), torch.randn(3, 8, 2)
        labels = torch.randint(2, (3, 8))
        velocity = labels.float().unsqueeze(-1).expand_as(source)
        time = torch.rand(3, 1, 1)
        residual, energy = diagnose.fm_errors(model, source, target, time, component_ids=labels)
        torch.testing.assert_close(residual, (velocity - target + source).square().mean((1, 2)))
        torch.testing.assert_close(energy, (target - source).square().mean((1, 2)))
        prediction, ratio, path = diagnose.rollout(model, source, 4, keep_path=True, component_ids=labels)
        torch.testing.assert_close(prediction, source + velocity, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=1e-6, atol=1e-6)
        self.assertEqual(path.shape, (5, 8, 2))
        self.assertTrue(all(row[2] is labels for row in model.seen))
        model.seen.clear()
        prediction, _, _, seconds = audit_generation.rollout_bank(
            model, source, 4, batch_size=2, component_ids=labels)
        torch.testing.assert_close(prediction, source + velocity, rtol=1e-6, atol=1e-6)
        self.assertGreaterEqual(seconds, 0)
        # Every microbatch keeps the same labels through all of its Euler steps.
        for start in range(0, len(model.seen), 4):
            group = model.seen[start:start + 4]
            self.assertTrue(all(row[2] is group[0][2] for row in group))

    def test_templates_isolate_id_conditioning_from_prior_bank_beta_and_backbone(self):
        root = Path(__file__).resolve().parents[1]
        for directory, filename, baseline_filename in (
                ("checkerboard_experiments", "nsot_anchor_prior_id_k8_n256_seed0.yaml",
                 "nsot_anchor_prior_k8_n256_seed0.yaml"),
                ("horse_experiments", "horse_nsot_anchor_prior_id_k8_n256_seed0.yaml",
                 "horse_nsot_anchor_prior_k8_n256_seed0.yaml")):
            config = yaml.safe_load((root / directory / filename).read_text(encoding="utf-8"))
            baseline = yaml.safe_load((root / directory / baseline_filename).read_text(encoding="utf-8"))
            with self.subTest(dataset=directory):
                self.assertEqual(config["model"]["anchor_id_count"], config["num_regions"])
                self.assertNotIn("directional_hybrid", config["nsot"])
                self.assertIsNotNone(anchor_conditioning.settings(config))
                compare = copy.deepcopy(config)
                compare["model"].pop("anchor_id_count")
                compare["checkpoint"] = baseline["checkpoint"]
                self.assertEqual(compare, baseline)


class AnchorConditioningIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.previous = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        os.chdir(self.temp.name)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self):
        os.chdir(self.previous)
        self.temp.cleanup()
        torch.set_num_threads(self.previous_threads)

    @staticmethod
    def prepare(config, dataset="checkerboard"):
        with contextlib.redirect_stdout(io.StringIO()):
            return nsot.prepare(config, dataset)

    def test_same_bank_is_reused_and_returned_labels_do_not_advance_rng(self):
        baseline = tiny_config(conditioned=False)
        path, _ = self.prepare(baseline)
        original_bytes = path.read_bytes()
        config = tiny_config()
        with patch.object(nsot, "exact_superset_permutation", side_effect=AssertionError("OT refit")):
            self.prepare(config)
        self.assertEqual(path.read_bytes(), original_bytes)
        sampler = nsot.NSOTPairSampler(config, "checkerboard", "cpu", torch.float32)
        plain_generator = torch.Generator().manual_seed(42)
        source, target = sampler.sample(3, generator=plain_generator)
        label_generator = torch.Generator().manual_seed(42)
        actual_source, actual_target, labels = sampler.sample(
            3, generator=label_generator, return_components=True)
        torch.testing.assert_close(source, actual_source, rtol=0, atol=0)
        torch.testing.assert_close(target, actual_target, rtol=0, atol=0)
        torch.testing.assert_close(plain_generator.get_state(), label_generator.get_state(), rtol=0, atol=0)
        replay = torch.Generator().manual_seed(42)
        indices = torch.randint(64, (3, 8), generator=replay)
        torch.testing.assert_close(labels, sampler.source_components[indices], rtol=0, atol=0)

    def test_loss_and_training_step_fail_if_conditioned_labels_are_missing(self):
        config = tiny_config()
        self.prepare(config)
        sampler = nsot.NSOTPairSampler(config, "checkerboard", "cpu", torch.float32)
        source, target, labels = sampler.sample(2, return_components=True)
        model = PointSetTransformer(**config["model"])
        optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaises(ValueError):
            train.train_step(model, optimizer, target, coupling="nsot", paired_noise=source,
                             anchor_config=config)
        loss = train.train_step(model, optimizer, target, coupling="nsot", paired_noise=source,
                                anchor_config=config, component_ids=labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.anchor_embedding.weight.detach().abs().sum().item(), 0)

    def test_analytic_mean_field_rejects_conditioned_bank_law_before_loading_checkpoint(self):
        path = Path("conditioned.yaml")
        path.write_text(yaml.safe_dump(tiny_config()), encoding="utf-8")
        with patch.object(audit_mean_field, "load_verified_model", side_effect=AssertionError("Loaded checkpoint")), \
                self.assertRaisesRegex(ValueError, "anchor-ID-conditioned NSOT"):
            audit_mean_field.audit(path, "checkerboard", clouds=1, targets=2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_original_component_sampling_and_embedding_backward(self):
        config = tiny_config()
        self.prepare(config)
        device = torch.device("cuda")
        sampler = nsot.NSOTPairSampler(config, "checkerboard", device, torch.float32)
        generator = torch.Generator(device=device).manual_seed(42)
        source, target, labels = sampler.sample(2, generator=generator, return_components=True)
        replay = torch.Generator(device=device).manual_seed(42)
        indices = torch.randint(64, (2, 8), device=device, generator=replay)
        torch.testing.assert_close(labels, sampler.source_components[indices], rtol=0, atol=0)
        self.assertEqual(labels.device.type, "cuda")
        self.assertEqual(labels.dtype, torch.long)
        for model_class in (PointSetTransformer, train_horse.HorsePointSetTransformer):
            model = model_class(**config["model"]).to(device)
            with self.assertRaises(ValueError):
                model(source, torch.zeros(2, 1, 1, device=device), component_ids=labels.cpu())
            loss = train.train_step(model, torch.optim.AdamW(model.parameters()), target,
                                    coupling="nsot", paired_noise=source,
                                    anchor_config=config, component_ids=labels)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(model.anchor_embedding.weight.grad).all())
            prediction = sample.integrate_velocity(model, source, num_steps=2, component_ids=labels)
            self.assertTrue(torch.isfinite(prediction).all())

    def test_both_datasets_train_checkpoint_reload_diagnostics_and_cache_free_eval(self):
        with contextlib.redirect_stdout(io.StringIO()):
            for dataset, trainer, evaluator, model_class in (
                    ("checkerboard", train.main, eval_checkerboard.main, PointSetTransformer),
                    ("horse", train_horse.main, eval_horse.main, train_horse.HorsePointSetTransformer)):
                config = tiny_config(dataset)
                config_path = Path(f"{dataset}.yaml")
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                self.prepare(config, dataset)
                with patch.object(train, "coupled_points", side_effect=AssertionError("Online OT")):
                    run = trainer(config_path, steps=1)
                model, saved, checkpoint, metadata = experiment.load_model(
                    run / "config.yaml", model_class, dataset)
                self.assertEqual(saved["model"]["anchor_id_count"], 2)
                self.assertTrue(metadata["training_config_verified"])
                self.assertIn("anchor_prior_id", run.name)
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.assertIn("anchor_embedding.weight", payload["model_state_dict"])
                self.assertEqual(payload["config"]["model"], saved["model"])
                self.assertGreater(model.anchor_embedding.weight.detach().abs().sum().item(), 0)
                data = DiagnosticData(saved, dataset, "cpu", torch.float32, 2026)
                fm = audit_generation.fm_summary(model, saved, data, 1, 2, 2026)
                self.assertTrue(all(np.isfinite(row["mse"]["mean"]) for row in fm["time_errors"]))
                with patch.object(audit_generation, "render"), patch("horse_regions.HorseRegions.render"):
                    directory = audit_generation.audit(
                        run / "config.yaml", dataset, clouds=2, batch_size=1,
                        nfes=(1, 2), reference_nfe=2, max_reference_nfe=8,
                        fm_batches=1, matching_batch_size=2, seed=2026)
                diagnostic = json.loads((directory / "diagnostics.json").read_text(encoding="utf-8"))
                expected_source, expected_target, expected_ids = data.bank_with_components(2)
                self.assertEqual(diagnostic["noise_sha256"], tensor_sha256(expected_source))
                self.assertEqual(diagnostic["target_sha256"], tensor_sha256(expected_target))
                self.assertEqual(diagnostic["source_components_sha256"], tensor_sha256(expected_ids))
                self.assertEqual(sum(diagnostic["source_component_counts"]), 16)
                pair_cache = Path(saved["nsot"]["cache"])
                pair_cache.rename(pair_cache.with_suffix(".backup"))
                with patch.object(nsot, "load_cache", side_effect=AssertionError("Inference cache IO")), \
                        patch.object(anchor_flow, "fit_prior_centers", side_effect=AssertionError("Inference refit")):
                    if dataset == "checkerboard":
                        with patch.object(eval_checkerboard, "render_density"):
                            output = evaluator(run / "config.yaml", 2, render=True)
                    else:
                        with patch.object(eval_horse, "render_comparison"), patch("horse_regions.HorseRegions.render"):
                            output = evaluator(run / "config.yaml", 2)
                result = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
                self.assertTrue(np.isfinite(result["chamfer"]))
                self.assertEqual(result["config"]["model"]["anchor_id_count"], 2)
                self.assertEqual(result["coupling_details"], metadata["coupling_details"])
                changed = copy.deepcopy(saved)
                changed["model"].pop("anchor_id_count")
                tampered_path = run / "tampered.yaml"
                tampered_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "training settings"):
                    experiment.load_model(tampered_path, model_class, dataset)


if __name__ == "__main__":
    unittest.main()
