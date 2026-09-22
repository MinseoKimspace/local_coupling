"""CPU checks only; actual PSF CUDA kernels are checked by psf_adapter.py."""
import ast
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from eval_3d import cd_matrix, cd_metrics, cloud_cd
from prepare_psf import PSF, main as prepare, provenance
from psf_adapter import PSFVelocity, create_backbone, shapenet_dataset
from sample import integrate_velocity
from train_3d import (ResumableBatchSampler, accumulated_update, restore_rng,
                      resume_signature, rng_state)


class PSFBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        prepare(check=True)

    def test_patch_idempotent(self):
        before = provenance()
        prepare()
        self.assertEqual(before, provenance())

    def test_architecture_matches_upstream(self):
        class Base(torch.nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.arguments = kwargs
        with patch("psf_adapter.upstream_module", return_value=SimpleNamespace(PVCNN2Base=Base)), \
                patch("torch.cuda.is_available", return_value=True):
            model = create_backbone()
        tree = ast.parse((PSF / "train_flow.py").read_text(encoding="utf-8"))
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PVCNN2")
        for node in original.body:
            if isinstance(node, ast.Assign) and node.targets[0].id in ("sa_blocks", "fp_blocks"):
                self.assertEqual(getattr(model, node.targets[0].id), ast.literal_eval(node.value))
        self.assertEqual(model.arguments["extra_feature_channels"], 0)
        self.assertEqual(model.arguments["num_classes"], 3)

    def test_layout_time_scale_and_integration(self):
        class Backbone(torch.nn.Module):
            def forward(self, points, time):
                self.observed = (points.shape, time.clone())
                return torch.ones_like(points)
        backbone = Backbone()
        with patch("psf_adapter.create_backbone", return_value=backbone):
            model = PSFVelocity()
        points = torch.zeros(2, 1024, 3)
        # Simulates only the interface. It does not exercise GPU kernels.
        with patch.object(torch.Tensor, "is_cuda", new=property(lambda self: True)):
            velocity = model(points, torch.tensor([0.0, 0.5]).reshape(2, 1, 1))
            self.assertEqual(backbone.observed[0], (2, 3, 1024))
            torch.testing.assert_close(backbone.observed[1], torch.tensor([0.0, 499.5]))
            self.assertEqual(velocity.shape, points.shape)
            torch.testing.assert_close(integrate_velocity(model, points, num_steps=4), points + 1)
        self.assertTrue(model.training)

    def test_cd_matches_dense_and_distribution_metrics(self):
        torch.manual_seed(3)
        x, y = torch.randn(3, 13, 3), torch.randn(3, 17, 3)
        d = torch.cdist(x, y).square()
        expected = d.amin(2).mean(1) + d.amin(1).mean(1)
        torch.testing.assert_close(cloud_cd(x, y, chunk=4), expected)
        cross = cd_matrix(x, y, batch_size=2)
        torch.testing.assert_close(cross.diagonal(), expected)
        # Identical sets of two distinct clouds: cross-set nearest neighbor, no self matches.
        shapes = torch.stack((torch.zeros(8, 3), torch.ones(8, 3)))
        scores = cd_metrics(shapes, shapes, batch_size=1)
        self.assertEqual(scores, {"mmd_cd": 0.0, "cov_cd": 1.0, "one_nna_cd": 0.0})

    def test_upstream_loader_reuses_training_normalization(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(4)
            for split in ("train", "val"):
                directory = root / "03001627" / split
                directory.mkdir(parents=True)
                for i in (1, 0):
                    np.save(directory / f"shape{i}.npy", rng.normal(size=(15000, 3)).astype("float32"))
            with contextlib.redirect_stdout(io.StringIO()):
                train = shapenet_dataset(root, "chair", 1024)
                stats = {"mean": train.all_points_mean.tolist(), "std": train.all_points_std.tolist()}
                val = shapenet_dataset(root, "chair", 1024, split="val", normalization=stats)
            np.testing.assert_allclose(val.all_points_mean, train.all_points_mean)
            np.testing.assert_allclose(val.all_points_std, train.all_points_std)
            self.assertEqual(val[0]["train_points"].shape, (1024, 3))
            # No CUDA backend / Open3D / EMD is needed to use this dataset path.
            with self.assertRaises(ValueError):
                shapenet_dataset(root, "chair", 15000)


class TrainingProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_configs_differ_only_in_coupling(self):
        import yaml
        root = Path(__file__).resolve().parents[1]
        configs = [yaml.safe_load((root / "psf_experiments" / name).read_text())
                   for name in ("independent.yaml", "target_guided.yaml")]
        self.assertEqual([c.pop("coupling") for c in configs], ["independent", "target_guided_exact_optimized"])
        self.assertEqual(configs[0], configs[1])
        config = configs[0]
        self.assertEqual(config["training"]["num_steps"], 600000)
        self.assertEqual(config["data"]["batch_size"] * config["training"]["accumulation_steps"], 256)
        self.assertEqual(config["training"]["learning_rate"], 2e-4)
        self.assertNotIn("lr_gamma", config["training"])
        self.assertNotIn("ema_decay", config["training"])

    def test_accumulation_matches_full_batch_adam_update(self):
        torch.manual_seed(4)
        model = torch.nn.Linear(3, 3).double()
        reference = copy.deepcopy(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(.5, .999))
        full_optimizer = torch.optim.Adam(reference.parameters(), lr=2e-4, betas=(.5, .999))
        x = torch.randn(8, 3, dtype=torch.float64)
        loss_fn = lambda net, points: (net(points) - points).square().mean()
        value = accumulated_update(model, optimizer, iter(x.split(2)), 4, loss_fn)
        full_optimizer.zero_grad()
        expected = loss_fn(reference, x)
        expected.backward()
        full_optimizer.step()
        self.assertAlmostEqual(value, expected.item(), places=12)
        for actual, wanted in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=1e-12)
            self.assertEqual(optimizer.state[actual]["step"].item(), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 2e-4)

    def test_nonfinite_loss_does_not_update_weights(self):
        model = torch.nn.Linear(3, 3)
        before = copy.deepcopy(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters())
        batches = iter([torch.ones(2, 3), torch.full((2, 3), float("nan"))])
        with self.assertRaises(FloatingPointError):
            accumulated_update(model, optimizer, batches, 2, lambda net, x: net(x).square().mean())
        self.assertEqual(optimizer.state, {})
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        with self.assertRaisesRegex(ValueError, "equal microbatch"):
            accumulated_update(model, optimizer, iter([torch.ones(2, 3), torch.ones(1, 3)]), 2,
                               lambda net, x: net(x).square().mean())

    def test_sampler_roundtrip_across_epoch(self):
        sampler = ResumableBatchSampler(13, 4, 2)
        iterator = iter(sampler)
        first, second = next(iterator), next(iterator)
        self.assertEqual(len(set(first + second)), 8)
        state = sampler.state_dict()
        expected = [next(iterator) for _ in range(5)]
        resumed = ResumableBatchSampler(13, 4, 999)
        resumed.load_state_dict(state)
        iterator = iter(resumed)
        self.assertEqual([next(iterator) for _ in range(5)], expected)

    def test_resume_signature_rejects_training_changes(self):
        config = {"coupling": "independent", "data": {"root": "a", "batch_size": 16},
                  "training": {"num_steps": 100, "accumulation_steps": 16, "learning_rate": 2e-4}}
        changed = copy.deepcopy(config)
        changed["data"]["root"] = "b"
        changed["training"]["num_steps"] = 600000
        self.assertEqual(resume_signature(config), resume_signature(changed))
        changed["training"]["learning_rate"] = 1e-4
        self.assertNotEqual(resume_signature(config), resume_signature(changed))

    def test_toy_resume_matches_uninterrupted_training(self):
        # CPU toy network: tests accumulation + optimizer + dropout + loader/RNG
        # restoration, NOT PVCNN CUDA kernels or the POT assignment solver.
        class Dataset(torch.utils.data.Dataset):
            def __len__(self):
                return 13
            def __getitem__(self, i):
                return torch.tensor(np.random.normal(size=3) + random.random() + i / 13, dtype=torch.float32)

        def setup():
            net = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.2), torch.nn.Linear(5, 3))
            opt = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(.5, .999))
            sampler = ResumableBatchSampler(13, 2, 4)
            loader = torch.utils.data.DataLoader(Dataset(), batch_sampler=sampler, num_workers=0,
                                                 generator=torch.Generator().manual_seed(8))
            return net, opt, sampler, loader, torch.Generator().manual_seed(7)

        def advance(net, opt, batches, pair_rng):
            def loss_fn(net, target):
                noise = torch.randn_like(target)
                t = torch.rand(len(target), 1)
                # A separate pairing stream is saved independently from noise/time.
                target = target[:, torch.randperm(3, generator=pair_rng)]
                return (net((1-t)*noise + t*target) - (target-noise)).square().mean()
            return accumulated_update(net, opt, batches, 3, loss_fn)

        with patch("torch.cuda.is_available", return_value=False):
            random.seed(1)
            np.random.seed(1)
            torch.manual_seed(1)
            net, opt, sampler, loader, pair_rng = setup()
            batches = iter(loader)
            for _ in range(2):
                advance(net, opt, batches, pair_rng)
            buffer = io.BytesIO()
            torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(),
                        "sampler": sampler.state_dict(), "rng": rng_state(pair_rng)}, buffer)
            expected_losses = [advance(net, opt, batches, pair_rng) for _ in range(3)]
            buffer.seek(0)
            saved = torch.load(buffer, weights_only=True)
            resumed, resumed_opt, sampler, loader, pair_rng = setup()
            resumed.load_state_dict(saved["model"])
            resumed_opt.load_state_dict(saved["optimizer"])
            sampler.load_state_dict(saved["sampler"])
            restore_rng(saved["rng"], pair_rng)
            batches = iter(loader)
            actual_losses = [advance(resumed, resumed_opt, batches, pair_rng) for _ in range(3)]
            self.assertEqual(actual_losses, expected_losses)
            for a, b in zip(net.parameters(), resumed.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                for key in ("step", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(opt.state[a][key], resumed_opt.state[b][key], rtol=0, atol=0)
            self.assertEqual(resumed_opt.param_groups[0]["lr"], 2e-4)


if __name__ == "__main__":
    unittest.main()
