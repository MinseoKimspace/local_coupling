"""CPU checks only; actual PSF CUDA kernels are checked by psf_adapter.py."""
import ast
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from eval_3d import cd_matrix, cd_metrics, cloud_cd
from prepare_psf import PSF, main as prepare, provenance
from psf_adapter import PSFVelocity, create_backbone, shapenet_dataset
from sample import integrate_velocity


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


if __name__ == "__main__":
    unittest.main()
