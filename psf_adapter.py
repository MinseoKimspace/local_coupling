"""PSF/PVCNN bridge. Upstream: https://github.com/klightz/PSF (MIT).

PVCNN2 block configuration below is from upstream train_flow.py.
Copyright (c) 2023 Lemeng Wu; full license: third_party/PSF/LICENSE.
No import of upstream train_flow/test_flow, Open3D, PyTorch3D, or EMD.
"""
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from prepare_psf import PSF, ROOT, provenance


def upstream_module(name, relative_path):
    if name not in sys.modules:
        provenance()
        spec = importlib.util.spec_from_file_location(name, PSF / relative_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            del sys.modules[name]
            raise
    return sys.modules[name]


def create_backbone(embed_dim=64, use_att=True, dropout=0.1):
    if not torch.cuda.is_available():
        raise RuntimeError("PSF PVCNN requires CUDA; there is no CPU fallback.")
    # PSF uses absolute 'modules.*' imports. Avoid its 'model' package:
    # this repository already has a model.py, also imported by train.py.
    existing = sys.modules.get("modules")
    if existing is not None and Path(existing.__file__).resolve().parent != PSF / "modules":
        raise RuntimeError("An unrelated 'modules' package is already imported; use a fresh process.")
    cache = ROOT / ".torch_extensions"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(cache))
    os.environ.setdefault("MAX_JOBS", "2")
    sys.path.insert(0, str(PSF))
    try:
        base = upstream_module("_psf_pvcnn_generation", "model/pvcnn_generation.py").PVCNN2Base
    finally:
        sys.path.remove(str(PSF))

    class PVCNN2(base):
        sa_blocks = [
            ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
            ((64, 3, 16), (256, 0.2, 32, (64, 128))),
            ((128, 3, 8), (64, 0.4, 32, (128, 256))),
            (None, (16, 0.8, 32, (256, 256, 512))),
        ]
        fp_blocks = [
            ((256, 256), (256, 3, 8)),
            ((256, 256), (256, 3, 8)),
            ((256, 128), (128, 2, 16)),
            ((128, 128, 64), (64, 2, 32)),
        ]

    return PVCNN2(num_classes=3, embed_dim=embed_dim, use_att=use_att,
                  dropout=dropout, extra_feature_channels=0)


class PSFVelocity(nn.Module):
    """Repository interface [B,N,3], t in [0,1]; PSF interface [B,3,N], t*999."""
    def __init__(self, embed_dim=64, use_att=True, dropout=0.1):
        super().__init__()
        self.backbone = create_backbone(embed_dim, use_att, dropout)

    def forward(self, points, time):
        if points.ndim != 3 or points.shape[-1] != 3 or points.shape[1] < 1024:
            raise ValueError("Stock PSF requires [B,N,3] with N >= 1024.")
        if points.dtype != torch.float32 or not points.is_cuda:
            raise ValueError("PSF custom kernels require CUDA float32 tensors.")
        t = time.reshape(points.shape[0]) * 999.0
        return self.backbone(points.transpose(1, 2).contiguous(), t).transpose(1, 2).contiguous()


def shapenet_dataset(root, category, n_points, *, split="train", normalization=None):
    if not 1024 <= n_points <= 10000:
        raise ValueError("PSF loader training subset supports 1024 <= N <= 10000.")
    module = upstream_module("_psf_shapenet", "datasets/shapenet_data_pc.py")
    if category not in module.cate_to_synsetid:
        raise ValueError(f"Unknown ShapeNet category: {category}")
    directory = Path(root) / module.cate_to_synsetid[category] / split
    if not directory.is_dir() or not any(directory.glob("*.npy")):
        raise FileNotFoundError(f"Expected ShapeNet PC15k .npy files in {directory}")
    stats = {} if normalization is None else {
        "all_points_mean": np.asarray(normalization["mean"]),
        "all_points_std": np.asarray(normalization["std"]),
    }
    dataset = module.ShapeNet15kPointClouds(
        root_dir=str(root), categories=[category], split=split,
        tr_sample_size=n_points, te_sample_size=min(n_points, 5000),
        random_subsample=True, normalize_per_shape=False, normalize_std_per_axis=False,
        reflow=False, use_mask=False, **stats)
    if not np.isfinite(dataset.all_points).all():
        raise ValueError("Nonfinite ShapeNet coordinates after normalization")
    return dataset


def smoke_test(coupling, n_points):
    from train import train_step
    torch.manual_seed(0)
    model = PSFVelocity().cuda()
    target = torch.randn(2, n_points, 3, device="cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss = train_step(model, optimizer, target, coupling=coupling, num_regions=8,
                      coupling_generator=torch.Generator(device="cuda").manual_seed(1))
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if (not torch.isfinite(loss) or not gradients
            or not all(torch.isfinite(g).all().item() for g in gradients)
            or not any(torch.count_nonzero(g).item() for g in gradients)):
        raise RuntimeError("Nonfinite loss/gradients or missing nonzero gradients")
    torch.cuda.synchronize()
    print(f"PSF_FORWARD_BACKWARD_OK coupling={coupling} loss={loss.item():.6f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run actual PVCNN forward/backward kernels; no dataset needed.")
    parser.add_argument("--coupling", choices=["independent", "target_guided_exact_optimized"], default="independent")
    parser.add_argument("--n-points", type=int, default=2048)
    args = parser.parse_args()
    smoke_test(args.coupling, args.n_points)
