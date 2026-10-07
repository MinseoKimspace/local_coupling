"""Shared, coupling-independent draws for read-only 2D diagnostics."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from data import sample_checkerboard
from experiment import load_model
from model import PointSetTransformer
from train_horse import HorsePointSetTransformer, load_horse_mask, sample_horse


def tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str((value.shape, value.dtype.str)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def load_verified_model(config_path, dataset):
    if dataset not in ("checkerboard", "horse"):
        raise ValueError("Diagnostics support checkerboard and horse only")
    model_class = HorsePointSetTransformer if dataset == "horse" else PointSetTransformer
    model, config, checkpoint, metadata = load_model(config_path, model_class, dataset)
    if not metadata["training_config_verified"]:
        raise ValueError("Diagnostics require a verified checkpoint and its matching runs/.../config.yaml")
    if config["model"]["point_dim"] != 2:
        raise ValueError("These diagnostics require 2D point clouds")
    return model, config, checkpoint, metadata


class DiagnosticData:
    """Actual project samplers, isolated by purpose and individual cloud index.

    Matching batch size, permutation draws, model initialization and evaluation
    microbatching cannot advance these streams. CPU/CUDA values can differ;
    report their hashes and compare methods on the same device/dtype.
    """

    def __init__(self, config, dataset, device, dtype, seed):
        if dataset not in ("checkerboard", "horse"):
            raise ValueError("Expected checkerboard or horse")
        self.config, self.dataset = config, dataset
        self.device, self.dtype, self.seed = torch.device(device), dtype, int(seed)
        self.n = int(config["data"]["n_points"])
        self.mask = load_horse_mask(self.device, dtype) if dataset == "horse" else None

    def draw_seed(self, purpose, index):
        value = f"diagnostic_draw_v1:{self.seed}:{purpose}:{int(index)}".encode()
        return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**63 - 1)

    def source(self, index, purpose="evaluation"):
        generator = torch.Generator(device=self.device).manual_seed(self.draw_seed(purpose + ":source", index))
        return torch.randn(self.n, 2, device=self.device, dtype=self.dtype, generator=generator)

    def target(self, index, purpose="evaluation"):
        cuda_devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] \
            if self.device.type == "cuda" else []
        # Use the original sampler without changing its API or global RNG state.
        with torch.random.fork_rng(devices=cuda_devices):
            seed = self.draw_seed(purpose + ":target", index)
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
            if cuda_devices:
                state = torch.Generator(device=self.device).manual_seed(seed).get_state()
                torch.cuda.set_rng_state(state, self.device)
            if self.dataset == "horse":
                return sample_horse(self.mask, 1, self.n)[0]
            return sample_checkerboard(1, self.n, self.device, self.dtype,
                                       self.config["data"]["grid_size"])[0]

    def bank(self, count, purpose="evaluation"):
        if count < 1:
            raise ValueError("Cloud count must be positive")
        return (torch.stack([self.source(i, purpose) for i in range(count)]),
                torch.stack([self.target(i, purpose) for i in range(count)]))


def diagnostic_metadata(config_path, config, checkpoint, metadata, dataset, device):
    from experiment import environment
    return {
        "dataset": dataset, "config": config,
        "config_path": str(Path(config_path).resolve()), "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_config_verified": metadata["training_config_verified"],
        "coupling_details": metadata["coupling_details"], "environment": environment(device),
        "training_source_sha256": metadata.get("source_sha256"),
        "diagnostic_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(Path(__file__).parent.glob("*.py"))},
    }


def finite_scores(scores):
    """Undefined metrics (e.g. conditioned cell TV) are null, never fake zero."""
    return {key: float(value) if value is not None and np.isfinite(value) else None
            for key, value in scores.items()}
