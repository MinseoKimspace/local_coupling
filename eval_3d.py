"""Small PSF generation check with CD-based set metrics, never importing EMD.

This is not the paper's full benchmark protocol. Metrics use normalized coordinates.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch

from prepare_psf import ROOT, provenance
from psf_adapter import PSFVelocity, shapenet_dataset
from sample import integrate_velocity


@torch.no_grad()
def cloud_cd(x, y, chunk=512):
    """Batched sum of directional mean squared distances; bounded point memory."""
    def directional(a, b):
        return torch.cat([torch.cdist(part, b).square().amin(-1)
                          for part in a.split(chunk, dim=1)], dim=1).mean(-1)
    return directional(x, y) + directional(y, x)


@torch.no_grad()
def cd_matrix(x, y, batch_size=4):
    result = torch.empty(len(x), len(y), device=x.device)
    for i in range(len(x)):
        for j in range(0, len(y), batch_size):
            targets = y[j:j + batch_size]
            result[i, j:j + len(targets)] = cloud_cd(x[i:i+1].expand(len(targets), -1, -1), targets)
    return result


@torch.no_grad()
def cd_metrics(generated, reference, batch_size=4):
    """MMD: reference->generated min; COV: generated->reference unique NN.

    1-NNA is leave-one-out balanced two-sample classification accuracy; ideal ~0.5.
    MMD here means Minimum Matching Distance, not kernel Maximum Mean Discrepancy.
    """
    cross = cd_matrix(generated, reference, batch_size)
    gg = cd_matrix(generated, generated, batch_size)
    rr = cd_matrix(reference, reference, batch_size)
    gg.fill_diagonal_(float("inf"))
    rr.fill_diagonal_(float("inf"))
    distances = torch.cat((torch.cat((gg, cross), 1), torch.cat((cross.T, rr), 1)), 0)
    labels = torch.arange(len(distances), device=distances.device) >= len(generated)
    correct = labels[distances.argmin(1)] == labels
    return {"mmd_cd": cross.amin(0).mean().item(),
            "cov_cd": cross.argmin(1).unique().numel() / len(reference),
            "one_nna_cd": correct.float().mean().item()}


def render_clouds(points, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 6))
    bound = max(float(np.abs(points[:8]).max()), 1e-6)
    for i, cloud in enumerate(points[:8]):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        ax.scatter(cloud[:, 0], cloud[:, 2], cloud[:, 1], s=1)
        ax.set(xlim=(-bound, bound), ylim=(-bound, bound), zlim=(-bound, bound))
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main(checkpoint, dataroot=None, nfe=(1, 2, 4, 8, 16, 32, 64, 128),
         samples=32, batch_size=4, seed=1, split="val"):
    if min(samples, batch_size, *nfe) < 1 or samples < 2:
        raise ValueError("Expected samples >= 2 and positive batch_size/NFE")
    from experiment import environment, synchronize
    checkpoint = Path(checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") not in ("psf_tg_v1", "psf_tg_v2"):
        raise ValueError("Expected this adapter's PSF checkpoint, not a 2D/upstream checkpoint.")
    psf = provenance()
    if payload["psf"] != psf:
        raise ValueError("PSF version/patch differs from the checkpoint's training version.")
    config = payload["config"]
    data = config["data"]
    dataset = shapenet_dataset(dataroot or data["root"], data["category"], data["n_points"],
                               split=split, normalization=payload["normalization"])
    if len(dataset) < samples:
        raise ValueError(f"Only {len(dataset)} reference shapes; reduce --samples (no duplicate shapes).")
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))[:samples]
    # Held-out SHAPES, sampled from their first 10k points, as in the upstream loader.
    reference = torch.stack([dataset[int(i)]["train_points"] for i in indices]).cuda()
    torch.manual_seed(seed)
    model = PSFVelocity(**config["model"]).cuda().eval()
    model.load_state_dict(payload["model_state_dict"])
    del payload["model_state_dict"]
    # Training resume states are not needed for sampling.
    for key in ("optimizer_state_dict", "sampler_state", "rng_state"):
        payload.pop(key, None)
    noise = torch.randn(samples, data["n_points"], 3, device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(seed))
    with torch.no_grad():
        model(noise[:batch_size], noise.new_zeros(min(batch_size, samples), 1, 1))
    identifier = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    directory = ROOT / "eval_results" / "psf3d" / f"{checkpoint.parent.name}_{identifier[:12]}"
    directory.mkdir(parents=True, exist_ok=True)
    for steps in sorted(set(nfe)):
        synchronize(noise.device)
        start = perf_counter()
        generated = torch.cat([integrate_velocity(model, batch, num_steps=steps)
                               for batch in noise.split(batch_size)])
        synchronize(noise.device)
        seconds = perf_counter() - start
        if not torch.isfinite(generated).all():
            raise FloatingPointError(f"Nonfinite generation at NFE={steps}")
        scores = cd_metrics(generated, reference, batch_size)
        name = f"nfe_{steps:03d}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
        array = generated.cpu().numpy()
        np.save(directory / f"{name}.npy", array)
        render_clouds(array, directory / f"{name}.png")
        result = {"dataset": "shapenet_psf3d", "checkpoint": str(checkpoint), "checkpoint_sha256": identifier,
                  "config": config, "psf": psf, "training_steps": payload["step"],
                  "training_seconds": payload["training_seconds"], "evaluation_environment": environment("cuda"),
                  "evaluation_weights": "raw_no_ema", "training_schedule": payload.get("schedule"),
                  "effective_batch_size": payload.get("effective_batch_size", data["batch_size"]),
                  "normalization": payload["normalization"], "evaluation_seed": seed, "split": split,
                  "reference_shape_ids": [dataset.all_cate_mids[int(i)] for i in indices],
                  "samples": samples, "batch_size": batch_size, "nfe": steps, "solver": "Euler",
                  "inference_seconds": seconds, "emd_evaluated": False,
                  "protocol": "Pilot only: same normalization as training; equal generated/reference counts; not full PSF benchmark.",
                  "metric_definitions": {"cd": "sum of directional mean squared Euclidean distances",
                    "mmd_cd": "mean over reference clouds of minimum CD to generated clouds; lower is better",
                    "cov_cd": "fraction of references selected by generated nearest neighbors; higher is better",
                    "one_nna_cd": "leave-one-out two-sample accuracy; near 0.5 desirable; NOT lower-is-always-better"},
                  "generated_points": f"{name}.npy", "image": f"{name}.png", **scores}
        with (directory / f"{name}.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(f"nfe={steps} {scores} saved={directory / (name + '.json')}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--dataroot")
    parser.add_argument("--nfe", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    main(**vars(parser.parse_args()))
