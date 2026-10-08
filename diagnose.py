import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from anchor_flow import sample_source
from experiment import evaluation_title
from summarize_results import output_directory, save_json, save_table, statistics, formatted, plt


TIMES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0)
BINS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 1.0))
NFES = (1, 2, 4, 8, 16, 32, 64, 128)


@torch.no_grad()
def fm_errors(model, source, target, time, *, config=None):
    if config is not None and (config.get("anchor_flow") or {}).get("mode") == "anchor_waypoint":
        raise ValueError("Linear-path FM residuals do not apply to anchor_waypoint; "
                         "use audit_generation.py --skip-fm for generation/integration diagnostics")
    velocity = target - source
    prediction = model((1 - time) * source + time * target, time)
    return ((prediction - velocity).square().mean((1, 2)), velocity.square().mean((1, 2)))


@torch.no_grad()
def rollout(model, noise, steps, keep_path=False):
    x = noise.clone()
    lengths = noise.new_zeros(noise.shape[:2])
    path = [x[0, :24].cpu().clone()] if keep_path else []
    for step in range(steps):
        t = noise.new_full((noise.shape[0], 1, 1), step / steps)
        dx = model(x, t) / steps
        x = x + dx
        lengths += dx.norm(dim=-1)
        if keep_path:
            path.append(x[0, :24].cpu().clone())
    distance = (x - noise).norm(dim=-1).sum(1)
    valid = distance > torch.finfo(noise.dtype).eps
    # Sum lengths / sum endpoint distances within each cloud, not mean of
    # per-point ratios (unstable for particles whose endpoints barely move).
    ratios = lengths.sum(1)[valid] / distance[valid]
    return x, ratios, torch.stack(path) if keep_path else None


def render(payload, path, directory):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    rows = payload["time_errors"]
    axes[0, 0].plot([r["t"] for r in rows], [r["mse"]["mean"] for r in rows], "o-")
    axes[0, 0].set(xlabel="Flow time t", ylabel="FM MSE per coordinate")
    endpoints = payload["endpoint_errors"]
    axes[0, 1].plot([r["nfe"] for r in endpoints], [r["mse"]["mean"] for r in endpoints], "o-")
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set(xlabel="NFE (Euler)", ylabel=f"Endpoint MSE vs Euler {payload['reference_nfe']}")
    axes[1, 0].bar([f"{r['interval'][0]}–{r['interval'][1]}" for r in payload["time_bins"]],
                   [r["mse"]["mean"] for r in payload["time_bins"]])
    axes[1, 0].set(xlabel="Uniformly sampled time bin", ylabel="FM MSE per coordinate")
    for particle in range(path.shape[1]):
        axes[1, 1].plot(path[:, particle, 0], path[:, particle, 1], lw=0.8, alpha=0.7)
    axes[1, 1].scatter(path[0, :, 0], path[0, :, 1], marker="x", s=15, c="gray", label="noise")
    axes[1, 1].scatter(path[-1, :, 0], path[-1, :, 1], s=12, c="black", label="generated")
    axes[1, 1].set_aspect("equal", adjustable="datalim")
    axes[1, 1].set_title("First cloud, first 24 particles (not target paths)")
    axes[1, 1].legend()
    config = payload["config"]
    title = f"{payload['dataset']} | {evaluation_title(config)}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(directory / "diagnostics.png", dpi=170)
    plt.close(fig)
    table = [[f"t={r['t']}", formatted(r["mse"])] for r in rows]
    table += [[f"NFE={r['nfe']} vs reference", formatted(r["mse"])] for r in endpoints]
    table += [["Path length / displacement", formatted(payload["path_ratio"])],
              ["Reference vs doubled NFE MSE", formatted(payload["reference_check_mse"])]]
    save_table(directory / "table.png", ["Diagnostic", "Mean ± cloud SD"], table, title)


def diagnose(config_path, dataset, *, batches=8, batch_size=None, seed=2026, reference_nfe=128,
             output="analysis_results"):
    from coupling import CLOUD_METHODS, TG_CACHED_METHODS, coupled_points
    from data import checkerboard_centers, sample_checkerboard
    from experiment import environment, load_model
    from model import PointSetTransformer
    from train_horse import HorsePointSetTransformer, load_horse_mask, sample_horse

    if batches < 1 or (batch_size is not None and batch_size < 1) or reference_nfe < max(NFES):
        raise ValueError("batches/batch_size must be positive; reference_nfe must be >=128")
    model_class = HorsePointSetTransformer if dataset == "horse" else PointSetTransformer
    model, config, checkpoint, metadata = load_model(config_path, model_class, dataset)
    if (config.get("anchor_flow") or {}).get("mode") == "anchor_waypoint":
        raise ValueError("diagnose.py includes linear-path FM residuals, which are invalid for anchor_waypoint; "
                         "use audit_generation.py --skip-fm instead")
    if not metadata["training_config_verified"]:
        raise ValueError("Diagnostics require a verified checkpoint and its matching run config.yaml")
    cloud_coupling = config["coupling"] in CLOUD_METHODS
    if batch_size is None:
        batch_size = config["data"]["batch_size"] if cloud_coupling else 16
    if cloud_coupling and batch_size != config["data"]["batch_size"]:
        raise ValueError("Cloud OT diagnostics require the training data.batch_size to preserve the coupling law")
    parameter = next(model.parameters())
    device, dtype = parameter.device, parameter.dtype
    torch.manual_seed(seed)  # reset AFTER model initialization; common draws across methods
    np.random.seed(seed + 1)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    time_generator = torch.Generator(device=device).manual_seed(seed + 2)
    n = config["data"]["n_points"]
    pair_sampler = None
    if config["coupling"] == "nsot":
        from nsot import NSOTPairSampler
        pair_sampler = NSOTPairSampler(config, dataset, device, dtype)
    elif config["coupling"] in TG_CACHED_METHODS:
        from tg_cache import TGCachedPairSampler
        pair_sampler = TGCachedPairSampler(config, dataset, device, dtype)
    centers = None
    if dataset == "horse":
        mask = load_horse_mask(device, dtype)
        sample_target = lambda: sample_horse(mask, batch_size, n)
    else:
        grid = config["data"]["grid_size"]
        centers = checkerboard_centers(grid, device, dtype)
        sample_target = lambda: sample_checkerboard(batch_size, n, device, dtype, grid)
    errors = [[] for _ in TIMES]
    bin_errors = [[] for _ in BINS]
    endpoint_errors = {nfe: [] for nfe in NFES}
    energy, ratios, reference_errors = [], [], []
    first_path = None
    for batch in range(batches):
        target = sample_target()
        noise = sample_source(config, batch_size, device=device, dtype=dtype)
        if pair_sampler is None:
            paired_noise, target = coupled_points(noise, target, coupling=config["coupling"],
                num_regions=config.get("num_regions"), target_centers=centers,
                sinkhorn_epsilon=config.get("sinkhorn_epsilon", 0.1),
                sinkhorn_iterations=config.get("sinkhorn_iterations", 100), generator=generator)
        else:
            paired_noise, target = pair_sampler.sample(batch_size, generator=generator)
        for index, t in enumerate(TIMES):
            error, baseline = fm_errors(model, paired_noise, target, noise.new_full((batch_size, 1, 1), t))
            errors[index].extend(error.tolist())
            if index == 0:
                energy.extend(baseline.tolist())
        for index, (lo, hi) in enumerate(BINS):
            t = torch.rand(batch_size, 1, 1, device=device, dtype=dtype, generator=time_generator) * (hi - lo) + lo
            error, _ = fm_errors(model, paired_noise, target, t)
            bin_errors[index].extend(error.tolist())
        reference, ratio, path = rollout(model, noise, reference_nfe, keep_path=batch == 0)
        ratios.extend(ratio.tolist())
        if batch == 0:
            first_path = path.numpy()
        doubled_reference, _, _ = rollout(model, noise, 2 * reference_nfe)
        reference_errors.extend((reference - doubled_reference).square().mean((1, 2)).tolist())
        for nfe in NFES:
            prediction = reference if nfe == reference_nfe else rollout(model, noise, nfe)[0]
            endpoint_errors[nfe].extend((prediction - reference).square().mean((1, 2)).tolist())
        print(f"diagnostic_batch={batch + 1}/{batches}", flush=True)
    all_values = energy + ratios + reference_errors + sum(errors, []) + sum(bin_errors, []) + sum(endpoint_errors.values(), [])
    if not all(torch.isfinite(torch.tensor(all_values))):
        raise FloatingPointError("Nonfinite diagnostic values; results were not saved")
    payload = {
        "dataset": dataset, "config": config, "checkpoint": str(checkpoint),
        "config_path": str(Path(config_path).resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "coupling_details": metadata["coupling_details"], "environment": environment(device),
        "training_source_sha256": metadata.get("source_sha256"),
        "diagnostic_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in Path(__file__).parent.glob("*.py")},
        "diagnostic_seed": seed, "batches": batches, "batch_size": batch_size,
        "reference_nfe": reference_nfe, "reference_check_nfe": 2 * reference_nfe,
        "definitions": {
            "fm_sampling_scope": ("fresh pairing draws from fixed training cache; NOT unseen source/target pool"
                                  if pair_sampler is not None else "fresh source and target clouds"),
            "mse": "mean squared error over points and coordinates per cloud; then mean across fresh clouds",
            "fm": "held-out interpolation velocity residual against this checkpoint's own coupling; NOT conditional variance or training-history loss",
            "std": "sample SD across diagnostic clouds, NOT across training seeds or confidence interval",
            "endpoint": "same indexed initial particles; endpoint MSE vs this model's finite-step Euler reference, NOT ground truth",
            "path_ratio": "per cloud: sum particle path lengths / sum particle endpoint displacements; measured on reference rollout; zero-displacement clouds excluded",
            "reference_check": "reference vs doubled-step endpoint MSE; nonzero indicates reference is not numerically converged",
        },
        "target_velocity_energy": statistics(energy),
        "time_errors": [{"t": t, "mse": statistics(v)} for t, v in zip(TIMES, errors)],
        "time_bins": [{"interval": bounds, "mse": statistics(v)} for bounds, v in zip(BINS, bin_errors)],
        "endpoint_errors": [{"nfe": nfe, "mse": statistics(v)} for nfe, v in endpoint_errors.items()],
        "path_ratio": statistics(ratios), "zero_displacement_clouds": batches * batch_size - len(ratios),
        "reference_check_mse": statistics(reference_errors),
        "example_trajectory": {"times": [i / reference_nfe for i in range(reference_nfe + 1)],
                               "points": first_path.tolist()},
    }
    directory = output_directory(Path(output) / dataset, checkpoint.stem + "_diagnostic")
    save_json(directory / "diagnostics.json", payload)
    render(payload, first_path, directory)
    print(f"saved={directory}")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Held-out FM error and learned-trajectory diagnostics; no training.")
    parser.add_argument("config", help="Saved runs/.../config.yaml")
    parser.add_argument("--dataset", required=True, choices=["checkerboard", "horse"])
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Default: training batch size for cloud OT, otherwise 16")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--reference-nfe", type=int, default=128)
    parser.add_argument("--output", default="analysis_results")
    parser.add_argument("--quality", action="store_true",
                        help="Joint common-bank quality, refined reference, and horse ROI diagnostics")
    from audit_generation import arguments as quality_arguments
    quality_arguments(parser)
    args = parser.parse_args()
    if args.quality:
        from audit_generation import audit
        audit(args.config, args.dataset, clouds=args.clouds, batch_size=args.eval_batch_size,
              nfes=args.nfes, reference_nfe=args.reference_nfe, max_reference_nfe=args.max_reference_nfe,
              endpoint_tolerance=args.endpoint_tolerance, quality_tolerance=args.quality_tolerance,
              fm_batches=0 if args.skip_fm else args.batches, matching_batch_size=args.batch_size,
              seed=args.seed, roi_file=args.roi_file, output=args.output)
    else:
        diagnose(args.config, args.dataset, batches=args.batches, batch_size=args.batch_size,
                 seed=args.seed, reference_nfe=args.reference_nfe, output=args.output)
