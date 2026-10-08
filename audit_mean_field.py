"""Read-only, Rao-Blackwellized t=0 mean-field audit on fresh 2D clouds.

Use a VERIFIED saved runs/.../config.yaml, not a training-template YAML.
The estimator is exact over internal random pairing and Monte Carlo over
target point sets. It does NOT estimate the mean field at t>0.
"""

import argparse
import hashlib
import math
from pathlib import Path

import torch

from coupling import (assign_regions, balanced_partition_solver, balanced_target_partition,
                      canonical_method, coupling_permutation, farthest_point_sample,
                      region_centroids)
from diagnostic_data import DiagnosticData, diagnostic_metadata, load_verified_model, tensor_sha256
from experiment import evaluation_title, read_config
from summarize_results import formatted, output_directory, plt, save_json, save_table, statistics


SUPPORTED = {"independent", "target_guided", "target_guided_exact_optimized",
             "geometry_aware_ot", "global_ot"}


@torch.no_grad()
def conditional_moments(source, target, config):
    """Return pairing-marginalized velocity, fine variance, and source labels.

    TG conditions on the realized target tensor and its actual partition,
    including any order-dependent FPS anchor choices. Only the internal
    random bijection is analytically averaged here.

    Independent is symmetrized over target labels. This is law-equivalent for
    the project's exchangeable iid samplers; its fine/between split conditions
    on the UNORDERED target set, not the original ordered target tensor.
    """
    method = canonical_method(config["coupling"])
    if method not in SUPPORTED:
        raise ValueError("t=0 analytic audit supports " + ", ".join(sorted(SUPPORTED))
                         + "; cloud minibatch OT/EFM require a different conditional estimator")
    if source.shape != target.shape or source.ndim != 3 or any(size < 1 for size in source.shape) \
            or not torch.isfinite(target).all() \
            or not torch.isfinite(source).all():
        raise ValueError("Expected matching finite [B,N,D] source/target tensors")
    if method == "global_ot":
        permutation = coupling_permutation(source, target, coupling=method)
        ordered = target.gather(1, permutation[..., None].expand_as(target))
        return ordered.double() - source.double(), source.new_zeros(source.shape[0], dtype=torch.float64), None
    if method == "independent":
        center = target.double().mean(1, keepdim=True)
        fine = (target.double() - center).square().mean((1, 2))
        return center.expand_as(target) - source.double(), fine, None

    k = int(config["num_regions"])
    if method == "geometry_aware_ot":
        rows = torch.arange(target.shape[0], device=target.device)[:, None]
        anchors = target[rows, farthest_point_sample(target, k)]
        labels = torch.cdist(target, anchors).argmin(-1)
        centers, capacities = region_centroids(target, labels, k)
    else:
        _, labels, centers, capacities = balanced_target_partition(
            target, k, solver=balanced_partition_solver(method))
    source_labels = assign_regions(source, centers, capacities,
                                   solver="exact_batched" if method == "target_guided_exact_optimized" else "exact")
    # Keep training's assignment arithmetic, then calculate statistical moments
    # in float64 using the ACTUAL target values and target membership.
    precise_centers, _ = region_centroids(target.double(), labels, k)
    mapped = precise_centers.gather(1, source_labels[..., None].expand_as(source))
    target_centers = precise_centers.gather(1, labels[..., None].expand_as(target))
    fine = (target.double() - target_centers).square().mean((1, 2))
    return mapped - source.double(), fine, source_labels


class MomentAccumulator:
    """Float64 Welford moments with an unbiased MC correction for fixed f(X)."""

    def __init__(self, prediction):
        self.prediction = prediction.detach().cpu().double()
        if not self.prediction.numel() or not torch.isfinite(self.prediction).all():
            raise ValueError("Prediction must be nonempty and finite")
        self.mean = torch.zeros_like(self.prediction)
        self.m2 = torch.zeros_like(self.prediction)
        self.count, self.fine_sum = 0, 0.0

    def update(self, mean_velocity, fine_variance):
        value = mean_velocity.detach().cpu().double()
        fine = float(fine_variance)
        if value.shape != self.mean.shape or not torch.isfinite(value).all() \
                or not math.isfinite(fine) or fine < 0:
            raise ValueError("Nonfinite or invalid conditional moments")
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.fine_sum += fine

    def scores(self):
        if self.count < 2:
            raise ValueError("At least two independent target realizations are needed")
        between = self.m2.mean().item() / (self.count - 1)
        fine = self.fine_sum / self.count
        uncorrected = (self.prediction - self.mean).square().mean().item()
        correction = between / self.count
        corrected = uncorrected - correction
        return {
            "targets": self.count, "fine_variance": fine, "between_variance": between,
            "total_conditional_variance": fine + between,
            "mean_field_mse_uncorrected": uncorrected,
            "mc_mean_variance": correction, "mc_mean_rms_uncertainty": correction ** .5,
            "mean_field_mse_corrected": corrected,
            # E_R residual averaged over sampled Y. The decomposition below
            # holds algebraically for this sample, not only asymptotically.
            "expected_fm_mse": uncorrected + (1 - 1 / self.count) * between + fine,
            "decomposition_residual": (uncorrected + (1 - 1 / self.count) * between + fine)
                                      - (corrected + fine + between),
        }


def render(payload, directory):
    keys = ("fine_variance", "between_variance", "mean_field_mse_corrected", "mc_mean_variance")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, key in zip(axes.flat, keys):
        rows = payload["summary_by_targets"]
        ax.errorbar([r["targets"] for r in rows], [r[key]["mean"] for r in rows],
                    yerr=[r[key]["std"] or 0 for r in rows], marker="o", capsize=3)
        ax.axhline(0, color="gray", lw=.6)
        ax.set(xlabel="Target MC realizations per fixed source", ylabel=key)
        ax.grid(alpha=.2)
    title = f"t=0 only | {payload['dataset']} | {evaluation_title(payload['config'])}"
    fig.suptitle(title + "\nMean +/- source-cloud SD, NOT training-seed uncertainty")
    fig.tight_layout()
    fig.savefig(directory / "mean_field.png", dpi=160)
    plt.close(fig)
    rows = [[r["targets"], *[formatted(r[key]) for key in keys]] for r in payload["summary_by_targets"]]
    save_table(directory / "table.png", ["Target draws", *keys], rows, title)


@torch.no_grad()
def audit(config_path, dataset, *, clouds=16, targets=128, batch_size=8, seed=2026,
          prefixes=None, output="analysis_results"):
    if clouds < 1 or targets < 2 or batch_size < 1:
        raise ValueError("clouds/batch_size must be positive; targets must be >=2")
    template = read_config(config_path)
    if canonical_method(template["coupling"]) not in SUPPORTED:
        raise ValueError("Analytic t=0 audit supports " + ", ".join(sorted(SUPPORTED)))
    if prefixes is None:
        prefixes = [p for p in (16, 32, 64, 128) if p <= targets]
    if any(isinstance(p, bool) or int(p) != p or not 2 <= p <= targets for p in prefixes):
        raise ValueError("Prefix counts must be integers in [2,targets]")
    prefixes = sorted(set([int(p) for p in prefixes] + [targets]))
    model, config, checkpoint, metadata = load_verified_model(config_path, dataset)
    parameter = next(model.parameters())
    data = DiagnosticData(config, dataset, parameter.device, parameter.dtype, seed)
    sources = torch.stack([data.source(i, "mean_field") for i in range(clouds)])
    records, target_hash = [], hashlib.sha256()
    for cloud in range(clouds):
        source = sources[cloud:cloud + 1]
        prediction = model(source, source.new_zeros(1, 1, 1))[0]
        accumulator = MomentAccumulator(prediction)
        snapshots = []
        for start in range(0, targets, batch_size):
            draws = [data.target(r, f"mean_field:source{cloud}")
                     for r in range(start, min(start + batch_size, targets))]
            for draw in draws:
                target_hash.update(tensor_sha256(draw).encode())
            target = torch.stack(draws)
            means, fines, _ = conditional_moments(source.expand_as(target), target, config)
            for value, fine in zip(means, fines):
                accumulator.update(value, fine)
                if accumulator.count in prefixes:
                    snapshots.append(accumulator.scores())
        records.append({"source_cloud": cloud, "by_targets": snapshots})
        print(f"mean_field_source={cloud + 1}/{clouds} target_draws={targets}", flush=True)
        if cloud == 0:
            example = {"source": source[0].cpu().tolist(), "prediction": prediction.cpu().tolist(),
                       "estimated_mean_velocity": accumulator.mean.tolist()}
    keys = [key for key in records[0]["by_targets"][-1] if key != "targets"]
    summary = [{"targets": p, **{key: statistics([r["by_targets"][i][key] for r in records])
                                 for key in keys}} for i, p in enumerate(prefixes)]
    payload = {
        **diagnostic_metadata(config_path, config, checkpoint, metadata, dataset, parameter.device),
        "format_version": 1, "diagnostic_seed": seed, "source_clouds": clouds,
        "target_draws_per_source": targets, "target_batch_size": batch_size,
        "source_sha256": tensor_sha256(sources), "target_draws_sha256": target_hash.hexdigest(),
        "definitions": {
            "scope": "t=0 ONLY; model observes the entire source cloud, not Y or patch labels",
            "normalization": "mean squared error over N points and d coordinates; fine=WCSS/(N*d)",
            "conditioning": "fixed whole X0 and realized target tensor/partition, including order-dependent FPS anchors; internal uniform random bijection analytically marginalized",
            "independent": "target-order symmetrization is law-equivalent for the project's iid exchangeable target samplers",
            "between": "variance over fresh target sets, including target sampling, partition and source assignment; NOT isolated assignment variance or 3D shape-mode variance",
            "correction": "||prediction-mean(m)||^2 - unbiased sample Var(m)/M; independent fresh target MC draws",
            "negative_corrected_mse": "possible for finite MC; retained WITHOUT clipping, not a negative population risk",
            "std": "sample SD across fresh source clouds, NOT training seeds or a confidence interval",
            "prefixes": "nested prefixes of the SAME target draws; inspect stability, not independent repetitions",
            "rng": "diagnostic_draw_v1; source and target indexed separately; invariant to target microbatch size",
        },
        "summary_by_targets": summary, "records": records, "first_source_example": example,
    }
    directory = output_directory(Path(output) / dataset, checkpoint.stem + "_mean_field")
    save_json(directory / "mean_field.json", payload)
    render(payload, directory)
    print(f"saved={directory}")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Verified saved runs/.../config.yaml")
    parser.add_argument("--dataset", required=True, choices=["checkerboard", "horse"])
    parser.add_argument("--clouds", type=int, default=16, help="Fresh fixed source clouds")
    parser.add_argument("--targets", type=int, default=128, help="Fresh target cloud draws per source; NOT N points")
    parser.add_argument("--batch-size", type=int, default=8, help="Target-draw compute microbatch; does not change draws")
    parser.add_argument("--prefixes", nargs="+", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="analysis_results")
    args = parser.parse_args()
    audit(args.config, args.dataset, clouds=args.clouds, targets=args.targets, batch_size=args.batch_size,
          prefixes=args.prefixes, seed=args.seed, output=args.output)
