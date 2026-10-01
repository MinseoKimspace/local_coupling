"""Training-free CH/DB/WCSS comparison on the actual TG target partitions."""
import argparse
import hashlib
from numbers import Integral
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
import torch

from coupling import balanced_partition_solver, balanced_target_partition, coupling_info
from data import sample_checkerboard
from experiment import environment, read_config
from summarize_results import formatted, output_directory, plt, save_json, statistics
from train_horse import load_horse_mask, sample_horse


METRICS = ("ch", "db", "normalized_wcss")


def partition_scores(points, labels):
    """Float64 scores; undefined or infinite indices are None, never ranked."""
    points = np.asarray(points, dtype=np.float64)
    labels = np.asarray(labels)
    if (points.ndim != 2 or min(points.shape) < 1 or not np.isfinite(points).all()
            or labels.shape != (len(points),) or not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError("Expected finite [N, D] points and integer [N] labels")
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    n, d = points.shape
    k = len(counts)
    centers = np.zeros((k, d))
    np.add.at(centers, inverse, points)
    centers /= counts[:, None]
    residual = points - centers[inverse]
    wcss = float(np.square(residual).sum())
    mean = points.mean(0)
    total = float(np.square(points - mean).sum())
    between = float((counts[:, None] * np.square(centers - mean)).sum())
    if not np.isfinite([wcss, total, between]).all():
        raise FloatingPointError("Nonfinite partition dispersion")
    ch = db = None
    if 1 < k < n:
        if wcss > 0:
            ch = between / wcss * (n - k) / (k - 1)
        # DB uses MEAN Euclidean distance, not squared distance or RMS radius.
        radii = np.bincount(inverse, weights=np.linalg.norm(residual, axis=1)) / counts
        distances = cdist(centers, centers)
        np.fill_diagonal(distances, np.inf)
        if not (distances == 0).any():
            db = float(((radii[:, None] + radii[None, :]) / distances).max(1).mean())
    return {"k": k, "capacities": counts.tolist(), "wcss": wcss,
            "wcss_per_point": wcss / n, "total_scatter": total, "between_scatter": between,
            "normalized_wcss": wcss / total if total > 0 else None,
            "ch": float(ch) if ch is not None and np.isfinite(ch) else None,
            "db": db if db is not None and np.isfinite(db) else None}


def render(summary, dataset, directory):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title in zip(axes, METRICS,
            ("CH (higher is better)", "DB (lower is better)", "WCSS / total scatter (not generation quality)")):
        rows = [row for row in summary if row[metric]["mean"] is not None]
        if rows:
            ax.errorbar([row["k"] for row in rows], [row[metric]["mean"] for row in rows],
                        yerr=[row[metric]["std"] or 0 for row in rows], marker="o", capsize=3)
        ax.set(title=title, xlabel="K")
        ax.set_xticks([row["k"] for row in summary])
        ax.grid(alpha=0.25)
    fig.suptitle(f"{dataset} | same target clouds across K | mean +/- cloud SD, not training seeds")
    fig.tight_layout()
    fig.savefig(directory / "scores.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def audit(config_path, dataset, *, ks=(4, 8, 16, 32), clouds=32, seed=0,
          device=None, output="analysis_results"):
    if dataset not in ("checkerboard", "horse"):
        raise ValueError("Choose checkerboard or horse")
    if isinstance(clouds, bool) or not isinstance(clouds, Integral) or clouds < 1:
        raise ValueError("clouds must be a positive integer")
    config = read_config(config_path)
    if config["coupling"] not in ("target_guided", "target_guided_exact_optimized"):
        raise ValueError("This audit requires an exact TG training config")
    if config["model"]["point_dim"] != 2:
        raise ValueError("The dataset samplers in this audit are 2D-only")
    n = config["data"]["n_points"]
    ks = tuple(ks)
    if not ks or any(isinstance(k, bool) or not isinstance(k, Integral) or not 2 <= k < n for k in ks):
        raise ValueError("Every candidate must be an integer with 2 <= K < N")
    ks = sorted(set(int(k) for k in ks))
    device = torch.device(config["device"] if device is None else device)
    dtype = getattr(torch, config["dtype"])
    torch.manual_seed(seed)
    mask = load_horse_mask(device, dtype) if dataset == "horse" else None
    grid = config["data"].get("grid_size", 4)
    solver = balanced_partition_solver(config["coupling"])
    records = []
    for index in range(clouds):
        # Sampling is outside the K loop. Every candidate sees identical points.
        target = (sample_horse(mask, 1, n) if mask is not None
                  else sample_checkerboard(1, n, device, dtype, grid))
        points = target[0].cpu().numpy()
        scores = []
        for k in ks:
            _, labels, _, _ = balanced_target_partition(target, k, solver=solver)
            assignment = labels[0].cpu().numpy()
            scores.append({**partition_scores(points, assignment),
                           "labels_sha256": hashlib.sha256(assignment.tobytes()).hexdigest()})
        records.append({"cloud": index, "target_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
                        "scores": scores})
        print(f"k_audit_cloud={index + 1}/{clouds}", flush=True)
    summary = [{"k": k, **{metric: statistics([r["scores"][j][metric] for r in records])
                             for metric in (*METRICS, "wcss", "wcss_per_point")}}
               for j, k in enumerate(ks)]
    selections = {}
    for metric, select in (("ch", max), ("db", min)):
        valid = [row for row in summary if row[metric]["n"] == clouds]
        selections[metric] = select(valid, key=lambda row: row[metric]["mean"])["k"] if valid else None
    payload = {
        "dataset": dataset, "config": config, "config_path": str(Path(config_path).resolve()),
        "candidate_ks": ks, "clouds": clouds, "audit_seed": seed,
        "environment": environment(device), "coupling_details": coupling_info(config["coupling"]),
        "definitions": {
            "scope": "Only TG target FPS + exact balanced assignment; no source, checkpoint, training or inference. Candidate K overrides config.num_regions only for this audit.",
            "wcss": "Sum ||y_j - c_label(j)||^2; centroids are recomputed in float64 from TG membership, NOT the FPS anchors.",
            "normalized_wcss": "WCSS / sum ||y_j - mean(Y)||^2; dimensionless. WCSS/N is saved separately and is NOT this normalization.",
            "ch": "[sum n_k ||c_k-mean(Y)||^2 / WCSS] * (N-K)/(K-1); maximize.",
            "db": "Mean_k max_l!=k (s_k+s_l)/||c_k-c_l||; s_k = mean Euclidean distance to centroid; minimize.",
            "undefined": "CH is null for WCSS=0 or K outside 2..N-1; DB is null for coincident centroids or invalid K; normalized WCSS is null for total scatter=0. Nonfinite indices are null, not sklearn's degeneracy sentinels.",
            "averaging": "Scores computed per cloud, then unweighted mean and sample SD across fresh target clouds, NOT training seeds. Same points reused for every K.",
            "selection": "CH argmax / DB argmin of mean scores among candidates defined on ALL clouds; exact ties choose smaller K. No automatic training config change and no generation-optimality claim."},
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("audit_k.py", "coupling.py", "data.py", "train_horse.py")},
        "summary": summary, "selected_k": selections, "records": records}
    directory = output_directory(Path(output) / dataset, "k_audit")
    save_json(directory / "k_audit.json", payload)
    render(summary, dataset, directory)
    for row in summary:
        print(f"K={row['k']} " + " | ".join(f"{m}={formatted(row[m])}" for m in METRICS))
    print(f"selected_k (partition scores only)={selections}")
    print(f"saved={directory}")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--dataset", required=True, choices=("checkerboard", "horse"))
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--clouds", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0, help="Fresh-target audit seed, not training seed")
    parser.add_argument("--device", default=None, help="Optional sampling/partition device override, e.g. cpu")
    parser.add_argument("--output", default="analysis_results")
    args = parser.parse_args()
    audit(args.config, args.dataset, ks=args.ks, clouds=args.clouds, seed=args.seed,
          device=args.device, output=args.output)
