"""Training-free target partition audit; geometry proxies, not generation metrics."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import torch

from coupling import balanced_target_partition
from data import sample_checkerboard
from experiment import environment, read_config
from summarize_results import output_directory, save_json, statistics, plt
from train_horse import load_horse_mask, sample_horse


def cell_ids(points, grid):
    columns = np.floor((points[..., 0] + 1) * grid / 2).astype(int)
    rows = np.floor((1 - points[..., 1]) * grid / 2).astype(int)
    valid = (rows >= 0) & (rows < grid) & (columns >= 0) & (columns < grid)
    valid &= (rows + columns) % 2 == 0
    return np.where(valid, rows * grid + columns, -1)


def support_query(dataset, *, grid=4, mask=None):
    """Return foreground membership and chord sampling spacing in data coordinates."""
    if dataset == "checkerboard":
        return lambda q: cell_ids(q, grid) >= 0, 2 / grid / 64
    if dataset != "horse" or mask is None:
        raise ValueError("Horse support requires its foreground mask")
    height, width = mask.shape
    scale = max(height, width)

    def inside(q):
        columns = np.floor(q[..., 0] * scale / 2 + width / 2).astype(int)
        rows = np.floor(height / 2 - q[..., 1] * scale / 2).astype(int)
        valid = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
        return valid & mask[np.clip(rows, 0, height - 1), np.clip(columns, 0, width - 1)]

    return inside, 1 / scale  # Half a pixel; finite-resolution test, not exact visibility.


def chord_outside(points, pairs, inside, spacing):
    """Fraction of interior samples outside support for each straight chord."""
    values = np.empty(len(pairs))
    for start in range(0, len(pairs), 128):
        pair = pairs[start:start + 128]
        a, b = points[pair[:, 0]], points[pair[:, 1]]
        segments = max(2, int(np.ceil(np.linalg.norm(b - a, axis=1).max() / spacing)))
        t = np.arange(1, segments) / segments
        q = a[:, None] + t[None, :, None] * (b - a)[:, None]
        values[start:start + len(pair)] = (~inside(q)).mean(1)
    return values


def neighbor_edges(points, neighbors):
    """Undirected union-kNN candidates; O(N*k) storage, valid in 2D or 3D."""
    n = len(points)
    if n < 2:
        return np.empty((0, 2), dtype=int), np.empty(0), 0.0
    count = min(neighbors, n - 1)
    distances, indices = cKDTree(points).query(points, k=count + 1)
    # Remove by index, not distance: coincident but distinct points are neighbors.
    valid = indices != np.arange(n)[:, None]
    keep = valid & (np.cumsum(valid, axis=1) <= count)
    pairs = np.column_stack((np.repeat(np.arange(n), count + 1), indices.ravel()))
    pairs = np.unique(np.sort(pairs[keep.ravel()], axis=1), axis=0)
    lengths = np.linalg.norm(points[pairs[:, 0]] - points[pairs[:, 1]], axis=1)
    return pairs, lengths, float(np.median(distances[keep].reshape(n, count)[:, -1]))


def connectivity(n, pairs, labels, k):
    graph = csr_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    graph = graph.maximum(graph.T)
    total, global_labels = connected_components(graph, directed=False)
    patches = []
    for patch in range(k):
        ids = np.flatnonzero(labels == patch)
        if not len(ids):
            continue
        count, components = connected_components(graph[ids][:, ids], directed=False)
        patches.append({"patch": patch, "points": len(ids), "components": int(count),
                        "largest_component_fraction": float(np.bincount(components).max() / len(ids)),
                        "global_components_spanned": int(len(np.unique(global_labels[ids])))})
    return {"global_components": int(total), "isolated_point_fraction": float((graph.getnnz(1) == 0).mean()),
            "fragmented_patch_fraction": float(np.mean([p["components"] > 1 for p in patches])),
            "mean_largest_component_fraction": float(np.mean([p["largest_component_fraction"] for p in patches])),
            "patches": patches}


def sample_pairs(ids, limit, rng):
    n = len(ids)
    if n < 2:
        return np.empty((0, 2), dtype=int)
    if n * (n - 1) // 2 <= limit:
        a, b = np.triu_indices(n, 1)
    else:
        a = rng.integers(n, size=limit)
        b = rng.integers(n - 1, size=limit)
        b += b >= a
    return np.column_stack((ids[a], ids[b]))


def audit_partition(points, labels, k, edges, lengths, scale, edge_outside,
                    inside, spacing, factors, pair_limit, rng, cells=None):
    patches, bad_chords = [], []
    for patch in range(k):
        ids = np.flatnonzero(labels == patch)
        pairs = sample_pairs(ids, pair_limit, rng)
        outside = chord_outside(points, pairs, inside, spacing)
        row = {"patch": patch, "points": len(ids), "sampled_pairs": len(pairs),
               "chord_exit_fraction": float((outside > 0).mean()) if len(pairs) else None,
               "mean_chord_outside_fraction": float(outside.mean()) if len(pairs) else None}
        if cells is not None and len(ids):
            _, counts = np.unique(cells[ids], return_counts=True)
            row.update(cells_touched=len(counts), cell_mixing_fraction=float(1 - counts.max() / len(ids)))
        patches.append(row)
        bad_chords.extend(pairs[outside > 0][:3].tolist())
    graphs = []
    for factor in factors:
        eligible = lengths <= factor * scale
        kept = edges[eligible & (edge_outside == 0)]
        graphs.append({"radius_factor": factor, "radius": factor * scale,
                       "candidate_edges": int(eligible.sum()), "support_valid_edges": len(kept),
                       "support_rejected_edge_fraction": float((edge_outside[eligible] > 0).mean()) if eligible.any() else None,
                       **connectivity(len(points), kept, labels, k)})
    pair_count = sum(p["sampled_pairs"] for p in patches)
    result = {"capacities": np.bincount(labels, minlength=k).tolist(), "patches": patches, "graphs": graphs,
              "chord_exit_fraction": sum((p["chord_exit_fraction"] or 0) * p["sampled_pairs"] for p in patches) / pair_count if pair_count else None}
    if cells is not None:
        result["point_weighted_cell_mixing_fraction"] = sum(p.get("cell_mixing_fraction", 0) * p["points"] for p in patches) / len(points)
    return result, bad_chords[:24]


def render(payload, inside, directory):
    example = payload["example"]
    points, anchors = np.array(example["points"]), np.array(example["anchors"])
    axis = np.linspace(-1.05, 1.05, 600)
    xx, yy = np.meshgrid(axis, axis)
    background = inside(np.stack((xx, yy), axis=-1))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, name in zip(axes, ("balanced", "nearest_reference")):
        ax.imshow(background, origin="lower", extent=(-1.05, 1.05, -1.05, 1.05), cmap="Greys", alpha=0.14)
        for a, b in example[name]["example_exiting_chords"]:
            ax.plot(points[[a, b], 0], points[[a, b], 1], color="red", alpha=0.3, lw=0.7)
        ax.scatter(*points.T, c=example[name]["labels"], cmap="tab20", vmin=0, vmax=max(1, len(anchors) - 1), s=17)
        ax.scatter(*anchors.T, marker="*", s=90, c="black")
        for patch, anchor in enumerate(anchors):
            ax.annotate(str(patch), anchor, xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set(title=name.replace("_", " "), xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect="equal")
    fig.suptitle(f"{payload['dataset']} | first cloud | stars: shared FPS anchors; red: chords crossing background")
    fig.tight_layout()
    fig.savefig(directory / "patches.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ("balanced", "nearest_reference"):
        rows = payload["summary"][name]["graphs"]
        for ax, metric in zip(axes, ("fragmented_patch_fraction", "mean_largest_component_fraction")):
            ax.errorbar([r["radius_factor"] for r in rows], [r[metric]["mean"] for r in rows],
                        yerr=[r[metric]["std"] or 0 for r in rows], marker="o", capsize=3, label=name)
            ax.set(xlabel="Radius / median k-neighbor distance", ylabel=metric.replace("_", " "))
            ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Support-filtered target graph | mean +/- cloud SD (not training seeds)")
    fig.tight_layout()
    fig.savefig(directory / "connectivity.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def audit(config_path, dataset, *, clouds=16, seed=2026, neighbors=8, radius_factors=(0.75, 1.0, 1.5),
          pairs_per_patch=256, output="analysis_results"):
    if dataset not in ("checkerboard", "horse") or min(clouds, neighbors, pairs_per_patch) < 1:
        raise ValueError("Choose a supported dataset and positive clouds/neighbors/pairs_per_patch")
    factors = sorted(set(float(f) for f in radius_factors))
    if not factors or not all(np.isfinite(f) and f > 0 for f in factors):
        raise ValueError("radius_factors must be finite and positive")
    config = read_config(config_path)
    if config["model"]["point_dim"] != 2:
        raise ValueError("This support-mask audit is for 2D; the graph helpers also accept 3D")
    n, k = config["data"]["n_points"], config["num_regions"]
    device, dtype = torch.device(config["device"]), getattr(torch, config["dtype"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    mask = load_horse_mask(device, dtype) if dataset == "horse" else None
    grid = config["data"].get("grid_size", 4)
    inside, spacing = support_query(dataset, grid=grid, mask=mask.cpu().numpy().astype(bool) if mask is not None else None)
    records, example = [], None
    for index in range(clouds):
        target = sample_horse(mask, 1, n) if mask is not None else sample_checkerboard(1, n, device, dtype, grid)
        anchors, labels, _, _ = balanced_target_partition(target, k)
        nearest = torch.cdist(target, anchors).argmin(-1)[0].cpu().numpy()
        points, balanced = target[0].cpu().numpy(), labels[0].cpu().numpy()
        edges, lengths, scale = neighbor_edges(points, neighbors)
        outside = chord_outside(points, edges, inside, spacing)
        cells = cell_ids(points, grid) if dataset == "checkerboard" else None
        record = {"cloud": index, "neighbor_scale": scale}
        preview = {"points": points.tolist(), "anchors": anchors[0].cpu().tolist()}
        if cells is not None:
            record["true_cell_counts"] = np.bincount(cells, minlength=grid * grid).tolist()
        for name, assignment in (("balanced", balanced), ("nearest_reference", nearest)):
            result, bad = audit_partition(points, assignment, k, edges, lengths, scale, outside,
                inside, spacing, factors, pairs_per_patch, rng, cells)
            record[name] = result
            preview[name] = {"labels": assignment.tolist(), "example_exiting_chords": bad}
        records.append(record)
        if example is None:
            example = preview
        print(f"patch_audit_cloud={index + 1}/{clouds}", flush=True)
    summary = {}
    for name in ("balanced", "nearest_reference"):
        summary[name] = {"chord_exit_fraction": statistics([r[name]["chord_exit_fraction"] for r in records]),
            "graphs": [{"radius_factor": factor, **{metric: statistics([r[name]["graphs"][j][metric] for r in records])
                for metric in ("fragmented_patch_fraction", "mean_largest_component_fraction", "global_components", "isolated_point_fraction")}}
                for j, factor in enumerate(factors)]}
        if dataset == "checkerboard":
            summary[name]["point_weighted_cell_mixing_fraction"] = statistics([r[name]["point_weighted_cell_mixing_fraction"] for r in records])
    payload = {"dataset": dataset, "config": config, "config_path": str(Path(config_path).resolve()),
        "audit_seed": seed, "clouds": clouds, "neighbors": neighbors, "radius_factors": factors,
        "pairs_per_patch": pairs_per_patch, "chord_spacing": spacing, "environment": environment(device),
        "partition": "TG: target FPS then exact capacity-constrained squared distance to anchors; nearest uses the SAME anchors",
        "definitions": {
            "scope": "No checkpoint, source noise, model training, or learned trajectories. Always audits exact TG target partition; source options are unused.",
            "graph": "Union-kNN, radius-filtered, then remove chords crossing known background. Patch connectivity is on induced subgraphs. Full graph fragmentation is reported as a sampling confound.",
            "chords": "Distinct-endpoint within-patch target chords, not FM paths. All unordered pairs when feasible; otherwise uniform pairs with replacement. Interior support tests have finite spacing and can miss smaller gaps.",
            "averaging": "Chords pooled by sampled pair count within each cloud; graph summary is unweighted across nonempty patches. Then mean and sample SD across fresh clouds, not training seeds.",
            "interpretation": "Nonconvex but connected shapes can have exiting chords. Graph fragments can reflect sparse sampling. Neither proves a bad anatomical partition or generation failure. Unequal checkerboard cell counts can force mixing under equal capacities.",
            "3d": "Graph construction is dimension-independent; oracle support testing here is 2D-only and does not claim a 3D inside/outside test."},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        "summary": summary, "records": records, "example": example}
    directory = output_directory(Path(output) / dataset, "target_patch_audit")
    save_json(directory / "patch_audit.json", payload)
    render(payload, inside, directory)
    print(f"saved={directory}")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--dataset", required=True, choices=("checkerboard", "horse"))
    parser.add_argument("--clouds", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--radius-factors", nargs="+", type=float, default=[0.75, 1.0, 1.5])
    parser.add_argument("--pairs-per-patch", type=int, default=256)
    parser.add_argument("--output", default="analysis_results")
    args = parser.parse_args()
    audit(args.config, args.dataset, clouds=args.clouds, seed=args.seed, neighbors=args.neighbors,
          radius_factors=args.radius_factors, pairs_per_patch=args.pairs_per_patch, output=args.output)
