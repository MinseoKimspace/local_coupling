"""Experimental overlapping boundary-window reassignment of a fixed MS-W2 bank.

This is NOT a smoothing penalty or globally exact OT. Each bounded window is
solved exactly against its CURRENT partners. Accepted local changes preserve
the original empirical marginals and decrease a specified multiscale point
cost. Lower coupling cost does not guarantee improved generated samples.
"""

import copy
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
import torch

import multiscale_wasserstein as msw
from nsot import file_sha256, source_moments

METHOD = "multiscale_overlap"
FORMAT_VERSION = 1
IMPLEMENTATION = "fixed_multiscale_boundary_windows_current_pool_exact_v1"
ACCEPTANCE_ATOL = 1e-12
ACCEPTANCE_RTOL = 1e-12


def base_config(config):
    """The unchanged MS-W2 endpoint bank and initial permutation configuration."""
    result = copy.deepcopy(config)
    result["coupling"] = msw.METHOD
    result.pop(METHOD, None)
    return result


def _options(raw, size, levels, max_cost_mib):
    if not isinstance(raw, dict):
        raise ValueError("multiscale_overlap must be a mapping")
    unknown = raw.keys() - {"cache", "cache_sha256", "window_sizes", "max_windows_per_scale",
                            "sweeps", "neighbors", "max_local_solves"}
    if unknown:
        raise ValueError("Unsupported overlap settings: " + ", ".join(sorted(unknown)))
    sizes = raw.get("window_sizes", [64, 128])
    if not isinstance(sizes, (list, tuple)) or not sizes:
        raise ValueError("window_sizes must be a nonempty increasing list")
    sizes = [msw._integer(value, "window size", minimum=2, maximum=size - 1) for value in sizes]
    if sizes != sorted(set(sizes)):
        raise ValueError("window_sizes must be unique and increasing")
    maximum = msw._integer(raw.get("max_windows_per_scale", 32), "max_windows_per_scale", maximum=10000)
    sweeps = msw._integer(raw.get("sweeps", 2), "sweeps", minimum=0, maximum=100)
    neighbors = msw._integer(raw.get("neighbors", 16), "neighbors", maximum=size - 1)
    budget = msw._integer(raw.get("max_local_solves", 1000), "max_local_solves", maximum=1000000)
    # Both source-anchored AND target-anchored windows, all widths/scales/sweeps.
    upper = 2 * levels * len(sizes) * maximum * sweeps
    if upper > budget:
        raise ValueError(f"Overlap worst-case work {upper} exceeds max_local_solves={budget}")
    memory = max(sizes) ** 2 * 8 / 2**20
    if memory > max_cost_mib:
        raise ValueError("Overlap local cost matrix exceeds max_cost_mib")
    return {"window_sizes": sizes, "max_windows_per_scale": maximum, "sweeps": sweeps,
            "neighbors": neighbors, "max_local_solves": budget,
            "local_solve_upper_bound": upper, "largest_local_cost_mib": memory}


def settings(config):
    if config.get("coupling") != METHOD:
        raise ValueError(f"Expected coupling: {METHOD}")
    baseline = msw.settings(base_config(config))
    raw = config.get(METHOD, {})
    opts = _options(raw, baseline["superset_size"], len(baseline["partition_counts"]),
                    baseline["max_cost_mib"])
    path = raw.get("cache")
    if not isinstance(path, str) or not path.strip() or path == "auto":
        raise ValueError("A concrete overlap cache path is required")
    if Path(path).resolve() == Path(baseline["cache"]).resolve():
        raise ValueError("Overlap cache must differ from the unchanged baseline cache")
    return {"cache": path, **opts}


def _permutation(value, size, name="permutation"):
    value = np.asarray(value)
    if value.shape != (size,) or not np.issubdtype(value.dtype, np.integer) \
            or not np.array_equal(np.sort(value), np.arange(size)):
        raise ValueError(f"{name} must be a bijective integer permutation")
    return value.astype(np.int64, copy=True)


def boundary_windows(points, levels, window_sizes, max_windows_per_scale, neighbors):
    """Deterministic fixed kNN boundary windows; not exhaustive boundary coverage.

    Boundary anchors are nearest cross-label kNN edges. Farthest-point anchor
    selection spreads a bounded number of windows spatially. Each window
    includes both edge endpoints, so it actually crosses this scale's labels.
    Multiple widths at a shared anchor deliberately overlap.
    """
    points = msw._points(points).astype(np.float64)
    size = len(points)
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=min(size, neighbors + 1), workers=1)
    windows, reports = [], []
    for level, labels in enumerate(levels):
        pairs = {}
        for i in range(size):
            candidates = [(float(d), int(j)) for d, j in zip(distances[i], indices[i])
                          if j != i and labels[j] != labels[i]]
            if candidates:
                distance, j = min(candidates)
                edge = tuple(sorted((i, j)))
                pairs[edge] = distance
        edges = sorted(pairs, key=lambda pair: (pairs[pair], pair))
        centers = np.array([(points[a] + points[b]) * .5 for a, b in edges])
        # Resolve all equal-distance choices by the stable edge order.
        selected, minimum_distance = [], np.full(len(edges), np.inf)
        for _ in range(min(max_windows_per_scale, len(edges))):
            chosen = 0 if not selected else int(np.argmax(minimum_distance))
            selected.append(chosen)
            current = np.square(centers - centers[chosen]).sum(axis=1)
            minimum_distance = np.minimum(minimum_distance, current)
            minimum_distance[selected] = -1.
        for width in window_sizes:
            seen, covered, count = set(), np.zeros(size, dtype=bool), 0
            for anchor in selected:
                edge = edges[anchor]
                _, near = tree.query(centers[anchor], k=min(size, width + 2), workers=1)
                near = sorted((int(j) for j in np.atleast_1d(near)),
                              key=lambda j: (float(np.square(points[j] - centers[anchor]).sum()), j))
                chosen = list(edge)
                chosen.extend(j for j in near if j not in edge)
                members = np.array(sorted(chosen[:width]), dtype=np.int64)
                key = tuple(members)
                if key in seen:
                    continue
                seen.add(key)
                if len(np.unique(labels[members])) < 2:
                    raise RuntimeError("Internal error: boundary window contains only one patch")
                covered[members] = True
                count += 1
                windows.append({"level": level, "width": width, "indices": members,
                                "anchor": list(edge)})
            reports.append({"level": level, "partition_count": int(np.max(labels)) + 1,
                            "width": width, "boundary_candidate_edges": len(edges),
                            "windows": count, "covered_points": int(covered.sum()),
                            "coverage_fraction": float(covered.mean())})
    return windows, reports


class _PointCost:
    """Reuse exact nonleaf patch tables; never allocate a global M-by-M matrix."""
    def __init__(self, source, target, counts, weights, progress=None):
        self.source = source.astype(np.float64)
        self.target = target.astype(np.float64)
        self.source_levels = msw.balanced_partitions(source, counts)
        self.target_levels = msw.balanced_partitions(target, counts)
        self.weights = weights
        self.tables, self.reports = [], []
        for level, (count, weight) in enumerate(zip(counts[:-1], weights[:-1])):
            started = perf_counter()
            table = np.zeros((count, count), dtype=np.float64)
            if weight:
                if progress:
                    progress(f"overlap nonleaf scale={level + 1} exact_patch_solves={count ** 2}")
                source_groups = [self.source[self.source_levels[level] == i] for i in range(count)]
                target_groups = [self.target[self.target_levels[level] == i] for i in range(count)]
                for i, a in enumerate(source_groups):
                    for j, b in enumerate(target_groups):
                        table[i, j] = msw.patch_wasserstein_squared(a, b)
            self.tables.append(table)
            self.reports.append({"level": level, "partition_count": count,
                                 "exact_patch_solves": count ** 2 if weight else 0,
                                 "cost_seconds": perf_counter() - started})

    def matrix(self, source_ids, target_ids):
        result = self.weights[-1] * cdist(self.source[source_ids], self.target[target_ids], "sqeuclidean")
        for level, table in enumerate(self.tables):
            result += self.weights[level] * table[
                self.source_levels[level][source_ids, None], self.target_levels[level][target_ids][None, :]]
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite overlap costs")
        return result

    def objective(self, permutation):
        result = self.weights[-1] * np.square(self.source - self.target[permutation]).sum(axis=1)
        for level, table in enumerate(self.tables):
            result += self.weights[level] * table[self.source_levels[level], self.target_levels[level][permutation]]
        return float(result.mean())


def _apply_windows(permutation, windows, cost, sweeps, progress=None):
    """Sequential current pools are essential when fixed windows overlap."""
    permutation = permutation.copy()
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))
    reports, attempted, accepted = [], 0, 0
    for sweep in range(sweeps):
        sweep_before = cost.objective(permutation)
        sweep_accepts = 0
        for side, window in windows:
            if side == "source":
                source_ids = window["indices"]
                target_ids = permutation[source_ids].copy()
            else:
                target_ids = window["indices"]
                source_ids = inverse[target_ids].copy()
            matrix = cost.matrix(source_ids, target_ids)
            # Current source_ids[k] -> target_ids[k] on both sides by construction.
            old = float(np.trace(matrix))
            rows, cols = linear_sum_assignment(matrix)
            new = float(matrix[rows, cols].sum())
            attempted += 1
            threshold = ACCEPTANCE_ATOL + ACCEPTANCE_RTOL * max(abs(old), abs(new))
            if old - new > threshold:
                source_order = source_ids[rows]
                target_order = target_ids[cols]
                permutation[source_order] = target_order
                inverse[target_order] = source_order
                accepted += 1
                sweep_accepts += 1
        after = cost.objective(permutation)
        if after > sweep_before + 1e-10 * max(1., abs(sweep_before)):
            raise RuntimeError("Overlap objective increased")
        reports.append({"sweep": sweep + 1, "accepted": sweep_accepts,
                        "objective_before": sweep_before, "objective_after": after})
        if progress:
            progress(f"overlap sweep={sweep + 1}/{sweeps} accepted={sweep_accepts} objective={after:.9g}")
        if not sweep_accepts:
            break
    return permutation, {"local_solves": attempted, "accepted_local_solves": accepted,
                         "sweeps_completed": len(reports), "sweep_reports": reports}


def overlap_permutation(source, target, base_permutation, partition_counts, scale_weights,
                        window_sizes=(64, 128), max_windows_per_scale=32, sweeps=2,
                        neighbors=16, max_local_solves=1000, max_cost_mib=512, progress=None):
    source, target = msw._points(source), msw._points(target)
    if source.shape != target.shape:
        raise ValueError("Source and target must have matching shapes")
    size = len(source)
    counts, weights = msw._scales(size, partition_counts, scale_weights)
    msw._memory_guard(size, counts, max_cost_mib)
    opts = _options({"window_sizes": window_sizes, "max_windows_per_scale": max_windows_per_scale,
                     "sweeps": sweeps, "neighbors": neighbors, "max_local_solves": max_local_solves},
                    size, len(counts), max_cost_mib)
    base_permutation = _permutation(base_permutation, size, "base_permutation")
    started = perf_counter()
    cost = _PointCost(source, target, counts, weights, progress)
    table_seconds = perf_counter() - started
    windows, coverage = [], []
    window_start = perf_counter()
    for side, points, levels in (("source", source, cost.source_levels), ("target", target, cost.target_levels)):
        selected, reports = boundary_windows(points, levels, opts["window_sizes"],
                                            opts["max_windows_per_scale"], opts["neighbors"])
        windows.extend((side, window) for window in selected)
        coverage.extend({"side": side, **report} for report in reports)
    window_seconds = perf_counter() - window_start
    before = cost.objective(base_permutation)
    solve_start = perf_counter()
    permutation, reports = _apply_windows(base_permutation, windows, cost, opts["sweeps"], progress)
    _permutation(permutation, size)
    if reports["local_solves"] > opts["max_local_solves"]:
        raise RuntimeError("Internal error: local solve budget exceeded")
    changed_leaf = cost.target_levels[-1][base_permutation] != cost.target_levels[-1][permutation]
    diagnostics = {**reports, **opts, "nonleaf_cost_tables": cost.reports,
                   "nonleaf_cost_seconds": table_seconds, "window_seconds": window_seconds,
                   "local_assignment_seconds": perf_counter() - solve_start,
                   "matching_seconds": perf_counter() - started,
                   "objective_before": before, "objective_after": cost.objective(permutation),
                   "objective_definition": "normalized nonleaf exact patch W2 costs plus finest weight times point squared distance",
                   "multiscale_objective": cost.objective(permutation),
                   "point_transport_cost": float(np.square(source.astype(np.float64) - target[permutation]).sum(1).mean()),
                   "changed_fraction": float(np.mean(base_permutation != permutation)),
                   "cross_leaf_fraction": float(changed_leaf.mean()), "window_count": len(windows),
                   "window_coverage": coverage,
                   "window_signature_sha256": msw._array_hash(np.array(
                       [json.dumps({"side": side, "level": w["level"], "width": w["width"],
                                    "indices": w["indices"].tolist(), "anchor": w["anchor"]}, sort_keys=True)
                        for side, w in windows], dtype="U")),
                   "boundary_detection": "nearest cross-label edge among bounded kNN; spatially spread anchors, not exhaustive coverage",
                   "acceptance_atol": ACCEPTANCE_ATOL, "acceptance_rtol": ACCEPTANCE_RTOL,
                   "matching_scope": "exact assignment within sequential current-partner windows; bounded local optimization, NOT global exact OT or guaranteed local optimum",
                   "quality_guarantee": "none; cost decrease does not guarantee improved fitting or generated density"}
    return permutation, diagnostics


def cache_spec(config, dataset):
    opts = settings(config)
    base_spec = msw.cache_spec(base_config(config), dataset)
    return {"format_version": FORMAT_VERSION, "implementation": IMPLEMENTATION,
            "base_spec": base_spec,
            **{key: base_spec[key] for key in ("superset_size", "cache_seed", "partition_counts",
                                               "scale_weights", "dataset_spec")},
            "overlap_settings": {key: value for key, value in opts.items() if key != "cache"}}


def load_cache(config, dataset):
    opts = settings(config)
    path = Path(opts["cache"])
    if not path.is_file():
        raise FileNotFoundError(f"Overlap cache missing: {path}; prepare the boundary study first")
    digest = file_sha256(path)
    expected_hash = config[METHOD].get("cache_sha256")
    if expected_hash is not None and expected_hash != digest:
        raise ValueError("Overlap cache SHA256 differs from the checkpoint training cache")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"source", "target", "base_permutation", "permutation", "metadata"}:
            raise ValueError("Overlap cache has missing/extra arrays")
        source, target = archive["source"], archive["target"]
        base_permutation, permutation = archive["base_permutation"], archive["permutation"]
        metadata = json.loads(str(archive["metadata"].item()))
    for key, value in cache_spec(config, dataset).items():
        if metadata.get(key) != value:
            raise ValueError(f"Overlap cache/config mismatch: {key}")
    size = metadata["base_spec"]["superset_size"]
    if source.shape != (size, 2) or target.shape != source.shape \
            or source.dtype != np.float32 or target.dtype != np.float32 \
            or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Overlap cache contains invalid endpoints")
    _permutation(base_permutation, size, "base_permutation")
    _permutation(permutation, size)
    if metadata.get("endpoint_sha256") != {"source": msw._array_hash(source), "target": msw._array_hash(target)}:
        raise ValueError("Overlap endpoint hashes do not match stored arrays")
    if metadata.get("base_permutation_sha256") != msw._array_hash(base_permutation) \
            or metadata.get("permutation_sha256") != msw._array_hash(permutation):
        raise ValueError("Overlap permutation hashes do not match stored mapping")
    if metadata.get("target_coordinates") != "unchanged original sampled points; no barycentric projection":
        raise ValueError("Overlap cache has unsupported target coordinates")
    if metadata.get("source_coordinates") != "unchanged Gaussian draws; no jitter, whitening or perturbation":
        raise ValueError("Overlap cache has unsupported source coordinates")
    original_hash = config[msw.METHOD].get("cache_sha256")
    if original_hash is not None and original_hash != metadata.get("base_cache_sha256"):
        raise ValueError("Overlap baseline cache SHA256 differs from configured baseline")
    return source, target, permutation, metadata, digest


def prepare(config, dataset):
    opts, spec = settings(config), cache_spec(config, dataset)
    path = Path(opts["cache"])
    # Never resample, silently recreate or modify the baseline for this arm.
    source, target, base_permutation, base_metadata, base_digest = msw.load_cache(base_config(config), dataset)
    if path.exists():
        _, _, _, metadata, digest = load_cache(config, dataset)
        if metadata.get("base_cache_sha256") != base_digest:
            raise ValueError("Existing overlap cache was built from a different baseline file")
        print(f"reused_overlap_cache={path} sha256={digest}", flush=True)
        return path, metadata
    started = perf_counter()
    base_opts = msw.settings(base_config(config))
    permutation, diagnostics = overlap_permutation(
        source, target, base_permutation, base_opts["partition_counts"], base_opts["scale_weights"],
        **{key: opts[key] for key in ("window_sizes", "max_windows_per_scale", "sweeps", "neighbors", "max_local_solves")},
        max_cost_mib=base_opts["max_cost_mib"], progress=lambda message: print(message, flush=True))
    baseline_objective = base_metadata.get("multiscale_objective")
    if not isinstance(baseline_objective, (int, float)) or not math.isfinite(baseline_objective) \
            or not np.isclose(diagnostics["objective_before"], baseline_objective, atol=1e-10, rtol=1e-8):
        raise ValueError("Overlap starting point objective does not agree with baseline MS-W2 objective")
    metadata = {**spec, **diagnostics, "base_cache_sha256": base_digest,
                "base_multiscale_objective": baseline_objective,
                "base_permutation_sha256": msw._array_hash(base_permutation),
                "endpoint_sha256": {"source": msw._array_hash(source), "target": msw._array_hash(target)},
                "permutation_sha256": msw._array_hash(permutation),
                "source_moments": source_moments(source), "precompute_seconds": perf_counter() - started,
                "precompute_timing_scope": "overlap preparation only; excludes original bank preparation, cache IO/loading and training",
                "base_precompute_seconds": base_metadata.get("precompute_seconds"),
                "marginals": "original finite source/target empirical measures preserved exactly; population marginals approximate",
                "source_coordinates": "unchanged Gaussian draws; no jitter, whitening or perturbation",
                "target_coordinates": "unchanged original sampled points; no barycentric projection",
                "source_sha256": {name: file_sha256(Path(__file__).parent / name) for name in
                                  ("multiscale_overlap.py", "multiscale_wasserstein.py", "nsot.py", "data.py", "train_horse.py")}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, source=source, target=target, base_permutation=base_permutation,
                            permutation=permutation, metadata=np.array(json.dumps(metadata, allow_nan=False)))
    print(f"saved_overlap_cache={path} sha256={file_sha256(path)}", flush=True)
    return path, metadata


class MultiscaleOverlapSampler:
    def __init__(self, config, dataset, device, dtype):
        source, target, permutation, self.metadata, self.cache_sha256 = load_cache(config, dataset)
        self.source = torch.as_tensor(source, device=device, dtype=dtype)
        self.target = torch.as_tensor(target[permutation], device=device, dtype=dtype)
        self.n_points = config["data"]["n_points"]

    @torch.no_grad()
    def sample(self, batch_size, *, generator=None):
        msw._integer(batch_size, "batch_size")
        indices = torch.randint(len(self.source), (batch_size, self.n_points),
                                device=self.source.device, generator=generator)
        return self.source[indices], self.target[indices]

    def details(self):
        return {**self.metadata, "cache_sha256": self.cache_sha256,
                "plan_sampling": "iid pairs with replacement from the fixed bijective bank",
                "inference_source": "fresh iid standard Gaussian; no bank or patch labels", "beta": 0.0}
