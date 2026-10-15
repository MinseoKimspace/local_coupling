"""Exact multiscale patch-distribution matching, with an unchanged finite bank.

Each spatial scale contains nested equal-mass patches. Full empirical patch
W2^2 costs (NOT centroid/sliced/entropic approximations) are lifted to the leaf
patches and averaged with fixed weights. Exact leaf assignment is followed by
exact point assignment inside selected leaf pairs. This is a block-constrained
coupling, not global Euclidean OT, FGW, or a guarantee of learnability.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch

from data import validate_gaussian_source
from nsot import dataset_spec, file_sha256, source_moments

METHOD = "multiscale_wasserstein"
FORMAT_VERSION = 1
IMPLEMENTATION = "nested_equal_mass_patch_w2_exact_v1"
PCA_IMPLEMENTATION = "nested_equal_mass_patch_w2_exact_pca_v1"
ZERO_WEIGHT_IMPLEMENTATION = "nested_equal_mass_patch_w2_exact_skip_zero_v1"
PCA_ZERO_WEIGHT_IMPLEMENTATION = "nested_equal_mass_patch_w2_exact_pca_skip_zero_v1"
PARTITION_METHODS = ("axis", "pca")
PCA_EIGENGAP_TOL = 1e-6


def partition_settings(source="axis", target="axis", eigengap_tol=PCA_EIGENGAP_TOL):
    for name, value in (("source_partition", source), ("target_partition", target)):
        if not isinstance(value, str) or value not in PARTITION_METHODS:
            raise ValueError(f"{name} must be axis or pca")
    if isinstance(eigengap_tol, bool) or not isinstance(eigengap_tol, (int, float)) \
            or not math.isfinite(eigengap_tol) or not 0 <= eigengap_tol < 1:
        raise ValueError("pca_eigengap_tol must be finite and in [0,1)")
    return {"source_partition": source, "target_partition": target,
            "pca_eigengap_tol": float(eigengap_tol)}


def _integer(value, name, minimum=1, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) \
            or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum or 'inf'}]")
    return int(value)


def _scales(count, partition_counts, scale_weights):
    if not isinstance(partition_counts, (list, tuple)) or not partition_counts:
        raise ValueError("partition_counts must be a nonempty list")
    counts, previous = [], 1
    for level, value in enumerate(partition_counts):
        value = _integer(value, "partition count", maximum=count)
        if value % previous or (level and value <= previous) or count % value:
            raise ValueError("partition_counts must increase, be nested multiples and divide superset_size")
        counts.append(value)
        previous = value
    if not isinstance(scale_weights, (list, tuple)) or len(scale_weights) != len(counts):
        raise ValueError("scale_weights must have one value per partition count")
    if any(isinstance(w, bool) or not isinstance(w, (int, float, np.number))
           or not np.isreal(w) or not np.isfinite(w) or w < 0 for w in scale_weights):
        raise ValueError("scale_weights must be finite nonnegative numbers")
    weights = np.asarray(scale_weights, dtype=np.float64)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("scale_weights must have a finite positive sum")
    return counts, (weights / total).tolist()


def _memory_guard(count, counts, max_cost_mib, weights=None):
    if isinstance(max_cost_mib, bool) or not isinstance(max_cost_mib, (int, float)) \
            or not math.isfinite(max_cost_mib) or max_cost_mib <= 0:
        raise ValueError("max_cost_mib must be finite and positive")
    # Bound one dense cost matrix, NOT the total process peak (solver workspaces
    # and several small leaf-cost matrices also consume memory).
    # All partitions are still built, but zero-weight scales allocate no patch
    # costs. Leaf assignment and final within-leaf exact OT always run.
    active = counts if weights is None else [n for n, w in zip(counts, weights) if w > 0]
    elements = [counts[-1] ** 2, (count // counts[-1]) ** 2]
    elements.extend((count // n) ** 2 for n in active)
    largest = max(elements) * 8 / 2**20
    if largest > max_cost_mib:
        raise ValueError(f"Largest dense cost matrix {largest:.1f} MiB exceeds max_cost_mib={max_cost_mib}")
    return largest


def settings(config):
    if config.get("coupling") != METHOD:
        raise ValueError(f"Expected coupling: {METHOD}")
    raw = config.get(METHOD, {})
    if not isinstance(raw, dict):
        raise ValueError(f"{METHOD} must be a mapping")
    unknown = raw.keys() - {"cache", "cache_sha256", "superset_size", "cache_seed",
                            "partition_counts", "scale_weights", "max_cost_mib",
                            "source_partition", "target_partition", "pca_eigengap_tol"}
    if unknown:
        raise ValueError("Unsupported multiscale settings: " + ", ".join(sorted(unknown)))
    size = _integer(raw.get("superset_size", 10000), "superset_size", maximum=10000)
    seed = _integer(raw.get("cache_seed", 0), "cache_seed", minimum=0, maximum=2**63 - 1)
    counts, weights = _scales(size, raw.get("partition_counts", [4, 20, 100]),
                              raw.get("scale_weights", [1, 1, 1]))
    memory = raw.get("max_cost_mib", 512)
    _memory_guard(size, counts, memory, weights)
    path = raw.get("cache")
    if not isinstance(path, str) or not path.strip() or path == "auto":
        raise ValueError("A concrete multiscale cache path is required")
    if config["model"]["point_dim"] != 2:
        raise ValueError("Multiscale bank currently supports 2D only")
    _integer(config["data"]["n_points"], "n_points", maximum=size)
    validate_gaussian_source(config)
    partitions = partition_settings(raw.get("source_partition", "axis"),
                                    raw.get("target_partition", "axis"),
                                    raw.get("pca_eigengap_tol", PCA_EIGENGAP_TOL))
    return {"cache": path, "superset_size": size, "cache_seed": seed,
            "partition_counts": counts, "scale_weights": weights, "max_cost_mib": float(memory),
            **partitions}


def _points(points):
    value = np.asarray(points)
    if value.ndim != 2 or value.shape[1] != 2 or not len(value) \
            or not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value) \
            or not np.isfinite(value).all():
        raise ValueError("Expected finite nonempty real [M,2] points")
    return value


def _partition_projection(points, indices, strategy, eigengap_tol, diagnostics):
    """Choose a sorting coordinate only; never transform OT/training endpoints."""
    patch = points[indices].astype(np.float64)
    axis = int(np.argmax(np.ptp(patch, axis=0)))
    if diagnostics is not None:
        diagnostics["splits"] += 1
    if strategy == "axis":
        return points[indices, axis]  # Preserve the original baseline exactly.
    centered = patch - patch.mean(axis=0)
    covariance = centered.T @ centered / len(patch)
    if not np.isfinite(covariance).all():
        raise ValueError("Nonfinite PCA covariance")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = float(eigenvalues[-1])
    gap = float((largest - eigenvalues[-2]) / largest) if largest > 0 else 0.
    if diagnostics is not None:
        previous = diagnostics["min_relative_eigengap"]
        diagnostics["min_relative_eigengap"] = gap if previous is None else min(previous, gap)
    if largest <= 0 or gap <= eigengap_tol:
        if diagnostics is not None:
            diagnostics["axis_fallbacks"] += 1
        return points[indices, axis]
    direction = eigenvectors[:, -1].copy()
    # A sign flip changes recursive 2/3 splits, not just their label order.
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0:
        direction *= -1
    if diagnostics is not None:
        diagnostics["pca_splits"] += 1
    return centered @ direction


def _split_equal(points, indices, groups, strategy="axis", eigengap_tol=PCA_EIGENGAP_TOL,
                 diagnostics=None):
    """Balanced recursive subdivision; original-index ties are stable."""
    if groups == 1:
        return [indices]
    projection = _partition_projection(points, indices, strategy, eigengap_tol, diagnostics)
    order = np.lexsort((indices, projection))
    ordered = indices[order]
    left_groups = groups // 2
    cut = (len(indices) // groups) * left_groups
    return (_split_equal(points, ordered[:cut], left_groups, strategy, eigengap_tol, diagnostics)
            + _split_equal(points, ordered[cut:], groups - left_groups, strategy, eigengap_tol, diagnostics))


def balanced_partitions(points, partition_counts, *, strategy="axis",
                        pca_eigengap_tol=PCA_EIGENGAP_TOL, diagnostics=None):
    points = _points(points)
    partition_settings(strategy, strategy, pca_eigengap_tol)
    if diagnostics is not None:
        diagnostics.update(strategy=strategy, splits=0, pca_splits=0, axis_fallbacks=0,
                           min_relative_eigengap=None)
    if not isinstance(partition_counts, (list, tuple)) or not partition_counts:
        raise ValueError("partition_counts must be a nonempty list")
    counts, _ = _scales(len(points), partition_counts, [1] * len(partition_counts))
    groups, previous, result = [np.arange(len(points), dtype=np.int64)], 1, []
    for count in counts:
        groups = [child for parent in groups for child in _split_equal(
            points, parent, count // previous, strategy, pca_eigengap_tol, diagnostics)]
        labels = np.empty(len(points), dtype=np.int64)
        for label, indices in enumerate(groups):
            labels[indices] = label
        result.append(labels)
        previous = count
    return result


def _assignment(a, b):
    cost = cdist(a, b, metric="sqeuclidean")
    if not np.isfinite(cost).all():
        raise ValueError("Nonfinite squared Euclidean costs")
    rows, cols = linear_sum_assignment(cost)
    order = np.empty(len(a), dtype=np.int64)
    order[rows] = cols
    return order, float(cost[rows, cols].mean())


def patch_wasserstein_squared(a, b):
    """Exact W2^2 of equally sized, uniform empirical patch distributions."""
    a, b = _points(a), _points(b)
    if a.shape != b.shape:
        raise ValueError("Patch W2 requires equal-size supports")
    return _assignment(a, b)[1]


def multiscale_permutation(source, target, partition_counts, scale_weights,
                          max_cost_mib=512, progress=None, *, source_partition="axis",
                          target_partition="axis", pca_eigengap_tol=PCA_EIGENGAP_TOL):
    partitions = partition_settings(source_partition, target_partition, pca_eigengap_tol)
    source, target = _points(source), _points(target)
    if source.shape != target.shape:
        raise ValueError("Source and target must have matching shapes")
    size = len(source)
    counts, weights = _scales(size, partition_counts, scale_weights)
    largest = _memory_guard(size, counts, max_cost_mib, weights)
    start = perf_counter()
    source_diagnostics, target_diagnostics = {}, {}
    source_labels = balanced_partitions(source, counts, strategy=source_partition,
                                       pca_eigengap_tol=pca_eigengap_tol, diagnostics=source_diagnostics)
    target_labels = balanced_partitions(target, counts, strategy=target_partition,
                                       pca_eigengap_tol=pca_eigengap_tol, diagnostics=target_diagnostics)
    partition_seconds = perf_counter() - start
    leaves = counts[-1]
    source_leaves = [np.flatnonzero(source_labels[-1] == i) for i in range(leaves)]
    target_leaves = [np.flatnonzero(target_labels[-1] == i) for i in range(leaves)]
    leaf_cost = np.zeros((leaves, leaves), dtype=np.float64)
    reports, scale_costs = [], []
    for level, (count, weight) in enumerate(zip(counts, weights)):
        if weight == 0:
            # Keep the hierarchy (and therefore the finest partition) exactly
            # the same. Only omit a cost that contributes zero to the objective.
            reports.append({"partition_count": count, "points_per_patch": size // count,
                            "weight": weight, "exact_patch_solves": 0, "cost_seconds": 0.,
                            "cost_status": "skipped_zero_weight",
                            "selected_mean_w2_squared": None})
            scale_costs.append(None)
            if progress:
                progress(f"scale={level + 1}/{len(counts)} patches={count} skipped_zero_weight")
            continue
        level_start = perf_counter()
        source_groups = [source[source_labels[level] == i] for i in range(count)]
        target_groups = [target[target_labels[level] == j] for j in range(count)]
        matrix = np.empty((count, count), dtype=np.float64)
        if progress:
            progress(f"scale={level + 1}/{len(counts)} patches={count} points/patch={size // count} "
                     f"exact_patch_solves={count * count}")
        for i, a in enumerate(source_groups):
            for j, b in enumerate(target_groups):
                matrix[i, j] = patch_wasserstein_squared(a, b)
            if progress and ((i + 1) % max(1, count // 10) == 0 or i + 1 == count):
                progress(f"scale={level + 1} rows={i + 1}/{count} seconds={perf_counter() - level_start:.1f}")
        # Read actual memberships; no assumption that independent partitions nest.
        src_parent = np.array([source_labels[level][indices[0]] for indices in source_leaves])
        tgt_parent = np.array([target_labels[level][indices[0]] for indices in target_leaves])
        lifted = matrix[src_parent[:, None], tgt_parent[None, :]]
        leaf_cost += weight * lifted
        scale_costs.append((matrix, src_parent, tgt_parent))
        reports.append({"partition_count": count, "points_per_patch": size // count,
                        "weight": weight, "exact_patch_solves": count**2,
                        "cost_seconds": perf_counter() - level_start})
    assignment_start = perf_counter()
    rows, cols = linear_sum_assignment(leaf_cost)
    leaf_permutation = np.empty(leaves, dtype=np.int64)
    leaf_permutation[rows] = cols
    leaf_assignment_seconds = perf_counter() - assignment_start
    refinement_start = perf_counter()
    permutation = np.empty(size, dtype=np.int64)
    for i, j in enumerate(leaf_permutation):
        source_ids, target_ids = source_leaves[i], target_leaves[j]
        local, _ = _assignment(source[source_ids], target[target_ids])
        permutation[source_ids] = target_ids[local]
    if not np.array_equal(np.sort(permutation), np.arange(size)):
        raise RuntimeError("Internal error: matching is not bijective")
    for report, scale_cost in zip(reports, scale_costs):
        if scale_cost is None:
            continue
        matrix, a, b = scale_cost
        report["selected_mean_w2_squared"] = float(matrix[a, b[leaf_permutation]].mean())
    diagnostics = {"partition_seconds": partition_seconds, "scales": reports,
                   "partition_settings": partitions,
                   "partition_diagnostics": {"source": source_diagnostics, "target": target_diagnostics},
                   "leaf_assignment_seconds": leaf_assignment_seconds,
                   "point_assignment_seconds": perf_counter() - refinement_start,
                   "matching_seconds": perf_counter() - start,
                   "largest_dense_cost_mib": largest,
                   "memory_estimate_scope": "one dense cost matrix only, excludes solver and other arrays",
                   "leaf_permutation": leaf_permutation.tolist(),
                   "multiscale_objective": float(leaf_cost[rows, cols].mean()),
                   "point_transport_cost": float(np.square(source.astype(np.float64) - target[permutation]).sum(1).mean()),
                   "matching_scope": "exact leaf assignment for multiscale cost, then exact within-leaf point assignment; not global point W2"}
    return permutation, diagnostics


def draw_supersets(config, dataset):
    """Draw the original IID endpoints, byte-identical to NSOT."""
    from nsot import draw_supersets as draw, SOLVER
    opts = settings(config)
    baseline = copy.deepcopy(config)
    baseline["coupling"] = "nsot"
    baseline["nsot"] = {"cache": opts["cache"], "superset_size": opts["superset_size"],
                        "cache_seed": opts["cache_seed"], "beta": 0.0, "solver": SOLVER}
    return draw(baseline, dataset)


def cache_spec(config, dataset):
    opts = settings(config)
    # Earlier ablations computed all costs even when their weight was zero.
    # Their mapping is equivalent, but their preprocessing timings are not.
    # Fail closed instead of silently reusing those old timing measurements.
    skips_zero = any(weight == 0 for weight in opts["scale_weights"])
    implementation = ZERO_WEIGHT_IMPLEMENTATION if skips_zero else IMPLEMENTATION
    spec = {"format_version": FORMAT_VERSION, "implementation": implementation,
            "superset_size": opts["superset_size"], "cache_seed": opts["cache_seed"],
            "partition_counts": opts["partition_counts"], "scale_weights": opts["scale_weights"],
            "dataset_spec": dataset_spec(config, dataset)}
    # Old axis/axis caches remain valid. PCA caches have a distinct identity and
    # cannot be silently accepted by a baseline config (or vice versa).
    if "pca" in (opts["source_partition"], opts["target_partition"]):
        spec.update(implementation=PCA_ZERO_WEIGHT_IMPLEMENTATION if skips_zero else PCA_IMPLEMENTATION,
                    partition_settings=partition_settings(opts["source_partition"],
                                                          opts["target_partition"], opts["pca_eigengap_tol"]))
    return spec


def _array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_cache(config, dataset):
    opts = settings(config)
    path = Path(opts["cache"])
    if not path.is_file():
        raise FileNotFoundError(f"Multiscale cache missing: {path}; run python run_multiscale.py --stage prepare")
    digest = file_sha256(path)
    expected_hash = config[METHOD].get("cache_sha256")
    if expected_hash is not None and expected_hash != digest:
        raise ValueError("Multiscale cache SHA256 differs from the checkpoint training cache")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"source", "target", "permutation", "metadata"}:
            raise ValueError("Multiscale cache has missing/extra arrays")
        source, target, permutation = archive["source"], archive["target"], archive["permutation"]
        metadata = json.loads(str(archive["metadata"].item()))
    expected_spec = cache_spec(config, dataset)
    for key, value in expected_spec.items():
        if metadata.get(key) != value:
            raise ValueError(f"Multiscale cache/config mismatch: {key}")
    if metadata.get("target_coordinates") != "unchanged original sampled points; no barycentric projection":
        raise ValueError("Multiscale cache has unsupported target coordinates")
    size = opts["superset_size"]
    if source.shape != (size, 2) or target.shape != source.shape \
            or source.dtype != np.float32 or target.dtype != np.float32 \
            or not np.isfinite(source).all() or not np.isfinite(target).all() \
            or permutation.shape != (size,) or not np.issubdtype(permutation.dtype, np.integer) \
            or not np.array_equal(np.sort(permutation), np.arange(size)):
        raise ValueError("Multiscale cache contains invalid points or non-bijective permutation")
    if metadata.get("endpoint_sha256") != {"source": _array_hash(source), "target": _array_hash(target)}:
        raise ValueError("Multiscale endpoint hashes do not match stored arrays")
    if metadata.get("permutation_sha256") != _array_hash(permutation):
        raise ValueError("Multiscale permutation hash does not match stored mapping")
    return source, target, permutation, metadata, digest


def prepare(config, dataset):
    """Prepare once, validating rather than overwriting any existing cache."""
    opts, spec = settings(config), cache_spec(config, dataset)
    path = Path(opts["cache"])
    if path.exists():
        _, _, _, metadata, digest = load_cache(config, dataset)
        print(f"reused_multiscale_cache={path} sha256={digest}", flush=True)
        return path, metadata
    start = perf_counter()
    source, target = draw_supersets(config, dataset)
    sampling_seconds = perf_counter() - start
    permutation, diagnostics = multiscale_permutation(
        source, target, opts["partition_counts"], opts["scale_weights"], opts["max_cost_mib"],
        progress=lambda message: print(message, flush=True),
        source_partition=opts["source_partition"], target_partition=opts["target_partition"],
        pca_eigengap_tol=opts["pca_eigengap_tol"])
    metadata = {**spec, **diagnostics, "sampling_seconds": sampling_seconds,
                "precompute_seconds": perf_counter() - start,
                "precompute_timing_scope": "sampling and matching; excludes cache IO/loading and training",
                "endpoint_sha256": {"source": _array_hash(source), "target": _array_hash(target)},
                "permutation_sha256": _array_hash(permutation),
                "source_moments": source_moments(source),
                "marginals": "original finite source/target empirical measures preserved exactly; population marginals approximate",
                "source_coordinates": "unchanged Gaussian draws; no jitter, whitening or perturbation",
                "target_coordinates": "unchanged original sampled points; no barycentric projection",
                "source_sha256": {name: file_sha256(Path(__file__).parent / name) for name in
                                  ("multiscale_wasserstein.py", "nsot.py", "data.py", "train_horse.py")}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, source=source, target=target, permutation=permutation,
                            metadata=np.array(json.dumps(metadata, allow_nan=False)))
    print(f"saved_multiscale_cache={path} sha256={file_sha256(path)} "
          f"precompute_seconds={metadata['precompute_seconds']:.3f}", flush=True)
    return path, metadata


class MultiscaleWassersteinSampler:
    def __init__(self, config, dataset, device, dtype):
        settings(config)
        source, target, permutation, self.metadata, self.cache_sha256 = load_cache(config, dataset)
        self.source = torch.as_tensor(source, device=device, dtype=dtype)
        self.target = torch.as_tensor(target[permutation], device=device, dtype=dtype)
        self.n_points = config["data"]["n_points"]

    @torch.no_grad()
    def sample(self, batch_size, *, generator=None):
        _integer(batch_size, "batch_size")
        indices = torch.randint(len(self.source), (batch_size, self.n_points),
                                device=self.source.device, generator=generator)
        return self.source[indices], self.target[indices]

    def details(self):
        partitions = self.metadata.get("partition_settings", partition_settings())
        descriptions = {"axis": "nested_equal_mass_spatial_kd", "pca": "nested_equal_mass_spatial_pca"}
        return {**self.metadata, "cache_sha256": self.cache_sha256,
                "source_partition": descriptions[partitions["source_partition"]],
                "target_partition": descriptions[partitions["target_partition"]],
                "plan_sampling": "iid pairs with replacement from the fixed bijective bank",
                "inference_source": "fresh iid standard Gaussian; no bank or patch labels",
                "beta": 0.0}
