"""Point-level sparse refinement of a prepared multiscale Wasserstein bank.

Keep all point edges in the originally matched leaf blocks. Add symmetric swap
edges from source and target spatial k-nearest neighbors across leaf boundaries,
then solve one balanced sparse assignment. Original coordinates are never moved
or averaged. This is a neighborhood-support experiment, not Gerber--Maggioni's
full multiscale solver, capacity propagation, or potential-based refinement.

The additive point cost agrees with the original multiscale objective on its
block-optimal assignment: ancestor patch W2 costs at all non-leaf scales plus
the finest weight times pointwise squared distance. Relaxing the blocks cannot
increase this specified objective, but need not improve learned flow quality.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree
import torch

import multiscale_wasserstein as base
from nsot import file_sha256


METHOD = "multiscale_refined"
FORMAT_VERSION = 1
IMPLEMENTATION = "multiscale_cross_leaf_sparse_assignment_v1"
OBJECTIVE = ("sum of normalized non-leaf ancestor patch W2_squared costs plus "
             "normalized finest weight times point squared Euclidean distance; mean over source points")


def base_config(config):
    result = copy.deepcopy(config)
    result["coupling"] = base.METHOD
    result.pop(METHOD, None)
    return result


def _integer(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) \
            or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum or 'inf'}]")
    return int(value)


def _edge_limit(size, leaves, neighbors, max_edges):
    neighbors = _integer(neighbors, "neighbors", maximum=size - 1)
    max_edges = _integer(max_edges, "max_edges", minimum=1)
    upper = size * (size // leaves + 4 * neighbors)
    if upper > max_edges:
        raise ValueError(f"Candidate edge upper bound {upper} exceeds max_edges={max_edges}; "
                         "this conservative guard counts proposals before deduplication")
    return neighbors, max_edges, upper


def settings(config):
    if config.get("coupling") != METHOD:
        raise ValueError(f"Expected coupling: {METHOD}")
    baseline = base.settings(base_config(config))
    raw = config.get(METHOD, {})
    if not isinstance(raw, dict):
        raise ValueError(f"{METHOD} must be a mapping")
    unknown = raw.keys() - {"cache", "cache_sha256", "neighbors", "max_edges"}
    if unknown:
        raise ValueError("Unsupported refinement settings: " + ", ".join(sorted(unknown)))
    path = raw.get("cache")
    if not isinstance(path, str) or not path.strip() or path == "auto":
        raise ValueError("A concrete refinement cache path is required")
    if Path(path).resolve() == Path(baseline["cache"]).resolve():
        raise ValueError("Refined cache must not overwrite the original multiscale cache")
    neighbors, max_edges, upper = _edge_limit(
        baseline["superset_size"], baseline["partition_counts"][-1],
        raw.get("neighbors", 8), raw.get("max_edges", 2000000))
    return {"cache": path, "neighbors": neighbors, "max_edges": max_edges,
            "proposal_upper_bound": upper, "base": baseline}


def _permutation(value, size, name="permutation"):
    value = np.asarray(value)
    if value.shape != (size,) or not np.issubdtype(value.dtype, np.integer) \
            or not np.array_equal(np.sort(value), np.arange(size)):
        raise ValueError(f"{name} must be a full bijective permutation")
    return value.astype(np.int64, copy=False)


def _labels(value, size):
    value = np.asarray(value)
    if value.shape != (size,) or not np.issubdtype(value.dtype, np.integer):
        raise ValueError("Leaf labels must be integer [M] arrays")
    unique, counts = np.unique(value, return_counts=True)
    if not np.array_equal(unique, np.arange(len(unique))) or not np.all(counts == counts[0]):
        raise ValueError("Leaf labels must be contiguous, nonnegative and equal mass")
    return value, len(unique)


def _matched_blocks(order, source_labels, target_labels):
    size = len(order)
    source_labels, leaves = _labels(source_labels, size)
    target_labels, target_leaves = _labels(target_labels, size)
    if leaves != target_leaves:
        raise ValueError("Source and target must have the same number of leaf patches")
    blocks, matched = [], []
    for leaf in range(leaves):
        source_ids = np.flatnonzero(source_labels == leaf)
        target_leaf = np.unique(target_labels[order[source_ids]])
        if len(target_leaf) != 1:
            raise ValueError("Base permutation must map each source leaf wholly to one target leaf")
        matched.append(int(target_leaf[0]))
        blocks.append((source_ids, np.flatnonzero(target_labels == target_leaf[0])))
    if len(set(matched)) != leaves:
        raise ValueError("Base leaf assignment must be bijective")
    return blocks, source_labels, target_labels


def _nearest_other(tree, points, index, neighbors):
    """Exactly k nearest other indices, resolving equal distances by index.

    Only one point's query/tie set is materialized. Querying the boundary ball
    avoids cKDTree's unspecified selection when more than k points tie.
    """
    distances, indices = tree.query(points[index], k=neighbors + 1, workers=1)
    distances, indices = np.atleast_1d(distances), np.atleast_1d(indices)
    other = indices != index
    boundary = np.sort(distances[other])[neighbors - 1]
    radius = boundary + 32 * np.finfo(np.float64).eps * max(1.0, float(boundary))
    candidates = np.asarray(tree.query_ball_point(points[index], radius), dtype=np.int64)
    candidates = candidates[candidates != index]
    squared = np.square(points[candidates] - points[index]).sum(axis=1)
    if len(candidates) < neighbors or not np.isfinite(squared).all():
        raise ValueError("Could not construct finite deterministic nearest neighbors")
    return candidates[np.lexsort((candidates, squared))[:neighbors]]


def candidate_edges(source, target, base_permutation, source_leaf_labels, target_leaf_labels,
                    *, neighbors=8, max_edges=2000000):
    """Sorted unique sparse candidate edges and construction diagnostics.

    For each source point, take k nearest OTHER source points, then keep only
    neighbors in a different source leaf. Add both swapped original partners.
    Repeat in target space using the inverse original permutation and different
    target leaves. This is NOT 'search until k cross-leaf neighbors are found'.
    All within-original-matched-block edges are retained. The early max_edges
    guard is the conservative raw bound M*(M/L + 4*k), before deduplication.
    """
    source, target = base._points(source), base._points(target)
    if source.shape != target.shape:
        raise ValueError("Source and target must have matching shapes")
    size = len(source)
    order = _permutation(base_permutation, size, "base_permutation")
    blocks, source_labels, target_labels = _matched_blocks(
        order, source_leaf_labels, target_leaf_labels)
    neighbors, _, upper = _edge_limit(size, len(blocks), neighbors, max_edges)
    encoded = np.empty(upper, dtype=np.int64)
    cursor = 0
    for source_ids, target_ids in blocks:
        for source_id in source_ids:
            stop = cursor + len(target_ids)
            encoded[cursor:stop] = source_id * size + target_ids
            cursor = stop
    block_edges = cursor
    counts = {"source_neighbor_proposals": 0, "target_neighbor_proposals": 0}
    if neighbors:
        inverse = np.empty(size, dtype=np.int64)
        inverse[order] = np.arange(size)
        for name, points, labels in (("source", source, source_labels), ("target", target, target_labels)):
            points = np.asarray(points, dtype=np.float64)
            tree = cKDTree(points)
            for index in range(size):
                adjacent = _nearest_other(tree, points, index, neighbors)
                adjacent = adjacent[labels[adjacent] != labels[index]]
                count = len(adjacent)
                if not count:
                    continue
                if name == "source":
                    encoded[cursor:cursor + count] = index * size + order[adjacent]
                    encoded[cursor + count:cursor + 2 * count] = adjacent * size + order[index]
                else:
                    encoded[cursor:cursor + count] = inverse[index] * size + adjacent
                    encoded[cursor + count:cursor + 2 * count] = inverse[adjacent] * size + index
                cursor += 2 * count
                counts[name + "_neighbor_proposals"] += 2 * count
    encoded = np.unique(encoded[:cursor])
    base_edges = np.arange(size, dtype=np.int64) * size + order
    locations = np.searchsorted(encoded, base_edges)
    if np.any(locations >= len(encoded)) or not np.array_equal(encoded[locations], base_edges):
        raise RuntimeError("Candidate graph lost an original permutation edge")
    rows, columns = encoded // size, encoded % size
    return rows, columns, {"edges": len(encoded), "base_block_edges": block_edges,
                           "proposal_upper_bound": upper, "raw_proposals": cursor,
                           "neighbors": neighbors, **counts,
                           "candidate_rule": "full original matched blocks plus symmetric source/target kNN cross-leaf swaps; nearest-other ties by original index",
                           "max_edges_scope": "conservative raw proposal bound before deduplication"}


def _cost_tables(source, target, source_levels, target_levels, counts, weights, progress):
    tables, reports = [], []
    for level, (count, weight) in enumerate(zip(counts[:-1], weights[:-1])):
        start = perf_counter()
        if not weight:
            tables.append(None)
            reports.append({"partition_count": count, "weight": weight,
                            "exact_patch_solves": 0, "cost_seconds": 0.0})
            continue
        matrix = np.empty((count, count), dtype=np.float64)
        source_groups = [source[source_levels[level] == i] for i in range(count)]
        target_groups = [target[target_levels[level] == i] for i in range(count)]
        if progress:
            progress(f"refinement_coarse_scale={level + 1} patches={count} points/patch={len(source) // count} exact_solves={count**2}")
        for i, left in enumerate(source_groups):
            for j, right in enumerate(target_groups):
                matrix[i, j] = base.patch_wasserstein_squared(left, right)
            if progress and ((i + 1) % max(1, count // 10) == 0 or i + 1 == count):
                progress(f"refinement_coarse_scale={level + 1} rows={i + 1}/{count} seconds={perf_counter() - start:.1f}")
        tables.append(matrix)
        reports.append({"partition_count": count, "weight": weight,
                        "exact_patch_solves": count**2, "cost_seconds": perf_counter() - start})
    return tables, reports


def _edge_costs(source, target, rows, columns, source_levels, target_levels, tables, weights):
    result = np.zeros(len(rows), dtype=np.float64)
    # Bound temporary coordinate/index arrays independently of graph size.
    for start in range(0, len(rows), 65536):
        stop = min(len(rows), start + 65536)
        left, right = rows[start:stop], columns[start:stop]
        if weights[-1]:
            delta = source[left].astype(np.float64) - target[right].astype(np.float64)
            result[start:stop] = weights[-1] * np.square(delta).sum(axis=1)
        for level, matrix in enumerate(tables):
            if matrix is not None:
                result[start:stop] += weights[level] * matrix[source_levels[level][left], target_levels[level][right]]
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError("Refinement costs must be finite nonnegative values")
    return result


def refine_permutation(source, target, base_permutation, partition_counts, scale_weights,
                       *, neighbors=8, max_edges=2000000, max_cost_mib=512,
                       progress=None, expected_base_objective=None):
    """Refine a base block bijection without changing either endpoint array."""
    source, target = base._points(source), base._points(target)
    if source.shape != target.shape:
        raise ValueError("Source and target must have matching shapes")
    size = len(source)
    original = _permutation(base_permutation, size, "base_permutation")
    counts, weights = base._scales(size, partition_counts, scale_weights)
    largest = base._memory_guard(size, counts, max_cost_mib)
    _edge_limit(size, counts[-1], neighbors, max_edges)
    start = perf_counter()
    source_levels = base.balanced_partitions(source, counts)
    target_levels = base.balanced_partitions(target, counts)
    partition_seconds = perf_counter() - start
    graph_start = perf_counter()
    rows, columns, graph = candidate_edges(
        source, target, original, source_levels[-1], target_levels[-1],
        neighbors=neighbors, max_edges=max_edges)
    graph_seconds = perf_counter() - graph_start
    if progress:
        progress(f"refinement_edges={len(rows)} base_block_edges={graph['base_block_edges']} raw_upper_bound={graph['proposal_upper_bound']}")
    cost_start = perf_counter()
    tables, reports = _cost_tables(source, target, source_levels, target_levels, counts, weights, progress)
    costs = _edge_costs(source, target, rows, columns, source_levels, target_levels, tables, weights)
    cost_seconds = perf_counter() - cost_start
    encoded = rows * size + columns
    original_positions = np.searchsorted(encoded, np.arange(size) * size + original)
    before = float(costs[original_positions].mean())
    tolerance = 1e-9 * max(1.0, abs(before))
    if expected_base_objective is not None:
        if not isinstance(expected_base_objective, (int, float, np.number)) \
                or isinstance(expected_base_objective, bool) or not np.isfinite(expected_base_objective) \
                or abs(before - float(expected_base_objective)) > tolerance:
            raise ValueError("Refinement initial objective differs from the original multiscale objective")
    solver_start = perf_counter()
    scale = max(1.0, float(costs.max()))
    if neighbors == 0:
        # Explicit no-refinement control: do not replace a tied original map.
        order = original.copy()
    else:
        # SciPy sparse matching treats stored zero weights specially. A uniform
        # positive shift changes every full matching by the same M*shift.
        positive = costs / scale + 1.0
        matrix = coo_matrix((positive, (rows, columns)), shape=(size, size)).tocsr()
        matched_rows, matched_columns = min_weight_full_bipartite_matching(matrix)
        if len(matched_rows) != size:
            raise RuntimeError("Sparse matching did not return a full assignment")
        order = np.empty(size, dtype=np.int64)
        order[matched_rows] = matched_columns
        _permutation(order, size)
    solver_seconds = perf_counter() - solver_start
    selected = np.arange(size) * size + order
    selected_positions = np.searchsorted(encoded, selected)
    if np.any(selected_positions >= len(encoded)) or not np.array_equal(encoded[selected_positions], selected):
        raise RuntimeError("Sparse assignment returned an edge outside the candidate graph")
    after = float(costs[selected_positions].mean())
    if after > before + tolerance:
        raise RuntimeError(f"Sparse refinement increased its objective: {before} -> {after}")
    target_leaf = target_levels[-1]
    diagnostics = {**graph, "partition_seconds": partition_seconds, "graph_seconds": graph_seconds,
                   "cost_seconds": cost_seconds, "coarse_scales": reports,
                   "sparse_assignment_seconds": solver_seconds,
                   "refinement_seconds": perf_counter() - start,
                   "objective": OBJECTIVE, "objective_before": before, "objective_after": after,
                   "objective_tolerance": tolerance, "base_multiscale_objective": expected_base_objective,
                   "changed_fraction": float(np.mean(order != original)),
                   "cross_leaf_fraction": float(np.mean(target_leaf[order] != target_leaf[original])),
                   "point_transport_cost_before": float(np.square(source.astype(np.float64) - target[original]).sum(axis=1).mean()),
                   "point_transport_cost_after": float(np.square(source.astype(np.float64) - target[order]).sum(axis=1).mean()),
                   "cost_scale": scale, "solver_uniform_shift": 1.0,
                   "solver": "scipy.sparse.csgraph.min_weight_full_bipartite_matching",
                   "no_refinement_control": neighbors == 0,
                   "largest_dense_cost_mib": largest,
                   "memory_scope": "coarse patch W2 obeys the original dense-matrix guard; fine assignment uses a guarded sparse graph, not a dense M-by-M point cost matrix",
                   "matching_scope": "global unit assignment on expanded sparse support; not certified globally optimal over omitted edges and no learnability/Jacobian guarantee"}
    return order, diagnostics


def cache_spec(config, dataset):
    opts = settings(config)
    return {"format_version": FORMAT_VERSION, "implementation": IMPLEMENTATION,
            "base_spec": base.cache_spec(base_config(config), dataset),
            "neighbors": opts["neighbors"], "max_edges": opts["max_edges"]}


def _array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_cache(config, dataset):
    """Load the self-contained refined bank; the original NPZ need not exist."""
    opts = settings(config)
    path = Path(opts["cache"])
    if not path.is_file():
        raise FileNotFoundError(f"Refined cache missing: {path}; prepare the refinement experiment first")
    digest = file_sha256(path)
    expected = config[METHOD].get("cache_sha256")
    if expected is not None and expected != digest:
        raise ValueError("Refined cache SHA256 differs from the checkpoint training cache")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"source", "target", "base_permutation", "permutation", "metadata"}:
            raise ValueError("Refined cache has missing/extra arrays")
        source, target = archive["source"], archive["target"]
        original, order = archive["base_permutation"], archive["permutation"]
        metadata = json.loads(str(archive["metadata"].item()))
    for key, value in cache_spec(config, dataset).items():
        if metadata.get(key) != value:
            raise ValueError(f"Refined cache/config mismatch: {key}")
    expected_base = config[base.METHOD].get("cache_sha256")
    if expected_base is not None and metadata.get("base_cache_sha256") != expected_base:
        raise ValueError("Refined bank was built from a different original cache SHA256")
    size = opts["base"]["superset_size"]
    if source.shape != (size, 2) or target.shape != source.shape \
            or source.dtype != np.float32 or target.dtype != np.float32 \
            or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Refined cache contains invalid endpoint arrays")
    _permutation(original, size, "base_permutation")
    _permutation(order, size)
    if metadata.get("endpoint_sha256") != {"source": _array_hash(source), "target": _array_hash(target)}:
        raise ValueError("Refined endpoint hashes do not match stored arrays")
    if metadata.get("base_permutation_sha256") != _array_hash(original):
        raise ValueError("Refined base permutation hash does not match stored mapping")
    if metadata.get("permutation_sha256") != _array_hash(order):
        raise ValueError("Refined permutation hash does not match stored mapping")
    return source, target, order, metadata, digest


def prepare(config, dataset):
    """Require a prepared original bank; never overwrite either cache."""
    opts, spec = settings(config), cache_spec(config, dataset)
    path = Path(opts["cache"])
    if path.exists():
        _, _, _, metadata, digest = load_cache(config, dataset)
        print(f"reused_refined_cache={path} sha256={digest}", flush=True)
        return path, metadata
    start = perf_counter()
    try:
        source, target, original, original_metadata, original_hash = base.load_cache(base_config(config), dataset)
    except FileNotFoundError as error:
        raise FileNotFoundError("Prepare the original multiscale Wasserstein bank before refinement; "
                                "the original cache will not be regenerated or substituted automatically") from error
    base_load_seconds = perf_counter() - start
    baseline = opts["base"]
    order, diagnostics = refine_permutation(
        source, target, original, baseline["partition_counts"], baseline["scale_weights"],
        neighbors=opts["neighbors"], max_edges=opts["max_edges"], max_cost_mib=baseline["max_cost_mib"],
        progress=lambda message: print(message, flush=True),
        expected_base_objective=original_metadata["multiscale_objective"])
    metadata = {**spec, **diagnostics,
                "base_cache_sha256": original_hash, "base_cache_path": str(Path(baseline["cache"]).resolve()),
                "base_permutation_sha256": _array_hash(original), "permutation_sha256": _array_hash(order),
                "endpoint_sha256": {"source": _array_hash(source), "target": _array_hash(target)},
                "base_precompute_seconds": original_metadata["precompute_seconds"],
                "base_load_seconds": base_load_seconds, "precompute_seconds": perf_counter() - start,
                "precompute_timing_scope": "incremental refinement including base loading and coarse-cost recomputation; excludes original bank construction and refined cache IO",
                "source_moments": original_metadata["source_moments"],
                "marginals": "unchanged original finite source/target empirical measures; population approximation remains",
                "source_coordinates": "unchanged original Gaussian bank; no jitter or perturbation",
                "target_coordinates": "unchanged original target bank; no barycentric averaging",
                "source_sha256": {name: file_sha256(Path(__file__).parent / name) for name in
                                  ("multiscale_refinement.py", "multiscale_wasserstein.py", "nsot.py", "data.py", "train_horse.py")}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, source=source, target=target, base_permutation=original,
                            permutation=order, metadata=np.array(json.dumps(metadata, allow_nan=False)))
    print(f"saved_refined_cache={path} sha256={file_sha256(path)} precompute_seconds={metadata['precompute_seconds']:.3f}", flush=True)
    return path, metadata


class MultiscaleRefinedSampler:
    def __init__(self, config, dataset, device, dtype):
        settings(config)
        source, target, order, self.metadata, self.cache_sha256 = load_cache(config, dataset)
        self.source = torch.as_tensor(source, device=device, dtype=dtype)
        self.target = torch.as_tensor(target[order], device=device, dtype=dtype)
        self.n_points = config["data"]["n_points"]

    @torch.no_grad()
    def sample(self, batch_size, *, generator=None):
        _integer(batch_size, "batch_size", minimum=1)
        indices = torch.randint(len(self.source), (batch_size, self.n_points),
                                device=self.source.device, generator=generator)
        return self.source[indices], self.target[indices]

    def details(self):
        return {**self.metadata, "cache_sha256": self.cache_sha256,
                "plan_sampling": "iid pairs with replacement; same draw law as original multiscale sampler",
                "inference_source": "fresh iid standard Gaussian; no bank or patch labels",
                "beta": 0.0}
