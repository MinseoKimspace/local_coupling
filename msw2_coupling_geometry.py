"""Read-only geometry probes for one fixed MS-W2 bank coupling.

These probes describe a finite assignment and a nearest-bank extension proxy.
Neither a large velocity jump nor the proxy alone establishes a model failure.
All distances and vector norms are Euclidean sums over coordinates, not
per-coordinate means. Computation uses kNN trees and never a dense M by M cost.
"""

import hashlib

import numpy as np
from scipy.spatial import cKDTree


def _integer(value, name, minimum=1, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) \
            or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer >= {minimum}" +
                         (f" and <= {maximum}" if maximum is not None else ""))
    return int(value)


def _paired(source, target_paired, minimum=2):
    source, target = np.asarray(source), np.asarray(target_paired)
    for points in (source, target):
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum \
                or not np.issubdtype(points.dtype, np.number) \
                or np.iscomplexobj(points) or not np.isfinite(points).all():
            raise ValueError(f"Expected finite real [M,2] points with M >= {minimum}")
    if source.shape != target.shape:
        raise ValueError("source and target_paired must have the same shape")
    source, target = source.astype(np.float64), target.astype(np.float64)
    if len(np.unique(source, axis=0)) != len(source):
        raise ValueError("Duplicate source coordinates have nonidentifiable nearest-bank ties")
    return source, target


def _statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "p10": None, "median": None,
                "p90": None, "p99": None, "max": None}
    return {"count": int(len(values)), "mean": float(values.mean()),
            "p10": float(np.quantile(values, .1)),
            "median": float(np.median(values)), "p90": float(np.quantile(values, .9)),
            "p99": float(np.quantile(values, .99)), "max": float(values.max())}


def _neighbors(points, count):
    # Query one extra candidate, then explicitly remove self. This also works
    # when interpolation brings several distinct particles to identical sites.
    _, candidates = cKDTree(points).query(points, k=count + 1, workers=1)
    candidates = np.asarray(candidates).reshape(len(points), count + 1)
    result = np.empty((len(points), count), dtype=np.int64)
    for point, row in enumerate(candidates):
        result[point] = row[row != point][:count]
    return result


def _edges(neighbors):
    first = np.repeat(np.arange(len(neighbors)), neighbors.shape[1])
    pairs = np.column_stack((first, neighbors.ravel()))
    pairs.sort(axis=1)
    return np.unique(pairs, axis=0).T


def _levels(source_levels, count):
    if not isinstance(source_levels, (list, tuple)) or not source_levels:
        raise ValueError("source_levels must be a nonempty list of labels")
    levels = []
    for labels in source_levels:
        labels = np.asarray(labels)
        if labels.shape != (count,) or not np.issubdtype(labels.dtype, np.integer) \
                or np.issubdtype(labels.dtype, np.bool_) or (labels < 0).any():
            raise ValueError("Each source level must contain M nonnegative integer labels")
        levels.append(labels.astype(np.int64))
    # A finer patch must have exactly one ancestor at every previous level.
    for coarser, finer in zip(levels, levels[1:]):
        for label in np.unique(finer):
            if len(np.unique(coarser[finer == label])) != 1:
                raise ValueError("source_levels must be nested from coarse to fine")
    return levels


def geometry_audit(source, target_paired, source_levels, *, neighbors=8,
                   times=(0., .25, .5, .75, .9, .95, .99), distance_bins=8):
    """Return summary, tabular rows and arrays for diagnostics 1 and 3 linkage.

    Source edges are the union of directed kNN edges, counted once each. Same
    and cross-patch comparisons use common source-distance quantile strata;
    strata lacking either group are reported and excluded from standardized
    averages. Matching in broad bins is approximate distance adjustment, not
    exact matching or a causal comparison.
    """
    source, target = _paired(source, target_paired)
    count = len(source)
    neighbors = _integer(neighbors, "neighbors", maximum=count - 1)
    distance_bins = _integer(distance_bins, "distance_bins")
    levels = _levels(source_levels, count)
    try:
        times = list(times)
    except TypeError as exc:
        raise ValueError("times must be a nonempty sequence in [0,1]") from exc
    if not times or any(isinstance(t, (bool, np.bool_))
                        or not isinstance(t, (int, float, np.integer, np.floating))
                        or not np.isfinite(t) or not 0 <= t <= 1
                        for t in times):
        raise ValueError("times must contain finite real values in [0,1]")
    times = [float(t) for t in times]
    if len(set(times)) != len(times):
        raise ValueError("times must not contain duplicates")
    velocity = target - source
    epsilon = 1e-12 * max(1., float(np.max(np.abs(source))),
                          float(np.max(np.abs(target))))
    directed = _neighbors(source, neighbors)
    edge_i, edge_j = _edges(directed)
    distance = np.linalg.norm(source[edge_i] - source[edge_j], axis=1)
    target_distance = np.linalg.norm(target[edge_i] - target[edge_j], axis=1)
    jump = np.linalg.norm(velocity[edge_i] - velocity[edge_j], axis=1)
    ratio = jump / (distance + epsilon)
    per_point_distance = np.linalg.norm(source[:, None] - source[directed], axis=2)
    per_point_jump = np.linalg.norm(velocity[:, None] - velocity[directed], axis=2)
    arrays = {"edge_i": edge_i, "edge_j": edge_j, "source_distance": distance,
              "target_distance": target_distance, "velocity_jump": jump,
              "velocity_ratio": ratio, "source_labels_finest": levels[-1],
              "point_boundary_fraction": (levels[-1][:, None] != levels[-1][directed]).mean(1),
              "point_velocity_jump_mean": per_point_jump.mean(1),
              "point_velocity_ratio_mean": (per_point_jump / (per_point_distance + epsilon)).mean(1)}
    # Repeated distance quantiles collapse, so tied distances stay together.
    cuts = np.unique(np.quantile(distance, np.linspace(0, 1, distance_bins + 1)))
    if len(cuts) == 1:
        cuts = np.array([cuts[0], cuts[0]])
    bin_ids = np.searchsorted(cuts[1:-1], distance, side="right")
    quantities = {"source_distance": distance, "target_distance": target_distance,
                  "velocity_jump": jump, "velocity_ratio": ratio}
    strata_rows, comparisons = [], []
    for level, labels in enumerate(levels):
        cross = labels[edge_i] != labels[edge_j]
        matched_rows, unmatched = [], []
        for group in range(len(cuts) - 1):
            within = bin_ids == group
            same, other = within & ~cross, within & cross
            row = {"level": level, "partition_count": int(len(np.unique(labels))),
                   "distance_bin": group, "distance_lower": float(cuts[group]),
                   "distance_upper": float(cuts[group + 1]),
                   "same_count": int(same.sum()), "cross_count": int(other.sum()),
                   "matched": bool(same.any() and other.any())}
            for name, values in quantities.items():
                row[f"same_{name}_mean"] = float(values[same].mean()) if same.any() else None
                row[f"cross_{name}_mean"] = float(values[other].mean()) if other.any() else None
            strata_rows.append(row)
            if row["matched"]:
                matched_rows.append(row)
            else:
                unmatched.append(group)
        matched_count = sum(r["same_count"] + r["cross_count"] for r in matched_rows)
        standardized = {}
        for name in quantities:
            values = {}
            for prefix in ("same", "cross"):
                values[prefix] = (sum((r["same_count"] + r["cross_count"]) *
                                     r[f"{prefix}_{name}_mean"] for r in matched_rows) / matched_count
                                  if matched_count else None)
            values["cross_minus_same"] = (values["cross"] - values["same"]
                                           if matched_count else None)
            standardized[name] = values
        comparisons.append({"level": level, "partition_count": int(len(np.unique(labels))),
                            "same_edges": int((~cross).sum()), "cross_edges": int(cross.sum()),
                            "matched_edge_count": int(matched_count),
                            "matched_edge_fraction": float(matched_count / len(distance)),
                            "unmatched_distance_bins": unmatched,
                            "distance_standardized_means": standardized})
    time_rows = []
    for index, time in enumerate(times):
        interpolated = source + time * velocity
        first, second = _edges(_neighbors(interpolated, neighbors))
        gap = np.linalg.norm(interpolated[first] - interpolated[second], axis=1)
        jump_t = np.linalg.norm(velocity[first] - velocity[second], axis=1)
        ratio_t = jump_t / (gap + epsilon)
        finest_cross = levels[-1][first] != levels[-1][second]
        row = {"time_index": index, "t": time, "edge_count": int(len(first)),
               "finest_cross_fraction": float(finest_cross.mean()),
               "zero_distance_edge_count": int((gap == 0).sum()),
               "positive_distance_edge_count": int((gap > 0).sum()),
               "velocity_ratio_scope": "strictly positive separation edges only"}
        for name, values in (("distance", gap), ("velocity_jump", jump_t),
                             ("velocity_ratio", ratio_t)):
            summarized = values[gap > 0] if name == "velocity_ratio" else values
            for statistic, value in _statistics(summarized).items():
                row[f"{name}_{statistic}"] = value
            arrays[f"time_{index}_{name}"] = values
        for statistic, value in _statistics(jump_t[gap == 0]).items():
            row[f"coincident_velocity_jump_{statistic}"] = value
        arrays[f"time_{index}_edge_i"] = first
        arrays[f"time_{index}_edge_j"] = second
        time_rows.append(row)
    summary = {"bank_size": count, "neighbors": neighbors, "times": times,
               "ratio_epsilon": epsilon, "requested_distance_bins": distance_bins,
               "effective_distance_bins": len(cuts) - 1,
               "source_edge_count": int(len(distance)),
               "source_edges": {name: _statistics(values) for name, values in quantities.items()},
               "partition_comparisons": comparisons,
               "definitions": {
                   "edge_set": "Union of directed kNN edges, unique undirected pairs; self excluded",
                   "velocity": "target_paired - source; full 2D Euclidean vector norm",
                   "velocity_ratio": "Euclidean velocity jump / (Euclidean separation + ratio_epsilon)",
                   "point_arrays": "Means and crossing fraction over each point's directed kNN",
                   "strata": "Shared source-distance quantile bins; both groups standardized to pooled matched-bin weights",
                   "time_conflict": "Recomputed same-time kNN on bank straight interpolants, not a conditional variance estimate",
                   "time_ratio_statistics": "Only strictly positive separation edges; coincident-edge velocity jumps reported separately",
                   "time_ratio_arrays": "Raw epsilon-regularized ratios include coincidences, whose values depend on epsilon and must not enter positive-distance ratio statistics",
                   "interpretation": "Descriptive coupling geometry; necessary target separation and sparse tails may also yield large jumps; not failure or causality proof",
                   "duplicate_policy": "Duplicate source coordinates rejected; interpolant coincidences allowed and counted"}}
    return summary, {"distance_strata": strata_rows, "time_conflicts": time_rows}, arrays


def gaussian_routing_audit(source, target_paired, *, draws=100000,
                           neighbors=(1, 8), seed=2026, chunk_size=10000):
    """Nearest-bank Gaussian probability allocation, diagnostic 2 only.

    q1 estimates Gaussian Voronoi-cell mass. For k>1 each query contributes 1/k
    to its k nearest sources, providing a sensitivity proxy. Neither predicts
    the transformer routing rule. Query identities are returned so downstream
    region Monte Carlo SE uses query-level contributions rather than treating
    the k neighbors of a query as independent observations.
    """
    source, _ = _paired(source, target_paired, minimum=1)
    draws = _integer(draws, "draws", minimum=2)
    chunk_size = _integer(chunk_size, "chunk_size")
    seed = _integer(seed, "seed", minimum=0, maximum=2**63 - 1)
    if not isinstance(neighbors, (list, tuple)) or not neighbors:
        raise ValueError("neighbors must be a nonempty list or tuple")
    counts = [_integer(k, "routing neighbors", maximum=len(source)) for k in neighbors]
    if len(set(counts)) != len(counts):
        raise ValueError("routing neighbors must be unique")
    max_count = max(counts)
    tree = cKDTree(source)
    rng = np.random.default_rng(seed)
    query_indices = np.empty((draws, max_count), dtype=np.int64)
    nearest_distance = np.empty(draws, dtype=np.float64)
    query_hasher = hashlib.sha256()
    for start in range(0, draws, chunk_size):
        stop = min(start + chunk_size, draws)
        query = rng.standard_normal((stop - start, 2))
        query_hasher.update(query.astype("<f8", copy=False).tobytes(order="C"))
        distances, indices = tree.query(query, k=max_count, workers=1)
        query_indices[start:stop] = np.asarray(indices).reshape(stop - start, max_count)
        nearest_distance[start:stop] = np.asarray(distances).reshape(stop - start, max_count)[:, 0]
    arrays = {"query_neighbor_indices": query_indices, "query_nearest_distance": nearest_distance}
    allocations = {}
    uniform = np.full(len(source), 1. / len(source))
    for k in counts:
        incidence = np.bincount(query_indices[:, :k].ravel(), minlength=len(source))
        weight = incidence.astype(np.float64) / (draws * k)
        arrays[f"count{k}"] = incidence
        arrays[f"q{k}"] = weight
        midpoint = .5 * (weight + uniform)
        positive = weight > 0
        js = .5 * float((weight[positive] * np.log(weight[positive] / midpoint[positive])).sum()
                         + (uniform * np.log(uniform / midpoint)).sum())
        allocations[str(k)] = {
            "weight_sum": float(weight.sum()), "incidence_sum": int(incidence.sum()),
            "unobserved_bank_points": int((incidence == 0).sum()),
            "effective_bank_points": float(1 / np.square(weight).sum()),
            "total_variation_vs_uniform": float(.5 * np.abs(weight - uniform).sum()),
            "js_nats_vs_uniform": js,
            "relative_weight_vs_uniform": _statistics(weight * len(source))}
    summary = {"bank_size": len(source), "draws": draws, "neighbors": counts,
               "seed": seed, "chunk_size": chunk_size,
               "gaussian_query_sha256": query_hasher.hexdigest(),
               "gaussian_query_hash_format": "Concatenated row-major little-endian float64 [draws,2] bytes",
               "nearest_source_distance": _statistics(nearest_distance),
               "allocations": allocations,
               "definitions": {
                   "source_law": "Independent standard Gaussian N(0,I_2)",
                   "allocation": "Each Gaussian query allocates 1/k probability to its k nearest bank sources",
                   "countk": "Neighbor incidence counts, sum = draws*k; qk=countk/(draws*k)",
                   "region_mc_se": "Compute each query's fraction of neighbors whose paired target lies in the region, then sample SD/sqrt(draws)",
                   "interpretation": "Nearest-bank extension sensitivity proxy; not the neural model's routing rule or a causal test",
                   "sampling_limit": "Coverage, effective count, TV and JS are finite-Monte-Carlo estimates; unobserved does not imply zero true mass",
                   "duplicate_policy": "Duplicate source coordinates rejected as nonidentifiable nearest-bank ties"}}
    return summary, arrays
