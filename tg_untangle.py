"""Offline within-patch permutation search using supervision conflict only.

The graph is an undirected source kNN graph, frozen before permutation search.
Cross-patch edges are retained even though moves only exchange two targets in
the same patch. This makes each accepted swap an exact descent step for one
fixed sparse objective, rather than changing the objective's neighbor graph.

For U = Y[permutation] - X and Z_t = X + t U, the objective is the edge/time
mean of [||U_i-U_j|| - L(t)||Z_t,i-Z_t,j||]_+**2. It measures individual
supervision conflicts with a Lipschitz budget, not the exact mean field's
Lipschitz constant or a guarantee of improved generation. No teacher,
Jacobian, OT cost, transport penalty, or coordinate perturbation is used.
"""

from collections.abc import Mapping

import numpy as np


DEFAULTS = {
    "neighbors": 8,
    "times": [0.05, 0.15, 0.3],
    "lipschitz": [1.0, 1.0, 1.0],
    "permutations": 4,
    "swap_steps": 256,
    "seed": 0,
}


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not minimum <= int(value) < 2**63):
        raise ValueError(f"{name} must be an integer >= {minimum} and < 2**63")
    return int(value)


def _real(value, name, minimum=0.0):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number >= {minimum}") from error
    if not np.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return result


def normalize_options(value):
    """Canonical, JSON-serializable options; missing/False means disabled.

    True or a mapping enables search unless the mapping says enabled: false.
    A scalar lipschitz budget is broadcast to all times. Times must be unique
    and strictly interior: at either endpoint permutations cannot usefully
    change endpoint geometry. Configured neighbor count is checked against N
    when the graph/search is constructed, rather than silently clipped.
    """
    if value is None or value is False:
        supplied = {"enabled": False}
    elif value is True:
        supplied = {"enabled": True}
    elif isinstance(value, Mapping):
        supplied = dict(value)
    else:
        raise ValueError("tg_cache.untangle must be null, boolean, or a mapping")
    unknown = set(supplied) - ({"enabled"} | set(DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown untangle options: {', '.join(sorted(map(str, unknown)))}")
    enabled = supplied.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("untangle.enabled must be a boolean")
    opts = {**DEFAULTS, **supplied, "enabled": enabled}
    opts["neighbors"] = _integer(opts["neighbors"], "untangle.neighbors", 1)
    opts["permutations"] = _integer(opts["permutations"], "untangle.permutations", 1)
    opts["swap_steps"] = _integer(opts["swap_steps"], "untangle.swap_steps")
    opts["seed"] = _integer(opts["seed"], "untangle.seed")
    raw_times = opts["times"]
    if not isinstance(raw_times, (list, tuple, np.ndarray)) or len(raw_times) == 0:
        raise ValueError("untangle.times must be a nonempty sequence")
    times = [_real(item, "untangle.times") for item in raw_times]
    if any(not 0.0 < item < 1.0 for item in times) or len(set(times)) != len(times):
        raise ValueError("untangle.times must be distinct and strictly between 0 and 1")
    raw_lipschitz = opts["lipschitz"]
    # If only times are changed, retain the default constant budget rather than
    # require users to duplicate its value once per newly configured time.
    if "lipschitz" not in supplied:
        raw_lipschitz = 1.0
    if isinstance(raw_lipschitz, (list, tuple, np.ndarray)):
        if len(raw_lipschitz) != len(times):
            raise ValueError("untangle.lipschitz must have one value per time")
        lipschitz = [_real(item, "untangle.lipschitz") for item in raw_lipschitz]
    else:
        lipschitz = [_real(raw_lipschitz, "untangle.lipschitz")] * len(times)
    opts["times"], opts["lipschitz"] = times, lipschitz
    return opts


def _points(value, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind == "c":
            raise ValueError("Complex coordinates are not supported")
        result = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite N x D array") from error
    if (result.ndim != 2 or min(result.shape) < 1
            or not np.isfinite(result).all()):
        raise ValueError(f"{name} must be a finite nonempty N x D array")
    return result


def _labels(value, name, n, k):
    labels = np.asarray(value)
    if (labels.shape != (n,) or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels >= k)):
        raise ValueError(f"{name} must be an integer length-N array in [0, K)")
    return labels.astype(np.int64, copy=False)


def _patch_groups(source_labels, target_labels, k):
    k = _integer(k, "k", 1)
    raw_source, raw_target = np.asarray(source_labels), np.asarray(target_labels)
    if raw_source.ndim != 1 or len(raw_source) < 1 or k > len(raw_source):
        raise ValueError("Patch labels require a nonempty cloud and 1 <= K <= N")
    n = len(raw_source)
    source_labels = _labels(raw_source, "source_labels", n, k)
    target_labels = _labels(raw_target, "target_labels", n, k)
    source_groups = [np.flatnonzero(source_labels == patch) for patch in range(k)]
    target_groups = [np.flatnonzero(target_labels == patch) for patch in range(k)]
    if any(len(a) != len(b) for a, b in zip(source_groups, target_groups)):
        raise ValueError("Source and target patch counts differ")
    return source_groups, target_groups


def random_valid_permutation(source_labels, target_labels, k, rng):
    """Uniform patchwise bijection without touching NumPy's global RNG."""
    source_groups, target_groups = _patch_groups(source_labels, target_labels, k)
    permutation = np.empty(len(source_labels), dtype=np.int32)
    for source, target in zip(source_groups, target_groups):
        permutation[source] = rng.permutation(target)
    return permutation


def build_source_edges(source, neighbors):
    """Frozen undirected kNN union, unique lexicographic int64 edges.

    All source points participate regardless of patch membership. Ties use
    source index order, making graph construction deterministic.
    """
    source = _points(source, "source")
    neighbors = _integer(neighbors, "neighbors", 1)
    n = len(source)
    if neighbors >= n:
        raise ValueError("neighbors must be smaller than the cloud's point count")
    differences = source[:, None, :] - source[None, :, :]
    with np.errstate(over="ignore", invalid="ignore"):
        distance_squared = np.einsum("ijd,ijd->ij", differences, differences)
    if not np.isfinite(distance_squared).all():
        raise ValueError("Source coordinates exceed the supported numerical range")
    np.fill_diagonal(distance_squared, np.inf)
    nearest = np.argsort(distance_squared, axis=1, kind="stable")[:, :neighbors]
    edges = np.stack((np.repeat(np.arange(n), neighbors), nearest.reshape(-1)), axis=1)
    edges.sort(axis=1)
    return np.unique(edges, axis=0).astype(np.int64, copy=False)


def _permutation(value, n):
    permutation = np.asarray(value)
    if (permutation.shape != (n,) or permutation.dtype.kind not in "iu"
            or np.any(permutation < 0) or np.any(permutation >= n)
            or len(np.unique(permutation)) != n):
        raise ValueError("permutation must be a bijection of indices 0 through N-1")
    return permutation.astype(np.int64, copy=False)


def _edges(value, n):
    edges = np.asarray(value)
    if (edges.ndim != 2 or edges.shape[1] != 2 or edges.dtype.kind not in "iu"
            or np.any(edges < 0) or np.any(edges >= n)
            or np.any(edges[:, 0] == edges[:, 1])):
        raise ValueError("edges must be an integer E x 2 array of distinct valid endpoints")
    if len(np.unique(np.sort(edges, axis=1), axis=0)) != len(edges):
        raise ValueError("edges must not contain duplicate undirected edges")
    return edges.astype(np.int64, copy=False)


def _edge_scores(source_difference, target_difference, times, lipschitz):
    velocity_difference = target_difference - source_difference
    with np.errstate(over="ignore", invalid="ignore"):
        velocity_norm = np.linalg.norm(velocity_difference, axis=1)
        interpolated = source_difference[None, :, :] + times[:, None, None] * velocity_difference[None, :, :]
        position_norm = np.linalg.norm(interpolated, axis=2)
        excess = np.maximum(velocity_norm[None, :] - lipschitz[:, None] * position_norm, 0.0)
        scores = np.mean(excess * excess, axis=0)
    if not np.isfinite(scores).all():
        raise ValueError("Conflict scores exceed the supported numerical range")
    return scores


def conflict_score(source, target, permutation, edges, options):
    """Validated edge/time mean score; never changes the supplied arrays."""
    opts = normalize_options(options)
    source, target = _points(source, "source"), _points(target, "target")
    if source.shape != target.shape:
        raise ValueError("source and target must have the same N x D shape")
    permutation, edges = _permutation(permutation, len(source)), _edges(edges, len(source))
    if len(edges) == 0:
        return 0.0
    left, right = edges.T
    scores = _edge_scores(source[left] - source[right],
                          target[permutation[left]] - target[permutation[right]],
                          np.asarray(opts["times"]), np.asarray(opts["lipschitz"]))
    return float(np.mean(scores))


def optimize_permutations(source, target, source_labels, target_labels, k, options, seed):
    """Greedy random same-patch swaps; return int32[P,N] pool and diagnostics.

    Every pool entry starts from an independently sampled uniform valid
    permutation. Only edges incident to either swapped source point change,
    so exact affected-edge deltas are inexpensive. Strictly improving moves
    are accepted; this is a bounded local search, not a global optimum.
    """
    opts = normalize_options(options)
    if not opts["enabled"]:
        raise ValueError("Permutation optimization requires untangle.enabled: true")
    seed = _integer(seed, "seed")
    source, target = _points(source, "source"), _points(target, "target")
    if source.shape != target.shape:
        raise ValueError("source and target must have the same N x D shape")
    n = len(source)
    if n >= 2**31:
        raise ValueError("Permutation pools require N < 2**31")
    source_groups, target_groups = _patch_groups(source_labels, target_labels, k)
    if len(source_labels) != n:
        raise ValueError("Patch labels must have the cloud's point count")
    edges = build_source_edges(source, opts["neighbors"])
    left, right = edges.T
    source_difference = source[left] - source[right]
    incident = [np.flatnonzero((left == point) | (right == point)) for point in range(n)]
    movable = np.concatenate([group for group in source_groups if len(group) >= 2]) if any(
        len(group) >= 2 for group in source_groups) else np.empty(0, dtype=np.int64)
    group_for_point = np.empty(n, dtype=np.int64)
    position_in_group = np.empty(n, dtype=np.int64)
    for patch, group in enumerate(source_groups):
        group_for_point[group] = patch
        position_in_group[group] = np.arange(len(group))
    times, lipschitz = np.asarray(opts["times"]), np.asarray(opts["lipschitz"])
    pool = np.empty((opts["permutations"], n), dtype=np.int32)
    records = []
    rng_seeds = np.random.SeedSequence([opts["seed"], seed]).spawn(opts["permutations"])
    for pool_index, rng_seed in enumerate(rng_seeds):
        rng = np.random.default_rng(rng_seed)
        permutation = np.empty(n, dtype=np.int32)
        for source_group, target_group in zip(source_groups, target_groups):
            permutation[source_group] = rng.permutation(target_group)
        paired = target[permutation].copy()
        edge_scores = _edge_scores(source_difference, paired[left] - paired[right], times, lipschitz)
        initial_score = float(edge_scores.mean())
        proposed = accepted = 0
        for _ in range(opts["swap_steps"] if len(movable) else 0):
            a = int(movable[rng.integers(len(movable))])
            group = source_groups[group_for_point[a]]
            partner = int(rng.integers(len(group) - 1))
            if partner >= position_in_group[a]:
                partner += 1
            b = int(group[partner])
            affected = np.union1d(incident[a], incident[b])
            previous_sum = float(edge_scores[affected].sum())
            paired[[a, b]] = paired[[b, a]]
            replacement = _edge_scores(source_difference[affected],
                                       paired[left[affected]] - paired[right[affected]], times, lipschitz)
            replacement_sum = float(replacement.sum())
            tolerance = 1e-12 * max(abs(previous_sum), abs(replacement_sum), 1e-12)
            proposed += 1
            if replacement_sum < previous_sum - tolerance:
                permutation[[a, b]] = permutation[[b, a]]
                edge_scores[affected] = replacement
                accepted += 1
            else:
                paired[[a, b]] = paired[[b, a]]
        final_score = float(_edge_scores(source_difference, paired[left] - paired[right],
                                         times, lipschitz).mean())
        if final_score > initial_score + 1e-12 * max(abs(initial_score), 1e-12):
            raise RuntimeError("Conflict optimization increased its fixed objective")
        pool[pool_index] = permutation
        records.append({"initial_score": initial_score, "final_score": final_score,
                        "proposed_swaps": proposed, "accepted_swaps": accepted})
    stats = {
        "objective": "mean_edge_time_squared_positive_supervision_lipschitz_excess",
        "graph": "fixed_undirected_source_knn_union_including_cross_patch_edges",
        "edge_count": int(len(edges)),
        "cross_patch_edge_count": int(np.count_nonzero(np.asarray(source_labels)[left] != np.asarray(source_labels)[right])),
        "options": opts,
        "cloud_seed": seed,
        "per_permutation": records,
        "initial_score_mean": float(np.mean([record["initial_score"] for record in records])),
        "final_score_mean": float(np.mean([record["final_score"] for record in records])),
        "proposed_swaps": int(sum(record["proposed_swaps"] for record in records)),
        "accepted_swaps": int(sum(record["accepted_swaps"] for record in records)),
        "search": "bounded_greedy_same_patch_swaps; no_global_optimality_guarantee",
    }
    return pool, stats
