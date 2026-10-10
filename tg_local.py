"""Offline coupling search using actual paths and held-out local affine fits.

Original coordinates are never moved. Returned tables contain balanced fine
group labels, NOT point permutations: training must sample a fresh uniform
bijection inside those groups on every visit. Search permutations are temporary
Monte Carlo samples, and are never returned or cached for training.

The score measures leave-center-out prediction of velocity from nearby path
positions. Affine contraction, expansion and rotation can be explained by the
fit; no Jacobian norm or common-velocity penalty is used. Target adjacency and
local distance gates are empirical safeguards, not topology certificates. This
small-cloud implementation uses quadratic distance matrices (e.g. N=256).
"""

import numbers

import numpy as np


DEFAULTS = {
    "subpatches": 1,
    "num_tables": 3,
    "score_permutations": 2,
    "candidate_count": 4,
    "max_swaps": 4,
    "times": [0.0, 0.5, 0.75, 0.9],
    "neighbors": 12,
    "ridge": 1e-3,
    "max_cost_increase": 0.02,
    "target_neighbors": 2,
    "max_target_distance_factor": 2.0,
    "destination_budget": 0.35,
}


def settings(value=None):
    """Normalize a strict, JSON-safe schema; only one/two children are supported."""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("tg local options must be a mapping")
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown tg local options: {sorted(unknown)}")
    result = {**DEFAULTS, **value}
    for name, minimum in (("subpatches", 1), ("num_tables", 2),
                          ("score_permutations", 1), ("candidate_count", 1),
                          ("max_swaps", 0), ("neighbors", 4), ("target_neighbors", 1)):
        item = result[name]
        if isinstance(item, bool) or not isinstance(item, numbers.Integral) or item < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        result[name] = int(item)
    if result["subpatches"] not in (1, 2):
        raise ValueError("subpatches must be 1 or 2")
    for name in ("ridge", "max_cost_increase", "max_target_distance_factor", "destination_budget"):
        item = result[name]
        if isinstance(item, bool) or not isinstance(item, numbers.Real) or not np.isfinite(item):
            raise ValueError(f"{name} must be a finite number")
        positive = name in {"ridge", "max_target_distance_factor"}
        if item < 0 or (positive and item == 0):
            raise ValueError(f"{name} must be {'> 0' if positive else '>= 0'}")
        result[name] = float(item)
    times = result["times"]
    if not isinstance(times, (list, tuple)) or not times:
        raise ValueError("times must be a nonempty list")
    if any(isinstance(item, bool) or not isinstance(item, numbers.Real)
           or not np.isfinite(item) or not 0 <= item < 1 for item in times):
        raise ValueError("times must contain finite numbers in [0, 1)")
    result["times"] = [float(item) for item in times]
    if result["times"] != sorted(set(result["times"])):
        raise ValueError("times must be strictly increasing and unique")
    return result


def _validate_inputs(source, target, source_labels, target_labels, k, seed):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (2,) or target.shape != source.shape or not len(source):
        raise ValueError("local clouds must have matching nonempty [N, 2] shapes")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("local clouds must contain only finite coordinates")
    if isinstance(k, bool) or not isinstance(k, numbers.Integral) or not 1 <= k <= len(source):
        raise ValueError("k must be an integer between 1 and N")
    labels = []
    for name, raw in (("source_labels", source_labels), ("target_labels", target_labels)):
        raw = np.asarray(raw)
        if raw.shape != (len(source),) or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"{name} must be an integer [N] array")
        if np.any(raw < 0) or np.any(raw >= k):
            raise ValueError(f"{name} contains an out-of-range patch ID")
        labels.append(raw.astype(np.int32, copy=True))
    counts = [np.bincount(item, minlength=k) for item in labels]
    if not np.array_equal(*counts) or np.any(counts[0] == 0):
        raise ValueError("source and target must have identical positive capacities for every patch")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("local seed must be a nonnegative integer")
    return source, target, labels[0], labels[1], int(k), int(seed)


def _squared_distances(points):
    return np.sum((points[:, None] - points[None, :]) ** 2, axis=-1)


def _canonical_axis(points):
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered
    eigenvalues, vectors = np.linalg.eigh(covariance)
    # Isotropic/constant geometry has no preferred PCA direction; use x.
    if eigenvalues[-1] - eigenvalues[0] <= 1e-12 * max(1.0, eigenvalues[-1]):
        return np.array([1.0, 0.0])
    axis = vectors[:, -1]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    return axis


def target_children(target, target_labels, k, subpatches):
    """Deterministic PCA rank split of each parent; unequal halves are allowed."""
    if subpatches == 1:
        return np.asarray(target_labels, dtype=np.int32).copy()
    result = np.empty(len(target), dtype=np.int32)
    for patch in range(k):
        indices = np.flatnonzero(target_labels == patch)
        if len(indices) < subpatches:
            raise ValueError("Every target patch needs at least subpatches points")
        order = np.argsort(target[indices] @ _canonical_axis(target[indices]), kind="stable")
        for child, part in enumerate(np.array_split(indices[order], subpatches)):
            result[part] = patch * subpatches + child
    return result


def source_children(source, coarse_labels, target, fine_target_labels, k, subpatches):
    """Offline balanced rank allocation along the two child-centroid direction.

    This matches target child capacities and uses no pointwise OT. It is cached
    per coarse table; the subsequent within-child bijection is always random.
    """
    if subpatches == 1:
        return np.asarray(coarse_labels, dtype=np.int32).copy()
    result = np.empty(len(source), dtype=np.int32)
    for patch in range(k):
        indices = np.flatnonzero(coarse_labels == patch)
        left = target[fine_target_labels == patch * subpatches]
        right = target[fine_target_labels == patch * subpatches + 1]
        axis = right.mean(axis=0) - left.mean(axis=0)
        if np.linalg.norm(axis) <= 1e-12:
            axis = _canonical_axis(np.concatenate((left, right)))
        order = indices[np.argsort(source[indices] @ axis, kind="stable")]
        result[order[:len(left)]] = patch * subpatches
        result[order[len(left):]] = patch * subpatches + 1
    return result


def _permutations(source_labels, target_labels, groups, count, seed):
    """Temporary common-random-number samples for scoring, never cached."""
    rng = np.random.default_rng(seed)
    result = np.empty((count, len(source_labels)), dtype=np.int64)
    for group in range(groups):
        source = np.flatnonzero(source_labels == group)
        target = np.flatnonzero(target_labels == group)
        if len(source) != len(target):
            raise ValueError("Fine source/target capacities differ")
        for row in range(count):
            result[row, source] = rng.permutation(target)
    return result


def _transport_permutations(permutations, old_labels, new_labels, groups, seed):
    """Couple score samples while retaining every unchanged group's point pair.

    For each group, only targets released by members leaving that group are
    uniformly reassigned to incoming members. For fixed old/new label tables,
    a uniform old bijection maps to a uniform new bijection. This reduces MC
    noise from reshuffling unaffected points; it is not a training pairing.
    """
    rng = np.random.default_rng(seed)
    moved = np.asarray(old_labels) != np.asarray(new_labels)
    result = np.asarray(permutations).copy()
    for group in range(groups):
        leaving = np.flatnonzero(moved & (old_labels == group))
        entering = np.flatnonzero(moved & (new_labels == group))
        if len(leaving) != len(entering):
            raise ValueError("Transported fine tables must preserve group capacities")
        for row in range(len(result)):
            result[row, entering] = rng.permutation(permutations[row, leaving])
    return result


def _target_geometry(target, target_labels, centroids, opt):
    k, n = len(centroids), len(target)
    patch_rms = np.array([np.sqrt(np.mean(np.sum((target[target_labels == a] - centroids[a]) ** 2,
                                                axis=1))) for a in range(k)])
    distances = np.sqrt(_squared_distances(centroids))
    nearest = distances.copy()
    np.fill_diagonal(nearest, np.inf)
    adjacency = np.zeros((k, k), dtype=bool)
    if k > 1:
        neighbors = np.argsort(nearest, axis=1, kind="stable")[:, :min(opt["target_neighbors"], k - 1)]
        adjacency[np.arange(k)[:, None], neighbors] = True
    adjacency &= adjacency.T
    adjacency &= distances <= opt["max_target_distance_factor"] * (patch_rms[:, None] + patch_rms[None, :]) + 1e-12
    np.fill_diagonal(adjacency, True)
    target_squared = _squared_distances(target)
    np.fill_diagonal(target_squared, np.inf)
    width = min(opt["neighbors"], n - 1)
    if width:
        local_radius_squared = np.sort(target_squared, axis=1)[:, width - 1]
        radius = np.maximum(local_radius_squared[:, None], local_radius_squared[None, :])
        local_gate = target_squared <= opt["max_target_distance_factor"] ** 2 * radius + 1e-12
    else:
        local_gate = np.zeros((n, n), dtype=bool)
    local_gate &= adjacency[target_labels[:, None], target_labels[None, :]]
    np.fill_diagonal(local_gate, False)
    return adjacency, local_gate


def local_affine_score(positions, velocities, neighbors=12, ridge=1e-3,
                       allowed=None, normalization=1.0):
    """Leave-center-out weighted affine residual, with stabilized 3x3 fits.

    Each center's velocity is held out. At least three valid neighbors are
    needed to fit a 2D affine field; centers with fewer are explicitly omitted.
    Position offsets are normalized by their local radius. A very small ridge
    applies to the slopes only, so common translation is fitted without bias.
    """
    positions, velocities = np.asarray(positions, dtype=np.float64), np.asarray(velocities, dtype=np.float64)
    n = len(positions)
    if n < 4:
        return 0.0, 0.0
    width = min(neighbors, n - 1)
    squared = _squared_distances(positions)
    np.fill_diagonal(squared, np.inf)
    if allowed is not None:
        squared = np.where(allowed, squared, np.inf)
    indices = np.argsort(squared, axis=1, kind="stable")[:, :width]
    selected_distances = np.take_along_axis(squared, indices, axis=1)
    valid = np.isfinite(selected_distances)
    counts = valid.sum(axis=1)
    scored = counts >= 3
    if not np.any(scored):
        return 0.0, 0.0
    radius_squared = np.max(np.where(valid, selected_distances, 0.0), axis=1)
    radius_squared = np.maximum(radius_squared, np.finfo(np.float64).eps)
    weights = np.where(valid, 1 / (1 + np.where(valid, selected_distances, 0) / radius_squared[:, None]), 0.0)
    offsets = (positions[indices] - positions[:, None]) / np.sqrt(radius_squared)[:, None, None]
    features = np.concatenate((np.ones((n, width, 1)), offsets), axis=2)
    # Responses are centered on the neighbor average to avoid precision loss
    # for a large common velocity; the held-out velocity is not used in fitting.
    total_weight = np.maximum(weights.sum(axis=1), np.finfo(np.float64).eps)
    mean_velocity = np.sum(weights[:, :, None] * velocities[indices], axis=1) / total_weight[:, None]
    response = velocities[indices] - mean_velocity[:, None]
    gram = np.einsum("nki,nk,nkj->nij", features, weights, features)
    rhs = np.einsum("nki,nk,nkj->nij", features, weights, response)
    penalty = np.diag([1e-12, ridge, ridge])[None] * total_weight[:, None, None]
    coefficients = np.linalg.solve(gram + penalty, rhs)
    prediction = coefficients[:, 0] + mean_velocity
    residual = np.sum((prediction[scored] - velocities[scored]) ** 2, axis=1)
    return float(residual.mean() / max(float(normalization), np.finfo(np.float64).eps)), float(scored.mean())


def path_score(source, target, permutations, options=None, *, target_gate=None):
    """Score each actual fine bijection at each time, THEN average its scores."""
    opt = settings(options)
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    normalization = float(np.mean(np.sum((target - target.mean(axis=0)) ** 2, axis=1)))
    values, coverage = [], []
    for permutation in np.asarray(permutations):
        paired = target[permutation]
        velocity = paired - source
        allowed = None if target_gate is None else target_gate[np.ix_(permutation, permutation)]
        row, row_coverage = [], []
        for time in opt["times"]:
            score, fraction = local_affine_score(
                source + time * velocity, velocity, opt["neighbors"], opt["ridge"],
                allowed, normalization)
            row.append(score)
            row_coverage.append(fraction)
        values.append(row)
        coverage.append(row_coverage)
    values, coverage = np.asarray(values), np.asarray(coverage)
    return float(values.mean()), values.mean(axis=0).tolist(), coverage.mean(axis=0).tolist()


def _expected_fine_cost(source, target, source_labels, target_labels, groups):
    """Exact expected squared cost under uniform random within-group pairing."""
    total = 0.0
    for group in range(groups):
        left, right = source[source_labels == group], target[target_labels == group]
        center = right.mean(axis=0)
        total += np.sum((left - center) ** 2) + len(left) * np.mean(np.sum((right - center) ** 2, axis=1))
    return float(total / len(source))


def build_tables(source, target, source_labels, target_labels, k, options=None, seed=0):
    """Return balanced fine-label tables, target fine labels, and honest audits.

    Each row starts from the original coarse TG assignment. Row zero is never
    coarse-swapped. Child allocation is deterministic for each row. Remaining
    rows use bounded, disjoint coarse swaps selected by actual-path MC scores.
    Common actual bijections are transported across each row's candidate swaps;
    independent seeds produce the held-out diagnostics after all selection.
    Neither the search sample nor the held-out sample is a training pairing.
    """
    opt = settings(options)
    source, target, original, target_labels, k, seed = _validate_inputs(
        source, target, source_labels, target_labels, k, seed)
    rng = np.random.default_rng(seed)
    n, r, subpatches = len(source), opt["num_tables"], opt["subpatches"]
    groups = k * subpatches
    fine_target = target_children(target, target_labels, k, subpatches)
    centroids = np.stack([target[target_labels == a].mean(axis=0) for a in range(k)])
    target_rms_squared = float(np.mean(np.sum((target - target.mean(axis=0)) ** 2, axis=1)))
    numerical = 1e-12 * max(1.0, target_rms_squared)
    base_cost = float(np.mean(np.sum((source - centroids[original]) ** 2, axis=1)))
    adjacency, target_gate = _target_geometry(target, target_labels, centroids, opt)
    source_squared = _squared_distances(source)
    np.fill_diagonal(source_squared, np.inf)
    nn = np.argsort(source_squared, axis=1, kind="stable")[:, :min(opt["neighbors"], n - 1)]
    edges = np.unique(np.sort(np.column_stack((np.repeat(np.arange(n), nn.shape[1]), nn.ravel())), axis=1), axis=0)
    cross = original[edges[:, 0]] != original[edges[:, 1]]
    allowed = cross & adjacency[original[edges[:, 0]], original[edges[:, 1]]]
    candidates = edges[allowed]
    i, j = candidates.T
    shifts = centroids[original[j]] - centroids[original[i]]
    shift_squared = np.sum(shifts ** 2, axis=1)
    cost_delta = (np.sum((source[i] - centroids[original[j]]) ** 2, axis=1)
                  + np.sum((source[j] - centroids[original[i]]) ** 2, axis=1)
                  - np.sum((source[i] - centroids[original[i]]) ** 2, axis=1)
                  - np.sum((source[j] - centroids[original[j]]) ** 2, axis=1)) / n
    coarse = np.repeat(original[None], r, axis=0)
    tables = np.repeat(source_children(source, original, target, fine_target, k, subpatches)[None], r, axis=0)
    baseline_fine_cost = _expected_fine_cost(source, target, tables[0], fine_target, groups)
    fine_cost_limit = baseline_fine_cost * (1 + opt["max_cost_increase"])
    swaps, traces = np.zeros(r, dtype=np.int32), []
    costs = np.full(r, base_cost)
    drift_sum = np.zeros(n)
    score_seeds = rng.integers(0, 2**63 - 1, size=r, dtype=np.int64)
    heldout_seeds = rng.integers(0, 2**63 - 1, size=r, dtype=np.int64)
    heldout_transport_seeds = rng.integers(0, 2**63 - 1, size=r, dtype=np.int64)
    baseline_scores, scores, baseline_heldout, heldout = [], [], [], []
    score_by_time, heldout_by_time, coverage, baseline_coverage = [], [], [], []
    baseline_score_by_time, baseline_heldout_by_time = [], []
    candidate_evaluations = 0
    reject_counts = {"coverage_decrease": 0, "non_improving": 0,
                     "expected_fine_cost_budget": 0, "rows_without_accepted_swaps": 0}

    def evaluate(permutations):
        return path_score(source, target, permutations, opt, target_gate=target_gate)

    for row in range(r):
        current_permutations = _permutations(tables[row], fine_target, groups,
                                            opt["score_permutations"], int(score_seeds[row]))
        current_result = evaluate(current_permutations)
        current_score, _, current_coverage = current_result
        baseline_scores.append(current_score)
        baseline_score_by_time.append(current_result[1])
        baseline_coverage.append(current_coverage)
        trace = [current_score]
        used = np.zeros(n, dtype=bool)
        if row:
            for _ in range(min(opt["max_swaps"], n // 2)):
                feasible = ~(used[i] | used[j])
                feasible &= costs[row] + cost_delta <= base_cost * (1 + opt["max_cost_increase"]) + numerical
                feasible &= (drift_sum[i] + shift_squared) / r <= opt["destination_budget"] ** 2 * target_rms_squared + numerical
                feasible &= (drift_sum[j] + shift_squared) / r <= opt["destination_budget"] ** 2 * target_rms_squared + numerical
                pool = np.flatnonzero(feasible)
                if not len(pool):
                    break
                pool = rng.choice(pool, min(opt["candidate_count"], len(pool)), replace=False)
                selected, selected_score, selected_labels, selected_permutations = None, current_score, None, None
                selected_result = None
                for index in pool:
                    a, b = candidates[index]
                    candidate_coarse = coarse[row].copy()
                    candidate_coarse[a], candidate_coarse[b] = candidate_coarse[b], candidate_coarse[a]
                    candidate_labels = source_children(source, candidate_coarse, target, fine_target, k, subpatches)
                    candidate_fine_cost = _expected_fine_cost(source, target, candidate_labels, fine_target, groups)
                    if candidate_fine_cost > fine_cost_limit + numerical:
                        reject_counts["expected_fine_cost_budget"] += 1
                        continue
                    candidate_permutations = _transport_permutations(
                        current_permutations, tables[row], candidate_labels, groups,
                        int(rng.integers(0, 2**63 - 1)))
                    candidate_result = evaluate(candidate_permutations)
                    score, _, candidate_coverage = candidate_result
                    candidate_evaluations += 1
                    # With the fixed target gate and a complete bijection,
                    # coverage is invariant under permutations. Keep this
                    # guard explicit if neighborhood rules change later.
                    if np.any(np.asarray(candidate_coverage) < np.asarray(current_coverage) - 1e-12):
                        reject_counts["coverage_decrease"] += 1
                        continue
                    if score < selected_score - 1e-10 * max(1.0, current_score):
                        selected, selected_score, selected_labels = int(index), score, candidate_labels
                        selected_permutations = candidate_permutations
                        selected_result = candidate_result
                    else:
                        reject_counts["non_improving"] += 1
                if selected is None:
                    # This is a bounded stochastic candidate search, not a
                    # claim that every feasible swap has been examined.
                    break
                a, b = candidates[selected]
                coarse[row, a], coarse[row, b] = coarse[row, b], coarse[row, a]
                tables[row] = selected_labels
                current_permutations = selected_permutations
                current_result = selected_result
                used[[a, b]] = True
                drift_sum[[a, b]] += shift_squared[selected]
                costs[row] += cost_delta[selected]
                swaps[row] += 1
                current_score = selected_score
                trace.append(current_score)
            if swaps[row] == 0:
                reject_counts["rows_without_accepted_swaps"] += 1
        traces.append(trace)
        score, by_time, fraction = current_result
        scores.append(score)
        score_by_time.append(by_time)
        coverage.append(fraction)
        heldout_permutations = _permutations(tables[0], fine_target, groups,
                                            opt["score_permutations"], int(heldout_seeds[row]))
        baseline_heldout_result = evaluate(heldout_permutations)
        baseline_heldout.append(baseline_heldout_result[0])
        baseline_heldout_by_time.append(baseline_heldout_result[1])
        final_heldout_permutations = _transport_permutations(
            heldout_permutations, tables[0], tables[row], groups,
            int(heldout_transport_seeds[row]))
        hold_score, hold_time, _ = (baseline_heldout_result if np.array_equal(tables[0], tables[row])
                                  else evaluate(final_heldout_permutations))
        heldout.append(hold_score)
        heldout_by_time.append(hold_time)
    actual_costs = np.mean(np.sum((source[None] - centroids[coarse]) ** 2, axis=2), axis=1)
    destination_rms = np.sqrt(np.mean(np.sum((centroids[coarse] - centroids[original][None]) ** 2, axis=2), axis=0))
    expected_costs = [_expected_fine_cost(source, target, table, fine_target, groups) for table in tables]
    original_fine_cost = _expected_fine_cost(source, target, original, target_labels, k)
    stats = {
        "algorithm": "actual_path_leave_center_out_affine_v1",
        "settings": opt, "seed": seed,
        "fine_groups": groups,
        "row_zero": "unchanged original coarse assignment" if subpatches == 1 else "unchanged original coarse assignment with deterministic balanced child allocation",
        "subpatch_assignment": "none" if subpatches == 1 else "target PCA rank halves; source balanced rank along target child-centroid axis",
        "baseline_score": float(np.mean(baseline_scores)),
        "guided_score": float(np.mean(scores)),
        "heldout_baseline_score": float(np.mean(baseline_heldout)),
        "heldout_guided_score": float(np.mean(heldout)),
        "baseline_scores_per_table": baseline_scores,
        "scores_per_table": scores,
        "heldout_baseline_scores_per_table": baseline_heldout,
        "heldout_scores_per_table": heldout,
        "score_trace_per_table": traces,
        "score_by_time": np.mean(score_by_time, axis=0).tolist(),
        "baseline_score_by_time": np.mean(baseline_score_by_time, axis=0).tolist(),
        "heldout_score_by_time": np.mean(heldout_by_time, axis=0).tolist(),
        "heldout_baseline_score_by_time": np.mean(baseline_heldout_by_time, axis=0).tolist(),
        "scored_fraction_by_time": np.mean(coverage, axis=0).tolist(),
        "baseline_scored_fraction_by_time": np.mean(baseline_coverage, axis=0).tolist(),
        "min_scored_fraction": float(np.min(coverage)),
        "score_sampling": "common samples transported across label swaps, retaining unaffected pairs; independent held-out samples transported to the final table; samples never used during training",
        "baseline_centroid_cost": base_cost,
        "baseline_cost": base_cost,
        "guided_cost": float(actual_costs.mean()),
        "centroid_cost_per_table": actual_costs.tolist(),
        "original_tg_expected_fine_cost": original_fine_cost,
        "baseline_expected_fine_cost": expected_costs[0],
        "expected_fine_cost_limit": fine_cost_limit,
        "expected_fine_cost_per_table": expected_costs,
        "swaps_per_table": swaps.tolist(),
        "accepted_swaps": int(swaps.sum()),
        "coarse_changed_points_per_table": np.count_nonzero(coarse != original[None], axis=1).tolist(),
        "candidate_evaluations": candidate_evaluations,
        "eligible_source_edges": len(candidates),
        "allowed_target_patch_pairs": np.argwhere(np.triu(adjacency, k=1)).tolist(),
        "max_destination_rms": float(destination_rms.max()),
        "destination_rms_limit": float(opt["destination_budget"] * np.sqrt(target_rms_squared)),
        "no_feasible_change": bool(swaps.sum() == 0),
        "reject_counts": reject_counts,
        "guard_notes": "Target adjacency/local distance, centroid and expected fine-cost budgets, nondecreasing score coverage and ensemble centroid-drift limits are empirical safeguards; no topology, neural Jacobian or endpoint-quality guarantee. No accepted change does not certify exhaustive local optimality.",
    }
    return tables.astype(np.int32, copy=False), fine_target.astype(np.int32, copy=False), stats
