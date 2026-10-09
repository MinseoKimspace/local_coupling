"""Offline, capacity-preserving coarse TG boundary experiments.

Only patch labels are changed.  The caller must keep the existing fresh uniform
within-patch permutation during training.  The first table is the original TG
assignment; the other tables form an ensemble, not an averaged target point.

The guards below limit *added* centroid ambiguity.  They are not certificates
of target support, topology preservation, neural smoothness, or sample quality.
"""

import numbers

import numpy as np


DEFAULTS = {
    "num_tables": 4,
    "source_neighbors": 8,
    "max_swaps": 16,
    "tau": 2.0,
    "distance_epsilon": 1e-3,
    "max_cost_increase": 0.02,
    "cost_match_tolerance": 0.005,
    "destination_budget": 0.35,
    "target_neighbors": 2,
    "max_target_distance_factor": 2.0,
}


def settings(value=None):
    """Validate and normalize the small, JSON-safe boundary option schema."""
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("tg boundary options must be a mapping")
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown tg boundary options: {sorted(unknown)}")
    result = dict(DEFAULTS)
    result.update(value)
    for name, minimum in (("num_tables", 2), ("source_neighbors", 1),
                          ("max_swaps", 0), ("target_neighbors", 1)):
        item = result[name]
        if isinstance(item, bool) or not isinstance(item, numbers.Integral) or item < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
        result[name] = int(item)
    for name in set(DEFAULTS) - {"num_tables", "source_neighbors", "max_swaps", "target_neighbors"}:
        item = result[name]
        if isinstance(item, bool) or not isinstance(item, numbers.Real) or not np.isfinite(item):
            raise ValueError(f"{name} must be a finite number")
        item = float(item)
        minimum = 0.0
        if item < minimum or (name in {"distance_epsilon", "max_target_distance_factor"} and item == 0):
            sign = "> 0" if name in {"distance_epsilon", "max_target_distance_factor"} else ">= 0"
            raise ValueError(f"{name} must be {sign}")
        result[name] = item
    return result


def source_graph(source, neighbors):
    """Return unique undirected edges of the union/symmetric source kNN graph.

    The second array contains source distances (epsilon is added by the caller).
    Stable index ordering resolves exact-distance ties deterministically.
    """
    source = np.asarray(source, dtype=np.float64)
    n = len(source)
    if n < 2:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float64)
    squared = np.sum((source[:, None] - source[None, :]) ** 2, axis=-1)
    np.fill_diagonal(squared, np.inf)
    nn = np.argsort(squared, axis=1, kind="stable")[:, :min(int(neighbors), n - 1)]
    left = np.repeat(np.arange(n), nn.shape[1])
    edges = np.unique(np.sort(np.column_stack((left, nn.ravel())), axis=1), axis=0)
    distances = np.linalg.norm(source[edges[:, 0]] - source[edges[:, 1]], axis=1)
    return edges.astype(np.int32), distances


def graph_score(source, centroids, tables, edges, denominators, tau):
    """Evaluate the ensemble centroid-velocity squared-hinge graph objective.

    ``denominators`` must include the configured positive distance epsilon.
    """
    if len(edges) == 0:
        return 0.0
    velocity = np.asarray(centroids)[np.asarray(tables)].mean(axis=0) - np.asarray(source)
    differences = velocity[edges[:, 0]] - velocity[edges[:, 1]]
    hinge = np.maximum(np.linalg.norm(differences, axis=1) / denominators - tau, 0.0)
    return float(np.mean(hinge ** 2))


def _validate_inputs(source, target, source_labels, target_labels, k, seed):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (2,) or target.shape != source.shape or len(source) == 0:
        raise ValueError("boundary clouds must have matching nonempty [N, 2] shapes")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("boundary clouds must contain only finite coordinates")
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
        raise ValueError("boundary seed must be a nonnegative integer")
    return source, target, labels[0], labels[1], int(k), int(seed)


def _incident_edges(edges, n):
    lists = [[] for _ in range(n)]
    for index, (i, j) in enumerate(edges):
        lists[i].append(index)
        lists[j].append(index)
    width = max((len(item) for item in lists), default=0)
    padded = np.full((n, width), -1, dtype=np.int32)
    for i, item in enumerate(lists):
        padded[i, :len(item)] = item
    return padded


def _score_deltas(velocity, changes, candidates, edges, denominators, tau, incident):
    """Exact graph-score changes for all candidate swaps, using incident edges.

    Moving i by d and j by -d changes no other velocity.  An edge joining i/j
    appears in both incidence lists and is explicitly counted only once.
    """
    if len(candidates) == 0:
        return np.empty(0, dtype=np.float64)
    i, j = candidates.T
    first, second = incident[i], incident[j]
    indices = np.concatenate((first, second), axis=1)
    valid = indices >= 0
    safe = np.maximum(indices, 0)
    endpoints = edges[safe]
    width = first.shape[1]
    second_endpoints = endpoints[:, width:]
    duplicate = ((second_endpoints[:, :, 0] == i[:, None]) & (second_endpoints[:, :, 1] == j[:, None]))
    duplicate |= ((second_endpoints[:, :, 0] == j[:, None]) & (second_endpoints[:, :, 1] == i[:, None]))
    valid[:, width:] &= ~duplicate
    a, b = endpoints[:, :, 0], endpoints[:, :, 1]
    before = velocity[a] - velocity[b]
    factors = ((a == i[:, None]).astype(np.int8) - (a == j[:, None]).astype(np.int8)
               - (b == i[:, None]).astype(np.int8) + (b == j[:, None]).astype(np.int8))
    after = before + factors[:, :, None] * changes[:, None, :]
    den = denominators[safe]
    old = np.maximum(np.linalg.norm(before, axis=2) / den - tau, 0.0) ** 2
    new = np.maximum(np.linalg.norm(after, axis=2) / den - tau, 0.0) ** 2
    return np.sum(np.where(valid, new - old, 0.0), axis=1) / len(edges)


def build_tables(source, target, source_labels, target_labels, k, options=None, seed=0):
    """Build paired guided/control coarse table ensembles without global RNG.

    Row zero stays exactly TG.  Each remaining row uses at most ``max_swaps``
    disjoint source-edge swaps.  A guided swap is accepted only when a feasible
    random control swap can be paired with it, so actual row swap counts match.
    Control choices use feasibility and centroid-cost matching, never the graph
    objective.  Both incremental and cumulative paired cost differences are
    bounded by ``cost_match_tolerance * baseline_mean_centroid_cost``.

    Per-source destination guard:
        mean_r ||c[table[r, i]] - c[original[i]]||^2
          <= destination_budget^2 * mean_j ||target[j] - target.mean()||^2.
    """
    opt = settings(options)
    source, target, original, target_labels, k, seed = _validate_inputs(
        source, target, source_labels, target_labels, k, seed)
    rng = np.random.default_rng(seed)
    n, r = len(source), opt["num_tables"]
    centroids = np.stack([target[target_labels == a].mean(axis=0) for a in range(k)])
    patch_rms = np.array([np.sqrt(np.mean(np.sum((target[target_labels == a] - centroids[a]) ** 2,
                                                axis=1))) for a in range(k)])
    target_rms_squared = float(np.mean(np.sum((target - target.mean(axis=0)) ** 2, axis=1)))
    destination_limit_squared = opt["destination_budget"] ** 2 * target_rms_squared
    guided = np.repeat(original[None, :], r, axis=0)
    control = guided.copy()
    base_cost = float(np.mean(np.sum((source - centroids[original]) ** 2, axis=1)))
    guided_cost = np.full(r, base_cost)
    control_cost = np.full(r, base_cost)
    cost_limit = (1 + opt["max_cost_increase"]) * base_cost
    match_limit = opt["cost_match_tolerance"] * base_cost
    numerical = 1e-12 * max(1.0, base_cost, target_rms_squared)
    all_edges, source_distances = source_graph(source, opt["source_neighbors"])
    denominators = source_distances + opt["distance_epsilon"]
    incident = _incident_edges(all_edges, n)

    centroid_distances = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=2)
    nearest_distances = centroid_distances.copy()
    np.fill_diagonal(nearest_distances, np.inf)
    target_nn = np.zeros((k, k), dtype=bool)
    if k > 1:
        neighbors = np.argsort(nearest_distances, axis=1, kind="stable")[:, :min(opt["target_neighbors"], k - 1)]
        target_nn[np.arange(k)[:, None], neighbors] = True
    target_gate = target_nn & target_nn.T
    target_gate &= centroid_distances <= opt["max_target_distance_factor"] * (patch_rms[:, None] + patch_rms[None, :]) + numerical
    np.fill_diagonal(target_gate, False)
    first_labels, second_labels = original[all_edges[:, 0]], original[all_edges[:, 1]]
    cross_patch = first_labels != second_labels
    allowed = target_gate[first_labels, second_labels] & cross_patch
    candidates = all_edges[allowed]
    i, j = candidates.T
    shifts = centroids[original[j]] - centroids[original[i]]
    shift_squared = np.sum(shifts ** 2, axis=1)
    candidate_cost_delta = (
        np.sum((source[i] - centroids[original[j]]) ** 2, axis=1)
        + np.sum((source[j] - centroids[original[i]]) ** 2, axis=1)
        - np.sum((source[i] - centroids[original[i]]) ** 2, axis=1)
        - np.sum((source[j] - centroids[original[j]]) ** 2, axis=1)) / n
    guided_drift_sum = np.zeros(n)
    control_drift_sum = np.zeros(n)
    velocity = centroids[original] - source
    initial_score = graph_score(source, centroids, guided, all_edges, denominators, opt["tau"])
    swaps_per_table = np.zeros(r, dtype=np.int32)
    paired_delta_differences = []
    score_trace = [initial_score]
    reject_counts = {"same_patch_edges": int(np.count_nonzero(~cross_patch)),
                     "target_gate_edges": int(np.count_nonzero(cross_patch & ~allowed)),
                     "guided_used": 0, "guided_destination_budget": 0, "guided_cost": 0,
                     "guided_non_improving": 0, "no_matched_control": 0,
                     "rows_without_accepted_swaps": 0}

    def feasible(used, drift_sum, row_cost):
        unused = ~(used[i] | used[j])
        budget = ((drift_sum[i] + shift_squared) / r <= destination_limit_squared + numerical)
        budget &= (drift_sum[j] + shift_squared) / r <= destination_limit_squared + numerical
        cost = row_cost + candidate_cost_delta <= cost_limit + numerical
        return unused & budget & cost, unused, budget, cost

    for row in range(1, r):
        guided_used = np.zeros(n, dtype=bool)
        control_used = np.zeros(n, dtype=bool)
        for _ in range(min(opt["max_swaps"], n // 2)):
            good, unused, budget, cost = feasible(guided_used, guided_drift_sum, guided_cost[row])
            reject_counts["guided_used"] += int(np.count_nonzero(~unused))
            reject_counts["guided_destination_budget"] += int(np.count_nonzero(unused & ~budget))
            reject_counts["guided_cost"] += int(np.count_nonzero(unused & budget & ~cost))
            eligible = np.flatnonzero(good)
            if eligible.size == 0:
                break
            score_delta = _score_deltas(velocity, shifts[eligible] / r, candidates[eligible],
                                        all_edges, denominators, opt["tau"], incident)
            improving = score_delta < -1e-12 * max(1.0, score_trace[-1])
            reject_counts["guided_non_improving"] += int(np.count_nonzero(~improving))
            if not np.any(improving):
                break
            improving_ids = eligible[improving]
            ranked = improving_ids[np.argsort(score_delta[improving], kind="stable")]
            control_good = feasible(control_used, control_drift_sum, control_cost[row])[0]
            selected = None
            for guide_index in ranked:
                desired_delta = candidate_cost_delta[guide_index]
                matched = control_good & (np.abs(candidate_cost_delta - desired_delta) <= match_limit + numerical)
                matched &= np.abs(control_cost[row] + candidate_cost_delta
                                  - guided_cost[row] - desired_delta) <= match_limit + numerical
                pool = np.flatnonzero(matched)
                if pool.size:
                    selected = (int(guide_index), int(rng.choice(pool)))
                    break
                reject_counts["no_matched_control"] += 1
            if selected is None:
                break
            guide_index, control_index = selected
            for table, used, drift, costs, index in (
                    (guided, guided_used, guided_drift_sum, guided_cost, guide_index),
                    (control, control_used, control_drift_sum, control_cost, control_index)):
                a, b = candidates[index]
                table[row, a], table[row, b] = table[row, b], table[row, a]
                used[[a, b]] = True
                drift[[a, b]] += shift_squared[index]
                costs[row] += candidate_cost_delta[index]
            a, b = candidates[guide_index]
            velocity[a] += shifts[guide_index] / r
            velocity[b] -= shifts[guide_index] / r
            swaps_per_table[row] += 1
            paired_delta_differences.append(float(abs(candidate_cost_delta[guide_index]
                                                      - candidate_cost_delta[control_index])))
            # Full graph recomputation only after accepting, to audit incremental deltas.
            score_trace.append(graph_score(source, centroids, guided, all_edges, denominators, opt["tau"]))
        if swaps_per_table[row] == 0:
            reject_counts["rows_without_accepted_swaps"] += 1

    guided_score = graph_score(source, centroids, guided, all_edges, denominators, opt["tau"])
    control_score = graph_score(source, centroids, control, all_edges, denominators, opt["tau"])
    # Recompute diagnostics from returned tables rather than relying on updates.
    guided_cost_actual = np.mean(np.sum((source[None] - centroids[guided]) ** 2, axis=2), axis=1)
    control_cost_actual = np.mean(np.sum((source[None] - centroids[control]) ** 2, axis=2), axis=1)
    guided_rms = np.sqrt(np.mean(np.sum((centroids[guided] - centroids[original][None]) ** 2, axis=2), axis=0))
    control_rms = np.sqrt(np.mean(np.sum((centroids[control] - centroids[original][None]) ** 2, axis=2), axis=0))
    cost_scale = max(base_cost, np.finfo(np.float64).tiny)
    target_scale = max(np.sqrt(target_rms_squared), np.finfo(np.float64).tiny)
    stats = {
        "algorithm": "paired_coarse_boundary_ensemble_v1",
        "settings": opt,
        "seed": seed,
        "baseline_score": initial_score,
        "guided_score": guided_score,
        "control_score": control_score,
        "guided_score_trace": [float(item) for item in score_trace],
        "baseline_centroid_cost": base_cost,
        "baseline_cost": base_cost,
        "guided_cost": float(guided_cost_actual.mean()),
        "control_cost": float(control_cost_actual.mean()),
        "guided_centroid_cost": guided_cost_actual.tolist(),
        "control_centroid_cost": control_cost_actual.tolist(),
        "swaps_per_table": swaps_per_table.tolist(),
        "accepted_swaps": int(swaps_per_table.sum()),
        "guided_changed_points": np.count_nonzero(guided != original[None], axis=1).tolist(),
        "control_changed_points": np.count_nonzero(control != original[None], axis=1).tolist(),
        "guided_max_destination_rms": float(guided_rms.max()),
        "control_max_destination_rms": float(control_rms.max()),
        "guided_max_destination_rms_fraction": float(guided_rms.max() / target_scale),
        "control_max_destination_rms_fraction": float(control_rms.max() / target_scale),
        "destination_rms_limit": float(np.sqrt(destination_limit_squared)),
        "target_global_rms": float(np.sqrt(target_rms_squared)),
        "max_pair_cost_delta_difference_fraction": float(max(paired_delta_differences, default=0.0) / cost_scale),
        "max_table_cost_difference_fraction": float(np.max(np.abs(guided_cost_actual - control_cost_actual)) / cost_scale),
        "max_paired_cost_gap": float(np.max(np.abs(guided_cost_actual - control_cost_actual))),
        "max_guided_destination_drift": float(guided_rms.max()),
        "max_control_destination_drift": float(control_rms.max()),
        "source_graph_edges": int(len(all_edges)),
        "eligible_source_edges": int(len(candidates)),
        "allowed_target_patch_pairs": np.argwhere(np.triu(target_gate, k=1)).tolist(),
        "reject_counts": reject_counts,
        "no_feasible_change": bool(swaps_per_table.sum() == 0),
        "guard_notes": "Centroid drift and adjacency are empirical safeguards, not support/topology guarantees.",
    }
    return guided.astype(np.int32, copy=False), control.astype(np.int32, copy=False), stats
