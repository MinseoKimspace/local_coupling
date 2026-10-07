"""Capacity- and marginal-preserving bipartite cycle rounding (float64).

Each alternating-cycle update is a martingale and makes at least one edge
integral. Integer row/column degrees are preserved. This is NOT categorical
sampling, greedy rounding, or a Gibbs sampler over permutations.
Reference: Gandhi et al., https://www.cs.umd.edu/~srin/PDF/2005/depround-dec05.pdf
Numerical claims are subject to the explicit float64 tolerances below.
"""

import numpy as np
import ot
from scipy.optimize import minimize
from scipy.special import logsumexp


MARGINAL_TOLERANCE = 1e-8
EDGE_TOLERANCE = 1e-12


def integer_capacities(counts, n, k):
    counts = np.asarray(counts)
    if counts.shape != (k,) or not np.isfinite(counts).all() \
            or (counts < 0).any() or not np.equal(counts, np.rint(counts)).all() \
            or counts.sum() != n:
        raise ValueError("Integer nonnegative capacities must sum to N")
    return counts.astype(np.int64)


def feasible_plan(plan, counts):
    """Repair only small Sinkhorn residuals by downscaling + deficit outer product.

    This returns a feasible numerical plan, not an exactly unmodified entropic
    optimum. Large residuals are rejected rather than silently hidden.
    """
    p = np.array(plan, dtype=np.float64, copy=True)
    if p.ndim != 2 or not p.size or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Expected a finite nonnegative [N,K] plan")
    counts = integer_capacities(counts, *p.shape)
    before = max(np.abs(p.sum(1) - 1).max(), np.abs(p.sum(0) - counts).max())
    if before > 1e-6:
        raise ValueError(f"Sinkhorn marginal residual {before:.3g}; increase sinkhorn_iterations")
    p *= np.minimum(1 / np.maximum(p.sum(1), np.finfo(float).tiny), 1)[:, None]
    p *= np.minimum(counts / np.maximum(p.sum(0), np.finfo(float).tiny), 1)[None, :]
    rows, columns = np.maximum(1 - p.sum(1), 0), np.maximum(counts - p.sum(0), 0)
    mass = rows.sum()
    if mass > np.finfo(float).eps:
        p += np.outer(rows, columns) / mass
    after = max(np.abs(p.sum(1) - 1).max(), np.abs(p.sum(0) - counts).max())
    if after > MARGINAL_TOLERANCE:
        raise RuntimeError("Failed to repair Sinkhorn marginals")
    return p, float(before), float(after)


def entropic_plan(cost, counts, *, epsilon, iterations):
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("Expected finite nonempty [N,K] costs")
    counts = integer_capacities(counts, *matrix.shape)
    if not np.isfinite(epsilon) or epsilon <= 0 or iterations < 1:
        raise ValueError("epsilon/iterations must be positive")
    n, k = matrix.shape
    live = counts > 0
    plan = np.zeros((n, k), dtype=np.float64)
    if live.sum() == 1:
        plan[:, live] = 1
    else:
        candidate, log = ot.sinkhorn(
            np.full(n, 1 / n), counts[live] / n, matrix[:, live], epsilon,
            method="sinkhorn_log", numItermax=iterations, stopThr=1e-12, warn=False, log=True,
        )
        candidate *= n
        residual = max(np.abs(candidate.sum(1) - 1).max(),
                       np.abs(candidate.sum(0) - counts[live]).max())
        if residual > 1e-6:
            # Same entropic transportation objective, with row constraints
            # eliminated analytically. Only K-1 dual variables are optimized.
            # This avoids near-hard Sinkhorn's arbitrarily slow scaling rate.
            reduced, mass = matrix[:, live], counts[live]

            def dual(value):
                potentials = np.append(value, 0.)
                logits = (potentials - reduced) / epsilon
                normalizer = logsumexp(logits, axis=1)
                probabilities = np.exp(logits - normalizer[:, None])
                objective = epsilon * normalizer.sum() - mass @ potentials
                gradient = probabilities.sum(0) - mass
                return objective, gradient[:-1]

            initial = np.asarray(log["log_v"]) * epsilon
            initial -= initial[-1]
            solved = minimize(dual, initial[:-1], jac=True, method="BFGS",
                              options={"gtol": 1e-10, "maxiter": iterations})
            logits = (np.append(solved.x, 0.) - reduced) / epsilon
            candidate = np.exp(logits - logsumexp(logits, axis=1)[:, None])
        plan[:, live] = candidate
    return feasible_plan(plan, counts)


def _cycle(adjacency):
    """Find an alternating cycle in the fractional bipartite graph."""
    parent, depth = [-1] * len(adjacency), [-1] * len(adjacency)
    for start, neighbours in enumerate(adjacency):
        if not neighbours or depth[start] >= 0:
            continue
        depth[start] = 0
        stack = [(start, iter(neighbours))]
        while stack:
            vertex, iterator = stack[-1]
            neighbour = next(iterator, None)
            if neighbour is None:
                stack.pop()
                continue
            if neighbour == parent[vertex]:
                continue
            if depth[neighbour] < 0:
                parent[neighbour], depth[neighbour] = vertex, depth[vertex] + 1
                stack.append((neighbour, iter(adjacency[neighbour])))
            elif depth[neighbour] < depth[vertex]:
                nodes = [neighbour, vertex]
                while nodes[-1] != neighbour:
                    nodes.append(parent[nodes[-1]])
                return list(zip(nodes[:-1], nodes[1:]))
    return None


def dependent_round(plan, counts, rng):
    """Draw labels with exact counts and E[one_hot(labels)] = plan numerically.

    Source/patch traversal is randomized to avoid privileging fixed input
    indices. Original source order is restored on return; coordinates are
    never moved. Tiny edge snapping is bounded by float64 edge tolerance.
    """
    original = np.asarray(plan, dtype=np.float64)
    if original.ndim != 2 or not original.size or not np.isfinite(original).all() \
            or (original < 0).any() or (original > 1 + MARGINAL_TOLERANCE).any():
        raise ValueError("Expected a finite probability plan")
    n, k = original.shape
    counts = integer_capacities(counts, n, k)
    if max(np.abs(original.sum(1) - 1).max(),
           np.abs(original.sum(0) - counts).max()) > MARGINAL_TOLERANCE:
        raise ValueError("Dependent rounding needs feasible row/column marginals")
    row_order, col_order = rng.permutation(n), rng.permutation(k)
    p = original[np.ix_(row_order, col_order)].copy()
    adjacency = [set() for _ in range(n + k)]
    rows, columns = np.nonzero((p > EDGE_TOLERANCE) & (p < 1 - EDGE_TOLERANCE))
    for row, col in zip(rows.tolist(), columns.tolist()):
        adjacency[row].add(n + col)
        adjacency[n + col].add(row)
    for _ in range(len(rows) + 1):
        cycle = _cycle(adjacency)
        if cycle is None:
            break
        edges = [(min(u, v), max(u, v) - n) for u, v in cycle]
        plus, minus = edges[::2], edges[1::2]
        alpha = min(*[1 - p[r, c] for r, c in plus], *[p[r, c] for r, c in minus])
        beta = min(*[p[r, c] for r, c in plus], *[1 - p[r, c] for r, c in minus])
        if alpha <= 0 or beta <= 0:
            raise RuntimeError("Nonpositive alternating-cycle step")
        change = alpha if rng.random() < beta / (alpha + beta) else -beta
        for sign, group in ((1, plus), (-1, minus)):
            for row, col in group:
                p[row, col] += sign * change
                if p[row, col] <= EDGE_TOLERANCE or p[row, col] >= 1 - EDGE_TOLERANCE:
                    p[row, col] = float(p[row, col] > .5)
                    adjacency[row].discard(n + col)
                    adjacency[n + col].discard(row)
    labels = np.empty(n, dtype=np.int64)
    labels[row_order] = col_order[p.argmax(1)]
    if (p.max(1) < 1 - MARGINAL_TOLERANCE).any() \
            or not np.array_equal(np.bincount(labels, minlength=k), counts):
        raise RuntimeError("Fractional rounding did not reach an integral capacity-preserving assignment")
    return labels


def random_fine_permutation(source_labels, target_labels, k, rng):
    """Uniform random bijections per patch; source indices are unchanged."""
    permutation = np.empty(len(source_labels), dtype=np.int64)
    for patch in range(k):
        source = np.flatnonzero(source_labels == patch)
        target = np.flatnonzero(target_labels == patch)
        if len(source) != len(target):
            raise ValueError("Source and target patch counts differ")
        permutation[source] = rng.permutation(target)
    return permutation
