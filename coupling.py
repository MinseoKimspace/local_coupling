import warnings

import numpy as np
import ot
import torch
import torch.nn.functional as F

# Flamary et al., JMLR 22(78), 2021: https://jmlr.org/papers/v22/20-451.html
ALIASES = {"global_hungarian": "global_ot", "geometry_aware_hungarian": "geometry_aware_ot"}

METHODS = {
    "independent": ("none", "none", "none"),
    "regional": ("balanced", "regional", "random"),
    "target_guided": ("balanced", "exact", "random"),
    "target_guided_exact_optimized": ("balanced", "exact_batched", "random"),
    "target_guided_source_greedy": ("balanced", "greedy", "random"),
    "target_guided_source_sinkhorn": ("balanced", "sinkhorn", "random"),
    "target_guided_sinkhorn": ("balanced", "sinkhorn", "random"),
    "geometry_aware_sinkhorn": ("nearest", "sinkhorn", "random"),
    "geometry_aware_ot": ("nearest", "exact", "random"),
    "target_guided_strict": ("oracle", "exact", "random"),
    "target_guided_strict_local": ("oracle", "exact", "exact"),
    "target_guided_strict_balanced": ("oracle", "greedy", "exact"),
    "global_ot": ("none", "global", "none"),
}


def canonical_method(name: str) -> str:
    name = ALIASES.get(name, name)
    if name not in METHODS:
        raise ValueError(f"Unknown coupling: {name}")
    return name


def balanced_partition_solver(method):
    if method == "target_guided_sinkhorn":
        return "sinkhorn"
    if method == "target_guided_exact_optimized":
        return "exact_batched"
    return "exact"


def coupling_info(name: str) -> dict:
    name = canonical_method(name)
    partition, assignment, local = METHODS[name]
    target_solver = balanced_partition_solver(name) if partition == "balanced" else None
    exact = (assignment in ("exact", "exact_batched", "global", "regional") or local == "exact"
             or target_solver in ("exact", "exact_batched"))
    info = {
        "method": name,
        "implementation": "pot_v1",
        "target_partition": partition,
        "source_assignment": "exact" if assignment == "exact_batched" else assignment,
        "local_pairing": local,
        "cost": "squared_euclidean" if name != "independent" else None,
        "fps_start": "farthest_from_set_mean" if partition in ("balanced", "nearest") else None,
        "exact_solver": "POT/network_simplex" if exact else None,
        "sinkhorn_solver": "POT/sinkhorn_log" if assignment == "sinkhorn" else None,
        "rounding": "capacity_preserving_greedy" if assignment == "sinkhorn" else None,
    }
    if name in ("target_guided_exact_optimized", "target_guided_source_greedy", "target_guided_source_sinkhorn"):
        info.update(implementation="pot_tg_cost_v1", target_partition_solver="POT/network_simplex",
                    exact_transfer="batched" if assignment == "exact_batched" else "per_cloud")
        if assignment == "greedy":
            info["greedy_rule"] = "negative squared distance; descending best-vs-second margin; capacity preserving"
    return info


def farthest_point_sample(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    batch, n, _ = points.shape
    if not 1 <= num_samples <= n:
        raise ValueError("num_regions must satisfy 1 <= K <= N")
    rows = torch.arange(batch, device=points.device)
    indices = torch.empty(batch, num_samples, dtype=torch.long, device=points.device)
    distances = points.new_full((batch, n), float("inf"))
    farthest = ((points - points.mean(1, keepdim=True)) ** 2).sum(-1).argmax(-1)
    for i in range(num_samples):
        indices[:, i] = farthest
        anchor = points[rows, farthest].unsqueeze(1)
        distances = torch.minimum(distances, ((points - anchor) ** 2).sum(-1))
        distances[rows, farthest] = -1
        farthest = distances.argmax(-1)
    return indices


def exact_assignment(cost: torch.Tensor, capacities: torch.Tensor | None = None) -> torch.Tensor:
    n, k = cost.shape
    if capacities is None:
        capacities = torch.ones(k, dtype=torch.long, device=cost.device)
    capacity = capacities.detach().cpu().double().numpy()
    matrix = np.ascontiguousarray(cost.detach().cpu().double().numpy())
    return torch.as_tensor(_exact_assignment_numpy(matrix, capacity), dtype=torch.long, device=cost.device)


def _exact_assignment_numpy(matrix, capacity):
    n, k = matrix.shape
    if (capacity.shape != (k,) or np.any(capacity < 0)
            or not np.equal(capacity, np.rint(capacity)).all() or capacity.sum() != n):
        raise ValueError("Integer nonnegative capacities must sum to the number of points")
    if n == 0 or not np.isfinite(matrix).all():
        raise ValueError("Expected nonempty, finite assignment costs")
    plan, log = ot.emd(np.ones(n), capacity, matrix, log=True)
    hard = np.rint(plan)
    if (log.get("warning") or not np.allclose(plan, hard, atol=1e-7, rtol=0)
            or not np.equal(hard.sum(1), 1).all()
            or not np.equal(hard.sum(0), capacity).all()):
        raise RuntimeError(f"POT did not return an optimal integral assignment: {log.get('warning')}")
    return hard.argmax(1)


def exact_assignment_batched(costs, capacities):
    # Same float64 matrices and POT solves as exact_assignment; transfer the
    # entire batch once in each direction instead of once per cloud.
    matrices = np.ascontiguousarray(costs.detach().cpu().double().numpy())
    counts = capacities.detach().cpu().double().numpy()
    labels = np.stack([_exact_assignment_numpy(matrix, capacity) for matrix, capacity in zip(matrices, counts)])
    return torch.as_tensor(labels, dtype=torch.long, device=costs.device)


def round_balanced_plan(scores: torch.Tensor, capacities: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(scores).all():
        raise ValueError("Cannot round nonfinite assignment scores")
    n, k = scores.shape
    remaining = capacities.cpu().tolist()
    if any(c < 0 or int(c) != c for c in remaining) or sum(remaining) != n:
        raise ValueError("Integer nonnegative capacities must sum to N")
    ranked, preferences = scores.sort(dim=1, descending=True)
    margins = ranked[:, 0] - ranked[:, 1] if k > 1 else scores.new_zeros(n)
    order = margins.argsort(descending=True).cpu().tolist()
    preferences = preferences.cpu().tolist()
    labels = [-1] * n
    for i in order:
        for region in preferences[i]:
            if remaining[region] > 0:
                labels[i] = region
                remaining[region] -= 1
                break
    return torch.tensor(labels, dtype=torch.long, device=scores.device)


def assign_regions(points, centers, capacities, *, solver="exact", epsilon=0.1, iterations=100):
    if centers.ndim == 2:
        centers = centers.unsqueeze(0).expand(points.shape[0], -1, -1)
    costs = torch.cdist(points, centers).square()
    if solver == "exact_batched":
        return exact_assignment_batched(costs, capacities)
    labels = []
    for cost, capacity in zip(costs, capacities):
        if solver == "exact":
            labels.append(exact_assignment(cost, capacity))
            continue
        live = (torch.arange(cost.shape[1], device=cost.device) if solver == "greedy"
                else torch.where(capacity > 0)[0])
        reduced, counts = cost[:, live], capacity[live]
        if solver == "sinkhorn":
            if epsilon <= 0 or iterations < 1:
                raise ValueError("Sinkhorn epsilon and iterations must be positive")
            n = cost.shape[0]
            plan = ot.sinkhorn(
                cost.new_full((n,), 1.0 / n), counts.to(cost.dtype) / n,
                reduced, epsilon, method="sinkhorn_log", numItermax=iterations,
                stopThr=1e-6, warn=False,
            ) * n
            residual = torch.maximum((plan.sum(1) - 1).abs().max(), (plan.sum(0) - counts).abs().max())
            if residual.item() > 1e-3:
                warnings.warn("Sinkhorn marginal tolerance not reached; applying capacity-preserving greedy rounding.", stacklevel=2)
        elif solver == "greedy":
            plan = -reduced
        else:
            raise ValueError(f"Unknown assignment solver: {solver}")
        labels.append(live[round_balanced_plan(plan, counts)])
    return torch.stack(labels)


def region_centroids(points, labels, k):
    membership = F.one_hot(labels, k).to(points.dtype)
    counts = membership.sum(1)
    centers = membership.transpose(1, 2) @ points / counts.clamp_min(1).unsqueeze(-1)
    return centers, counts.long()


def balanced_target_partition(target, k, *, solver="exact", epsilon=0.1, iterations=100):
    """Shared by training and the training-free patch audit."""
    batch, n, _ = target.shape
    rows = torch.arange(batch, device=target.device).unsqueeze(1)
    anchors = target[rows, farthest_point_sample(target, k)]
    capacities = torch.full((batch, k), n // k, dtype=torch.long, device=target.device)
    capacities[:, :n % k] += 1
    labels = assign_regions(target, anchors, capacities, solver=solver, epsilon=epsilon, iterations=iterations)
    centers, capacities = region_centroids(target, labels, k)
    return anchors, labels, centers, capacities


def pair_within_regions(source, target, source_labels, target_labels, k, *, local="random", generator=None):
    permutation = torch.empty(source.shape[:2], dtype=torch.long, device=source.device)
    for b in range(source.shape[0]):
        for region in range(k):
            src = torch.where(source_labels[b] == region)[0]
            dst = torch.where(target_labels[b] == region)[0]
            if src.numel() != dst.numel():
                raise RuntimeError("Source and target patch capacities differ")
            if src.numel() == 0:
                continue
            order = (exact_assignment(torch.cdist(source[b, src], target[b, dst]).square())
                     if local == "exact" else torch.randperm(dst.numel(), device=source.device, generator=generator))
            permutation[b, src] = dst[order]
    return permutation


@torch.no_grad()
def coupling_permutation(source, target, *, coupling, num_regions=None, target_centers=None,
                         sinkhorn_epsilon=0.1, sinkhorn_iterations=100, generator=None):
    method = canonical_method(coupling)
    partition, assignment, local = METHODS[method]
    if source.ndim != 3 or source.shape != target.shape:
        raise ValueError("Source and target must have the same [B, N, D] shape")
    if method == "independent":
        return None
    if method == "global_ot":
        return torch.stack([exact_assignment(cost) for cost in torch.cdist(source, target).square()])
    batch, n, _ = target.shape
    rows = torch.arange(batch, device=target.device).unsqueeze(1)
    options = {"epsilon": sinkhorn_epsilon, "iterations": sinkhorn_iterations}
    if partition == "oracle":
        if target_centers is None:
            raise ValueError("Strict checkerboard experiments require known cell centers")
        centers = target_centers
        k = centers.shape[0]
        target_labels = torch.cdist(target, centers.unsqueeze(0)).argmin(-1)
        capacities = F.one_hot(target_labels, k).sum(1)
    else:
        if num_regions is None:
            raise ValueError("num_regions is required")
        k = num_regions
        if partition == "nearest":
            anchors = target[rows, farthest_point_sample(target, k)]
            target_labels = torch.cdist(target, anchors).argmin(-1)
            centers, capacities = region_centroids(target, target_labels, k)
        else:
            if method == "regional" and n % k:
                raise ValueError("Regional patch-to-patch matching requires N divisible by K")
            anchors, target_labels, centers, capacities = balanced_target_partition(
                target, k, solver=balanced_partition_solver(method), **options)
    if assignment == "regional":
        anchors = source[rows, farthest_point_sample(source, k)]
        source_labels = assign_regions(source, anchors, capacities)
        source_centers, _ = region_centroids(source, source_labels, k)
        matches = torch.stack([exact_assignment(cost) for cost in torch.cdist(source_centers, centers).square()])
        source_labels = matches.gather(1, source_labels)
    else:
        source_labels = assign_regions(source, centers, capacities, solver=assignment, **options)
    return pair_within_regions(source, target, source_labels, target_labels, k,
                               local=local, generator=generator)
