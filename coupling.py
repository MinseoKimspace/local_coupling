import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def farthest_point_sample(
    points: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    batch_size, num_points, _ = points.shape
    batch_indices = torch.arange(batch_size, device=points.device)
    sample_indices = torch.empty(
        batch_size,
        num_samples,
        dtype=torch.long,
        device=points.device,
    )
    min_distances = torch.full(
        (batch_size, num_points),
        float("inf"),
        dtype=points.dtype,
        device=points.device,
    )

    center = points.mean(dim=1, keepdim=True)
    farthest = ((points - center) ** 2).sum(dim=-1).argmax(dim=-1)

    for index in range(num_samples):
        sample_indices[:, index] = farthest
        anchor = points[batch_indices, farthest].unsqueeze(1)
        distances = ((points - anchor) ** 2).sum(dim=-1)
        min_distances = torch.minimum(min_distances, distances)
        farthest = min_distances.argmax(dim=-1)

    return sample_indices


def balanced_partition(
    points: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_points, _ = points.shape
    num_regions = anchors.shape[1]
    region_size = num_points // num_regions
    costs = torch.cdist(points, anchors).square()
    costs = costs.repeat_interleave(region_size, dim=-1).cpu().numpy()
    regions = torch.empty(
        batch_size,
        num_points,
        dtype=torch.long,
        device=points.device,
    )

    for batch_index in range(batch_size):
        _, anchor_slots = linear_sum_assignment(costs[batch_index])
        regions[batch_index] = torch.as_tensor(
            anchor_slots // region_size,
            device=points.device,
        )

    return regions


def _region_centroids(
    points: torch.Tensor,
    regions: torch.Tensor,
    num_regions: int,
) -> torch.Tensor:
    membership = F.one_hot(regions, num_regions).to(points.dtype)
    totals = membership.transpose(1, 2) @ points
    return totals / membership.sum(dim=1).unsqueeze(-1)


def match_regions(
    source: torch.Tensor,
    target: torch.Tensor,
    source_regions: torch.Tensor,
    target_regions: torch.Tensor,
    *,
    num_regions: int,
) -> torch.Tensor:
    source_centroids = _region_centroids(source, source_regions, num_regions)
    target_centroids = _region_centroids(target, target_regions, num_regions)
    costs = torch.cdist(source_centroids, target_centroids).square().cpu().numpy()
    matches = torch.empty(
        source.shape[0],
        num_regions,
        dtype=torch.long,
        device=source.device,
    )

    for batch_index in range(source.shape[0]):
        _, target_indices = linear_sum_assignment(costs[batch_index])
        matches[batch_index] = torch.as_tensor(
            target_indices,
            device=source.device,
        )

    return matches


@torch.no_grad()
def regional_permutation(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    num_regions: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    batch_size, num_points, _ = source.shape
    batch_indices = torch.arange(batch_size, device=source.device).unsqueeze(1)

    source_anchor_indices = farthest_point_sample(source, num_regions)
    target_anchor_indices = farthest_point_sample(target, num_regions)
    source_anchors = source[batch_indices, source_anchor_indices]
    target_anchors = target[batch_indices, target_anchor_indices]

    source_regions = balanced_partition(source, source_anchors)
    target_regions = balanced_partition(target, target_anchors)
    region_matches = match_regions(
        source,
        target,
        source_regions,
        target_regions,
        num_regions=num_regions,
    )

    permutation = torch.empty(
        batch_size,
        num_points,
        dtype=torch.long,
        device=source.device,
    )

    for batch_index in range(batch_size):
        for source_region in range(num_regions):
            target_region = region_matches[batch_index, source_region]
            source_indices = torch.where(source_regions[batch_index] == source_region)[0]
            target_indices = torch.where(target_regions[batch_index] == target_region)[0]
            order = torch.randperm(
                target_indices.numel(),
                device=source.device,
                generator=generator,
            )
            permutation[batch_index, source_indices] = target_indices[order]

    return permutation


@torch.no_grad()
def target_guided_permutation(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    num_regions: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    batch_size, num_points, _ = source.shape
    batch_indices = torch.arange(batch_size, device=source.device).unsqueeze(1)

    target_anchor_indices = farthest_point_sample(target, num_regions)
    target_anchors = target[batch_indices, target_anchor_indices]
    target_regions = balanced_partition(target, target_anchors)
    target_centroids = _region_centroids(target, target_regions, num_regions)
    source_regions = balanced_partition(source, target_centroids)

    permutation = torch.empty(
        batch_size,
        num_points,
        dtype=torch.long,
        device=source.device,
    )

    for batch_index in range(batch_size):
        for region in range(num_regions):
            source_indices = torch.where(source_regions[batch_index] == region)[0]
            target_indices = torch.where(target_regions[batch_index] == region)[0]
            order = torch.randperm(
                target_indices.numel(),
                device=source.device,
                generator=generator,
            )
            permutation[batch_index, source_indices] = target_indices[order]

    return permutation


@torch.no_grad()
def strict_target_guided_permutation(
    source: torch.Tensor,
    target: torch.Tensor,
    target_centers: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    batch_size, num_points, _ = source.shape
    target_regions = torch.cdist(target, target_centers.unsqueeze(0)).argmin(dim=-1)
    target_orders = torch.stack(
        [
            torch.randperm(
                num_points,
                device=target.device,
                generator=generator,
            )
            for _ in range(batch_size)
        ]
    )
    ordered_regions = torch.gather(target_regions, dim=1, index=target_orders)
    region_slots = target_centers[ordered_regions]
    costs = torch.cdist(source, region_slots).square().cpu().numpy()
    permutation = torch.empty(
        batch_size,
        num_points,
        dtype=torch.long,
        device=source.device,
    )

    for batch_index in range(batch_size):
        _, slot_indices = linear_sum_assignment(costs[batch_index])
        slot_indices = torch.as_tensor(slot_indices, device=source.device)
        permutation[batch_index] = target_orders[batch_index, slot_indices]

    return permutation


@torch.no_grad()
def strict_target_guided_local_permutation(
    source: torch.Tensor,
    target: torch.Tensor,
    target_centers: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_points, _ = source.shape
    num_regions = target_centers.shape[0]
    target_regions = torch.cdist(target, target_centers.unsqueeze(0)).argmin(dim=-1)
    region_slots = target_centers[target_regions]
    region_costs = torch.cdist(source, region_slots).square().cpu().numpy()
    source_regions = torch.empty_like(target_regions)

    for batch_index in range(batch_size):
        _, slot_indices = linear_sum_assignment(region_costs[batch_index])
        slot_indices = torch.as_tensor(slot_indices, device=source.device)
        source_regions[batch_index] = target_regions[batch_index, slot_indices]

    return _local_hungarian_permutation(
        source,
        target,
        source_regions,
        target_regions,
        num_regions,
    )


def _local_hungarian_permutation(
    source: torch.Tensor,
    target: torch.Tensor,
    source_regions: torch.Tensor,
    target_regions: torch.Tensor,
    num_regions: int,
) -> torch.Tensor:
    batch_size, num_points, _ = source.shape
    permutation = torch.empty(
        batch_size,
        num_points,
        dtype=torch.long,
        device=source.device,
    )

    for batch_index in range(batch_size):
        for region in range(num_regions):
            source_indices = torch.where(source_regions[batch_index] == region)[0]
            target_indices = torch.where(target_regions[batch_index] == region)[0]
            costs = torch.cdist(
                source[batch_index, source_indices],
                target[batch_index, target_indices],
            ).square().cpu().numpy()
            _, target_order = linear_sum_assignment(costs)
            target_order = torch.as_tensor(target_order, device=source.device)
            permutation[batch_index, source_indices] = target_indices[target_order]

    return permutation


def greedy_balanced_assignment(
    points: torch.Tensor,
    centers: torch.Tensor,
    capacities: torch.Tensor,
) -> torch.Tensor:
    costs = torch.cdist(points, centers.unsqueeze(0)).square()
    assignments = torch.empty(
        points.shape[0],
        points.shape[1],
        dtype=torch.long,
        device=points.device,
    )

    for batch_index in range(points.shape[0]):
        sorted_costs, preferences = costs[batch_index].sort(dim=1)
        margins = sorted_costs[:, 1] - sorted_costs[:, 0]
        point_order = margins.argsort(descending=True).cpu().tolist()
        preferences = preferences.cpu().tolist()
        remaining = capacities[batch_index].cpu().tolist()
        assignment = [-1] * points.shape[1]

        for point_index in point_order:
            for region in preferences[point_index]:
                if remaining[region] > 0:
                    assignment[point_index] = region
                    remaining[region] -= 1
                    break

        assignments[batch_index] = torch.tensor(
            assignment,
            device=points.device,
        )

    return assignments


@torch.no_grad()
def strict_target_guided_balanced_permutation(
    source: torch.Tensor,
    target: torch.Tensor,
    target_centers: torch.Tensor,
) -> torch.Tensor:
    num_regions = target_centers.shape[0]
    target_regions = torch.cdist(target, target_centers.unsqueeze(0)).argmin(dim=-1)
    capacities = F.one_hot(target_regions, num_regions).sum(dim=1)
    source_regions = greedy_balanced_assignment(
        source,
        target_centers,
        capacities,
    )

    return _local_hungarian_permutation(
        source,
        target,
        source_regions,
        target_regions,
        num_regions,
    )


@torch.no_grad()
def global_hungarian_permutation(
    source: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    costs = torch.cdist(source, target).square().cpu().numpy()
    permutation = torch.empty(
        source.shape[0],
        source.shape[1],
        dtype=torch.long,
        device=source.device,
    )

    for batch_index in range(source.shape[0]):
        _, target_indices = linear_sum_assignment(costs[batch_index])
        permutation[batch_index] = torch.as_tensor(
            target_indices,
            device=source.device,
        )

    return permutation
