"""Target-only capacity ranking; never moves/resamples source or target points."""

import math

import torch


CAPACITY_MODES = {
    "target_guided_capacity_large": "large",
    "target_guided_capacity_small": "small",
    "target_guided_capacity_random": "random",
}


def capacity_profile(n, k, profile=None):
    """One positive integer multiset, shared by all three adaptive conditions."""
    if not 1 <= k <= n:
        raise ValueError("Capacity profile requires 1 <= K <= N")
    if profile is None:
        # Reserve one point per patch, then use deterministic largest remainders.
        weights = [0.5 + i / (k - 1) for i in range(k)] if k > 1 else [1.0]
        quotas = [(n - k) * w / sum(weights) for w in weights]
        profile = [1 + math.floor(q) for q in quotas]
        order = sorted(range(k), key=lambda i: (-(quotas[i] % 1), i))
        for i in order[:n - sum(profile)]:
            profile[i] += 1
    if (len(profile) != k or any(type(v) is not int or v < 1 for v in profile)
            or sum(profile) != n):
        raise ValueError("capacity.profile must contain K positive integers summing to N")
    return sorted(profile)


def _directions(dim, count, reference):
    if type(count) is not int or count < 2:
        raise ValueError("num_directions must be an integer >= 2")
    i = torch.arange(count, device=reference.device, dtype=reference.dtype)
    if dim == 2:
        # Power is identical at +/- frequency: sample a half circle.
        angle = math.pi * i / count
        return torch.stack((angle.cos(), angle.sin()), -1)
    if dim == 3:
        z = 1 - 2 * (i + 0.5) / count
        angle = math.pi * (3 - math.sqrt(5)) * i
        radius = (1 - z.square()).sqrt()
        return torch.stack((radius * angle.cos(), radius * angle.sin(), z), -1)
    raise ValueError("Spectral capacity scoring supports 2D and 3D coordinates")


@torch.no_grad()
def local_spectral_scores(target, anchors, *, window_radius=0.35,
                          low_frequencies=(0.5, 1.0), high_frequencies=(2.0, 4.0),
                          num_directions=16, min_effective_points=8.0):
    """Gaussian-windowed, diagonal-subtracted band power ratio in spatial units.

    Fixed radius/frequencies are independent of the eventual target partition.
    Weighted pair power removes self-pairs, but is NOT an unbiased estimator of
    a continuum spectrum (windows/normalization depend on sampled points).
    Negative band estimates are clipped only AFTER averaging the whole band.
    Unreliable windows receive the mean reliable score (zero if none exist).
    """
    if (not math.isfinite(window_radius) or window_radius <= 0
            or not math.isfinite(min_effective_points) or min_effective_points < 2):
        raise ValueError("window_radius must be positive; min_effective_points must be >= 2")
    low, high = tuple(low_frequencies), tuple(high_frequencies)
    if (not low or not high or any(not math.isfinite(f) or f <= 0 for f in low + high)
            or max(low) >= min(high)):
        raise ValueError("Frequency bands must be positive, finite and ordered: low < high")
    if (target.ndim != 3 or anchors.ndim != 3 or target.shape[0] != anchors.shape[0]
            or target.shape[-1] != anchors.shape[-1] or target.shape[1] < 1):
        raise ValueError("Expected target [B,N,D] and anchors [B,K,D]")
    dtype = torch.float64 if target.dtype == torch.float64 else torch.float32
    points, centers = target.to(dtype), anchors.to(dtype)
    directions = _directions(points.shape[-1], num_directions, points)
    bands = points.new_tensor(low + high)
    frequencies = (bands[:, None, None] * directions[None]).reshape(-1, points.shape[-1])
    # Centering only stabilizes phase arithmetic; the power is translation invariant.
    phase = 2 * math.pi * (points - points.mean(1, keepdim=True)) @ frequencies.T
    weights = torch.softmax(-torch.cdist(centers, points).square() / (2 * window_radius**2), -1)
    diagonal = weights.square().sum(-1)
    real, imag = weights @ phase.cos(), weights @ phase.sin()
    eps = torch.finfo(dtype).eps
    pair_power = (real.square() + imag.square() - diagonal.unsqueeze(-1)) / (
        1 - diagonal).clamp_min(eps).unsqueeze(-1)
    boundary = len(low) * num_directions
    low_power = pair_power[..., :boundary].mean(-1).clamp_min(0)
    high_power = pair_power[..., boundary:].mean(-1).clamp_min(0)
    score = high_power / (low_power + high_power).clamp_min(eps)
    reliable = (diagonal.reciprocal() >= min_effective_points) & ((low_power + high_power) > eps)
    neutral = (score * reliable).sum(-1, keepdim=True) / reliable.sum(-1, keepdim=True).clamp_min(1)
    return torch.where(reliable, score, neutral)


def ranked_capacities(scores, profile, mode, *, generator=None):
    """Permute a single multiset; random tie order is independent of point order."""
    if mode not in ("large", "small", "random"):
        raise ValueError(f"Unknown capacity mode: {mode}")
    if scores.ndim != 2 or not torch.isfinite(scores).all():
        raise ValueError("Capacity scores must be finite [B,K]")
    values = torch.tensor(sorted(profile), dtype=torch.long, device=scores.device)
    if len(values) != scores.shape[1]:
        raise ValueError("Capacity profile length must equal the anchor count")
    if mode == "small":
        values = values.flip(0)
    capacities = torch.empty_like(scores, dtype=torch.long)
    for b, score in enumerate(scores):
        order = torch.randperm(len(values), device=scores.device, generator=generator)
        if mode != "random":
            order = order[score[order].argsort(stable=True)]
        capacities[b, order] = values
    return capacities


@torch.no_grad()
def adaptive_capacities(target, anchors, mode, *, options=None, generator=None):
    options = dict(options or {})
    profile = capacity_profile(target.shape[1], anchors.shape[1], options.pop("profile", None))
    scores = local_spectral_scores(target, anchors, **options)
    return ranked_capacities(scores, profile, mode, generator=generator)
