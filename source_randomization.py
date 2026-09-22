"""Finite random swap walk inside a per-cloud TG transport-cost budget.

This is NOT an exact uniform/Gibbs sampler over feasible assignments. Every
accepted swap preserves patch counts, regardless of whether the walk mixes.
"""
import math

import numpy as np
import torch


def randomization_settings(options=None):
    settings = dict(relative_budget=0.01, proposal_sweeps=4)
    options = {} if options is None else options
    if not isinstance(options, dict) or options.keys() - settings.keys():
        raise ValueError("Unknown source_randomization settings")
    settings.update(options)
    budget, sweeps = settings["relative_budget"], settings["proposal_sweeps"]
    if (isinstance(budget, bool) or not isinstance(budget, (int, float))
            or not math.isfinite(budget) or budget < 0):
        raise ValueError("relative_budget must be finite and nonnegative")
    if isinstance(sweeps, bool) or not isinstance(sweeps, int) or sweeps < 0:
        raise ValueError("proposal_sweeps must be a nonnegative integer")
    return settings


def randomize_assignment(costs, baseline, settings, *, generator=None, diagnostics=None):
    """CPU float64 costs [B,N,K]; vectorized over clouds, N proposals per sweep.

    The caller supplies an isolated CPU torch.Generator. Budget=0 or sweeps=0
    is an exact baseline control, including in the presence of cost ties.
    """
    if generator is not None and generator.device.type != "cpu":
        raise ValueError("assignment_generator must be a CPU torch.Generator")
    labels = baseline.copy()
    batch, n, _ = costs.shape
    rows, points = np.arange(batch), np.arange(n)
    initial = costs[rows[:, None], points, baseline].sum(1)
    current = initial.copy()
    limit = initial * (1 + settings["relative_budget"])
    if not np.isfinite(limit).all():
        raise ValueError("Nonfinite source assignment cost budget")
    accepted, cross_patch = np.zeros(batch, dtype=int), np.zeros(batch, dtype=int)
    sweeps = settings["proposal_sweeps"] if settings["relative_budget"] > 0 and n > 1 else 0
    for _ in range(sweeps):
        proposals = torch.randint(n, (n, batch, 2), generator=generator, device="cpu").numpy()
        for proposal in proposals:
            i, j = proposal.T
            a, b = labels[rows, i].copy(), labels[rows, j].copy()
            different = a != b
            cross_patch += different
            delta = costs[rows, i, b] + costs[rows, j, a] - costs[rows, i, a] - costs[rows, j, b]
            take = different & (current + delta <= limit)
            labels[rows[take], i[take]] = b[take]
            labels[rows[take], j[take]] = a[take]
            current[take] += delta[take]
            accepted += take
        # Recompute rather than accumulate floating-point error across sweeps.
        current = costs[rows[:, None], points, labels].sum(1)
    final = costs[rows[:, None], points, labels].sum(1)
    tolerance = 64 * np.finfo(np.float64).eps * np.maximum(1, np.abs(initial))
    if np.any(final > limit + tolerance):
        raise RuntimeError("Randomized assignment exceeded its cost budget")
    if diagnostics is not None:
        relative = np.divide(final - initial, initial, out=np.zeros(batch), where=initial > 0)
        changed = (labels != baseline).mean(1)
        diagnostics.update(settings=settings, clouds=batch, proposals_per_cloud=sweeps * n,
            mean_changed_fraction=float(changed.mean()), mean_accepted_swaps=float(accepted.mean()),
            mean_relative_cost_increase=float(relative.mean()), max_relative_cost_increase=float(relative.max()),
            per_cloud=[dict(baseline_mean_cost=float(initial[b] / n), final_mean_cost=float(final[b] / n),
                allowed_mean_cost=float(limit[b] / n), relative_cost_increase=float(relative[b]),
                accepted_swaps=int(accepted[b]), cross_patch_proposals=int(cross_patch[b]),
                changed_fraction=float(changed[b]), numerical_tolerance=float(tolerance[b] / n))
                for b in range(batch)],
            definition="Per-cloud source-to-centroid squared cost budget; finite random walk, not exact uniform sampling. Not a quality guarantee.")
    return labels
