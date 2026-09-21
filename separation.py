"""Soft separation-time constraints for fixed TG patches; no point movement.

For every eligible pair and EVERY within-patch bijection, the projected
conditional-path gap is at least (1-t)*gX + t*gY. This says nothing by itself
about the learned ODE or about anatomical gaps in the underlying shape.
"""
import math
from itertools import combinations

import torch


def separation_settings(options=None):
    settings = dict(time=0.5, margin_fraction=0.25, weight=10.0, min_target_gap=1e-6)
    options = {} if options is None else options
    if not isinstance(options, dict) or options.keys() - settings.keys():
        raise ValueError("Unknown separation settings")
    settings.update(options)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in settings.values()):
        raise ValueError("Separation settings must be finite numbers")
    if not (0 <= settings["time"] < 1 and 0 < settings["margin_fraction"] <= 1
            and settings["weight"] >= 0 and settings["min_target_gap"] >= 0):
        raise ValueError("Require 0<=time<1, 0<margin_fraction<=1, weight>=0, min_target_gap>=0")
    return settings


def projected_extrema(points, labels, k, ell, normal):
    projection = (points * normal[:, None]).sum(-1)
    left = projection.masked_fill(labels != k, -torch.inf).amax(1)
    right = projection.masked_fill(labels != ell, torch.inf).amin(1)
    return projection, left, right


def separation_penalty(source, target, anchors, baseline_labels, target_labels, settings):
    """Stream all unordered anchor pairs; keep an N x K unary cost, not N x N.

    b is frozen from baseline TG, NOT recomputed after the corrected solve.
    Penalties are summed over eligible pairs (not averaged over the pair count).
    """
    batch, n, _ = source.shape
    k_count = anchors.shape[1]
    penalty = source.new_zeros(batch, n, k_count)
    state = []
    for k, ell in combinations(range(k_count), 2):
        direction = anchors[:, ell] - anchors[:, k]
        length = direction.norm(dim=-1)
        normal = direction / length.clamp_min(torch.finfo(source.dtype).eps)[:, None]
        _, target_left, target_right = projected_extrema(target, target_labels, k, ell, normal)
        gap = target_right - target_left
        eligible = (length > torch.finfo(source.dtype).eps) & (gap > settings["min_target_gap"])
        projection, left, right = projected_extrema(source, baseline_labels, k, ell, normal)
        threshold = (left + right) / 2
        margin = settings["margin_fraction"] * gap
        required = (margin - settings["time"] * gap) / (1 - settings["time"])
        penalty[:, :, k] += (projection - (threshold - required / 2)[:, None]).clamp_min(0).square() * eligible[:, None]
        penalty[:, :, ell] += ((threshold + required / 2)[:, None] - projection).clamp_min(0).square() * eligible[:, None]
        state.append(dict(patches=(k, ell), normal=normal, eligible=eligible, target_gap=gap,
                          threshold=threshold, margin=margin, required_source_gap=required))
    return penalty, state


def separation_report(source, baseline_labels, corrected_labels, base_cost, penalty, state, settings,
                      corrected_solve_clouds):
    """Measured finite-set bounds, including failures; all values are JSON-safe."""
    batch = source.shape[0]
    pairs, eligible_clouds = [], set()
    field_names = ("source_gap", "gap_at_time", "separation_time", "margin_time", "margin_deficit",
                   "left_violation", "right_violation", "margin_satisfied")
    for pair in state:
        k, ell = pair["patches"]
        gap, margin, required, threshold = (pair[key] for key in
            ("target_gap", "margin", "required_source_gap", "threshold"))
        stages = []
        for labels in (baseline_labels, corrected_labels):
            _, left, right = projected_extrema(source, labels, k, ell, pair["normal"])
            source_gap = right - left
            at_time = (1 - settings["time"]) * source_gap + settings["time"] * gap
            # Denominators are strictly positive whenever the corresponding
            # branch is used: eligible gY>0, and m<=gY.
            denominator = (gap - source_gap).clamp_min(torch.finfo(source.dtype).tiny)
            t_sep = torch.where(source_gap < 0, -source_gap / denominator, 0)
            t_margin = torch.where(source_gap < margin, (margin - source_gap) / denominator, 0)
            stages.append(torch.stack((source_gap, at_time, t_sep, t_margin,
                (margin - at_time).clamp_min(0), (left - threshold + required / 2).clamp_min(0),
                (threshold + required / 2 - right).clamp_min(0), (at_time >= margin).to(source.dtype)), -1))
        values = torch.cat((gap[:, None], margin[:, None], required[:, None], threshold[:, None],
                            pair["normal"], *stages), -1).detach().cpu().tolist()
        dim = source.shape[-1]
        for cloud in pair["eligible"].nonzero().flatten().tolist():
            eligible_clouds.add(cloud)
            row = values[cloud]
            record = dict(cloud=cloud, patches=[k, ell], target_gap=row[0], margin=row[1],
                          required_source_gap=row[2], threshold=row[3], normal=row[4:4 + dim])
            for index, stage in enumerate(("baseline", "corrected")):
                offset = 4 + dim + index * len(field_names)
                record[stage] = dict(zip(field_names, row[offset:offset + len(field_names)]))
                record[stage]["margin_satisfied"] = bool(record[stage]["margin_satisfied"])
            pairs.append(record)
    summary = dict(clouds=batch, candidate_pairs=batch * len(state), eligible_pairs=len(pairs),
                   clouds_without_pairs=batch - len(eligible_clouds),
                   changed_source_fraction=(baseline_labels != corrected_labels).float().mean().item(),
                   corrected_solve_clouds=corrected_solve_clouds)
    summary.update(
        baseline_violating_pairs=sum(not p["baseline"]["margin_satisfied"] for p in pairs),
        resolved_pairs=sum(not p["baseline"]["margin_satisfied"] and p["corrected"]["margin_satisfied"] for p in pairs),
        newly_violating_pairs=sum(p["baseline"]["margin_satisfied"] and not p["corrected"]["margin_satisfied"] for p in pairs))
    for stage, labels in (("baseline", baseline_labels), ("corrected", corrected_labels)):
        rows = [p[stage] for p in pairs]
        summary[stage] = {name + "_mean": sum(r[name] for r in rows) / len(rows) if rows else None
                          for name in field_names[:5]}
        summary[stage].update(
            margin_satisfied_fraction=sum(r["margin_satisfied"] for r in rows) / len(rows) if rows else None,
            max_side_violation=max((max(r["left_violation"], r["right_violation"]) for r in rows), default=None),
            mean_base_cost=base_cost.gather(2, labels.unsqueeze(-1)).mean().item(),
            mean_penalty=penalty.gather(2, labels.unsqueeze(-1)).mean().item())
    return dict(settings=settings, summary=summary, pairs=pairs,
                scope="Worst projected gaps over all local random bijections for fixed empirical clouds; not learned ODE gaps.",
                objective_definition="Minimize D + weight*R; mean_base_cost is D/N, mean_penalty is unweighted R/N. Pair statistics pool eligible pairs, not clouds.",
                separation_time_definition="Zero-gap boundary; when gX<=0 strict separation holds only AFTER this time.")
