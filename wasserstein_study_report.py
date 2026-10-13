"""Read-only, paired analysis of an all-candidate Wasserstein-bank study.

Scores are taken only from the selection bank.  The final learned-model
endpoints use the independent validation bank and the normal generation
evaluator.  This module never trains a model or changes a saved experiment.
"""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import yaml


SCORES = ("transport", "displacement", "joint", "multiscale", "variance", "path_joint")
ENDPOINTS = ("final_validation_integrated_mse", "primary_chamfer")


def integrated_validation(points):
    """Trapezoidal time average; not an unweighted mean of nonuniform times."""
    if len(points) < 2:
        return None
    ordered = sorted((float(p["time"]), float(p["mse_per_coordinate"])) for p in points)
    times, values = np.asarray(ordered, dtype=float).T
    if not np.isfinite(times).all() or not np.isfinite(values).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Validation times and MSE must be finite, with distinct increasing times")
    if times[0] < 0 or times[-1] > 1 or np.any(values < 0):
        raise ValueError("Validation times must be in [0, 1] and MSE nonnegative")
    return float(np.trapz(values, times) / (times[-1] - times[0]))


def rank_correlation(x, y, control=None):
    """Spearman, or residual rank correlation after controlling one score.

    Constant ranks (including an exactly collinear score/control) return None;
    a numerical zero or a p-value would be misleading in that case.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Correlation inputs must be equal-length vectors")
    valid = np.isfinite(x) & np.isfinite(y)
    if control is not None:
        control = np.asarray(control, dtype=float)
        if control.shape != x.shape:
            raise ValueError("Control must have the same shape")
        valid &= np.isfinite(control)
    if int(valid.sum()) < (4 if control is not None else 3):
        return None
    xr, yr = rankdata(x[valid]), rankdata(y[valid])
    if control is not None:
        design = np.column_stack((np.ones(valid.sum()), rankdata(control[valid])))
        xr -= design @ np.linalg.lstsq(design, xr, rcond=None)[0]
        yr -= design @ np.linalg.lstsq(design, yr, rcond=None)[0]
    xr, yr = xr - xr.mean(), yr - yr.mean()
    scale = float(np.linalg.norm(xr) * np.linalg.norm(yr))
    if scale < 1e-10:
        return None
    return float(np.clip(np.dot(xr, yr) / scale, -1, 1))


def _path(value, base):
    path = Path(value)
    return path if path.is_absolute() else base / path


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _training_steps(job, child, base):
    config = _path(job["config"], base)
    expected_hash = job.get("config_sha256")
    if expected_hash is not None and hashlib.sha256(config.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Job config hash changed; saved study settings must not be edited")
    if child.get("training_steps") is not None:
        return int(child["training_steps"])
    with config.open(encoding="utf-8") as stream:
        return int(yaml.safe_load(stream)["training"]["num_steps"])


def _job_outcome(job, child, child_path, scores, nfe, fraction):
    base = child_path.parent
    candidate = int(job["candidate_id"])
    steps = _training_steps(job, child, base)
    row = {
        "dataset": child["dataset"], "bank_seed": int(child["bank_seed"]),
        "candidate_id": candidate, "training_seed": int(job["seed"]),
        "precompute_seconds_shared": _finite(child.get("precompute_seconds")),
        "training_steps": steps, "primary_nfe": nfe, "trained": False,
        "final_step": None, "final_validation_integrated_mse": None,
        "early_step_limit": int(math.floor(steps * fraction)), "early_step": None,
        "early_validation_integrated_mse": None, "early_elapsed_seconds": None,
        "validation_time_min": None, "validation_time_max": None,
        "primary_chamfer": None, "training_seconds": None, "inference_seconds": None,
        "validation_seconds": None, "training_seconds_excluding_validation": None,
        "quality_metrics_by_nfe_json": "{}",
        "inference_seconds_by_nfe_json": "{}",
        **{f"score_{key}": _finite(scores[candidate].get(key)) for key in SCORES},
    }
    run_config = job.get("run_config")
    if run_config:
        run_path = _path(run_config, base)
        row["trained"] = run_path.is_file()
        training_path = run_path.parent / "training.json"
        if training_path.is_file():
            training = _read_json(training_path)
            row["training_seconds"] = _finite(training.get("training_seconds"))
            for key in ("validation_seconds", "training_seconds_excluding_validation"):
                row[key] = _finite(training.get("coupling_details", {}).get(key))
        history_path = run_path.parent / "learning_curve.json"
        if history_path.is_file():
            history = _read_json(history_path)["history"]
            valid = [h for h in history if h.get("validation_by_time")]
            if valid:
                final = max(valid, key=lambda h: h["step"])
                row["final_step"] = int(final["step"])
                row["final_validation_integrated_mse"] = integrated_validation(final["validation_by_time"])
                row["validation_time_min"] = min(float(p["time"]) for p in final["validation_by_time"])
                row["validation_time_max"] = max(float(p["time"]) for p in final["validation_by_time"])
                # Never use a later checkpoint when logging missed the cutoff.
                eligible = [h for h in valid if 0 < h["step"] <= row["early_step_limit"]]
                if eligible:
                    early = max(eligible, key=lambda h: h["step"])
                    row["early_step"] = int(early["step"])
                    row["early_validation_integrated_mse"] = integrated_validation(early["validation_by_time"])
                    row["early_elapsed_seconds"] = _finite(early.get("elapsed_seconds"))
    quality_by_nfe, inference_by_nfe = {}, {}
    non_quality = {"training_seconds", "inference_seconds", "evaluation_seed", "evaluation_batch_size",
                   "n_points", "total_points", "euler_steps", "histogram_bins", "training_config_verified"}
    for steps_key, result_path in job.get("evaluations", {}).items():
        if not _path(result_path, base).is_file():
            continue
        result = _read_json(_path(result_path, base))
        quality = {key: _finite(value) for key, value in result.items()
                   if key not in non_quality and isinstance(value, (int, float)) and not isinstance(value, bool)}
        if "cell_mass_error" in result and result["cell_mass_error"] is None:
            quality["cell_mass_error"] = None
        quality_by_nfe[str(steps_key)] = quality
        inference_by_nfe[str(steps_key)] = _finite(result.get("inference_seconds"))
        if int(steps_key) != nfe:
            continue
        for output, key in (("primary_chamfer", "chamfer"), ("training_seconds", "training_seconds"),
                            ("inference_seconds", "inference_seconds")):
            value = _finite(result.get(key))
            if value is not None:
                row[output] = value
        for key in ("validation_seconds", "training_seconds_excluding_validation"):
            value = _finite(result.get("coupling_details", {}).get(key))
            if value is not None:
                row[key] = value
        row.update({f"primary_quality_{key}": value for key, value in quality.items()})
    row["quality_metrics_by_nfe_json"] = json.dumps(quality_by_nfe, sort_keys=True, allow_nan=False)
    row["inference_seconds_by_nfe_json"] = json.dumps(inference_by_nfe, sort_keys=True, allow_nan=False)
    return row


def _complete_row(row):
    return (row["trained"] and row["final_step"] == row["training_steps"]
            and all(row[key] is not None for key in ENDPOINTS)
            and row["validation_time_min"] == 0.0 and row["validation_time_max"] == 1.0)


def _write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize_study(path, manifest):
    """Write candidate_outcomes.csv, selector_regret.csv, and predictivity.json.

    ``path`` is the root manifest filename.  Its ``children`` contain dataset,
    bank_seed and child-manifest path. Declared bank_seeds/training_seeds and
    per-child planned_candidate_ids make incomplete coverage explicit.
    """
    path = Path(path).resolve()
    nfe = int(manifest.get("primary_nfe", 128))
    fraction = float(manifest.get("early_fraction", 0.05))
    if nfe < 1 or not 0 < fraction < 1:
        raise ValueError("Primary NFE must be positive and early fraction in (0, 1)")
    outcomes, groups, correlations, regret_rows, preparation_failures = [], [], [], [], []
    group_keys = set()
    for entry in manifest.get("children", []):
        if entry.get("manifest") is None:
            preparation_failures.append({"dataset": entry["dataset"], "bank_seed": entry["bank_seed"],
                                         "reason": "Child manifest unavailable; preparation is incomplete"})
            continue
        child_path = _path(entry["manifest"], path.parent)
        if not child_path.is_file():
            preparation_failures.append({"dataset": entry["dataset"], "bank_seed": entry["bank_seed"],
                                         "reason": "Declared child manifest file is missing"})
            continue
        child = _read_json(child_path)
        child.setdefault("bank_seed", entry["bank_seed"])
        if child["dataset"] != entry["dataset"] or int(child["bank_seed"]) != int(entry["bank_seed"]):
            raise ValueError("Child dataset/bank seed differs from study manifest")
        group_key = (child["dataset"], int(child["bank_seed"]))
        if group_key in group_keys:
            raise ValueError("Study has duplicate dataset/bank-seed children")
        group_keys.add(group_key)
        metadata = _read_json(child_path.parent / "bank_scores.json")
        score_rows = metadata["reports"]["selection"]["candidates"]
        scores = {int(row["candidate"]): row for row in score_rows}
        candidate_ids = sorted(int(i) for i in entry.get("planned_candidate_ids", child.get(
            "planned_candidate_ids", manifest.get("planned_candidate_ids", list(scores)))))
        seeds = sorted(int(i) for i in manifest.get("training_seeds", child.get(
            "training_seeds", sorted({job["seed"] for job in child["jobs"]}))))
        if len(set(candidate_ids)) != len(candidate_ids) or len(set(seeds)) != len(seeds):
            raise ValueError("Planned candidates and training seeds must be distinct")
        if not candidate_ids or not seeds or not set(candidate_ids).issubset(scores):
            raise ValueError("Study requires nonempty candidate/seed axes present in bank scores")
        rows, seen = [], set()
        for job in child["jobs"]:
            key = (int(job["candidate_id"]), int(job["seed"]))
            if key in seen:
                raise ValueError("Duplicate candidate/training-seed job")
            if key[0] not in candidate_ids or key[1] not in seeds:
                raise ValueError("Job is outside declared candidate/training-seed axes")
            seen.add(key)
            rows.append(_job_outcome(job, child, child_path, scores, nfe, fraction))
        expected = {(candidate, seed) for candidate in candidate_ids for seed in seeds}
        complete = seen == expected and all(_complete_row(row) for row in rows)
        group = {"dataset": child["dataset"], "bank_seed": int(child["bank_seed"]),
                 "planned_candidate_ids": candidate_ids, "training_seeds": seeds,
                 "expected_jobs": len(expected), "observed_jobs": len(rows),
                 "complete_jobs": sum(_complete_row(row) for row in rows), "complete": complete,
                 "missing_jobs": [{"candidate_id": c, "training_seed": s} for c, s in sorted(expected - seen)]}
        groups.append(group)
        outcomes.extend(rows)
        lookup = {(r["candidate_id"], r["training_seed"]): r for r in rows}
        available_scores = [key for key in SCORES if all(_finite(scores[c].get(key)) is not None for c in candidate_ids)]
        selected = {key: min(candidate_ids, key=lambda c: (scores[c][key], c)) for key in available_scores}
        random_id = metadata.get("selected", child.get("selected", {})).get("random")
        if random_id in candidate_ids:
            selected["random"] = int(random_id)
        for endpoint in ENDPOINTS:
            # A candidate is ranked only if every declared training seed exists.
            averaged = {c: float(np.mean([lookup[c, s][endpoint] for s in seeds]))
                        for c in candidate_ids if all((c, s) in lookup and lookup[c, s][endpoint] is not None
                                                     and _complete_row(lookup[c, s]) for s in seeds)}
            ids = sorted(averaged)
            for key in available_scores:
                transport = [scores[c]["transport"] for c in ids]
                correlations.append({"dataset": group["dataset"], "bank_seed": group["bank_seed"],
                                     "endpoint": endpoint, "score": key, "n_candidates": len(ids),
                                     "complete_candidate_coverage": len(ids) == len(candidate_ids),
                                     "spearman": rank_correlation([scores[c][key] for c in ids], [averaged[c] for c in ids]),
                                     "partial_rank_controlling_transport": rank_correlation(
                                         [scores[c][key] for c in ids], [averaged[c] for c in ids], transport)})
            for seed in seeds:
                if not all((c, seed) in lookup and _complete_row(lookup[c, seed]) for c in candidate_ids):
                    continue
                performance = {c: lookup[c, seed][endpoint] for c in candidate_ids}
                oracle = min(candidate_ids, key=lambda c: (performance[c], c))
                seed_selected = {**selected, "oracle": oracle}
                early_rows = [lookup[c, seed] for c in candidate_ids]
                if all(row["early_validation_integrated_mse"] is not None for row in early_rows):
                    seed_selected["early_validation"] = min(candidate_ids, key=lambda c: (
                        lookup[c, seed]["early_validation_integrated_mse"], c))
                for selector, chosen in seed_selected.items():
                    cost_seconds = None
                    cost_steps = 0
                    if selector == "early_validation":
                        cost_steps = sum(row["early_step"] for row in early_rows)
                        if all(row["early_elapsed_seconds"] is not None for row in early_rows):
                            cost_seconds = sum(row["early_elapsed_seconds"] for row in early_rows)
                    regret_rows.append({"dataset": group["dataset"], "bank_seed": group["bank_seed"],
                                        "training_seed": seed, "endpoint": endpoint, "selector": selector,
                                        "candidate_id": chosen, "value": performance[chosen], "oracle_candidate_id": oracle,
                                        "oracle_value": performance[oracle], "regret": performance[chosen] - performance[oracle],
                                        "early_total_optimizer_updates": cost_steps, "early_total_online_seconds": cost_seconds})
    declared_banks = sorted(int(s) for s in manifest.get("bank_seeds", sorted({g["bank_seed"] for g in groups})))
    declared_datasets = manifest.get("datasets", sorted({g["dataset"] for g in groups}))
    primary_scores = manifest.get("claim", {}).get("primary_scores", ["joint", "path_joint"])
    if not primary_scores or any(score not in SCORES[1:] for score in primary_scores):
        raise ValueError("Declare primary scores from the non-transport bank score set")
    decisions = []
    for dataset in declared_datasets:
        dataset_groups = [g for g in groups if g["dataset"] == dataset]
        coverage_complete = (set(g["bank_seed"] for g in dataset_groups) == set(declared_banks)
                             and len(declared_banks) > 0 and all(g["complete"] for g in dataset_groups))
        for score in SCORES[1:]:
            endpoint_checks = []
            for endpoint in ENDPOINTS:
                checks = []
                for bank_seed in declared_banks:
                    target = [r for r in regret_rows if r["dataset"] == dataset and r["bank_seed"] == bank_seed
                              and r["endpoint"] == endpoint and r["selector"] == score]
                    baseline = [r for r in regret_rows if r["dataset"] == dataset and r["bank_seed"] == bank_seed
                                and r["endpoint"] == endpoint and r["selector"] == "transport"]
                    corr = next((r for r in correlations if r["dataset"] == dataset and r["bank_seed"] == bank_seed
                                 and r["endpoint"] == endpoint and r["score"] == score), None)
                    paired = bool(target and baseline and {r["training_seed"] for r in target} == {r["training_seed"] for r in baseline})
                    own_mean = float(np.mean([r["regret"] for r in target])) if paired else None
                    base_mean = float(np.mean([r["regret"] for r in baseline])) if paired else None
                    checks.append({"bank_seed": bank_seed, "mean_regret": own_mean, "transport_mean_regret": base_mean,
                                   "improves_transport": own_mean < base_mean if paired else None,
                                   "positive_partial_rank": (corr["partial_rank_controlling_transport"] > 0
                                                             if corr and corr["partial_rank_controlling_transport"] is not None else None)})
                endpoint_checks.append({"endpoint": endpoint, "banks": checks,
                                        "improved_bank_count": sum(c["improves_transport"] is True for c in checks),
                                        "positive_partial_rank_all_banks": bool(checks) and all(c["positive_partial_rank"] is True for c in checks)})
            # The *same* bank seeds must satisfy both metrics and both conditions;
            # disjoint wins on the two endpoints do not pass this rule.
            qualifying_banks = [bank for index, bank in enumerate(declared_banks)
                                if all(check["banks"][index]["improves_transport"] is True
                                       and check["banks"][index]["positive_partial_rank"] is True
                                       for check in endpoint_checks)]
            eligible = (coverage_complete and len(declared_banks) >= 3 and not manifest.get("smoke", False)
                        and manifest.get("study_mode", manifest.get("mode")) not in ("pilot", "smoke"))
            passed = eligible and len(qualifying_banks) >= math.ceil(2 * len(declared_banks) / 3)
            primary = score in primary_scores
            status = ("secondary_only" if not primary else "passed_provisional_gate" if passed else
                      "not_supported" if eligible else "incomplete" if not coverage_complete else "pilot_only")
            decisions.append({"dataset": dataset, "score": score, "status": status,
                              "role": "primary" if primary else "secondary", "qualifying_bank_seeds": qualifying_banks,
                              "coverage_complete": coverage_complete, "endpoint_checks": endpoint_checks})
    report = {
        "format": "wasserstein_bank_predictivity_v1", "primary_nfe": nfe, "early_fraction": fraction,
        "primary_endpoints": list(ENDPOINTS), "primary_scores": primary_scores,
        "groups": groups, "preparation_failures": preparation_failures,
        "correlations": correlations, "decisions": decisions,
        "interpretation": {
            "lower_is_better": True, "positive_score_performance_rank_is_expected": True,
            "selector_scope": "Selection-bank scores only; ties choose the lowest candidate ID. No retraining per selector.",
            "oracle_scope": "Retrospective endpoint-specific best candidate per paired training seed; not deployable.",
            "early_selector_cost": f"All candidates must reach <={100 * fraction:g}% checkpoint; summed online seconds include validation. Not a free pre-training score.",
            "early_selector_caveat": "Early and final validation reuse the holdout bank; generation CD is a separate endpoint. Adaptive reuse can bias validation selection.",
            "integration": "Trapezoidal time average; confirm full [0,1] coverage for a complete primary endpoint.",
            "gate": "Only predeclared primary scores; at least 3 declared bank seeds and complete paired coverage. The same >=2/3 bank seeds must have lower regret than transport AND positive partial-rank correlation for BOTH primary endpoints. Other scores are secondary only.",
            "statistics": "Exploratory ranks and provisional effect gate; no p-values, independence assumption, or population generalization guarantee.",
        },
    }
    candidate_columns = list(dict.fromkeys(key for row in outcomes for key in row)) if outcomes else ["dataset", "bank_seed", "candidate_id", "training_seed"]
    regret_columns = list(regret_rows[0]) if regret_rows else ["dataset", "bank_seed", "training_seed", "endpoint", "selector", "regret"]
    _write_csv(path.parent / "candidate_outcomes.csv", outcomes, candidate_columns)
    _write_csv(path.parent / "selector_regret.csv", regret_rows, regret_columns)
    with (path.parent / "predictivity.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report
