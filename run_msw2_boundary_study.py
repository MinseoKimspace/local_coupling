"""Matched A/B/C/D MS-W2 study; preparation only unless --stage is explicit.

A: original coupling, uniform time loss. B: original coupling, late-time loss.
C: overlap-window coupling, uniform loss. D: overlap coupling, late-time loss.
All paths in this runner's manifests/configs are relative to the repository.
"""

import argparse
import copy
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml

from nsot import file_sha256


ROOT = Path(__file__).resolve().parent
FORMAT = "msw2_boundary_factorial_v1"
BASE = "multiscale_wasserstein"
OVERLAP = "multiscale_overlap"
ARMS = ("A", "B", "C", "D")
NFES = [1, 2, 4, 8, 16, 32, 128]
LATE = {"mode": "late", "threshold": 0.75, "multiplier": 2.0}


@contextmanager
def project_directory():
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        yield
    finally:
        os.chdir(previous)


def absolute(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def relative(path):
    return Path(os.path.relpath(absolute(path), ROOT)).as_posix()


def read_yaml(path):
    with absolute(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected configuration mapping: {path}")
    return config


def read_json(path):
    with absolute(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def save_manifest(path, value):
    path = absolute(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def artifact(path):
    return {"path": relative(path), "sha256": file_sha256(absolute(path))}


def verify_artifact(item, label):
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise ValueError(f"Invalid {label} artifact")
    path = absolute(item["path"])
    if not path.is_file() or file_sha256(path) != item.get("sha256"):
        raise ValueError(f"{label} changed or missing: {item['path']}")


def integers(values, label, minimum=0):
    if (not isinstance(values, list) or not values or len(values) != len(set(values)) or
            any(isinstance(v, bool) or not isinstance(v, int) or not minimum <= v < 2**32 - 2
                for v in values)):
        raise ValueError(f"{label} requires distinct integers in [{minimum}, 2**32-2)")
    return values


def _dataset_names(values):
    if (not values or len(values) != len(set(values)) or
            any(value not in ("checkerboard", "horse") for value in values)):
        raise ValueError("Choose distinct checkerboard/horse datasets")
    return values


def _resolve_existing_cache(value, config_path):
    candidate = Path(value)
    if candidate.is_absolute():
        candidates = [candidate]
    else:
        candidates = [ROOT / candidate, absolute(config_path).parent / candidate]
    for path in candidates:
        if path.is_file():
            return relative(path)
    raise FileNotFoundError(
        f"Base cache not found: {value}. Supply the lab repository's existing run/config.yaml; "
        "foreign-machine absolute paths are not rewritten or guessed.")


def create_experiment(*, datasets=None, seeds=None, bank_seed=0, device=None, steps=None,
                      smoke=False, nfe=None, base_checkerboard_config=None,
                      base_horse_config=None, output_root=None):
    import multiscale_wasserstein as msw
    import multiscale_overlap as overlap
    datasets = _dataset_names(["checkerboard", "horse"] if datasets is None else list(datasets))
    seeds = integers([0] if seeds is None else list(seeds), "seeds")
    integers([bank_seed], "bank_seed")
    nfes = integers(([1, 2] if smoke else NFES.copy()) if nfe is None else list(nfe), "nfe", 1)
    if steps is not None:
        integers([steps], "steps", 1)
    if smoke and any(value is not None for value in (device, steps, base_checkerboard_config, base_horse_config)):
        raise ValueError("--smoke supplies tiny CPU settings and cannot reuse full-size base configurations")
    directory = absolute(output_root or "eval_results/msw2_boundary_study") / (
        f"study_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:8]}")
    provided = {"checkerboard": base_checkerboard_config, "horse": base_horse_config}
    if any(provided[dataset] is not None and dataset not in datasets for dataset in provided):
        raise ValueError("A supplied base configuration must belong to a selected --datasets entry")
    proposed, origins = [], {}
    # Validate all settings and all supplied bank artifacts before creating files.
    for dataset in datasets:
        config_path = provided[dataset] or f"multiscale_experiments/{dataset}.yaml"
        config = read_yaml(config_path)
        if config.get("dataset") != dataset or config.get("coupling") != BASE:
            raise ValueError(f"{dataset}: base configuration must be original MS-W2")
        if config.get("training", {}).get("time_weighting", {"mode": "uniform"}) != {"mode": "uniform"}:
            raise ValueError("Base configuration must use uniform time weighting")
        config.pop(OVERLAP, None)
        options = config[BASE]
        if provided[dataset]:
            if options.get("cache_seed", 0) != bank_seed:
                raise ValueError("Supplied base bank cache_seed differs from --bank-seed")
            options["cache"] = _resolve_existing_cache(options["cache"], config_path)
            with project_directory():
                _, _, _, _, digest = msw.load_cache(config, dataset)
            options["cache_sha256"] = digest
            origins[dataset] = {"config": artifact(config_path), "cache": artifact(options["cache"])}
        else:
            options["cache"] = relative(directory / "banks" / f"{dataset}_base.npz")
            options["cache_seed"] = bank_seed
            options.pop("cache_sha256", None)
        if device is not None:
            config["device"] = device
        if steps is not None:
            config["training"]["num_steps"] = steps
        if smoke:
            config["device"] = "cpu"
            config["data"].update(batch_size=2, n_points=16)
            config["model"].update(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
            config["training"].update(num_steps=2, log_every=1)
            config["evaluation"].update(batch_size=6, histogram_bins=16)
            options.update(superset_size=64, partition_counts=[2, 4, 8], scale_weights=[1.0] * 3)
        expected = (64, [2, 4, 8], 16) if smoke else (10000, [4, 20, 100], 256)
        if (options["superset_size"], options["partition_counts"], config["data"]["n_points"]) != expected:
            raise ValueError("This study fixes M/N/partitions to 10000/256/[4,20,100] (or smoke settings)")
        checked = msw.settings(config)
        if not np.allclose(checked["scale_weights"], [1 / 3] * 3, rtol=0, atol=1e-15):
            raise ValueError("This study requires equal scale weights")
        overlap_options = {
            "cache": relative(directory / "banks" / f"{dataset}_overlap.npz"),
            "window_sizes": [8, 16] if smoke else [64, 128],
            "max_windows_per_scale": 4 if smoke else 32,
            "sweeps": 1 if smoke else 2, "neighbors": 4 if smoke else 16,
            "max_local_solves": 1000,
        }
        for seed in seeds:
            for arm in ARMS:
                job_config = copy.deepcopy(config)
                job_config["seed"] = seed
                job_config["training"].pop("time_weighting", None)
                job_config["training"]["time_seed"] = seed + 2
                if arm in ("B", "D"):
                    job_config["training"]["time_weighting"] = dict(LATE)
                if arm in ("C", "D"):
                    job_config["coupling"] = OVERLAP
                    job_config[OVERLAP] = dict(overlap_options)
                    overlap.settings(job_config)
                job_config["checkpoint"] = f"{dataset}_msw2_boundary_{arm}_seed{seed}.pt"
                proposed.append((dataset, seed, arm, job_config))
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "configs").mkdir()
    jobs = []
    for dataset, seed, arm, config in proposed:
        config_path = directory / "configs" / f"{dataset}_{arm}_seed{seed}.yaml"
        with config_path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        jobs.append({"dataset": dataset, "seed": seed, "arm": arm,
                     "config": artifact(config_path), "base_bank": None, "bank": None,
                     "run": None, "evaluations": {}})
    manifest = {
        "format": FORMAT, "created_at": datetime.now(timezone.utc).isoformat(), "smoke": smoke,
        "datasets": datasets, "training_seeds": seeds, "bank_seed": bank_seed,
        "evaluation_nfes": nfes, "origins": origins, "endpoint_checks": {}, "jobs": jobs,
        "protocol": {
            "arms": {"A": "base/uniform", "B": "base/late", "C": "overlap/uniform", "D": "overlap/late"},
            "late_weighting": LATE,
            "rng": "same model seed; bank draw generator seed+1; independent time generator seed+2",
            "coupling": "same bank coordinates; only permutation changes; no noise or barycentric projection",
            "evaluation": "standard evaluators and fresh Gaussian; matched seed/model/data/dtype/evaluation settings",
            "resume": "completed artifacts reused; interrupted training restarts, not optimizer-state resume",
            "inference_timing": "integration only, excluding sampling/loading/warmup/metrics/rendering",
        },
    }
    path = directory / "manifest.json"
    save_manifest(path, manifest)
    print(f"experiment_manifest={relative(path)}", flush=True)
    print(f"planned_models={len(jobs)}; prepare stage does not train", flush=True)
    return path, manifest


def verify_job(job):
    verify_artifact(job["config"], "Prepared YAML")
    config = read_yaml(job["config"]["path"])
    arm = job["arm"]
    if arm not in ARMS or config.get("coupling") != (OVERLAP if arm in ("C", "D") else BASE):
        raise ValueError("Manifest/config arm or method mismatch")
    if config.get("dataset") != job["dataset"] or config.get("seed") != job["seed"]:
        raise ValueError("Manifest/config dataset or seed mismatch")
    if config["training"].get("time_seed") != job["seed"] + 2:
        raise ValueError("Unmatched training time RNG")
    weighting = config["training"].get("time_weighting")
    if weighting != (LATE if arm in ("B", "D") else None):
        raise ValueError("Manifest/config time weighting mismatch")
    for key, method in (("base_bank", BASE), ("bank", config["coupling"])):
        if job[key]:
            verify_artifact(job[key], "Prepared bank")
            if (relative(config[method]["cache"]) != job[key]["path"] or
                    config[method].get("cache_sha256") != job[key]["sha256"]):
                raise ValueError("Prepared YAML and manifest disagree about bank")
    if job["run"]:
        for item in job["run"].values():
            verify_artifact(item, "Training artifact")
    for evaluation in job["evaluations"].values():
        for item in evaluation.values():
            verify_artifact(item, "Evaluation artifact")
    return config


def verify_manifest(manifest):
    if manifest.get("format") != FORMAT:
        raise ValueError("Unsupported study manifest format")
    _dataset_names(manifest["datasets"])
    integers(manifest["training_seeds"], "training_seeds")
    integers(manifest["evaluation_nfes"], "evaluation_nfes", 1)
    integers([manifest["bank_seed"]], "bank_seed")
    for origin in manifest["origins"].values():
        for item in origin.values():
            verify_artifact(item, "Original base input")
    expected = {(dataset, seed, arm) for dataset in manifest["datasets"]
                for seed in manifest["training_seeds"] for arm in ARMS}
    actual = [(job["dataset"], job["seed"], job["arm"]) for job in manifest["jobs"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Manifest must contain exactly A/B/C/D for every dataset and seed")
    matched = {}
    shared = {}
    for job in manifest["jobs"]:
        config = verify_job(job)
        comparable = copy.deepcopy(config)
        comparable.pop("coupling")
        comparable.pop("checkpoint")
        comparable.pop(OVERLAP, None)
        comparable["training"].pop("time_weighting", None)
        comparable[BASE].pop("cache_sha256", None)  # Partial preparation is resumable.
        key = (job["dataset"], job["seed"])
        if key in matched and comparable != matched[key]:
            raise ValueError("Factorial arms do not have matched model/data/training/evaluation settings")
        matched[key] = comparable
        for method in (BASE, OVERLAP):
            if method in config:
                options = dict(config[method])
                options.pop("cache_sha256", None)
                key = (job["dataset"], method)
                if key in shared and options != shared[key]:
                    raise ValueError("Dataset arms/seeds must share one base and one overlap bank")
                shared[key] = options


def _bank_artifact(path, metadata):
    return {**artifact(path), "metadata": metadata}


def _prepared_yaml(job, config):
    original = absolute(job["config"]["path"])
    path = original.with_name(original.stem + "_prepared.yaml")
    if path.exists():
        if read_yaml(path) != config:
            raise ValueError(f"Interrupted prepared YAML disagrees with bank: {relative(path)}")
    else:
        with path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
    return artifact(path)


def prepare_jobs(path, manifest):
    import multiscale_wasserstein as msw
    import multiscale_overlap as overlap
    verify_manifest(manifest)  # Verify every existing artifact before any write.
    changed = False
    base_banks, overlap_banks = {}, {}
    for job in manifest["jobs"]:
        config = verify_job(job)
        if job["bank"]:
            continue
        dataset = job["dataset"]
        if dataset not in base_banks:
            base_config = copy.deepcopy(config)
            base_config["coupling"] = BASE
            with project_directory():
                base_path, metadata = msw.prepare(base_config, dataset)
                base_banks[dataset] = _bank_artifact(base_path, metadata)
        base_bank = base_banks[dataset]
        config[BASE]["cache_sha256"] = base_bank["sha256"]
        bank = base_bank
        if config["coupling"] == OVERLAP:
            if dataset not in overlap_banks:
                with project_directory():
                    overlap_path, metadata = overlap.prepare(config, dataset)
                    overlap_banks[dataset] = _bank_artifact(overlap_path, metadata)
            bank = overlap_banks[dataset]
            config[OVERLAP]["cache_sha256"] = bank["sha256"]
        prepared = _prepared_yaml(job, config)
        job.update(config=prepared, base_bank=base_bank, bank=bank)
        save_manifest(path, manifest)
        changed = True
    for dataset in manifest["datasets"]:
        jobs = [job for job in manifest["jobs"] if job["dataset"] == dataset]
        baseline = next(job for job in jobs if job["arm"] == "A")["bank"]
        candidate = next(job for job in jobs if job["arm"] == "C")["bank"]
        with np.load(absolute(baseline["path"]), allow_pickle=False) as left, \
                np.load(absolute(candidate["path"]), allow_pickle=False) as right:
            for name in ("source", "target"):
                if not np.array_equal(left[name], right[name]):
                    raise ValueError(f"{dataset}: baseline/overlap {name} endpoints differ")
            for archive in (left, right):
                if not np.array_equal(np.sort(archive["permutation"]), np.arange(len(archive["source"]))):
                    raise ValueError("Bank permutation is not bijective")
        checks = {"same_source": True, "same_target": True, "both_bijective": True,
                  "base_sha256": baseline["sha256"], "overlap_sha256": candidate["sha256"]}
        if manifest["endpoint_checks"].get(dataset) != checks:
            manifest["endpoint_checks"][dataset] = checks
            changed = True
    if changed:
        save_manifest(path, manifest)


def train_jobs(path, manifest):
    verify_manifest(manifest)
    if any(job["bank"] is None for job in manifest["jobs"]):
        raise ValueError("Prepare all banks before training")
    for job in manifest["jobs"]:
        if job["run"]:
            print(f"reused_completed_training={job['dataset']}/{job['arm']}/seed{job['seed']}", flush=True)
            continue
        if job["dataset"] == "horse":
            from train_horse import main as train
        else:
            from train import main as train
        with project_directory():
            directory = absolute(train(job["config"]["path"]))
        saved = read_yaml(directory / "config.yaml")
        job["run"] = {"config": artifact(directory / "config.yaml"),
                      "checkpoint": artifact(directory / saved["checkpoint"]),
                      "training_metadata": artifact(directory / "training.json")}
        save_manifest(path, manifest)


def evaluate_jobs(path, manifest):
    verify_manifest(manifest)
    if any(job["run"] is None for job in manifest["jobs"]):
        raise ValueError("Run --stage train/all before evaluation")
    for job in manifest["jobs"]:
        if job["dataset"] == "horse":
            from eval_horse import main as evaluate
        else:
            from eval import main as evaluate
        for nfe in manifest["evaluation_nfes"]:
            if str(nfe) in job["evaluations"]:
                continue
            with project_directory():
                image = absolute(evaluate(job["run"]["config"]["path"], nfe))
            job["evaluations"][str(nfe)] = {"json": artifact(image.with_suffix(".json")), "png": artifact(image)}
            save_manifest(path, manifest)


_NON_METRICS = {"training_seconds", "inference_seconds", "evaluation_seed", "evaluation_batch_size",
                "n_points", "total_points", "euler_steps", "histogram_bins", "nfe"}


def _metrics(result):
    # The standard evaluators store all metrics, including horse ROI metrics, at top level.
    return {key: value for key, value in result.items() if key not in _NON_METRICS
            and (value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)))
            and key not in ("metric_metadata", "training_environment", "image")}


def _evaluation_protocol(result):
    keys = ("dataset", "evaluation_seed", "evaluation_batch_size", "n_points", "total_points",
            "euler_steps", "histogram_bins", "metric_definitions", "metric_metadata")
    return {key: result.get(key) for key in keys}


def write_summary(path, manifest):
    verify_manifest(manifest)
    banks, base_banks, overlap_banks, rows, paired = {}, {}, {}, [], {}
    for job in manifest["jobs"]:
        for key in ("base_bank", "bank"):
            bank = job[key]
            if bank:
                banks[bank["path"]] = bank
        if job["base_bank"]:
            base_banks[job["base_bank"]["path"]] = job["base_bank"]
        if job["bank"] and job["arm"] in ("C", "D"):
            overlap_banks[job["bank"]["path"]] = job["bank"]
        training = read_json(job["run"]["training_metadata"]["path"]) if job["run"] else {}
        base_metadata = job["base_bank"]["metadata"] if job["base_bank"] else {}
        bank_metadata = job["bank"]["metadata"] if job["bank"] else {}
        for nfe in manifest["evaluation_nfes"]:
            evaluation = job["evaluations"].get(str(nfe))
            result = read_json(evaluation["json"]["path"]) if evaluation else {}
            metrics = _metrics(result)
            seconds = result.get("inference_seconds")
            batch = result.get("evaluation_batch_size")
            row = {"dataset": job["dataset"], "seed": job["seed"], "arm": job["arm"], "nfe": nfe,
                   "status": "evaluated" if evaluation else "trained" if job["run"] else "prepared" if job["bank"] else "planned",
                   "coupling": OVERLAP if job["arm"] in ("C", "D") else BASE,
                   "time_weighting": "late" if job["arm"] in ("B", "D") else "uniform",
                   "training_seconds": training.get("training_seconds"),
                   "inference_seconds": seconds,
                   "inference_ms_per_cloud": 1000 * seconds / batch if seconds is not None and batch else None,
                   "base_precompute_seconds": base_metadata.get("precompute_seconds"),
                   "overlap_incremental_seconds": (bank_metadata.get("precompute_seconds")
                                                   if job["arm"] in ("C", "D") else 0.0),
                   "overlap_nonleaf_cost_seconds": bank_metadata.get("nonleaf_cost_seconds"),
                   "overlap_local_assignment_seconds": bank_metadata.get("local_assignment_seconds"),
                   "overlap_objective_before": bank_metadata.get("objective_before"),
                   "overlap_objective_after": bank_metadata.get("objective_after"),
                   "overlap_changed_fraction": bank_metadata.get("changed_fraction"),
                   "overlap_cross_leaf_fraction": bank_metadata.get("cross_leaf_fraction"),
                   "overlap_local_solves": bank_metadata.get("local_solves"),
                   "evaluation_json": evaluation["json"]["path"] if evaluation else None,
                   "evaluation_png": evaluation["png"]["path"] if evaluation else None,
                   **metrics}
            rows.append(row)
            if evaluation:
                paired.setdefault((job["dataset"], job["seed"], nfe), {})[job["arm"]] = (metrics, _evaluation_protocol(result))
    contrasts = []
    formulas = {"B-A": {"B": 1, "A": -1}, "C-A": {"C": 1, "A": -1},
                "D-C": {"D": 1, "C": -1}, "D-B": {"D": 1, "B": -1},
                "interaction_D-C-B+A": {"D": 1, "C": -1, "B": -1, "A": 1}}
    for (dataset, seed, nfe), arms in paired.items():
        for name, coefficients in formulas.items():
            if not all(arm in arms for arm in coefficients):
                continue
            protocols = [arms[arm][1] for arm in coefficients]
            if any(value != protocols[0] for value in protocols[1:]):
                raise ValueError("Cannot compute contrasts: evaluation protocols differ")
            keys = set.intersection(*(set(arms[arm][0]) for arm in coefficients))
            differences = {}
            for key in sorted(keys):
                values = [arms[arm][0][key] for arm in coefficients]
                differences[key] = (sum(coefficients[arm] * arms[arm][0][key] for arm in coefficients)
                                    if all(isinstance(value, (int, float)) and math.isfinite(value) for value in values) else None)
            contrasts.append({"dataset": dataset, "seed": seed, "nfe": nfe, "contrast": name, "metrics": differences})
    summary = {"format": FORMAT, "smoke": manifest["smoke"], "rows": rows, "contrasts": contrasts,
               "banks": banks, "endpoint_checks": manifest["endpoint_checks"],
               "unique_base_precompute_seconds": sum(bank["metadata"]["precompute_seconds"] for bank in base_banks.values()),
               "unique_overlap_incremental_seconds": sum(bank["metadata"]["precompute_seconds"] for bank in overlap_banks.values()),
               "timing_scope": {
                   "base_precompute_seconds": "historical shared base bank construction, counted once per unique base bank",
                   "overlap_incremental_seconds": "overlap-only compute, excluding original base construction; shared by C/D and seeds",
                   "training_seconds": "standard trainer loop; excludes offline bank preparation and checkpoint serialization",
                   "inference_seconds": "standard evaluator integration only; excludes sampling/warmup/loading/metrics/rendering",
                   "inference_ms_per_cloud": "batch integration time / clouds; throughput-normalized, not single-cloud latency"},
               "interpretation": "Raw paired differences, not statistical significance. Negative differences favor the first arm for lower-is-better metrics. Smoke is only a pipeline check."}
    save_manifest(absolute(path).parent / "summary.json", summary)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    target = absolute(path).parent / "comparison.csv"
    temporary = target.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "eval", "all"), default="prepare")
    parser.add_argument("--manifest", help="Resume exact saved jobs; never discovers the newest run")
    parser.add_argument("--datasets", nargs="+", choices=("checkerboard", "horse"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--bank-seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--nfe", nargs="+", type=int)
    parser.add_argument("--smoke", action="store_true", help="M64/N16, CPU, two updates; not quality evidence")
    parser.add_argument("--base-checkerboard-config", help="Existing lab MS-W2 config.yaml whose bank is reused unchanged")
    parser.add_argument("--base-horse-config", help="Existing lab MS-W2 config.yaml whose bank is reused unchanged")
    args = parser.parse_args(argv)
    if args.stage == "eval" and not args.manifest:
        parser.error("--stage eval requires --manifest")
    if args.manifest:
        if args.smoke or any(value is not None for value in (args.datasets, args.seeds, args.bank_seed,
                args.device, args.steps, args.base_checkerboard_config, args.base_horse_config)):
            parser.error("With --manifest only --stage and additional --nfe values may change")
        path = absolute(args.manifest)
        manifest = read_json(path)
        verify_manifest(manifest)
        if args.nfe is not None:
            nfes = integers(args.nfe, "nfe", 1)
            manifest["evaluation_nfes"] = sorted(set(manifest["evaluation_nfes"]) | set(nfes))
            save_manifest(path, manifest)
    else:
        path, manifest = create_experiment(
            datasets=args.datasets, seeds=args.seeds, bank_seed=args.bank_seed or 0, device=args.device,
            steps=args.steps, smoke=args.smoke, nfe=args.nfe,
            base_checkerboard_config=args.base_checkerboard_config, base_horse_config=args.base_horse_config)
    if manifest["smoke"]:
        torch.set_num_threads(1)
        print("SMOKE ONLY: M64/N16, CPU, two updates; no quality conclusions", flush=True)
    prepare_jobs(path, manifest)
    if args.stage in ("train", "all"):
        train_jobs(path, manifest)
    if args.stage in ("eval", "all"):
        evaluate_jobs(path, manifest)
    write_summary(path, manifest)
    print(f"experiment_manifest={relative(path)}", flush=True)
    print(f"summary={relative(path.parent / 'summary.json')}", flush=True)
    if args.stage == "prepare":
        print(f'next_command=python run_msw2_boundary_study.py --manifest "{relative(path)}" --stage all', flush=True)
    return path


if __name__ == "__main__":
    main()
