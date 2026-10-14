"""Run the five prespecified MS-W2 partition/weight ablations.

Default: write a plan only. Run training explicitly with --stage train/all.
"""

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml

import multiscale_wasserstein as msw
import run_multiscale as base


ROOT = base.ROOT
KIND = "msw2_scale_ablation_v1"
BASELINE = "baseline"
VARIANTS = {
    BASELINE: {"partition_counts": [4, 20, 100], "scale_weights": [1, 1, 1]},
    "fine_only": {"partition_counts": [4, 20, 100], "scale_weights": [0, 0, 1]},
    "fine_emphasis": {"partition_counts": [4, 20, 100], "scale_weights": [1, 1, 2]},
    "larger_leaf": {"partition_counts": [4, 20, 40], "scale_weights": [1, 1, 1]},
    "smaller_leaf": {"partition_counts": [4, 20, 200], "scale_weights": [1, 1, 1]},
}
EVALUATION_FIELDS = ("evaluation_seed", "evaluation_batch_size", "n_points", "euler_steps",
                     "histogram_bins", "metric_metadata", "metric_definitions")
QUALITY_METRICS = {"chamfer", "leakage", "histogram_js", "cell_mass_error",
                   "thin_region_mass_mae", "gap_region_leakage"}


def resolve(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def portable(value):
    path = resolve(value)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def variants(smoke=False):
    values = copy.deepcopy(VARIANTS)
    if smoke:
        for label, value in values.items():
            value["partition_counts"] = ([2, 4, 8] if label == "larger_leaf" else
                                         [2, 4, 40] if label == "smaller_leaf" else [2, 4, 20])
    return values


def create_experiment(*, datasets=None, seeds=None, bank_seed=0, device=None, steps=None,
                      nfe=None, smoke=False, output_root=None, results_root=None):
    datasets = ["checkerboard", "horse"] if datasets is None else list(datasets)
    if (not datasets or len(set(datasets)) != len(datasets)
            or any(value not in ("checkerboard", "horse") for value in datasets)):
        raise ValueError("Choose distinct checkerboard/horse datasets")
    seeds = base._integers([0] if seeds is None else seeds, "seeds")
    base._integers([bank_seed], "bank_seed")
    if steps is not None:
        base._integers([steps], "steps", 1)
    if smoke and (device is not None or steps is not None):
        raise ValueError("--smoke supplies its own CPU device and two-update settings")
    nfes = base._integers(([1, 2] if smoke else base.NFES) if nfe is None else nfe, "nfe", 1)
    definitions, proposed = variants(smoke), []
    for dataset in datasets:
        template = base._read_yaml(ROOT / "multiscale_experiments" / f"{dataset}.yaml")
        for label, definition in definitions.items():
            config = copy.deepcopy(template)
            config.update(dataset=dataset, coupling=msw.METHOD)
            config[msw.METHOD].update(definition, superset_size=80 if smoke else 10000,
                                      cache_seed=bank_seed)
            config[msw.METHOD].pop("cache_sha256", None)
            if device is not None:
                config["device"] = device
            if steps is not None:
                config["training"]["num_steps"] = steps
            if smoke:
                config["device"] = "cpu"
                config["data"].update(batch_size=2, n_points=8)
                config["model"].update(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
                config["training"].update(num_steps=2, log_every=1)
                config["evaluation"].update(batch_size=2, histogram_bins=8)
            msw.settings(config)
            config[msw.METHOD]["cache"] = portable(base._cache_path(config, dataset, msw.METHOD))
            for seed in seeds:
                job_config = copy.deepcopy(config)
                job_config["seed"] = seed
                job_config["checkpoint"] = f"{dataset}_msw2_{label}_bank{bank_seed}_seed{seed}.pt"
                proposed.append((dataset, label, seed, job_config))
    name = f"multiscale_ablation_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:8]}"
    directory = resolve(output_root or ROOT / "experiment_manifests") / name
    directory.mkdir(parents=True, exist_ok=False)
    jobs = []
    for dataset, label, seed, config in proposed:
        config_path = directory / f"{dataset}_{label}_seed{seed}.yaml"
        with config_path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        jobs.append({"dataset": dataset, "label": label, "seed": seed,
                     "config": portable(config_path), "config_sha256": base.file_sha256(config_path),
                     "bank": None, "run": None, "evaluations": {}})
    results = resolve(results_root or ROOT / "eval_results" / "multiscale_ablation") / name
    manifest = {"format": base.FORMAT, "experiment_type": KIND,
                "created_at": datetime.now(timezone.utc).isoformat(), "smoke": smoke,
                "datasets": datasets, "training_seeds": seeds, "bank_seed": bank_seed,
                "variants": definitions, "evaluation_nfes": sorted(nfes),
                "with_nsot": False, "with_refinement": False, "endpoint_checks": {},
                "results_directory": portable(results), "jobs": jobs,
                "protocol": {
                    "changed": "only partition_counts and scale_weights in the five prespecified conditions",
                    "fixed": "M, original IID source/target endpoints, paired training/evaluation settings and seeds",
                    "matching": "original exact multiscale block matching; recomputed partitions/permutation",
                    "evaluation": "original eval.py/eval_horse.py metrics and rendering",
                    "timing": "unique bank preprocessing, training, integration-only inference separately",
                }}
    path = directory / "manifest.json"
    base.save_manifest(path, manifest)
    print(f"experiment_manifest={portable(path)}", flush=True)
    print(f"planned_models={len(jobs)} smoke={smoke}; default plan performs no OT or training", flush=True)
    return path, manifest


def _paired_signature(config):
    signature = copy.deepcopy(config)
    signature.pop("checkpoint", None)
    options = signature[msw.METHOD]
    for key in ("cache", "cache_sha256", "partition_counts", "scale_weights"):
        options.pop(key, None)
    return signature


def verify_manifest(path, manifest):
    if manifest.get("format") != base.FORMAT or manifest.get("experiment_type") != KIND:
        raise ValueError("This runner requires its own scale-ablation manifest")
    definitions = variants(manifest.get("smoke", False))
    if manifest.get("variants") != definitions or manifest.get("with_nsot") or manifest.get("with_refinement"):
        raise ValueError("Manifest must contain only the five prespecified scale ablations")
    base._integers(manifest["training_seeds"], "seeds")
    base._integers([manifest["bank_seed"]], "bank_seed")
    base._integers(manifest["evaluation_nfes"], "nfe", 1)
    datasets = manifest["datasets"]
    if (not datasets or len(set(datasets)) != len(datasets)
            or any(value not in ("checkerboard", "horse") for value in datasets)):
        raise ValueError("Invalid ablation datasets")
    expected = {(dataset, seed, label) for dataset in datasets
                for seed in manifest["training_seeds"] for label in definitions}
    found, signatures = set(), {}
    with base.project_directory():
        for job in manifest["jobs"]:
            key = job["dataset"], job["seed"], job["label"]
            if key not in expected or key in found:
                raise ValueError("Missing, duplicate or unrelated scale-ablation job")
            found.add(key)
            config = base.verify_job(job)
            if config["coupling"] != msw.METHOD or config["seed"] != job["seed"]:
                raise ValueError("Scale-ablation job/config method or seed mismatch")
            opts = msw.settings(config)
            definition = definitions[job["label"]]
            if any(config[msw.METHOD][field] != value for field, value in definition.items()):
                raise ValueError("Scale-ablation partitions/weights differ from the prespecified condition")
            if opts["superset_size"] != (80 if manifest["smoke"] else 10000) or opts["cache_seed"] != manifest["bank_seed"]:
                raise ValueError("Scale ablations must share bank size and bank seed")
            pair = job["dataset"], job["seed"]
            signature = _paired_signature(config)
            if signatures.setdefault(pair, signature) != signature:
                raise ValueError("Paired model/data/training/evaluation settings differ")
            if job["run"]:
                saved = base._read_yaml(job["run"]["config"]["path"])
                if saved != config:
                    raise ValueError("Saved run config differs from its prepared ablation config")
    if found != expected:
        raise ValueError("Manifest is missing a prespecified scale-ablation job")


def verify_endpoints(path, manifest):
    """Compare actual arrays, not merely the bank seeds or reported hashes."""
    checks = {}
    with base.project_directory():
        for dataset in manifest["datasets"]:
            entries = [job for job in manifest["jobs"] if job["dataset"] == dataset]
            if any(job["bank"] is None for job in entries):
                raise ValueError("Prepare every ablation bank before training/evaluation")
            original, bank_checks = None, {}
            for job in entries:
                config = base.verify_job(job)
                source, target, _, metadata, digest = msw.load_cache(config, dataset)
                if original is None:
                    original = source, target
                elif not all(np.array_equal(before, after) for before, after in zip(original, (source, target))):
                    raise ValueError(f"{dataset}: scale-ablation source/target endpoints differ")
                bank_checks[digest] = metadata["endpoint_sha256"]
            checks[dataset] = {"same_source": True, "same_target": True,
                               "variants_checked": list(manifest["variants"]),
                               "endpoint_sha256": next(iter(bank_checks.values())),
                               "unique_bank_sha256": list(bank_checks)}
    if manifest.get("endpoint_checks") != checks:
        manifest["endpoint_checks"] = checks
        base.save_manifest(path, manifest)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def write_summary(path, manifest):
    verify_manifest(path, manifest)
    banks, rows = {}, []
    with base.project_directory():
        for job in manifest["jobs"]:
            config = base.verify_job(job)
            row = {"dataset": job["dataset"], "method": job["label"], "seed": job["seed"],
                   **manifest["variants"][job["label"]], "training_seconds": None,
                   "bank": job["bank"]["path"] if job["bank"] else None, "evaluations": {}}
            if job["bank"]:
                bank = job["bank"]
                banks[bank["sha256"]] = {"path": bank["path"], "precompute_seconds": bank["precompute_seconds"],
                                         "dataset": job["dataset"], "variant": job["label"],
                                         "scale_diagnostics": bank["metadata"].get("scales", [])}
            if job["run"]:
                with Path(job["run"]["training_metadata"]["path"]).open(encoding="utf-8") as stream:
                    row["training_seconds"] = json.load(stream)["training_seconds"]
            for nfe, artifacts in job["evaluations"].items():
                with Path(artifacts["json"]["path"]).open(encoding="utf-8") as stream:
                    result = json.load(stream)
                row["evaluations"][nfe] = {"result": result, "json": artifacts["json"]["path"],
                                            "png": artifacts["png"]["path"]}
            rows.append(row)
    paired, csv_rows = [], []
    for row in rows:
        if row["method"] == BASELINE:
            continue
        baseline = next(value for value in rows if value["dataset"] == row["dataset"]
                        and value["seed"] == row["seed"] and value["method"] == BASELINE)
        for nfe in sorted(set(baseline["evaluations"]) & set(row["evaluations"]), key=int):
            before = baseline["evaluations"][nfe]["result"]
            after = row["evaluations"][nfe]["result"]
            mismatches = [key for key in EVALUATION_FIELDS if before.get(key) != after.get(key)]
            if any(result.get("euler_steps", int(nfe)) != int(nfe) for result in (before, after)):
                mismatches.append("manifest_nfe")
            deltas = {}
            metrics = QUALITY_METRICS | {key for key in before.keys() | after.keys() if key.startswith("roi_")}
            for metric in sorted(metrics):
                left, right = before.get(metric), after.get(metric)
                if not _number(left) or not _number(right):
                    continue
                delta = right - left if not mismatches else None
                if not mismatches:
                    deltas[metric] = {"baseline": left, "variant": right, "delta": delta}
                csv_rows.append({"dataset": row["dataset"], "seed": row["seed"], "nfe": int(nfe),
                                 "variant": row["method"], "metric": metric, "baseline": left,
                                 "variant_value": right, "delta": delta,
                                 "evaluation_settings_match": not mismatches,
                                 "baseline_training_seconds": baseline["training_seconds"],
                                 "variant_training_seconds": row["training_seconds"],
                                 "baseline_inference_seconds": before.get("inference_seconds"),
                                 "variant_inference_seconds": after.get("inference_seconds")})
            paired.append({"dataset": row["dataset"], "seed": row["seed"], "nfe": int(nfe),
                           "variant": row["method"], "evaluation_settings_match": not mismatches,
                           "mismatched_fields": mismatches, "metrics_variant_minus_baseline": deltas})
    summary = {"format": KIND, "smoke": manifest["smoke"], "variants": manifest["variants"],
               "jobs": rows, "banks": banks, "endpoint_checks": manifest["endpoint_checks"],
               "paired_comparisons": paired,
               "unique_precompute_seconds": sum(bank["precompute_seconds"] for bank in banks.values()),
               "complete": all(job["run"] is not None and all(str(nfe) in job["evaluations"]
                                for nfe in manifest["evaluation_nfes"]) for job in manifest["jobs"]),
               "timing_note": "Each unique bank counted once across training seeds; historical preparation time. Inference excludes metrics/rendering.",
               "delta_note": "variant minus baseline; negative is favorable for errors, not necessarily for raw ROI mass. No statistical significance claim."}
    directory = resolve(manifest["results_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    base.save_manifest(directory / "summary.json", summary)
    base.save_manifest(path.parent / "summary.json", summary)
    fields = ["dataset", "seed", "nfe", "variant", "metric", "baseline", "variant_value", "delta",
              "evaluation_settings_match", "baseline_training_seconds", "variant_training_seconds",
              "baseline_inference_seconds", "variant_inference_seconds"]
    with (directory / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"comparison_results={portable(directory)}", flush=True)
    return summary


def run_stage(path, manifest, stage):
    if stage not in ("plan", "prepare", "train", "eval", "all"):
        raise ValueError("Unsupported ablation stage")
    path = resolve(path)
    verify_manifest(path, manifest)
    if manifest["smoke"]:
        torch.set_num_threads(1)
    with base.project_directory():
        if stage in ("prepare", "train", "all"):
            base.prepare_jobs(path, manifest)
        if stage != "plan":
            verify_endpoints(path, manifest)
        if stage in ("train", "all"):
            base.train_jobs(path, manifest)
        if stage in ("eval", "all"):
            base.evaluate_jobs(path, manifest)
        return write_summary(path, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "prepare", "train", "eval", "all"), default="plan")
    parser.add_argument("--manifest", help="Resume exactly the saved five-condition experiment")
    parser.add_argument("--datasets", nargs="+", choices=("checkerboard", "horse"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--bank-seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--nfe", nargs="+", type=int)
    parser.add_argument("--smoke", action="store_true", help="Tiny CPU pipeline settings; not a quality experiment")
    args = parser.parse_args(argv)
    if args.stage == "eval" and not args.manifest:
        parser.error("--stage eval requires --manifest for previously trained jobs")
    if args.manifest:
        if args.smoke or any(value is not None for value in
                             (args.datasets, args.seeds, args.bank_seed, args.device, args.steps)):
            parser.error("With --manifest only --stage and --nfe may change")
        path = resolve(args.manifest)
        with path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        verify_manifest(path, manifest)
        if args.nfe is not None:
            additions = base._integers(args.nfe, "nfe", 1)
            manifest["evaluation_nfes"] = sorted(set(manifest["evaluation_nfes"]) | set(additions))
            base.save_manifest(path, manifest)
    else:
        path, manifest = create_experiment(datasets=args.datasets, seeds=args.seeds,
                                          bank_seed=0 if args.bank_seed is None else args.bank_seed,
                                          device=args.device, steps=args.steps, nfe=args.nfe, smoke=args.smoke)
    run_stage(path, manifest, args.stage)
    return path


if __name__ == "__main__":
    main()
