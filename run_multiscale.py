"""Prepare/train/evaluate multiscale Wasserstein matching with a 10K pair bank.

Default is preparation only. Training is explicit (--stage train/all).
"""

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml

from nsot import file_sha256


ROOT = Path(__file__).resolve().parent
FORMAT = "multiscale_wasserstein_experiment_v1"
NFES = [1, 2, 4, 8, 16, 32, 64, 128]
REFINED = "multiscale_refined"


@contextmanager
def project_directory():
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        yield
    finally:
        os.chdir(previous)


def save_manifest(path, value):
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def _integers(values, label, minimum=0):
    if (not values or len(values) != len(set(values)) or
            any(isinstance(v, bool) or not isinstance(v, int) or not minimum <= v < 2**32 - 1
                for v in values)):
        raise ValueError(f"{label} must contain distinct integers in [{minimum},2**32-1)")
    return list(values)


def _read_yaml(path):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _cache_path(config, dataset, method):
    from nsot import dataset_spec
    options = {key: value for key, value in config[method].items()
               if key not in ("cache", "cache_sha256", "beta")}
    dependencies = (["nsot.py", "data.py", "train_horse.py"] if method == "nsot" else
                    ["multiscale_wasserstein.py", "nsot.py", "data.py", "train_horse.py"])
    if config[method].get("target_sampling", {}).get("mode") == "stratified":
        dependencies = [*dependencies, "target_mass_control.py"]
    specification = {"dataset": dataset_spec(config, dataset), "options": options,
                     "implementation_sha256": {name: file_sha256(ROOT / name) for name in dependencies}}
    digest = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:20]
    return ROOT / "coupling_cache" / method / f"{dataset}_{digest}.npz"


def _refinement_cache_path(config, dataset):
    baseline = config["multiscale_wasserstein"]
    base_hash = file_sha256(baseline["cache"])
    if baseline.get("cache_sha256") != base_hash:
        raise ValueError("Refinement requires the exact prepared baseline bank hash")
    options = {key: value for key, value in config[REFINED].items()
               if key not in ("cache", "cache_sha256")}
    specification = {"dataset": dataset, "base_cache_sha256": base_hash, "options": options,
                     "implementation_sha256": {name: file_sha256(ROOT / name) for name in
                                               ("multiscale_refinement.py", "multiscale_wasserstein.py")}}
    digest = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:20]
    return ROOT / "coupling_cache" / REFINED / f"{dataset}_{digest}.npz"


def _refined_config(baseline, neighbors):
    _integers([neighbors], "refinement_neighbors")
    config = copy.deepcopy(baseline)
    config["coupling"] = REFINED
    config[REFINED] = {"cache": "pending_prepared_base_hash", "neighbors": neighbors, "max_edges": 2000000}
    from multiscale_refinement import settings
    settings(config)
    return config


def _baseline_job(manifest, job):
    matches = [candidate for candidate in manifest["jobs"]
               if candidate["dataset"] == job["dataset"] and candidate["seed"] == job["seed"]
               and candidate["label"] == "multiscale"]
    if len(matches) != 1:
        raise ValueError("Every refined job requires exactly one matching dataset/seed baseline job")
    return matches[0]


def create_experiment(*, datasets=None, seeds=None, bank_seed=0, device=None, steps=None,
                      superset_size=None, partition_counts=None, scale_weights=None,
                      with_nsot=False, smoke=False, nfe=None, output_root=None,
                      with_refinement=False, refinement_neighbors=8):
    from multiscale_wasserstein import METHOD, settings
    datasets = ["checkerboard", "horse"] if datasets is None else list(datasets)
    if (not datasets or len(datasets) != len(set(datasets)) or
            any(d not in ("checkerboard", "horse") for d in datasets)):
        raise ValueError("Choose distinct checkerboard/horse datasets")
    seeds = _integers([0] if seeds is None else seeds, "seeds")
    _integers([bank_seed], "bank_seed")
    _integers([refinement_neighbors], "refinement_neighbors")
    nfe = _integers(([1, 2] if smoke else NFES) if nfe is None else nfe, "nfe", 1)
    if smoke and any(v is not None for v in (device, steps, superset_size, partition_counts, scale_weights)):
        raise ValueError("--smoke supplies its own device, steps, bank size and partition settings")
    if steps is not None:
        _integers([steps], "steps", 1)
    directory = Path(output_root or ROOT / "experiment_manifests")
    if not directory.is_absolute():
        directory = ROOT / directory
    directory = directory.resolve() / f"multiscale_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:8]}"
    # Validate all proposed settings before creating an experiment directory.
    proposed = []
    for dataset in datasets:
        config = _read_yaml(ROOT / "multiscale_experiments" / f"{dataset}.yaml")
        config["dataset"], config["coupling"] = dataset, METHOD
        options = config[METHOD]
        options["cache_seed"] = bank_seed
        if device is not None:
            config["device"] = device
        if steps is not None:
            config["training"]["num_steps"] = steps
        if superset_size is not None:
            options["superset_size"] = superset_size
        if partition_counts is not None:
            options["partition_counts"] = list(partition_counts)
        if scale_weights is not None:
            options["scale_weights"] = list(scale_weights)
        if smoke:
            config["device"] = "cpu"
            config["data"].update(batch_size=2, n_points=16)
            config["model"].update(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
            config["training"].update(num_steps=2, log_every=1)
            config["evaluation"].update(batch_size=6, histogram_bins=16)
            options.update(superset_size=64, partition_counts=[2, 4, 8], scale_weights=[1.0, 1.0, 1.0])
        settings(config)
        options["cache"] = str(_cache_path(config, dataset, METHOD))
        proposed.append((dataset, "multiscale", config))
        if with_refinement:
            proposed.append((dataset, "multiscale_refined", _refined_config(config, refinement_neighbors)))
        if with_nsot:
            for beta in (0.0, 0.2):
                baseline = copy.deepcopy(config)
                baseline.pop(METHOD)
                baseline["coupling"] = "nsot"
                baseline["nsot"] = {"cache": "pending", "superset_size": options["superset_size"],
                                    "cache_seed": bank_seed, "beta": beta,
                                    "solver": "scipy_linear_sum_assignment"}
                from nsot import settings as nsot_settings
                nsot_settings(baseline)
                baseline["nsot"]["cache"] = str(_cache_path(baseline, dataset, "nsot"))
                proposed.append((dataset, "nsot_beta0" if beta == 0 else "nsot_beta02", baseline))
    directory.mkdir(parents=True, exist_ok=False)
    jobs = []
    for dataset, label, config in proposed:
        for seed in seeds:
            job_config = copy.deepcopy(config)
            job_config["seed"] = seed
            job_config["checkpoint"] = f"{dataset}_{label}_bank{bank_seed}_seed{seed}.pt"
            config_path = directory / f"{dataset}_{label}_seed{seed}.yaml"
            with config_path.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(job_config, stream, sort_keys=False)
            jobs.append({"dataset": dataset, "label": label, "seed": seed,
                         "config": str(config_path), "config_sha256": file_sha256(config_path),
                         "bank": None, "run": None, "evaluations": {}})
    manifest = {"format": FORMAT, "created_at": datetime.now(timezone.utc).isoformat(),
                "smoke": smoke, "datasets": datasets, "bank_seed": bank_seed,
                "training_seeds": seeds, "with_nsot": with_nsot, "evaluation_nfes": nfe,
                "with_refinement": with_refinement, "refinement_neighbors": refinement_neighbors,
                "protocol": {"matching": "multiscale patch Wasserstein cost, restricted block permutation",
                             "source": "original standard Gaussian bank; fresh Gaussian inference",
                             "model_training": "unchanged original checkerboard/horse baseline settings unless overridden",
                             "online_perturbation": "none for multiscale; beta0 and beta0.2 controls only if requested",
                             "marginals": "exact finite empirical endpoints; not exact population marginals",
                             "timing": "precompute shared once per unique bank; training loop and inference separately"},
                "endpoint_checks": {}, "refinement_checks": {}, "jobs": jobs}
    path = directory / "manifest.json"
    save_manifest(path, manifest)
    print(f"experiment_manifest={path}", flush=True)
    print(f"planned_models={len(jobs)} smoke={smoke}; prepare stage does not train", flush=True)
    return path, manifest


def add_refinement(path, manifest, *, neighbors=8, output_root=None):
    """Clone an existing study while reusing verified baseline artifacts unchanged."""
    _integers([neighbors], "refinement_neighbors")
    if manifest.get("format") != FORMAT:
        raise ValueError("Unsupported source manifest")
    if any(job["label"] == "multiscale_refined" for job in manifest["jobs"]):
        raise ValueError("Manifest already contains refinement; resume it or clone the original baseline manifest")
    baseline_jobs = [job for job in manifest["jobs"] if job["label"] == "multiscale"]
    if not baseline_jobs:
        raise ValueError("No multiscale baseline jobs to refine")
    for job in manifest["jobs"]:
        verify_job(job)
    if any(job["bank"] is None for job in baseline_jobs):
        raise ValueError("Prepare the original baseline manifest before --add-refinement")
    directory = Path(output_root or path.parent.parent).resolve() / (
        f"multiscale_refined_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:8]}")
    directory.mkdir(parents=True, exist_ok=False)
    clone = copy.deepcopy(manifest)
    clone.update(created_at=datetime.now(timezone.utc).isoformat(), with_refinement=True,
                 refinement_neighbors=neighbors, refinement_checks={},
                 parent_manifest={"path": str(path.resolve()), "sha256": file_sha256(path)})
    for baseline in baseline_jobs:
        config = _refined_config(verify_job(baseline), neighbors)
        config["checkpoint"] = f"{baseline['dataset']}_multiscale_refined_bank{manifest['bank_seed']}_seed{baseline['seed']}.pt"
        config_path = directory / f"{baseline['dataset']}_multiscale_refined_seed{baseline['seed']}.yaml"
        with config_path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        clone["jobs"].append({"dataset": baseline["dataset"], "label": "multiscale_refined", "seed": baseline["seed"],
                              "config": str(config_path), "config_sha256": file_sha256(config_path),
                              "bank": None, "run": None, "evaluations": {}})
    clone_path = directory / "manifest.json"
    save_manifest(clone_path, clone)
    print(f"cloned_manifest={clone_path}; original manifest and completed baseline artifacts unchanged", flush=True)
    return clone_path, clone


def verify_job(job):
    path = Path(job["config"])
    if file_sha256(path) != job["config_sha256"]:
        raise ValueError(f"Prepared YAML changed: {path}; start a new experiment")
    config = _read_yaml(path)
    if config["coupling"] not in ("multiscale_wasserstein", "nsot", REFINED) or config["dataset"] != job["dataset"]:
        raise ValueError("Manifest/config dataset or method mismatch")
    if config["coupling"] == REFINED and config["multiscale_wasserstein"].get("cache_sha256"):
        baseline = config["multiscale_wasserstein"]
        if file_sha256(baseline["cache"]) != baseline["cache_sha256"]:
            raise ValueError("Refinement's baseline bank changed")
    if job["bank"]:
        bank = job["bank"]
        if file_sha256(bank["path"]) != bank["sha256"]:
            raise ValueError(f"Prepared bank changed: {bank['path']}")
        if config[config["coupling"]].get("cache_sha256") != bank["sha256"]:
            raise ValueError("Prepared YAML and manifest disagree about bank hash")
    if job["run"]:
        for name in ("config", "checkpoint", "training_metadata"):
            item = job["run"][name]
            if file_sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"Saved training artifact changed: {item['path']}")
    for result in job["evaluations"].values():
        for item in result.values():
            if file_sha256(item["path"]) != item["sha256"]:
                raise ValueError(f"Evaluation artifact changed: {item['path']}")
    return config


def prepare_jobs(path, manifest):
    # Baselines must be prepared before their derived banks, independent of manifest list order.
    for job in sorted(manifest["jobs"], key=lambda item: item["label"] == "multiscale_refined"):
        config = verify_job(job)
        if job["bank"] is not None:
            continue
        if config["coupling"] == "nsot":
            from nsot import prepare
        elif config["coupling"] == REFINED:
            baseline_job = _baseline_job(manifest, job)
            baseline = verify_job(baseline_job)
            if baseline_job["bank"] is None:
                raise ValueError("Prepare multiscale baseline before refinement")
            config["multiscale_wasserstein"] = copy.deepcopy(baseline["multiscale_wasserstein"])
            config[REFINED]["cache"] = str(_refinement_cache_path(config, job["dataset"]))
            from multiscale_refinement import prepare
        else:
            from multiscale_wasserstein import prepare
        with project_directory():
            cache, metadata = prepare(config, job["dataset"])
        cache = cache.resolve()
        digest = file_sha256(cache)
        config[config["coupling"]]["cache_sha256"] = digest
        # Keep the input YAML immutable. If interrupted before the manifest update,
        # the unreferenced prepared YAML can be verified and adopted on resume.
        prepared_path = Path(job["config"]).with_name(Path(job["config"]).stem + "_prepared.yaml")
        serialized = yaml.safe_dump(config, sort_keys=False)
        if prepared_path.exists():
            if _read_yaml(prepared_path) != config:
                raise ValueError(f"Stale prepared config disagrees with current bank: {prepared_path}")
        else:
            with prepared_path.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
        job["config"] = str(prepared_path)
        job["config_sha256"] = file_sha256(prepared_path)
        job["bank"] = {"path": str(cache), "sha256": digest,
                       "precompute_seconds": metadata["precompute_seconds"], "metadata": metadata}
        save_manifest(path, manifest)
    for job in manifest["jobs"]:
        if job["label"] != "multiscale_refined":
            continue
        baseline = _baseline_job(manifest, job)
        verify_job(job)
        verify_job(baseline)
        with np.load(baseline["bank"]["path"], allow_pickle=False) as original, np.load(job["bank"]["path"], allow_pickle=False) as refined:
            for key in ("source", "target"):
                if not np.array_equal(original[key], refined[key]):
                    raise ValueError(f"{job['dataset']}: refined {key} endpoints differ from baseline")
            base_permutation = original["permutation"]
            expected_permutation_hash = hashlib.sha256(np.ascontiguousarray(base_permutation).tobytes()).hexdigest()
            metadata = job["bank"]["metadata"]
            if (not np.array_equal(base_permutation, refined["base_permutation"])
                    or metadata.get("base_permutation_sha256") != expected_permutation_hash
                    or metadata.get("base_cache_sha256") != baseline["bank"]["sha256"]):
                raise ValueError("Refinement was not constructed from this exact baseline permutation/cache")
        manifest.setdefault("refinement_checks", {})[f"{job['dataset']}:seed{job['seed']}"] = {
            "same_source": True, "same_target": True, "same_base_permutation": True,
            "base_cache_sha256": baseline["bank"]["sha256"],
            "base_permutation_sha256": expected_permutation_hash, "refined_cache_sha256": job["bank"]["sha256"]}
        save_manifest(path, manifest)
    if manifest["with_nsot"]:
        for dataset in manifest["datasets"]:
            entries = [j for j in manifest["jobs"] if j["dataset"] == dataset]
            ms = next(j for j in entries if j["label"] == "multiscale")
            ns = next(j for j in entries if j["label"] == "nsot_beta0")
            with np.load(ms["bank"]["path"], allow_pickle=False) as left, np.load(ns["bank"]["path"], allow_pickle=False) as right:
                for key in ("source", "target"):
                    if not np.array_equal(left[key], right[key]):
                        raise ValueError(f"{dataset}: multiscale and NSOT {key} endpoints differ")
            manifest["endpoint_checks"][dataset] = {"same_source": True, "same_target": True,
                                                       "multiscale_bank_sha256": ms["bank"]["sha256"],
                                                       "nsot_bank_sha256": ns["bank"]["sha256"]}
        save_manifest(path, manifest)


def _artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def train_jobs(path, manifest):
    for job in manifest["jobs"]:
        verify_job(job)
        if job["bank"] is None:
            raise ValueError("Bank preparation is incomplete")
        if job["run"] is not None:
            print(f"reused_completed_training={job['run']['config']['path']}", flush=True)
            continue
        if job["dataset"] == "horse":
            from train_horse import main as train
        else:
            from train import main as train
        print(f"training={job['dataset']}/{job['label']} seed={job['seed']}", flush=True)
        with project_directory():
            directory = Path(train(job["config"])).resolve()
        saved = _read_yaml(directory / "config.yaml")
        job["run"] = {"config": _artifact(directory / "config.yaml"),
                      "checkpoint": _artifact(directory / saved["checkpoint"]),
                      "training_metadata": _artifact(directory / "training.json")}
        save_manifest(path, manifest)


def evaluate_jobs(path, manifest):
    for job in manifest["jobs"]:
        verify_job(job)
        if job["run"] is None:
            raise ValueError("Run --stage train/all before evaluation")
        if job["dataset"] == "horse":
            from eval_horse import main as evaluate
        else:
            from eval import main as evaluate
        for nfe in manifest["evaluation_nfes"]:
            existing = job["evaluations"].get(str(nfe))
            if existing:
                for artifact in existing.values():
                    if file_sha256(artifact["path"]) != artifact["sha256"]:
                        raise ValueError(f"Evaluation artifact changed: {artifact['path']}")
                continue
            with project_directory():
                image = Path(evaluate(job["run"]["config"]["path"], nfe)).resolve()
            job["evaluations"][str(nfe)] = {"json": _artifact(image.with_suffix(".json")),
                                            "png": _artifact(image)}
            save_manifest(path, manifest)


def write_summary(path, manifest):
    banks, rows = {}, []
    for job in manifest["jobs"]:
        verify_job(job)
        if job["bank"]:
            banks[job["bank"]["path"]] = {"sha256": job["bank"]["sha256"],
                                          "precompute_seconds": job["bank"]["precompute_seconds"],
                                          "kind": "incremental_refinement" if job["label"] == "multiscale_refined" else "base_bank"}
            if job["label"] == "multiscale_refined":
                baseline = _baseline_job(manifest, job)
                banks[job["bank"]["path"]]["base_bank"] = baseline["bank"]["path"]
        row = {"dataset": job["dataset"], "method": job["label"], "seed": job["seed"],
               "bank": job["bank"]["path"] if job["bank"] else None,
               "training_seconds": None, "evaluations": {}}
        if job["run"]:
            with Path(job["run"]["training_metadata"]["path"]).open(encoding="utf-8") as stream:
                row["training_seconds"] = json.load(stream)["training_seconds"]
        for nfe, artifacts in job["evaluations"].items():
            with Path(artifacts["json"]["path"]).open(encoding="utf-8") as stream:
                result = json.load(stream)
            # Retain the standard evaluator's complete metric schema, including horse ROI metrics.
            row["evaluations"][nfe] = {"result": result, "json": artifacts["json"]["path"],
                                        "png": artifacts["png"]["path"]}
        rows.append(row)
    paired = []
    for refined in (row for row in rows if row["method"] == "multiscale_refined"):
        baseline = next(row for row in rows if row["dataset"] == refined["dataset"]
                        and row["seed"] == refined["seed"] and row["method"] == "multiscale")
        for nfe in sorted(set(baseline["evaluations"]) & set(refined["evaluations"]), key=int):
            before, after = baseline["evaluations"][nfe]["result"], refined["evaluations"][nfe]["result"]
            setting_keys = ("evaluation_seed", "evaluation_batch_size", "n_points", "euler_steps",
                            "histogram_bins", "metric_metadata", "metric_definitions")
            mismatches = [key for key in setting_keys if before.get(key) != after.get(key)]
            metrics = {"chamfer", "leakage", "cell_mass_error", "histogram_js", "thin_region_mass_mae", "gap_region_leakage"}
            metrics.update(key for key in before.keys() | after.keys() if key.startswith("roi_"))
            deltas = {}
            for key in sorted(metrics):
                left, right = before.get(key), after.get(key)
                if (not mismatches and isinstance(left, (int, float)) and not isinstance(left, bool)
                        and isinstance(right, (int, float)) and not isinstance(right, bool)):
                    deltas[key] = {"baseline": left, "refined": right, "delta": right - left}
            comparison = {"dataset": refined["dataset"], "seed": refined["seed"], "nfe": int(nfe),
                          "evaluation_settings_match": not mismatches, "mismatched_fields": mismatches,
                          "metrics_refined_minus_baseline": deltas,
                          "baseline_json": baseline["evaluations"][nfe]["json"],
                          "refined_json": refined["evaluations"][nfe]["json"],
                          "baseline_training_seconds": baseline["training_seconds"],
                          "refined_training_seconds": refined["training_seconds"],
                          "baseline_inference_seconds": before.get("inference_seconds"),
                          "refined_inference_seconds": after.get("inference_seconds"),
                          "baseline_shared_precompute_seconds": banks[baseline["bank"]]["precompute_seconds"],
                          "incremental_refinement_precompute_seconds": banks[refined["bank"]]["precompute_seconds"]}
            for key in ("training_seconds", "inference_seconds"):
                left, right = comparison["baseline_" + key], comparison["refined_" + key]
                comparison[key + "_delta"] = right - left if left is not None and right is not None else None
            paired.append(comparison)
    save_manifest(path.parent / "summary.json", {"format": FORMAT, "smoke": manifest["smoke"],
                                                  "banks": banks, "jobs": rows,
                                                  "endpoint_checks": manifest["endpoint_checks"],
                                                  "refinement_checks": manifest.get("refinement_checks", {}),
                                                  "paired_refinement": paired,
                                                  "unique_precompute_seconds": sum(bank["precompute_seconds"] for bank in banks.values()),
                                                  "timing_note": "Base precompute counted once per unique bank; refined precompute is incremental only. Historical construction times, not current resume wall time. Inference excludes rendering/metrics.",
                                                  "delta_note": "refined minus baseline; negative is better for CD/leakage/JS/mass errors, not automatically for raw ROI mass; no statistical significance claim."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "eval", "all"), default="prepare")
    parser.add_argument("--manifest", help="Resume exact saved jobs without finding the latest run")
    parser.add_argument("--datasets", nargs="+", choices=("checkerboard", "horse"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--bank-seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--superset-size", type=int)
    parser.add_argument("--partition-counts", nargs="+", type=int)
    parser.add_argument("--scale-weights", nargs="+", type=float)
    parser.add_argument("--with-nsot", action="store_true", help="Also train matched-endpoint NSOT beta=0 and beta=0.2 controls")
    refinement_group = parser.add_mutually_exclusive_group()
    refinement_group.add_argument("--with-refinement", action="store_true", help="Add one sparse-refined comparison to each new baseline job")
    refinement_group.add_argument("--add-refinement", action="store_true", help="Clone --manifest and add refined jobs, reusing completed baseline runs")
    parser.add_argument("--refinement-neighbors", type=int, help="Cross-leaf candidate neighbors (default 8); 0 is an identity control")
    parser.add_argument("--smoke", action="store_true", help="Tiny CPU two-update pipeline check; not a quality experiment")
    parser.add_argument("--nfe", nargs="+", type=int)
    args = parser.parse_args()
    if args.stage == "eval" and not args.manifest:
        parser.error("--stage eval requires --manifest for previously trained jobs")
    if args.add_refinement and (not args.manifest or args.stage == "eval"):
        parser.error("--add-refinement requires --manifest and a prepare/train/all stage")
    if args.refinement_neighbors is not None and not (args.with_refinement or args.add_refinement):
        parser.error("--refinement-neighbors requires --with-refinement or --add-refinement")
    if args.manifest:
        if args.smoke or args.with_nsot or args.with_refinement or any(v is not None for v in (
                args.datasets, args.seeds, args.bank_seed, args.device, args.steps,
                args.superset_size, args.partition_counts, args.scale_weights)):
            parser.error("With --manifest only --stage/--nfe may change, or use --add-refinement to clone")
        path = Path(args.manifest).resolve()
        with path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("format") != FORMAT:
            raise ValueError("Unsupported manifest format")
        if args.add_refinement:
            path, manifest = add_refinement(path, manifest,
                                            neighbors=8 if args.refinement_neighbors is None else args.refinement_neighbors)
        if args.nfe is not None:
            additions = _integers(args.nfe, "nfe", 1)
            manifest["evaluation_nfes"] = sorted(set(manifest["evaluation_nfes"]) | set(additions))
            save_manifest(path, manifest)
    else:
        path, manifest = create_experiment(
            datasets=args.datasets, seeds=args.seeds, bank_seed=args.bank_seed or 0,
            device=args.device, steps=args.steps, superset_size=args.superset_size,
            partition_counts=args.partition_counts, scale_weights=args.scale_weights,
            with_nsot=args.with_nsot, smoke=args.smoke, nfe=args.nfe,
            with_refinement=args.with_refinement,
            refinement_neighbors=8 if args.refinement_neighbors is None else args.refinement_neighbors)
    if manifest["smoke"]:
        torch.set_num_threads(1)
        print("SMOKE ONLY: M=64, N=16, CPU, two optimizer updates; not quality evidence", flush=True)
    prepare_jobs(path, manifest)
    if args.stage in ("train", "all"):
        train_jobs(path, manifest)
    if args.stage in ("eval", "all"):
        evaluate_jobs(path, manifest)
    write_summary(path, manifest)
    print(f"experiment_manifest={path}", flush=True)
    print(f"summary={path.parent / 'summary.json'}", flush=True)
    if args.stage == "prepare":
        print(f'next_command=python run_multiscale.py --manifest "{path}" --stage all', flush=True)


if __name__ == "__main__":
    main()
