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
PARTITION_VARIANTS = (("axis_axis", "axis", "axis"),
                      ("axis_pca", "axis", "pca"),
                      ("pca_pca", "pca", "pca"))
COUPLING_VARIANTS = ("multiscale", "leaf_only", "qat")


def project_path(path):
    """Manifest paths are project-relative, never relative to the caller's cwd."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def portable_path(path):
    path = project_path(path).resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        # Programmatic tests may place outputs outside the project.
        return str(path)


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
    with project_path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _cache_path(config, dataset, method):
    from nsot import dataset_spec
    options = {key: value for key, value in config[method].items()
               if key not in ("cache", "cache_sha256", "beta")}
    dependencies = (["nsot.py", "data.py", "train_horse.py"] if method == "nsot" else
                    ["qat_coupling.py", "nsot.py", "data.py", "train_horse.py"] if method == "qat" else
                    ["multiscale_wasserstein.py", "nsot.py", "data.py", "train_horse.py"])
    specification = {"dataset": dataset_spec(config, dataset), "options": options,
                     "implementation_sha256": {name: file_sha256(ROOT / name) for name in dependencies}}
    digest = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()[:20]
    return ROOT / "coupling_cache" / method / f"{dataset}_{digest}.npz"


def create_experiment(*, datasets=None, seeds=None, bank_seed=0, device=None, steps=None,
                      superset_size=None, partition_counts=None, scale_weights=None,
                      with_nsot=False, smoke=False, nfe=None, output_root=None,
                      partition_study=False, source_partition=None, target_partition=None,
                      coupling_study=False):
    from multiscale_wasserstein import METHOD, settings
    datasets = ["checkerboard", "horse"] if datasets is None else list(datasets)
    if (not datasets or len(datasets) != len(set(datasets)) or
            any(d not in ("checkerboard", "horse") for d in datasets)):
        raise ValueError("Choose distinct checkerboard/horse datasets")
    seeds = _integers([0] if seeds is None else seeds, "seeds")
    _integers([bank_seed], "bank_seed")
    nfe = _integers(([1, 2] if smoke else NFES) if nfe is None else nfe, "nfe", 1)
    if partition_study and (with_nsot or source_partition is not None or target_partition is not None):
        raise ValueError("--partition-study supplies all three variants; do not combine with --with-nsot or partition overrides")
    if coupling_study and (partition_study or with_nsot or any(value is not None for value in
                           (partition_counts, scale_weights, source_partition, target_partition))):
        raise ValueError("--coupling-study fixes baseline/leaf-only/QAT; do not combine with other studies or partition/weight overrides")
    if smoke and any(v is not None for v in (device, steps, superset_size, partition_counts, scale_weights)):
        raise ValueError("--smoke supplies its own device, steps, bank size and partition settings")
    if steps is not None:
        _integers([steps], "steps", 1)
    default_root = (ROOT / "eval_results" / "coupling_study" if coupling_study else
                    ROOT / "eval_results" / "partition_study" if partition_study else ROOT / "experiment_manifests")
    directory = Path(output_root or default_root)
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
        if coupling_study:
            options.update(partition_counts=[4, 20, 100], scale_weights=[1.0, 1.0, 1.0])
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
        if source_partition is not None:
            options["source_partition"] = source_partition
        if target_partition is not None:
            options["target_partition"] = target_partition
        if smoke:
            config["device"] = "cpu"
            config["data"].update(batch_size=2, n_points=16)
            config["model"].update(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
            config["training"].update(num_steps=2, log_every=1)
            config["evaluation"].update(batch_size=6, histogram_bins=16)
            options.update(superset_size=64, partition_counts=[2, 4, 8], scale_weights=[1.0, 1.0, 1.0])
        if coupling_study:
            options.update(source_partition="axis", target_partition="axis")
            for label in COUPLING_VARIANTS:
                variant = copy.deepcopy(config)
                if label == "qat":
                    variant.pop(METHOD)
                    variant["coupling"] = "qat"
                    variant["qat"] = {"cache": "pending", "superset_size": options["superset_size"],
                                      "cache_seed": bank_seed}
                    from qat_coupling import settings as qat_settings
                    qat_settings(variant)
                else:
                    if label == "leaf_only":
                        variant[METHOD]["scale_weights"] = [0.0] * (len(options["partition_counts"]) - 1) + [1.0]
                    settings(variant)
                method = variant["coupling"]
                variant[method]["cache"] = portable_path(_cache_path(variant, dataset, method))
                proposed.append((dataset, label, variant))
        elif partition_study:
            for label, source, target in PARTITION_VARIANTS:
                variant = copy.deepcopy(config)
                variant[METHOD].update(source_partition=source, target_partition=target)
                settings(variant)
                variant[METHOD]["cache"] = str(_cache_path(variant, dataset, METHOD))
                proposed.append((dataset, label, variant))
        else:
            settings(config)
            options["cache"] = str(_cache_path(config, dataset, METHOD))
            proposed.append((dataset, "multiscale", config))
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
                         "config": portable_path(config_path) if coupling_study else str(config_path),
                         "config_sha256": file_sha256(config_path),
                         "bank": None, "run": None, "evaluations": {}})
    manifest = {"format": FORMAT, "created_at": datetime.now(timezone.utc).isoformat(),
                "smoke": smoke, "datasets": datasets, "bank_seed": bank_seed,
                "training_seeds": seeds, "with_nsot": with_nsot, "evaluation_nfes": nfe,
                "partition_study": partition_study,
                "coupling_study": coupling_study,
                "protocol": {"matching": "multiscale patch Wasserstein cost, restricted block permutation",
                             "source": "original standard Gaussian bank; fresh Gaussian inference",
                             "model_training": "unchanged original checkerboard/horse baseline settings unless overridden",
                             "online_perturbation": "none for multiscale; beta0 and beta0.2 controls only if requested",
                             "marginals": "exact finite empirical endpoints; not exact population marginals",
                             "timing": "precompute shared once per unique bank; training loop and inference separately"},
                "endpoint_checks": {}, "jobs": jobs}
    if partition_study:
        manifest["protocol"]["partition_comparison"] = {
            "variants": [label for label, _, _ in PARTITION_VARIANTS],
            "changed": "source/target partition directions only; PCA uses canonical sign and axis fallback",
            "fixed": "endpoint coordinates, M, scale counts/weights, exact solvers, model, training, evaluation",
            "rng": "same model seed, pair-index RNG and uniform-time RNG across variants of each dataset/seed",
            "interpretation": "single-seed pilot unless repeated; lower coupling cost is not proof of better generation"}
    if coupling_study:
        manifest["protocol"] = {
            "variants": list(COUPLING_VARIANTS),
            "fixed": "same target coordinates, M, N, dataset-specific model, optimizer, updates, uniform time and evaluation",
            "multiscale": "original nested axis partitions and [1,1,1] costs",
            "leaf_only": "identical nested partitions and endpoints; [0,0,1] costs, exact leaf and within-leaf assignment",
            "qat": "paper maximum-variance/mean split to singletons; aligned continuous Gaussian quantiles; no local OT",
            "source": "MS-W2/leaf-only share a finite Gaussian bank; QAT draws fresh conditional Gaussian pairs",
            "rng": "same initial model, uniform-time stream and pair-index RNG; QAT source has a private RNG",
            "inference": "identical fresh Gaussian input, reference data, Euler NFE and native renderer per dataset/seed",
            "scope": "coupling-only comparison using existing project backbone; not official QAT-FM benchmark reproduction",
            "timing": "unique preprocessing, training loop including online sampling, inference excluding metrics/rendering",
            "seeds": "training seeds vary with one fixed target bank; multiple bank seeds are separate experiments"}
    path = directory / "manifest.json"
    save_manifest(path, manifest)
    print(f"experiment_manifest={portable_path(path) if coupling_study else path}", flush=True)
    print(f"planned_models={len(jobs)} smoke={smoke}; prepare stage does not train", flush=True)
    return path, manifest


def verify_job(job):
    path = project_path(job["config"])
    if file_sha256(path) != job["config_sha256"]:
        raise ValueError(f"Prepared YAML changed: {path}; start a new experiment")
    config = _read_yaml(path)
    if config["coupling"] not in ("multiscale_wasserstein", "nsot", "qat") or config["dataset"] != job["dataset"]:
        raise ValueError("Manifest/config dataset or method mismatch")
    if job["bank"]:
        bank = job["bank"]
        if file_sha256(project_path(bank["path"])) != bank["sha256"]:
            raise ValueError(f"Prepared bank changed: {bank['path']}")
        if config[config["coupling"]].get("cache_sha256") != bank["sha256"]:
            raise ValueError("Prepared YAML and manifest disagree about bank hash")
    if job["run"]:
        for name in ("config", "checkpoint", "training_metadata"):
            item = job["run"][name]
            if file_sha256(project_path(item["path"])) != item["sha256"]:
                raise ValueError(f"Saved training artifact changed: {item['path']}")
    for result in job["evaluations"].values():
        for item in result.values():
            if file_sha256(project_path(item["path"])) != item["sha256"]:
                raise ValueError(f"Evaluation artifact changed: {item['path']}")
    return config


def verify_coupling_protocol(manifest):
    """Fail before expensive preparation/training if a comparison is no longer controlled."""
    if not manifest.get("coupling_study"):
        return
    expected_jobs = {(dataset, seed, label) for dataset in manifest["datasets"]
                     for seed in manifest["training_seeds"] for label in COUPLING_VARIANTS}
    actual_jobs = [(job["dataset"], job["seed"], job["label"]) for job in manifest["jobs"]]
    if len(actual_jobs) != len(expected_jobs) or set(actual_jobs) != expected_jobs:
        raise ValueError("Coupling study requires exactly MS-W2, leaf-only and QAT per dataset/seed")
    counts = [2, 4, 8] if manifest["smoke"] else [4, 20, 100]
    for dataset in manifest["datasets"]:
        for seed in manifest["training_seeds"]:
            configs = {job["label"]: verify_job(job) for job in manifest["jobs"]
                       if job["dataset"] == dataset and job["seed"] == seed}
            base = configs["multiscale"]
            for label, config in configs.items():
                for key in ("dataset", "seed", "device", "dtype", "model", "data", "training", "evaluation"):
                    if config.get(key) != base.get(key):
                        raise ValueError(f"Coupling study {dataset}/{label}: unequal {key}")
                if config["seed"] != seed:
                    raise ValueError("Coupling study config seed differs from manifest")
                method = "qat" if label == "qat" else "multiscale_wasserstein"
                if config["coupling"] != method:
                    raise ValueError(f"Coupling study {label}: incorrect coupling method")
                options = config[method]
                if options["superset_size"] != base["multiscale_wasserstein"]["superset_size"] \
                        or options["cache_seed"] != manifest["bank_seed"]:
                    raise ValueError("Coupling study requires identical M and bank seed")
                if label != "qat":
                    weights = [0, 0, 1] if label == "leaf_only" else [1, 1, 1]
                    if (options["partition_counts"] != counts or options["scale_weights"] != weights
                            or options.get("source_partition", "axis") != "axis"
                            or options.get("target_partition", "axis") != "axis"):
                        raise ValueError("Coupling study requires original axis hierarchy and fixed 111/001 weights")


def prepare_jobs(path, manifest):
    verify_coupling_protocol(manifest)
    for job in manifest["jobs"]:
        config = verify_job(job)
        if job["bank"] is not None:
            continue
        if config["coupling"] == "nsot":
            from nsot import prepare
        elif config["coupling"] == "qat":
            from qat_coupling import prepare
        else:
            from multiscale_wasserstein import prepare
        with project_directory():
            cache, metadata = prepare(config, job["dataset"])
        cache = project_path(cache).resolve()
        digest = file_sha256(cache)
        config[config["coupling"]]["cache_sha256"] = digest
        # Keep the input YAML immutable. If interrupted before the manifest update,
        # the unreferenced prepared YAML can be verified and adopted on resume.
        prepared_path = project_path(job["config"]).with_name(Path(job["config"]).stem + "_prepared.yaml")
        serialized = yaml.safe_dump(config, sort_keys=False)
        if prepared_path.exists():
            if _read_yaml(prepared_path) != config:
                raise ValueError(f"Stale prepared config disagrees with current bank: {prepared_path}")
        else:
            with prepared_path.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
        job["config"] = portable_path(prepared_path) if manifest.get("coupling_study") else str(prepared_path)
        job["config_sha256"] = file_sha256(prepared_path)
        job["bank"] = {"path": portable_path(cache) if manifest.get("coupling_study") else str(cache), "sha256": digest,
                       "precompute_seconds": metadata["precompute_seconds"], "metadata": metadata}
        save_manifest(path, manifest)
    if manifest.get("coupling_study"):
        for dataset in manifest["datasets"]:
            entries = [job for job in manifest["jobs"] if job["dataset"] == dataset]
            reference = next(job for job in entries if job["label"] == "multiscale")
            with np.load(project_path(reference["bank"]["path"]), allow_pickle=False) as bank:
                source, target = bank["source"], bank["target"]
            for job in entries:
                with np.load(project_path(job["bank"]["path"]), allow_pickle=False) as bank:
                    if not np.array_equal(target, bank["target"]):
                        raise ValueError(f"{dataset}/{job['label']}: target endpoints differ")
                    if job["label"] != "qat" and not np.array_equal(source, bank["source"]):
                        raise ValueError(f"{dataset}/{job['label']}: source endpoints differ")
                    if job["label"] == "qat" and "source" in bank.files:
                        raise ValueError("QAT must use a continuous Gaussian, not a finite source bank")
            manifest["endpoint_checks"][dataset] = {
                "same_target": True, "same_source_multiscale_leaf_only": True,
                "qat_source": "continuous Gaussian; no finite source bank",
                "bank_sha256": {job["label"]: job["bank"]["sha256"] for job in entries}}
        save_manifest(path, manifest)
    if manifest.get("partition_study"):
        for dataset in manifest["datasets"]:
            entries = [j for j in manifest["jobs"] if j["dataset"] == dataset]
            reference = entries[0]
            with np.load(reference["bank"]["path"], allow_pickle=False) as bank:
                source, target = bank["source"], bank["target"]
            for job in entries[1:]:
                with np.load(job["bank"]["path"], allow_pickle=False) as bank:
                    for key, expected in (("source", source), ("target", target)):
                        if not np.array_equal(expected, bank[key]):
                            raise ValueError(f"{dataset}/{job['label']}: partition study {key} endpoints differ")
            manifest["endpoint_checks"][dataset] = {
                "same_source": True, "same_target": True,
                "bank_sha256": {job["label"]: job["bank"]["sha256"] for job in entries}}
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


def _artifact(path, *, portable=False):
    path = project_path(path).resolve()
    return {"path": portable_path(path) if portable else str(path), "sha256": file_sha256(path)}


def train_jobs(path, manifest):
    verify_coupling_protocol(manifest)
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
        portable = manifest.get("coupling_study", False)
        job["run"] = {"config": _artifact(directory / "config.yaml", portable=portable),
                      "checkpoint": _artifact(directory / saved["checkpoint"], portable=portable),
                      "training_metadata": _artifact(directory / "training.json", portable=portable)}
        save_manifest(path, manifest)


def evaluate_jobs(path, manifest):
    verify_coupling_protocol(manifest)
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
                    if file_sha256(project_path(artifact["path"])) != artifact["sha256"]:
                        raise ValueError(f"Evaluation artifact changed: {artifact['path']}")
                continue
            with project_directory():
                image = Path(evaluate(job["run"]["config"]["path"], nfe)).resolve()
            portable = manifest.get("coupling_study", False)
            job["evaluations"][str(nfe)] = {"json": _artifact(image.with_suffix(".json"), portable=portable),
                                            "png": _artifact(image, portable=portable)}
            save_manifest(path, manifest)


def write_summary(path, manifest):
    verify_coupling_protocol(manifest)
    banks, rows = {}, []
    for job in manifest["jobs"]:
        verify_job(job)
        if job["bank"]:
            banks[job["bank"]["path"]] = {"sha256": job["bank"]["sha256"],
                                          "precompute_seconds": job["bank"]["precompute_seconds"],
                                          "kind": "quantile_tree" if job["label"] == "qat" else "base_bank"}
        row = {"dataset": job["dataset"], "method": job["label"], "seed": job["seed"],
               "bank": job["bank"]["path"] if job["bank"] else None,
               "training_seconds": None, "evaluations": {}}
        if job["run"]:
            with project_path(job["run"]["training_metadata"]["path"]).open(encoding="utf-8") as stream:
                row["training_seconds"] = json.load(stream)["training_seconds"]
        for nfe, artifacts in job["evaluations"].items():
            with project_path(artifacts["json"]["path"]).open(encoding="utf-8") as stream:
                result = json.load(stream)
            # Retain the standard evaluator's complete metric schema, including horse ROI metrics.
            row["evaluations"][nfe] = {"result": result, "json": artifacts["json"]["path"],
                                        "png": artifacts["png"]["path"]}
        rows.append(row)
    save_manifest(path.parent / "summary.json", {"format": FORMAT, "smoke": manifest["smoke"],
                                                  "partition_study": manifest.get("partition_study", False),
                                                  "coupling_study": manifest.get("coupling_study", False),
                                                  "protocol": manifest["protocol"],
                                                  "banks": banks, "jobs": rows,
                                                  "endpoint_checks": manifest["endpoint_checks"],
                                                  "unique_precompute_seconds": sum(bank["precompute_seconds"] for bank in banks.values()),
                                                  "timing_note": "Precompute counted once per unique bank. Historical construction times, not current resume wall time. Inference excludes rendering/metrics."})
    if manifest.get("partition_study") or manifest.get("coupling_study"):
        from multiscale_partition_report import write_report
        write_report(path.parent, manifest, root=ROOT)


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
    parser.add_argument("--partition-study", action="store_true",
                        help="Compare axis/axis, axis/PCA, PCA/PCA with identical endpoints and training settings")
    parser.add_argument("--coupling-study", action="store_true",
                        help="Compare original MS-W2, leaf-only block OT and paper-based QAT coupling")
    parser.add_argument("--source-partition", choices=("axis", "pca"))
    parser.add_argument("--target-partition", choices=("axis", "pca"))
    parser.add_argument("--with-nsot", action="store_true", help="Also train matched-endpoint NSOT beta=0 and beta=0.2 controls")
    parser.add_argument("--smoke", action="store_true", help="Tiny CPU two-update pipeline check; not a quality experiment")
    parser.add_argument("--nfe", nargs="+", type=int)
    args = parser.parse_args()
    if args.stage == "eval" and not args.manifest:
        parser.error("--stage eval requires --manifest for previously trained jobs")
    if args.manifest:
        if args.smoke or args.with_nsot or args.partition_study or args.coupling_study or any(v is not None for v in (
                args.datasets, args.seeds, args.bank_seed, args.device, args.steps,
                args.superset_size, args.partition_counts, args.scale_weights,
                args.source_partition, args.target_partition)):
            parser.error("With --manifest only --stage/--nfe may change")
        path = project_path(args.manifest).resolve()
        with path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("format") != FORMAT:
            raise ValueError("Unsupported manifest format")
        # Reject unsupported or altered saved jobs before modifying the manifest.
        for job in manifest["jobs"]:
            verify_job(job)
        verify_coupling_protocol(manifest)
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
            partition_study=args.partition_study, source_partition=args.source_partition,
            target_partition=args.target_partition, coupling_study=args.coupling_study)
    if manifest["smoke"]:
        torch.set_num_threads(1)
        print("SMOKE ONLY: M=64, N=16, CPU, two optimizer updates; not quality evidence", flush=True)
    prepare_jobs(path, manifest)
    if args.stage in ("train", "all"):
        train_jobs(path, manifest)
    if args.stage in ("eval", "all"):
        evaluate_jobs(path, manifest)
    write_summary(path, manifest)
    displayed_path = portable_path(path) if manifest.get("coupling_study") else str(path)
    displayed_summary = portable_path(path.parent / "summary.json") if manifest.get("coupling_study") else str(path.parent / "summary.json")
    print(f"experiment_manifest={displayed_path}", flush=True)
    print(f"summary={displayed_summary}", flush=True)
    if args.stage == "prepare":
        print(f'next_command=python run_multiscale.py --manifest "{displayed_path}" --stage all', flush=True)


if __name__ == "__main__":
    main()
