"""Prepare, train and evaluate baseline TG and two local-path coupling variants.

Default execution includes training. --stage eval requires the exact manifest
from a previous invocation; it never searches for a latest run. --dry-run is
read-only and imports no training dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
VARIANTS = ("baseline", "path_affine", "path_affine_subpatch")
DEFAULT_NFE = (1, 2, 4, 8, 16, 32, 64, 128)


def config_path(dataset, variant):
    prefix = "horse_" if dataset == "horse" else ""
    suffix = "" if variant == "baseline" else "_" + variant
    return ROOT / f"{dataset}_experiments" / f"{prefix}target_guided_cached{suffix}_k8_n256_seed0.yaml"


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_manifest(path, manifest):
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_file(path, digest, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    if _digest(path) != digest:
        raise ValueError(f"{label} changed since it was recorded: {path}")
    return path


def _create_manifest(args):
    import yaml

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if args.manifest is None:
        path = ROOT / "experiment_manifests" / f"tg_local_{stamp}_{uuid4().hex[:8]}" / "manifest.json"
    else:
        path = Path(args.manifest).resolve()
    if path.exists():
        raise FileExistsError(f"Manifest already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = path.parent / (path.stem + "_configs_" + uuid4().hex[:8])
    snapshot_dir.mkdir(exist_ok=False)
    datasets = ("checkerboard", "horse") if args.dataset in (None, "all") else (args.dataset,)
    manifest = {
        "format_version": 1, "experiment": "tg_local", "root": str(ROOT),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preflight_only": args.stage == "preflight", "sampling_override": args.sampling,
        "jobs": [],
    }
    for dataset in datasets:
        for variant in VARIANTS:
            source = config_path(dataset, variant)
            config = yaml.safe_load(source.read_text(encoding="utf-8"))
            if args.stage == "preflight":
                config["device"] = "cpu"
                config["tg_cache"].update({
                    "sampling": "bank", "num_clouds": args.clouds, "num_workers": 0,
                    "path": str(snapshot_dir / "cache" / f"{dataset}_{variant}"),
                })
                config["tg_cache"].pop("cache_sha256", None)
            snapshot = snapshot_dir / f"{dataset}_{variant}.yaml"
            with snapshot.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(config, stream, sort_keys=False)
            manifest["jobs"].append({
                "dataset": dataset, "variant": variant, "source_config": str(source),
                "input_config": str(snapshot), "input_sha256": _digest(snapshot),
                "prepared": False, "evaluations": {},
            })
    _write_manifest(path, manifest)
    return path, manifest


def _load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or manifest.get("experiment") != "tg_local":
        raise ValueError("Unsupported TG local manifest")
    if Path(manifest["root"]).resolve() != ROOT:
        raise ValueError("Manifest belongs to another project location")
    if not manifest.get("jobs"):
        raise ValueError("Manifest has no jobs")
    return path, manifest


def preflight_report(job, metadata):
    """Small, stable stdout report; the full diagnostic remains in metadata.json."""
    summary = metadata.get("local_summary", {})
    metrics = summary.get("metrics", {})

    def value(name, statistic="mean"):
        item = metrics.get(name)
        return item.get(statistic) if isinstance(item, dict) else item

    baseline = value("heldout_baseline_score")
    guided = value("heldout_guided_score")
    warning = None
    if baseline is not None and guided is not None and guided >= baseline:
        warning = "Heldout score did not improve; this geometry pilot provides no evidence of quality improvement."
    return {
        "dataset": job["dataset"], "variant": job["variant"],
        "cache_path": job["cache_path"], "metadata_path": job["metadata_path"],
        "precompute_seconds": metadata.get("precompute_seconds"),
        "local_guidance_seconds": metadata.get("local_guidance_seconds"),
        "coarse_noop_clouds": summary.get("noop_clouds"), "clouds": metadata.get("num_clouds"),
        "baseline_score_mean": value("baseline_score"), "guided_score_mean": value("guided_score"),
        "heldout_baseline_score_mean": baseline, "heldout_guided_score_mean": guided,
        "min_scored_fraction": value("min_scored_fraction", "min"),
        "accepted_swaps_mean": value("accepted_swaps"), "warning": warning,
    }


def _prepare(job, *, sampling, preflight):
    path, metadata = importlib.import_module("prepare_tg").main(
        job["input_config"], job["dataset"], sampling=None if preflight else sampling,
    )
    effective = Path(path) / "config.yaml" if sampling is not None and not preflight else Path(job["input_config"])
    if not effective.is_file():
        raise FileNotFoundError(f"Preparation did not produce its training config: {effective}")
    job.update({
        "prepared": True, "cache_path": str(Path(path).resolve()),
        "effective_config": str(effective.resolve()), "effective_sha256": _digest(effective),
        "precompute_seconds": metadata.get("precompute_seconds"),
        "metadata_path": str((Path(path) / "metadata.json").resolve()),
    })
    if preflight:
        print(json.dumps(preflight_report(job, metadata), indent=2, allow_nan=False), flush=True)


def _train(job):
    trainer = "train_horse" if job["dataset"] == "horse" else "train"
    run_dir = Path(importlib.import_module(trainer).main(job["effective_config"])).resolve()
    saved = run_dir / "config.yaml"
    if not saved.is_file():
        raise FileNotFoundError(f"Trainer did not save its run config: {saved}")
    job.update({"run_dir": str(run_dir), "run_config": str(saved), "run_config_sha256": _digest(saved)})


def _evaluate(job, nfe):
    evaluator = "eval_horse" if job["dataset"] == "horse" else "eval"
    output = Path(importlib.import_module(evaluator).main(job["run_config"], nfe)).resolve()
    result = output.with_suffix(".json")
    if not result.is_file():
        raise FileNotFoundError(f"Evaluator did not save its metrics: {result}")
    return {"json": str(result), "json_sha256": _digest(result), "image": str(output)}


def run(args):
    if args.stage in ("train", "eval") and args.manifest is None:
        raise ValueError(f"--stage {args.stage} requires --manifest from a previous prepare/all invocation")
    if args.clouds < 1 or any(nfe < 1 for nfe in args.nfe):
        raise ValueError("clouds and NFE values must be positive")
    if args.stage == "preflight" and args.sampling == "stream":
        raise ValueError("Preflight uses a small CPU bank; omit --sampling stream")
    existing = args.manifest is not None and Path(args.manifest).is_file()
    if existing:
        path, manifest = _load_manifest(args.manifest)
        preflight = bool(manifest["preflight_only"])
        if preflight != (args.stage == "preflight"):
            raise ValueError("A preflight manifest cannot be used to train or evaluate experiments")
        if args.sampling is not None and args.sampling != manifest["sampling_override"]:
            raise ValueError("Sampling override differs from this manifest; create a new comparison")
        jobs = [job for job in manifest["jobs"] if args.dataset in (None, "all", job["dataset"])]
        if not jobs:
            raise ValueError("Requested dataset is absent from this manifest")
    else:
        if args.stage in ("train", "eval"):
            raise FileNotFoundError(f"Manifest not found: {args.manifest}")
        path, manifest = None, None
        datasets = ("checkerboard", "horse") if args.dataset in (None, "all") else (args.dataset,)
        jobs = [{"dataset": dataset, "variant": variant, "input_config": str(config_path(dataset, variant))}
                for dataset in datasets for variant in VARIANTS]
    if args.dry_run:
        print("DRY RUN: no cache creation, training, evaluation or file writes", flush=True)
        print(f"stage={args.stage} sampling={args.sampling or (manifest or {}).get('sampling_override') or 'bank'}")
        for job in jobs:
            print(f"{job['dataset']} / {job['variant']}: {job['input_config']}")
            if args.stage in ("all", "prepare", "preflight"):
                print(f"  prepare CPU geometry{' on ' + str(args.clouds) + ' clouds' if args.stage == 'preflight' else ''}")
            if args.stage in ("all", "train"):
                print(f"  train -> exact returned runs/.../config.yaml recorded in manifest")
            if args.stage in ("all", "eval"):
                print(f"  evaluate {job.get('run_config', '<saved config returned by this training>')} NFE={args.nfe}")
        return path
    if manifest is None:
        path, manifest = _create_manifest(args)
        jobs = manifest["jobs"]
    print(f"manifest={path}", flush=True)
    previous_cwd = Path.cwd()
    os.chdir(ROOT)
    try:
        for job in jobs:
            _verify_file(job["input_config"], job["input_sha256"], "Input config snapshot")
        if args.stage in ("all", "prepare", "preflight"):
            for job in jobs:
                if not job["prepared"]:
                    _prepare(job, sampling=manifest["sampling_override"], preflight=manifest["preflight_only"])
                    _write_manifest(path, manifest)
                else:
                    _verify_file(job["effective_config"], job["effective_sha256"], "Prepared config")
                    if manifest["preflight_only"]:
                        metadata = json.loads(Path(job["metadata_path"]).read_text(encoding="utf-8"))
                        print(json.dumps(preflight_report(job, metadata), indent=2, allow_nan=False), flush=True)
        if args.stage in ("all", "train"):
            for job in jobs:
                if not job["prepared"]:
                    raise ValueError(f"Prepare first: {job['dataset']} / {job['variant']}")
                _verify_file(job["effective_config"], job["effective_sha256"], "Prepared config")
                if "run_config" not in job:
                    _train(job)
                    _write_manifest(path, manifest)
                else:
                    _verify_file(job["run_config"], job["run_config_sha256"], "Saved run config")
        if args.stage in ("all", "eval"):
            # Validate every selected run before starting any evaluations.
            for job in jobs:
                if "run_config" not in job:
                    raise ValueError(f"Train first: {job['dataset']} / {job['variant']}")
                _verify_file(job["run_config"], job["run_config_sha256"], "Saved run config")
            for job in jobs:
                for nfe in dict.fromkeys(args.nfe):
                    previous = job["evaluations"].get(str(nfe))
                    if previous and not args.rerun_eval:
                        _verify_file(previous["json"], previous["json_sha256"], "Evaluation result")
                        continue
                    job["evaluations"][str(nfe)] = _evaluate(job, nfe)
                    _write_manifest(path, manifest)
    finally:
        os.chdir(previous_cwd)
    print(f"complete stage={args.stage} manifest={path}", flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("checkerboard", "horse", "all"), default=None,
                        help="Default: both datasets; with an existing manifest, select its recorded jobs")
    parser.add_argument("--stage", choices=("all", "preflight", "prepare", "train", "eval"), default="all")
    parser.add_argument("--sampling", choices=("bank", "stream"), default=None)
    parser.add_argument("--manifest", help="Exact comparison manifest; required for train/eval-only")
    parser.add_argument("--nfe", type=int, nargs="+", default=list(DEFAULT_NFE))
    parser.add_argument("--clouds", type=int, default=8, help="CPU preflight only; never changes full-training bank size")
    parser.add_argument("--rerun-eval", action="store_true", help="Generate new evaluation files for recorded runs")
    parser.add_argument("--dry-run", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
