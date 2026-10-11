"""Four-way TG stream/dense-superset x no-noise/hybrid comparison.

Default command PREPARES, TRAINS and EVALUATES, not evaluation-only.
--dry-run is read-only. --stage smoke runs an isolated tiny CPU pipeline.
Resume via the exact manifest printed by a previous invocation, never latest.
"""

import argparse
import copy
import importlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml


ROOT = Path(__file__).resolve().parent
VARIANTS = ("stream", "stream_hybrid", "superset", "superset_hybrid")
NFES = (1, 2, 4, 8, 16, 32, 64, 128)


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(path, payload):
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify(path, expected):
    path = Path(path)
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"Recorded artifact missing or changed: {path}")
    return path


def planned_jobs(args):
    datasets = ("checkerboard", "horse") if args.dataset in (None, "all") else (args.dataset,)
    jobs = []
    for dataset in datasets:
        prefix = "horse_" if dataset == "horse" else ""
        for variant in args.variants:
            dense = variant.startswith("superset")
            method = "target_guided_superset" if dense else "target_guided_cached"
            template = ROOT / f"{dataset}_experiments" / f"{prefix}{method}_k8_n256_seed0.yaml"
            config = yaml.safe_load(template.read_text(encoding="utf-8"))
            config["seed"] = args.seed
            config["checkpoint"] = f"{dataset}_{variant}_seed{args.seed}.pt"
            beta = args.beta if variant.endswith("hybrid") else 0.0
            if dense:
                config["tg_superset"].update(
                    superset_size=args.superset_size, num_supersets=args.supersets, seed=args.cache_seed, beta=beta,
                    path=str(ROOT / "coupling_cache" /
                             f"{dataset}_tg_dense_m{args.superset_size}_r{args.supersets}_k8_seed{args.cache_seed}"))
            else:
                config["tg_cache"].update(
                    sampling="stream", num_clouds=None, seed=args.cache_seed, beta=beta,
                    path=str(ROOT / "coupling_cache" / f"{dataset}_tg_stream_k8_n256_seed{args.cache_seed}"))
            jobs.append({"dataset": dataset, "variant": variant, "template": str(template), "config": config})
    return jobs


def create_manifest(args, jobs):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = (Path(args.manifest).resolve() if args.manifest else
            ROOT / "experiment_manifests" / f"tg_superset_{stamp}_{uuid4().hex[:8]}" / "manifest.json")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    configs = path.parent / ("configs_" + uuid4().hex[:8])
    configs.mkdir(exist_ok=False)
    smoke = args.stage == "smoke"
    payload = {"format_version": 1, "experiment": "tg_superset", "root": str(ROOT),
               "created_at": datetime.now(timezone.utc).isoformat(), "smoke_only": smoke,
               "nfe": [1, 2] if smoke else args.nfe, "jobs": []}
    for job in jobs:
        config = copy.deepcopy(job["config"])
        dense = config["coupling"] == "target_guided_superset"
        if smoke:
            config["device"] = "cpu"
            config["data"].update(batch_size=2, n_points=32)
            config["model"].update(d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
            config["training"].update(num_steps=2, log_every=1)
            config["evaluation"].update(batch_size=2, histogram_bins=16)
            section = "tg_superset" if dense else "tg_cache"
            stem = "superset" if dense else "stream"
            config[section]["path"] = str(configs / "cache" / f"{job['dataset']}_{stem}")
            if dense:
                config[section].update(superset_size=128, num_supersets=2)
            else:
                config[section].update(num_workers=0, prepare_batch_size=2)
        file = configs / f"{job['dataset']}_{job['variant']}.yaml"
        with file.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        payload["jobs"].append({"dataset": job["dataset"], "variant": job["variant"],
                                "input_config": str(file), "input_sha256": digest(file), "evaluations": {}})
    write_manifest(path, payload)
    return path, payload


def load_manifest(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1 or payload.get("experiment") != "tg_superset" \
            or Path(payload.get("root", "")).resolve() != ROOT or not payload.get("jobs"):
        raise ValueError("Unsupported manifest or wrong project location")
    return path, payload


def prepare_job(job):
    from experiment import read_config
    from tg_cache import _fingerprint
    config = read_config(verify(job["input_config"], job["input_sha256"]))
    dense = config["coupling"] == "target_guided_superset"
    module, section = ("tg_superset", "tg_superset") if dense else ("tg_cache", "tg_cache")
    path, meta = importlib.import_module(module).prepare(config, job["dataset"])
    config[section]["cache_sha256"] = _fingerprint(meta)
    effective = Path(job["input_config"]).with_suffix(".prepared.yaml")
    if effective.exists():
        if yaml.safe_load(effective.read_text(encoding="utf-8")) != config:
            raise ValueError(f"Prepared config differs; do not overwrite: {effective}")
    else:
        with effective.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
    job.update(prepared_config=str(effective), prepared_sha256=digest(effective),
               cache_path=str(Path(path).resolve()), precompute_seconds=meta["precompute_seconds"])


def train_job(job):
    config_path = verify(job["prepared_config"], job["prepared_sha256"])
    module = "train_horse" if job["dataset"] == "horse" else "train"
    run = Path(importlib.import_module(module).main(config_path)).resolve()
    saved = run / "config.yaml"
    config = yaml.safe_load(saved.read_text(encoding="utf-8"))
    checkpoint = run / config["checkpoint"]
    job.update(run_config=str(saved), run_config_sha256=digest(saved),
               checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint))


def evaluate_job(job, nfe, *, smoke=False):
    verify(job["run_config"], job["run_config_sha256"])
    verify(job["checkpoint"], job["checkpoint_sha256"])
    module = "eval_horse" if job["dataset"] == "horse" else "eval"
    evaluator = importlib.import_module(module)
    # Horse evaluator always renders. Tiny smoke still exercises its real path.
    output = Path(evaluator.main(job["run_config"], nfe) if job["dataset"] == "horse" else
                  evaluator.main(job["run_config"], nfe, render=not smoke)).resolve()
    result = output.with_suffix(".json")
    return {"json": str(result), "sha256": digest(result)}


def run(args):
    if not math.isfinite(args.beta) or not 0 < args.beta <= 1 or min(args.supersets, args.superset_size) < 1 \
            or min(args.seed, args.cache_seed) < 0 or any(n < 1 for n in args.nfe):
        raise ValueError("Invalid beta, superset dimensions, seed or NFE")
    if args.stage in ("train", "eval") and not args.manifest:
        raise ValueError("train/eval requires --manifest from prepare/all")
    if args.manifest and Path(args.manifest).exists():
        path, payload = load_manifest(args.manifest)
    else:
        if args.stage in ("train", "eval"):
            raise FileNotFoundError(args.manifest)
        jobs = planned_jobs(args)
        if args.dry_run:
            print(json.dumps({"stage": args.stage, "training_included": args.stage in ("all", "smoke"),
                              "jobs": jobs}, indent=2), flush=True)
            return None
        path, payload = create_manifest(args, jobs)
    if args.dry_run:
        print(json.dumps(payload, indent=2), flush=True)
        return path
    if payload["smoke_only"] != (args.stage == "smoke"):
        raise ValueError("Smoke manifests cannot be resumed as full experiments; rerun --stage smoke instead")
    print(f"manifest={path}", flush=True)
    previous = Path.cwd()
    os.chdir(path.parent if payload["smoke_only"] else ROOT)
    try:
        for job in payload["jobs"]:
            if args.dataset not in (None, "all", job["dataset"]) or job["variant"] not in args.variants:
                continue
            print(f"job={job['dataset']}/{job['variant']}", flush=True)
            if args.stage in ("all", "prepare", "smoke"):
                prepare_job(job)
                write_manifest(path, payload)
            if args.stage in ("all", "train", "smoke"):
                if "prepared_config" not in job:
                    raise ValueError("Job not prepared; run --stage prepare with this manifest")
                if "run_config" not in job:
                    train_job(job)
                    write_manifest(path, payload)
                else:
                    verify(job["run_config"], job["run_config_sha256"])
                    verify(job["checkpoint"], job["checkpoint_sha256"])
            if args.stage in ("all", "eval", "smoke"):
                if "run_config" not in job:
                    raise ValueError("Job not trained; run --stage train with this manifest")
                for nfe in payload["nfe"]:
                    key = str(nfe)
                    if key in job["evaluations"] and not args.rerun_eval:
                        row = job["evaluations"][key]
                        verify(row["json"], row["sha256"])
                        continue
                    job["evaluations"][key] = evaluate_job(job, nfe, smoke=payload["smoke_only"])
                    write_manifest(path, payload)
    finally:
        os.chdir(previous)
    print(f"completed_manifest={path}", flush=True)
    return path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=("all", "checkerboard", "horse"), default=None)
    result.add_argument("--stage", choices=("all", "prepare", "train", "eval", "smoke"), default="all")
    result.add_argument("--manifest")
    result.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    result.add_argument("--beta", type=float, default=0.1)
    result.add_argument("--superset-size", type=int, default=8192)
    result.add_argument("--supersets", type=int, default=8)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--cache-seed", type=int, default=0)
    result.add_argument("--nfe", nargs="+", type=int, default=list(NFES))
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--rerun-eval", action="store_true")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
