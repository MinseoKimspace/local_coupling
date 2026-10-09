"""Run the four matched balanced-anchor vs constrained GMM-EM prior experiments.

Run from the experiment machine's repository root:
    python run_prior_comparison.py
Reevaluate only the exact runs recorded by a previous invocation:
    python run_prior_comparison.py --stage eval --manifest comparison_results/.../manifest.json

This runner calls existing Python APIs, so run-directory names never need to be
copied from screenshots or guessed. It does not remove or overwrite pair caches,
checkpoints or evaluation results. A unique manifest records each completed job
before proceeding, so its saved run configs remain available after a failure.
"""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
DEFAULT_NFES = (1, 2, 4, 8, 16, 32, 64, 128)
JOBS = (
    ("checkerboard", "balanced_anchor", "checkerboard_experiments/nsot_anchor_prior_k8_n256_seed0.yaml"),
    ("checkerboard", "gmm_em", "checkerboard_experiments/nsot_gmm_prior_k8_n256_seed0.yaml"),
    ("horse", "balanced_anchor", "horse_experiments/horse_nsot_anchor_prior_k8_n256_seed0.yaml"),
    ("horse", "gmm_em", "horse_experiments/horse_nsot_gmm_prior_k8_n256_seed0.yaml"),
)


def positive_integer(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def save_manifest(path, payload):
    # Only this invocation's newly created manifest is updated. The caller must
    # not pass an existing manifest here when requesting an evaluation-only run.
    # Never truncate the last valid recovery record before the new JSON is
    # complete. Both paths are in this invocation's own output directory.
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def new_manifest(stage, steps, jobs, *, parent=None):
    now = datetime.now(timezone.utc)
    directory = ROOT / "comparison_results" / f"prior_comparison_{now:%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "manifest.json"
    payload = {
        "format_version": 1,
        "comparison": "balanced-anchor vs equal-weight fixed-sigma GMM-EM prior; same NSOT",
        "created_at": now.isoformat(),
        "repository_root": str(ROOT),
        "stage": stage,
        "training_steps": steps,
        "parent_manifest": str(parent) if parent is not None else None,
        "jobs": jobs,
    }
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(f"comparison_manifest={path}", flush=True)
    return path, payload


def validate_saved_config(job):
    if job.get("dataset") not in ("checkerboard", "horse"):
        raise ValueError("Manifest contains an unsupported dataset")
    value = job.get("saved_config")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Manifest has no saved_config for a completed training job")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("Manifest saved_config must be the captured absolute run config path")
    if path.name != "config.yaml" or not path.is_file():
        raise FileNotFoundError(f"Saved training config not found: {path}")
    return path


def release_memory():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_jobs(path, payload):
    from experiment import read_config
    from nsot import prepare

    for job in payload["jobs"]:
        config = read_config(job["template_config"])
        cache_path = Path(config["nsot"]["cache"])
        existed = cache_path.exists()
        start = perf_counter()
        cache_path, metadata = prepare(config, job["dataset"])
        job["preparation"] = {
            "cache": str(cache_path.resolve()),
            "cache_reused": existed,
            "prepare_invocation_seconds": perf_counter() - start,
            "original_precompute_seconds": metadata.get("precompute_seconds"),
            "original_precompute_timing_scope": metadata.get("precompute_timing_scope"),
        }
        save_manifest(path, payload)


def train_jobs(path, payload, steps):
    from train import main as train_checkerboard
    from train_horse import main as train_horse

    for job in payload["jobs"]:
        trainer = train_horse if job["dataset"] == "horse" else train_checkerboard
        run_directory = Path(trainer(job["template_config"], steps=steps)).resolve()
        job["saved_config"] = str(run_directory / "config.yaml")
        job["training_json"] = str(run_directory / "training.json")
        validate_saved_config(job)
        save_manifest(path, payload)
        release_memory()


def evaluate_jobs(path, payload, nfes, roi_file):
    from eval import main as evaluate_checkerboard
    from eval_horse import main as evaluate_horse

    for job in payload["jobs"]:
        config_path = validate_saved_config(job)
        job["evaluations"] = []
        for nfe in nfes:
            if job["dataset"] == "horse":
                result = evaluate_horse(str(config_path), nfe, roi_file=roi_file)
            else:
                result = evaluate_checkerboard(str(config_path), nfe)
            json_path = Path(result).with_suffix(".json").resolve()
            if not json_path.is_file():
                raise FileNotFoundError(f"Evaluator returned no matching JSON: {json_path}")
            job["evaluations"].append({"nfe": nfe, "json": str(json_path)})
            save_manifest(path, payload)
            release_memory()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "prepare", "train", "eval"), default="all",
                        help="all=prepare/train/eval; train also prepares or validates pair caches")
    parser.add_argument("--steps", type=positive_integer, default=10000)
    parser.add_argument("--nfes", nargs="+", type=positive_integer, default=list(DEFAULT_NFES))
    parser.add_argument("--manifest", type=Path,
                        help="Required for --stage eval; uses only the exact saved runs in this manifest")
    parser.add_argument("--roi-file", type=Path,
                        help="Optional fixed horse ROI definition; shared by both horse experiments")
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        parser.error(f"Run this command from the repository root: {ROOT}")
    if args.manifest is not None and args.stage != "eval":
        parser.error("--manifest is only accepted with --stage eval")
    if args.stage == "eval" and args.manifest is None:
        parser.error("--stage eval requires --manifest; unresolved template YAMLs cannot be evaluated")
    if len(set(args.nfes)) != len(args.nfes):
        parser.error("--nfes must not contain duplicates")
    roi_file = str(args.roi_file.resolve()) if args.roi_file is not None else None
    if roi_file is not None and not Path(roi_file).is_file():
        parser.error(f"ROI definition does not exist: {roi_file}")

    if args.stage == "eval":
        parent = args.manifest.resolve()
        with parent.open(encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous.get("format_version") != 1 or not isinstance(previous.get("jobs"), list):
            parser.error("Unsupported comparison manifest")
        # A failed training run can have a valid subset. Evaluate completed jobs
        # only, preserving exact paths and leaving the original manifest intact.
        jobs = [{key: value for key, value in job.items() if key != "evaluations"}
                for job in previous["jobs"] if job.get("saved_config")]
        if not jobs:
            parser.error("Manifest contains no completed training runs")
        for job in jobs:
            validate_saved_config(job)
        path, payload = new_manifest(args.stage, previous.get("training_steps"), jobs, parent=parent)
    else:
        jobs = [{"dataset": dataset, "prior": prior, "template_config": str(ROOT / template)}
                for dataset, prior, template in JOBS]
        path, payload = new_manifest(args.stage, args.steps, jobs)
        prepare_jobs(path, payload)
        if args.stage in ("all", "train"):
            train_jobs(path, payload, args.steps)
    if args.stage in ("all", "eval"):
        evaluate_jobs(path, payload, args.nfes, roi_file)
    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    save_manifest(path, payload)
    print(f"completed_comparison_manifest={path}", flush=True)
    return path


if __name__ == "__main__":
    main()
