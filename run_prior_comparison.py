"""Matched K=8 full-GMM + NSOT versus balanced-anchor-prior + NSOT.

Default invocation PREPARES CACHES, TRAINS FOUR NEW MODELS, and EVALUATES them.
    python run_prior_comparison.py
    python run_prior_comparison.py --smoke
    python run_prior_comparison.py --stage eval --manifest comparison_results/.../manifest.json

The ordinary GMM learns weights, means and full covariance matrices from the
training reference. The anchor prior retains equal weights and fixed sigma.
Both use the same dataset-specific backbone, optimizer, NSOT size and beta.
This changes the prior AND its component-preserving kernel, not only coupling.
"""

import argparse
import copy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4


ROOT = Path(__file__).resolve().parent
FORMAT_VERSION = 2
DEFAULT_NFES = (1, 2, 4, 8, 16, 32, 64, 128)
JOBS = (
    ("checkerboard", "anchor_prior", "checkerboard_experiments/nsot_anchor_prior_k8_n256_seed0.yaml"),
    ("checkerboard", "gmm_full", "checkerboard_experiments/nsot_gmm_full_prior_k8_n256_seed0.yaml"),
    ("horse", "anchor_prior", "horse_experiments/horse_nsot_anchor_prior_k8_n256_seed0.yaml"),
    ("horse", "gmm_full", "horse_experiments/horse_nsot_gmm_full_prior_k8_n256_seed0.yaml"),
)
METRICS = ("chamfer", "leakage", "histogram_js", "cell_mass_error",
           "thin_region_mass_mae", "gap_region_leakage")


def positive_integer(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def target_seed(value):
    result = int(value)
    if not 0 <= result < 2**63:
        raise argparse.ArgumentTypeError("must be in [0, 2**63)")
    return result


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path, payload):
    """Atomically update only this invocation's own recovery/summary record."""
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def announce(stage, job, suffix=""):
    print(f"[{stage}] dataset={job['dataset']} prior={job['prior']} {suffix}", flush=True)


def new_manifest(stage, steps, jobs, *, parent=None, target_seed=2026, smoke=False):
    now = datetime.now(timezone.utc)
    directory = ROOT / "comparison_results" / f"prior_comparison_{now:%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "manifest.json"
    payload = {
        "format_version": FORMAT_VERSION,
        "comparison": "equal-weight fixed-sigma balanced-anchor prior vs learned-weight full-covariance GMM prior; NSOT",
        "created_at": now.isoformat(), "repository_root": str(ROOT),
        "stage": stage, "status": "running", "training_steps": steps,
        "target_seed": target_seed, "smoke": smoke,
        "parent_manifest": str(parent) if parent is not None else None,
        "jobs": jobs,
        "interpretation": "Single seed; no statistical or quality claim. NFE=0 is the initial prior, not a generated sample.",
    }
    save_json(path, payload)
    print(f"comparison_manifest={path}", flush=True)
    return path, payload


def validate_saved_config(job):
    if job.get("dataset") not in ("checkerboard", "horse"):
        raise ValueError("Manifest contains an unsupported dataset")
    if job.get("prior") not in ("anchor_prior", "gmm_full"):
        raise ValueError("Manifest contains an unsupported prior")
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


def create_smoke_templates(path, payload):
    """Never use or alter production caches/checkpoints in a smoke run."""
    import yaml

    directory = path.parent / "smoke_configs"
    directory.mkdir(exist_ok=False)
    for job in payload["jobs"]:
        with Path(job["template_config"]).open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        config["device"] = "cpu"
        config["data"].update(batch_size=2, n_points=32)
        config["model"].update(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
        config["training"].update(num_steps=payload["training_steps"], log_every=1)
        config["evaluation"].update(batch_size=2, histogram_bins=16)
        config["anchor_flow"]["reference_points"] = 256
        if job["prior"] == "gmm_full":
            config["anchor_flow"].update(n_init=1, max_iter=300)
        identity = f"{job['dataset']}_{job['prior']}"
        config["nsot"].update(superset_size=128, cache=str(path.parent / "smoke_cache" / f"{identity}.npz"))
        config["checkpoint"] = f"smoke_{identity}.pt"
        template = directory / f"{identity}.yaml"
        with template.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        job["production_template_config"] = job["template_config"]
        job["template_config"] = str(template)
    save_json(path, payload)


def prepare_jobs(path, payload):
    from experiment import read_config
    from nsot import prepare

    for job in payload["jobs"]:
        announce("prepare", job)
        config = read_config(job["template_config"])
        existed = Path(config["nsot"]["cache"]).exists()
        start = perf_counter()
        cache, metadata = prepare(config, job["dataset"])
        job["preparation"] = {
            "cache": str(Path(cache).resolve()), "cache_reused": existed,
            "prepare_invocation_seconds": perf_counter() - start,
            "original_precompute_seconds": metadata.get("precompute_seconds"),
            "original_prior_fit_seconds": metadata.get("prior_fit_seconds"),
            "original_ot_seconds": metadata.get("ot_seconds"),
            "gmm_fit_details": metadata.get("gmm_fit_details"),
            "original_precompute_timing_scope": metadata.get("precompute_timing_scope"),
        }
        save_json(path, payload)


def train_jobs(path, payload, steps):
    from train import main as train_checkerboard
    from train_horse import main as train_horse

    for job in payload["jobs"]:
        announce("train NEW model", job, f"updates={steps}")
        trainer = train_horse if job["dataset"] == "horse" else train_checkerboard
        run = Path(trainer(job["template_config"], steps=steps)).resolve()
        job["saved_config"] = str(run / "config.yaml")
        job["training_json"] = str(run / "training.json")
        validate_saved_config(job)
        save_json(path, payload)
        release_memory()


def write_summary(path, payload):
    """Keep raw single-run measurements; never average unrelated NFEs/seeds."""
    jobs = []
    for job in payload["jobs"]:
        training = read_json(job["training_json"]) if job.get("training_json") and Path(job["training_json"]).is_file() else {}
        preparation = job.get("preparation", {})
        details = training.get("coupling_details", {})
        precompute = preparation.get("original_precompute_seconds", details.get("precompute_seconds"))
        fit = preparation.get("original_prior_fit_seconds", details.get("prior_fit_seconds"))
        online = training.get("training_seconds")
        item = {
            "dataset": job["dataset"], "prior": job["prior"],
            "saved_config": job.get("saved_config"), "training_steps": payload["training_steps"],
            "training_seconds": online, "precompute_seconds": precompute,
            "final_training_loss": training.get("final_loss"),
            "prior_fit_seconds": fit,
            "ot_seconds": preparation.get("original_ot_seconds", details.get("ot_seconds")),
            "gmm_fit_details": preparation.get("gmm_fit_details", details.get("gmm_fit_details")),
            "precompute_plus_training_seconds": online + precompute if online is not None and precompute is not None else None,
            "prepare_invocation_seconds": preparation.get("prepare_invocation_seconds"),
            "cache_reused": preparation.get("cache_reused"), "source_baseline": None,
            "evaluations": [],
        }
        for entry in job.get("evaluations", []):
            result = read_json(entry["json"])
            item["evaluations"].append({
                "nfe": entry["nfe"], "json": entry["json"],
                **{key: result[key] for key in METRICS if key in result},
                "inference_seconds": result.get("inference_seconds"),
                "source_sampling_seconds": result.get("source_sampling_seconds"),
                "sampling_plus_inference_seconds": result.get("sampling_plus_inference_seconds"),
                "evaluation_batch_size": result.get("evaluation_batch_size"),
                "total_points": result.get("total_points"),
                "evaluation_target_sha256": result.get("evaluation_target_sha256"),
                "improvement_from_prior": {key: result["source_" + key] - result[key]
                    for key in METRICS if isinstance(result.get(key), (int, float))
                    and isinstance(result.get("source_" + key), (int, float))},
            })
            source = {"nfe": 0, "json": entry["json"],
                      **{key: result["source_" + key] for key in METRICS if "source_" + key in result}}
            if item["source_baseline"] is None:
                item["source_baseline"] = source
        jobs.append(item)
    summary = {
        "format_version": 1, "manifest": str(path), "smoke": payload["smoke"],
        "target_seed": payload["target_seed"], "jobs": jobs,
        "timing_scope": {
            "training_seconds": "online optimization loop, excludes cache load, fit, model construction and saving",
            "precompute_seconds": "original prior fit + source/target draws + exact OT, excludes cache IO; retained even if reused",
            "prior_fit_seconds": "subset of precompute_seconds; do not add twice",
            "ot_seconds": "subset of precompute_seconds; exact superset assignment only; do not add twice",
            "prepare_invocation_seconds": "this invocation, including cache validation/loading when reused",
            "precompute_plus_training_seconds": "compute-only cold-start sum; not end-to-end wall time",
            "inference_seconds": "Euler integration only; excludes sampling, warmup, loading, metrics and rendering",
            "source_sampling_seconds": "actual fresh prior draw; includes validation/Cholesky and possible first-use overhead",
            "sampling_plus_inference_seconds": "sum of fresh prior draw and warmed integration; excludes model warmup/loading/metrics/rendering",
        },
        "notes": ["All metrics are raw single-seed results; lower CD/leakage/JS is better.",
                  "NFE=0 is the initial prior against the same target, not an additional model evaluation.",
                  "improvement_from_prior = initial-prior error minus generated error; positive means improvement.",
                  "Full GMM learns weights/means/covariances; anchor prior uses equal weights and fixed isotropic sigma.",
                  "Smoke results validate wiring only, not convergence or generation quality."],
    }
    output = path.parent / "summary.json"
    save_json(output, summary)
    payload["summary_json"] = str(output)
    return output


def evaluate_jobs(path, payload, nfes, roi_file):
    from eval import main as evaluate_checkerboard
    from eval_horse import main as evaluate_horse

    for job in payload["jobs"]:
        config = validate_saved_config(job)
        job["evaluations"] = []
        for nfe in nfes:
            announce("eval", job, f"NFE={nfe} saved_config={config}")
            kwargs = {"target_seed": payload["target_seed"]}
            if job["dataset"] == "horse":
                result = evaluate_horse(str(config), nfe, roi_file=roi_file, **kwargs)
            else:
                result = evaluate_checkerboard(str(config), nfe, **kwargs)
            json_path = Path(result).with_suffix(".json").resolve()
            if not json_path.is_file():
                raise FileNotFoundError(f"Evaluator returned no matching JSON: {json_path}")
            record = read_json(json_path)
            digest = record.get("evaluation_target_sha256")
            if digest is not None:
                reference_hashes = payload.setdefault("evaluation_target_hashes", {})
                previous_hash = reference_hashes.setdefault(job["dataset"], digest)
                if previous_hash != digest:
                    raise ValueError("Comparison target coordinates differ across priors/NFEs; check saved evaluation batch/seed settings")
            job["evaluations"].append({"nfe": nfe, "json": str(json_path)})
            write_summary(path, payload)
            save_json(path, payload)
            release_memory()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "prepare", "train", "eval"), default="all",
                        help="Default all = prepare + FOUR NEW trainings + eval; train also validates/prepares caches")
    parser.add_argument("--steps", type=positive_integer, help="Default 10000 updates (smoke: 4)")
    parser.add_argument("--nfes", nargs="+", type=positive_integer, help="Default 1,2,4,8,16,32,64,128 (smoke: 1,2)")
    parser.add_argument("--manifest", type=Path, help="Required for stage eval; captured SAVED run configs only")
    parser.add_argument("--datasets", nargs="+", choices=("checkerboard", "horse"))
    parser.add_argument("--priors", nargs="+", choices=("anchor_prior", "gmm_full"))
    parser.add_argument("--target-seed", type=target_seed, help="Common held-out target RNG seed; default 2026 or parent manifest seed")
    parser.add_argument("--roi-file", type=Path, help="Same fixed horse ROI JSON for both priors")
    parser.add_argument("--smoke", action="store_true", help="CPU-only isolated tiny run; NOT a quality experiment")
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        parser.error(f"Run from the repository root: {ROOT}")
    if (args.stage == "eval") != (args.manifest is not None):
        parser.error("--stage eval requires --manifest; --manifest is not used for training")
    if args.smoke and args.stage == "eval":
        parser.error("To reevaluate a smoke run, use its manifest without --smoke")
    if args.stage == "eval" and args.steps is not None:
        parser.error("--steps cannot change an already-trained run")
    steps = args.steps if args.steps is not None else (4 if args.smoke else 10000)
    if args.smoke and steps > 10:
        parser.error("--smoke is limited to at most 10 updates")
    nfes = args.nfes if args.nfes is not None else ([1, 2] if args.smoke else list(DEFAULT_NFES))
    if len(set(nfes)) != len(nfes):
        parser.error("--nfes must not contain duplicates")
    roi_file = str(args.roi_file.resolve()) if args.roi_file is not None else None
    if roi_file is not None and not Path(roi_file).is_file():
        parser.error(f"ROI definition does not exist: {roi_file}")
    selected = lambda job: ((args.datasets is None or job["dataset"] in args.datasets)
                            and (args.priors is None or job["prior"] in args.priors))
    if args.stage == "eval":
        parent = args.manifest.resolve()
        previous = read_json(parent)
        if previous.get("format_version") != FORMAT_VERSION or not isinstance(previous.get("jobs"), list):
            parser.error("Expected a full-GMM comparison manifest version 2, not an old constrained-GMM manifest")
        jobs = [{key: copy.deepcopy(value) for key, value in job.items() if key != "evaluations"}
                for job in previous["jobs"] if job.get("saved_config") and selected(job)]
        if not jobs:
            parser.error("Manifest contains no selected completed training runs")
        for job in jobs:
            validate_saved_config(job)
        chosen_seed = args.target_seed if args.target_seed is not None else previous.get("target_seed", 2026)
        path, payload = new_manifest(args.stage, previous.get("training_steps"), jobs, parent=parent,
                                     target_seed=chosen_seed, smoke=previous.get("smoke", False))
    else:
        jobs = [{"dataset": dataset, "prior": prior, "template_config": str(ROOT / template)}
                for dataset, prior, template in JOBS]
        jobs = [job for job in jobs if selected(job)]
        chosen_seed = args.target_seed if args.target_seed is not None else 2026
        path, payload = new_manifest(args.stage, steps, jobs, target_seed=chosen_seed, smoke=args.smoke)
    prior_threads = None
    try:
        if args.smoke:
            import torch

            print("SMOKE ONLY: 128-point OT cache, 256-point reference, tiny CPU model; no quality claim.", flush=True)
            prior_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            create_smoke_templates(path, payload)
        if args.stage != "eval":
            print(f"Selected {len(jobs)} jobs; stage={args.stage}; fresh trainings={len(jobs) if args.stage in ('all', 'train') else 0}", flush=True)
            prepare_jobs(path, payload)
            if args.stage in ("all", "train"):
                train_jobs(path, payload, steps)
        if args.stage in ("all", "eval"):
            evaluate_jobs(path, payload, nfes, roi_file)
        payload["status"] = "complete"
        payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    except Exception as error:
        payload["status"] = "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_summary(path, payload)
        save_json(path, payload)
        if prior_threads is not None:
            import torch

            torch.set_num_threads(prior_threads)
    print(f"completed_comparison_manifest={path}", flush=True)
    print(f"comparison_summary={payload['summary_json']}", flush=True)
    return path


if __name__ == "__main__":
    main()
