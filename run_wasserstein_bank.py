"""Prepare, train and evaluate the same-bank Wasserstein selection experiment."""

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import yaml

from experiment import read_config
from model import PointSetTransformer
from train import train_model
from wasserstein_bank import METHOD, SELECTORS, cache_spec, file_sha256, prepare, settings


ROOT = Path(__file__).resolve().parent


def save_manifest(path, manifest):
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    temporary.replace(path)


def write_reports(directory, metadata):
    with (directory / "bank_scores.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
    with (directory / "candidate_scores.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["split", "candidate", "kind", "transport", "displacement", "joint", "multiscale", "variance_proxy"])
        writer.writeheader()
        for split, report in metadata["reports"].items():
            for row in report["candidates"]:
                writer.writerow({"split": split, "candidate": row["candidate"], "kind": row["specification"]["kind"],
                                 **{key: row[key] for key in ("transport", "displacement", "joint", "multiscale")},
                                 "variance_proxy": row["variance"]})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis_row, (split, report) in zip(axes, metadata["reports"].items()):
        rows = report["candidates"]
        for axis, metric in zip(axis_row, ("displacement", "joint", "multiscale")):
            axis.scatter([r[metric] for r in rows], [r["variance"] for r in rows])
            for row in rows:
                axis.annotate(str(row["candidate"]), (row[metric], row["variance"]))
            axis.set(xlabel=metric, ylabel="Local velocity variogram proxy", title=split.rstrip("_"))
    figure.suptitle("Candidate-bank diagnostics; labels are candidate IDs")
    figure.tight_layout()
    figure.savefig(directory / "bank_diagnostics.png", dpi=160)
    plt.close(figure)


def create_experiment(config_path, *, dataset=None, selectors=None, seeds=None, device=None, steps=None,
                      superset_size=None, backend=None, output_root="experiment_manifests", smoke=False):
    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    dataset = dataset or config.get("dataset", "mixture2d")
    if dataset != config.get("dataset", "mixture2d") and dataset in ("horse", "checkerboard"):
        template = (ROOT / "horse_experiments/horse_nsot_n256_seed0.yaml" if dataset == "horse"
                    else ROOT / "checkerboard_experiments/nsot.yaml")
        with template.open(encoding="utf-8") as stream:
            existing = yaml.safe_load(stream)
        existing.pop("nsot", None)
        existing["wasserstein_bank"] = config["wasserstein_bank"]
        config = existing
    config["dataset"], config["coupling"] = dataset, METHOD
    opts = config["wasserstein_bank"]
    if superset_size is not None:
        opts["superset_size"] = superset_size
    if backend is not None:
        opts["backend"] = backend
    if smoke:
        config["device"] = "cpu"
        config["data"].update(batch_size=2, n_points=16)
        config["model"].update(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
        config["training"].update(num_steps=2, log_every=1, validation_batch_size=2)
        config["evaluation"].update(batch_size=6, histogram_bins=16)
        opts.update(superset_size=64, pool_per_shape=16, subset_size=8, scales=[2, 4, 8],
                    repetitions=2, variance_times=[0.0, 0.5, 1.0], bandwidths=[0.1, 0.2], variance_bandwidth=0.1)
        torch.set_num_threads(1)
        print("smoke_test=CPU, tiny model, M=64, N=16, two optimizer updates; not a quality experiment", flush=True)
    if device is not None:
        config["device"] = device
    if steps is not None:
        if steps < 1:
            raise ValueError("steps must be positive")
        config["training"]["num_steps"] = steps
    opts.setdefault("selector", "joint")
    settings(config)
    if opts.get("cache", "auto") == "auto":
        import hashlib
        digest = hashlib.sha256(json.dumps(cache_spec(config, dataset), sort_keys=True).encode()).hexdigest()[:16]
        opts["cache"] = str((Path("coupling_cache") / "wasserstein_bank" / f"{dataset}_{digest}.npz").resolve())
    else:
        opts["cache"] = str(Path(opts["cache"]).resolve())
    cache_path, metadata = prepare(config, dataset)
    opts["cache_sha256"] = file_sha256(cache_path)
    selectors = selectors or ["displacement", "joint", "multiscale"]
    seeds = [0] if seeds is None else seeds
    if len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2**63 - 100003 for seed in seeds):
        raise ValueError("Provide distinct nonnegative training seeds below 2**63-100003")
    directory = Path(output_root) / f"wasserstein_bank_{dataset}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    write_reports(directory, metadata)
    jobs = []
    for selector in selectors:
        if selector not in SELECTORS:
            raise ValueError("Unknown selector")
        for seed in seeds:
            job_config = copy.deepcopy(config)
            job_config["seed"] = seed
            job_config["wasserstein_bank"]["selector"] = selector
            job_config["checkpoint"] = f"{dataset}_wasserstein_{selector}_seed{seed}.pt"
            path = directory / f"{selector}_seed{seed}.yaml"
            with path.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(job_config, stream, sort_keys=False)
            jobs.append({"selector": selector, "seed": seed, "config": str(path.resolve()),
                         "run_config": None, "evaluations": {}})
    manifest = {"format": "wasserstein_bank_experiment_v1", "dataset": dataset,
                "cache": str(cache_path.resolve()), "cache_sha256": opts["cache_sha256"],
                "selected": metadata["selected"], "precompute_seconds": metadata["precompute_seconds"],
                "jobs": jobs}
    path = directory / "manifest.json"
    save_manifest(path, manifest)
    print(f"manifest={path.resolve()}", flush=True)
    return path.resolve(), manifest


def train_jobs(path, manifest):
    for job in manifest["jobs"]:
        if job["run_config"] is not None:
            if not Path(job["run_config"]).is_file():
                raise FileNotFoundError("Completed job's saved run config is missing")
            print(f"reused_completed_training={job['run_config']}", flush=True)
            continue
        config = read_config(job["config"])
        torch.manual_seed(config["seed"])
        model_class = PointSetTransformer
        if manifest["dataset"] == "horse":
            from train_horse import HorsePointSetTransformer
            model_class = HorsePointSetTransformer
        model = model_class(**config["model"]).to(device=config["device"], dtype=getattr(torch, config["dtype"]))
        print(f"training_selector={job['selector']} seed={job['seed']}", flush=True)
        run_dir = train_model(model, config, None, dataset=manifest["dataset"], config_path=job["config"])
        job["run_config"] = str((run_dir / "config.yaml").resolve())
        save_manifest(path, manifest)
    plot_learning_curves(path, manifest)


def evaluate_jobs(path, manifest, nfe):
    if any(n < 1 for n in nfe):
        raise ValueError("NFE values must be positive")
    dataset = manifest["dataset"]
    if dataset == "horse":
        from eval_horse import main as evaluate
    elif dataset == "checkerboard":
        from eval import main as evaluate
    else:
        from eval_wasserstein_bank import main as evaluate
    for job in manifest["jobs"]:
        if job["run_config"] is None:
            raise ValueError("Run --stage train before evaluation")
        for steps in nfe:
            existing = job["evaluations"].get(str(steps))
            if existing and Path(existing).is_file():
                continue
            output = evaluate(job["run_config"], steps)
            job["evaluations"][str(steps)] = str(Path(output).with_suffix(".json").resolve())
            save_manifest(path, manifest)
    rows = []
    for job in manifest["jobs"]:
        for steps, result_path in job["evaluations"].items():
            with Path(result_path).open(encoding="utf-8") as stream:
                result = json.load(stream)
            rows.append({"selector": job["selector"], "seed": job["seed"], "nfe": int(steps),
                         "selected_candidate": manifest["selected"][job["selector"]],
                         "precompute_seconds_shared": manifest["precompute_seconds"],
                         **{key: result.get(key) for key in ("training_seconds", "inference_seconds", "chamfer", "leakage",
                                                            "histogram_js", "mmd_cd", "cov_cd", "one_nna_cd", "shape_mass_tv_proxy")}})
    if rows:
        with (path.parent / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_learning_curves(path, manifest)


def plot_learning_curves(path, manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for job in manifest["jobs"]:
        curve_path = Path(job["run_config"]).parent / "learning_curve.json"
        with curve_path.open(encoding="utf-8") as stream:
            history = json.load(stream)["history"]
        label = f"{job['selector']} seed={job['seed']}"
        validation = [np.mean([p["mse_per_coordinate"] for p in row["validation_by_time"]]) for row in history]
        axes[0].plot([r["step"] for r in history], validation, label=label)
        axes[1].plot([r["elapsed_seconds"] for r in history], validation, label=label)
    for axis, xlabel in zip(axes, ("Optimizer updates", "Online seconds including validation")):
        axis.set(xlabel=xlabel, ylabel="Holdout coupling validation MSE")
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path.parent / "learning_curves.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(ROOT / "wasserstein_experiments/mixture2d.yaml"))
    parser.add_argument("--stage", choices=("prepare", "train", "eval", "all"), default="prepare")
    parser.add_argument("--manifest", help="Resume an existing experiment without regenerating configs/banks")
    parser.add_argument("--dataset", choices=("checkerboard", "horse", "mixture2d"))
    parser.add_argument("--selectors", nargs="+", choices=SELECTORS)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--superset-size", type=int)
    parser.add_argument("--backend", choices=("exact", "sliced"))
    parser.add_argument("--smoke", action="store_true", help="Tiny CPU pipeline check, two updates; not a quality run")
    parser.add_argument("--nfe", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64, 128])
    args = parser.parse_args()
    if args.smoke and any(value is not None for value in (args.manifest, args.device, args.steps, args.superset_size)):
        parser.error("--smoke sets its own CPU/device/step/size settings and starts a new experiment")
    if args.manifest:
        if any(value is not None for value in (args.dataset, args.selectors, args.seeds, args.device, args.steps, args.superset_size, args.backend)):
            parser.error("With --manifest, settings come from saved configs; only --stage and --nfe may change")
        path = Path(args.manifest).resolve()
        with path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("format") != "wasserstein_bank_experiment_v1":
            raise ValueError("Unsupported experiment manifest")
        if file_sha256(Path(manifest["cache"])) != manifest["cache_sha256"]:
            raise ValueError("Experiment bank hash has changed")
    else:
        path, manifest = create_experiment(args.config, dataset=args.dataset, selectors=args.selectors,
                                          seeds=args.seeds, device=args.device, steps=args.steps,
                                          superset_size=args.superset_size, backend=args.backend, smoke=args.smoke)
    if args.stage in ("train", "all"):
        train_jobs(path, manifest)
    if args.stage in ("eval", "all"):
        evaluate_jobs(path, manifest, args.nfe)
    print(f"experiment_manifest={path}")
    if args.stage == "prepare":
        print(f'next_command=python run_wasserstein_bank.py --manifest "{path}" --stage all')


if __name__ == "__main__":
    main()
