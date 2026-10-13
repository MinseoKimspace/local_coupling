"""All-candidate learnability study using the existing checkerboard/horse protocol.

Default is preparation only. Full GPU training is explicit (--stage all/train).
"""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import yaml

from run_wasserstein_bank import ROOT, create_experiment, evaluate_jobs, save_manifest, train_jobs, verify_job_config
from wasserstein_bank import DEFAULT_CANDIDATES, file_sha256
from wasserstein_study_report import summarize_study


FORMAT = "wasserstein_candidate_study_v1"
NFES = [1, 2, 4, 8, 16, 32, 64, 128]
NOISE_LEVELS = [0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0]


def candidate_family(name):
    if name == "default":
        return copy.deepcopy(DEFAULT_CANDIDATES)
    if name == "cost_noise":
        return [{"kind": "random"}, {"kind": "exact"}] + [
            {"kind": "perturbed_exact", "noise_std": value} for value in NOISE_LEVELS]
    raise ValueError("family must be default or cost_noise")


def _seeds(values, name):
    if (not values or len(values) != len(set(values)) or
            any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 2**32 - 1 for v in values)):
        raise ValueError(f"{name} must be distinct nonnegative integers below 2**32-1")
    return list(values)


def create_study(*, datasets=None, bank_seeds=None, training_seeds=None, family="default",
                 pilot=False, smoke=False, device=None, steps=None, superset_size=None,
                 backend=None, nfe=None, output_root="experiment_manifests"):
    datasets = ["checkerboard", "horse"] if datasets is None else list(datasets)
    if not datasets or len(set(datasets)) != len(datasets) or any(d not in ("checkerboard", "horse") for d in datasets):
        raise ValueError("Provide distinct checkerboard/horse datasets")
    small = pilot or smoke
    bank_seeds = _seeds(([0] if small else [0, 1, 2]) if bank_seeds is None else bank_seeds, "bank_seeds")
    training_seeds = _seeds(([0] if small else [0, 1, 2]) if training_seeds is None else training_seeds, "training_seeds")
    if steps is not None and steps < 1:
        raise ValueError("steps must be positive")
    if smoke and any(v is not None for v in (device, steps, superset_size)):
        raise ValueError("smoke sets its own CPU/device/step/size settings")
    nfe = ([1, 2] if smoke else NFES) if nfe is None else nfe
    if not nfe or len(set(nfe)) != len(nfe) or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in nfe):
        raise ValueError("Provide distinct positive NFE values")
    candidates = candidate_family(family)
    middle = min((i for i, c in enumerate(candidates) if c["kind"] == "perturbed_exact"),
                 key=lambda i: abs(candidates[i]["noise_std"] - 0.2))
    ids = [0, 1, middle] if small else list(range(len(candidates)))
    directory = Path(output_root).resolve() / f"wasserstein_study_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    children, references = [], {}
    for dataset in datasets:
        template_path = ROOT / "wasserstein_experiments" / f"{dataset}.yaml"
        with template_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        config["wasserstein_bank"]["candidates"] = candidates
        if backend is not None:
            config["wasserstein_bank"]["backend"] = backend
        if superset_size is not None:
            config["wasserstein_bank"]["superset_size"] = superset_size
        if device is not None:
            config["device"] = device
        if steps is not None:
            config["training"]["num_steps"] = steps
        path = directory / f"{dataset}_template.yaml"
        with path.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        baseline = ROOT / ("checkerboard_experiments/nsot.yaml" if dataset == "checkerboard"
                           else "horse_experiments/horse_nsot_n256_seed0.yaml")
        with baseline.open(encoding="utf-8") as stream:
            baseline_config = yaml.safe_load(stream)
        same = all(config[key] == baseline_config[key] for key in ("data", "model", "evaluation")) and all(
            config["training"][key] == value for key, value in baseline_config["training"].items())
        references[dataset] = {"template": str(baseline), "sha256": file_sha256(baseline),
                               "original_model_data_training_evaluation_match": same and not smoke,
                               "note": "extra validation is observational; bank size/coupling differ; old baselines are not retrained"}
        for seed in bank_seeds:
            children.append({"dataset": dataset, "bank_seed": seed, "template": str(path),
                             "template_sha256": file_sha256(path), "manifest": None,
                             "planned_candidate_ids": ids, "training_seeds": training_seeds})
    manifest = {
        "format": FORMAT, "study_mode": "smoke" if smoke else "pilot" if pilot else "full", "smoke": smoke,
        "datasets": datasets, "bank_seeds": bank_seeds, "training_seeds": training_seeds,
        "family": family, "candidates": candidates, "planned_candidate_ids": ids, "evaluation_nfes": list(nfe),
        "primary_nfe": 128, "early_fraction": 0.05, "protocol_reference": references,
        "claim": {
            "question": "Do pretraining bank scores predict learnability and final generation beyond transport cost?",
            "primary_scores": ["joint", "path_joint"],
            "primary_endpoints": ["final_time_integrated_independent_bank_mse", "nfe_128_chamfer"],
            "decision_rule": "Exploratory support only: full study, >=3 complete bank seeds; a prespecified score must beat transport regret on >=ceil(2*B/3) banks and have positive transport-controlled rank association on the same banks, for both endpoints. Report per dataset; no p-value or universal claim.",
            "secondary": "other scores, alpha sensitivity, NFE sweep, density/leakage/horse ROIs; no post-hoc primary switching",
            "non_support": "transport wins/ties, rank sign inconsistent, primary endpoints disagree, or incomplete/pilot/smoke study",
        },
        "children": children,
    }
    path = directory / "manifest.json"
    save_manifest(path, manifest)
    count = len(children) * len(ids) * len(training_seeds)
    print(f"study_manifest={path}", flush=True)
    print(f"planned_models={count} datasets={datasets} candidates={ids} bank_seeds={bank_seeds} training_seeds={training_seeds}", flush=True)
    return path, manifest


def prepare_children(path, manifest):
    for child in manifest["children"]:
        if child["manifest"] is not None:
            load_child(child)
            continue
        if file_sha256(Path(child["template"])) != child["template_sha256"]:
            raise ValueError("Study template changed; start a new study instead of mixing settings")
        child_path, _ = create_experiment(
            child["template"], seeds=manifest["training_seeds"], bank_seed=child["bank_seed"],
            candidate_ids=child["planned_candidate_ids"], output_root=path.parent / "banks", smoke=manifest["smoke"])
        child["manifest"] = str(child_path)
        save_manifest(path, manifest)


def load_child(child):
    if child["manifest"] is None:
        raise ValueError("Study preparation is incomplete; run --stage prepare first")
    path = Path(child["manifest"])
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("dataset") != child["dataset"] or manifest.get("bank_seed") != child["bank_seed"]:
        raise ValueError("Child experiment dataset/bank_seed mismatch")
    expected = {(c, s) for c in child["planned_candidate_ids"] for s in child["training_seeds"]}
    actual = [(j.get("candidate_id"), j["seed"]) for j in manifest["jobs"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Child jobs differ from declared candidate/training-seed axes")
    for job in manifest["jobs"]:
        verify_job_config(job)
    if file_sha256(Path(manifest["cache"])) != manifest["cache_sha256"]:
        raise ValueError("Study bank hash changed")
    return path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "train", "eval", "all", "report"), default="prepare")
    parser.add_argument("--manifest", help="Resume the printed study manifest; no settings are regenerated")
    parser.add_argument("--datasets", nargs="+", choices=("checkerboard", "horse"))
    parser.add_argument("--bank-seeds", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, help="Model initialization/training seeds, separate from bank seeds")
    parser.add_argument("--family", choices=("default", "cost_noise"))
    parser.add_argument("--pilot", action="store_true", help="Only random, exact and noise=0.2; defaults one bank/train seed")
    parser.add_argument("--smoke", action="store_true", help="Two-update tiny CPU pipeline check, not quality evidence")
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--superset-size", type=int)
    parser.add_argument("--backend", choices=("exact", "sliced"))
    parser.add_argument("--nfe", nargs="+", type=int)
    args = parser.parse_args()
    if args.manifest:
        if args.pilot or args.smoke or any(v is not None for v in (
                args.datasets, args.bank_seeds, args.seeds, args.family, args.device, args.steps,
                args.superset_size, args.backend, args.nfe)):
            parser.error("With --manifest, only --stage may change; all settings including NFE are saved")
        path = Path(args.manifest).resolve()
        with path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("format") != FORMAT:
            raise ValueError("Unsupported study manifest")
    else:
        if args.stage == "report":
            parser.error("--stage report requires --manifest")
        path, manifest = create_study(
            datasets=args.datasets, bank_seeds=args.bank_seeds, training_seeds=args.seeds,
            family=args.family or "default", pilot=args.pilot, smoke=args.smoke, device=args.device,
            steps=args.steps, superset_size=args.superset_size, backend=args.backend, nfe=args.nfe)
    if args.stage != "report":
        prepare_children(path, manifest)
    for child in manifest["children"]:
        if args.stage in ("train", "eval", "all"):
            child_path, child_manifest = load_child(child)
            if args.stage in ("train", "all"):
                train_jobs(child_path, child_manifest)
            if args.stage in ("eval", "all"):
                evaluate_jobs(child_path, child_manifest, manifest["evaluation_nfes"])
    summarize_study(path, manifest)
    print(f"study_manifest={path}")
    if args.stage == "prepare":
        print(f'next_command=python run_wasserstein_study.py --manifest "{path}" --stage all')


if __name__ == "__main__":
    main()
