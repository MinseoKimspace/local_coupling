"""Separate same-M MS-W2 IID-vs-stratified target-bank experiment.

Default --stage audit only samples endpoints and checks mass (no OT/training).
Use --stage all explicitly for preparation, four default models, and evaluation.
"""

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

import multiscale_wasserstein as msw
import run_multiscale as base
from target_mass_control import TargetStrata

ROOT = base.ROOT
KIND = "msw2_target_mass_control_v1"
LABELS = ("multiscale_iid", "multiscale_stratified")


def portable(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def create_experiment(*, datasets=None, seeds=None, bank_seed=0, bins=16,
                      device=None, steps=None, smoke=False, nfe=None, output_root=None):
    datasets = ["checkerboard", "horse"] if datasets is None else list(datasets)
    if not datasets or len(set(datasets)) != len(datasets) or set(datasets) - {"checkerboard", "horse"}:
        raise ValueError("Choose distinct checkerboard/horse datasets")
    seeds = base._integers([0] if seeds is None else seeds, "seeds")
    base._integers([bank_seed], "bank_seed")
    if steps is not None:
        base._integers([steps], "steps", 1)
    if smoke and (device is not None or steps is not None):
        raise ValueError("--smoke supplies its own CPU device and two training steps")
    nfes = base._integers(([1, 2] if smoke else base.NFES) if nfe is None else nfe, "nfe", 1)
    proposed = []
    for dataset in datasets:
        config = base._read_yaml(ROOT / "multiscale_experiments" / f"{dataset}.yaml")
        config["dataset"] = dataset
        config[msw.METHOD]["cache_seed"] = bank_seed
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
            config[msw.METHOD].update(superset_size=64, partition_counts=[2, 4, 8])
        TargetStrata(config, dataset, bins)  # Validate all settings before writing.
        for label in LABELS:
            variant = copy.deepcopy(config)
            if label == LABELS[1]:
                variant[msw.METHOD]["target_sampling"] = {"mode": "stratified", "bins": bins}
            msw.settings(variant)
            variant[msw.METHOD]["cache"] = portable(base._cache_path(variant, dataset, msw.METHOD))
            proposed.append((dataset, label, variant))
    name = f"bank_mass_control_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:8]}"
    directory = Path(output_root or ROOT / "experiment_manifests") / name
    directory.mkdir(parents=True, exist_ok=False)
    jobs = []
    for dataset, label, config in proposed:
        for seed in seeds:
            config = copy.deepcopy(config)
            config["seed"] = seed
            config["checkpoint"] = f"{dataset}_{label}_bank{bank_seed}_seed{seed}.pt"
            path = directory / f"{dataset}_{label}_seed{seed}.yaml"
            with path.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(config, stream, sort_keys=False)
            jobs.append({"dataset": dataset, "label": label, "seed": seed,
                         "config": portable(path), "config_sha256": base.file_sha256(path),
                         "bank": None, "run": None, "evaluations": {}})
    manifest = {"format": base.FORMAT, "experiment_type": KIND, "smoke": smoke,
                "created_at": datetime.now(timezone.utc).isoformat(), "datasets": datasets,
                "bank_seed": bank_seed, "training_seeds": seeds, "mass_control_bins": bins,
                "evaluation_nfes": nfes, "with_nsot": False, "with_refinement": False,
                "endpoint_checks": {}, "jobs": jobs,
                "results_directory": portable(ROOT / "eval_results" / "bank_mass_control" / name),
                "protocol": {
                    "changed": "offline target sampling only: iid vs foreground-area-proportional grid quotas",
                    "fixed": "M, byte-identical source bank, MS-W2 algorithm/scales/weights, model, optimizer, steps, paired training/evaluation seeds",
                    "matching": "recomputed on each target bank; induced partitions/permutation may change",
                    "quota": "largest remainder; error below one point per grid bin, random seeded tie breaks",
                    "source": "original Gaussian bank; fresh Gaussian inference; NO noise injection",
                    "online": "unchanged iid pair sampling with replacement; NO per-cloud mass quota",
                    "evaluation": "original fresh iid target sampler and standard eval.py/eval_horse.py renderers",
                    "scope": "target grid is distinct from MS-W2 4/20/100 equal-mass patches; no quality guarantee"}}
    path = directory / "manifest.json"
    base.save_manifest(path, manifest)
    print(f"experiment_manifest={portable(path)}", flush=True)
    print(f"planned_models={len(jobs)}; smoke={smoke}", flush=True)
    return path, manifest


def paired_jobs(manifest):
    jobs = manifest["jobs"]
    expected = {(dataset, seed, label) for dataset in manifest["datasets"]
                for seed in manifest["training_seeds"] for label in LABELS}
    actual = [(job["dataset"], job["seed"], job["label"]) for job in jobs]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Mass-control manifest must have exactly one IID/stratified job per dataset/seed")
    for dataset in manifest["datasets"]:
        for seed in manifest["training_seeds"]:
            pair = [next(j for j in jobs if (j["dataset"], j["seed"], j["label"]) == (dataset, seed, label))
                    for label in LABELS]
            configs = [base.verify_job(j) for j in pair]
            before, after = configs
            for key in ("dataset", "coupling", "model", "data", "training", "evaluation", "seed", "dtype", "device"):
                if before.get(key) != after.get(key):
                    raise ValueError(f"Unmatched experiment setting: {dataset}/{seed}/{key}")
            a, b = msw.settings(before), msw.settings(after)
            expected_sampling = {"mode": "stratified", "bins": manifest["mass_control_bins"]}
            if a["target_sampling"] != {"mode": "iid"} or b["target_sampling"] != expected_sampling:
                raise ValueError("Expected IID and declared stratified target bank")
            fixed = ("superset_size", "cache_seed", "partition_counts", "scale_weights", "max_cost_mib")
            if any(a[key] != b[key] for key in fixed):
                raise ValueError("Unmatched MS-W2 bank/matching settings")
            yield pair, configs


def audit_banks(path, manifest):
    # Validate every seed even though bank construction is shared across seeds.
    pairs = list(paired_jobs(manifest))
    audit = {"experiment_type": KIND, "smoke": manifest["smoke"], "datasets": {},
             "note": "ENDPOINTS ONLY: no OT, training or generated-quality conclusion. TV uses chosen control grid."}
    for pair, configs in pairs:
        dataset = pair[0]["dataset"]
        if dataset in audit["datasets"]:
            continue
        strata = TargetStrata(configs[0], dataset, manifest["mass_control_bins"])
        draws = [msw.draw_supersets(c, dataset) for c in configs]
        if not np.array_equal(draws[0][0], draws[1][0]):
            raise ValueError("Gaussian source differs between conditions")
        record = {"same_source": True, "conditions": {}}
        for label, config, (source, target) in zip(LABELS, configs, draws):
            from metrics import checkerboard_metrics, horse_metrics
            points = torch.from_numpy(target).unsqueeze(0)
            report = strata.report(target)
            report["endpoint_sha256"] = {"source": msw._array_hash(source), "target": msw._array_hash(target)}
            if dataset == "checkerboard":
                leak, mass, js = checkerboard_metrics(points, config["data"]["grid_size"], 64)
                report["standard_metrics"] = {"leakage": leak, "cell_mass_error": mass, "histogram_js": js}
            else:
                from train_horse import load_horse_mask
                from horse_regions import HorseRegions
                mask = load_horse_mask("cpu", torch.float32)
                leak, js = horse_metrics(points, mask, 64)
                report["standard_metrics"] = {"leakage": leak, "histogram_js": js, **HorseRegions(mask).score(points)}
            record["conditions"][label] = report
        controlled = record["conditions"][LABELS[1]]
        if controlled["leakage"] != 0 or controlled["max_abs_count_error"] > 1 + 1e-9:
            raise ValueError("Target bank failed regional quota check")
        audit["datasets"][dataset] = record
        print(f"{dataset}: same_source=True M={controlled['size']} "
              f"region_TV iid={record['conditions'][LABELS[0]]['region_mass_tv']:.6f} "
              f"stratified={controlled['region_mass_tv']:.6f}; endpoints only", flush=True)
    directory = Path(manifest["results_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    audit_path = directory / "bank_audit.json"
    if manifest.get("bank_audit"):
        artifact = manifest["bank_audit"]
        if base.file_sha256(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Saved bank audit changed")
        if json.loads(Path(artifact["path"]).read_text(encoding="utf-8")) != audit:
            raise ValueError("Reproduced endpoints/audit differ; start a new experiment")
    else:
        base.save_manifest(audit_path, audit)
        manifest["bank_audit"] = {"path": portable(audit_path), "sha256": base.file_sha256(audit_path)}
        base.save_manifest(path, manifest)
    return audit


def verify_prepared_pairs(path, manifest, audit):
    for pair, configs in paired_jobs(manifest):
        dataset = pair[0]["dataset"]
        arrays = [msw.load_cache(c, dataset) for c in configs]
        if not np.array_equal(arrays[0][0], arrays[1][0]):
            raise ValueError("Prepared Gaussian source banks differ")
        for job, values in zip(pair, arrays):
            expected = audit["datasets"][dataset]["conditions"][job["label"]]["endpoint_sha256"]
            if {"source": msw._array_hash(values[0]), "target": msw._array_hash(values[1])} != expected:
                raise ValueError("Prepared bank differs from audited endpoints")
        manifest["endpoint_checks"][f"{dataset}:seed{pair[0]['seed']}"] = {
            "same_source": True, "same_M": True, "settings_match": True,
            "target_changed": not np.array_equal(arrays[0][1], arrays[1][1]),
            "audited_endpoints_match": True}
    base.save_manifest(path, manifest)


def write_summary(path, manifest):
    list(paired_jobs(manifest))
    base.write_summary(path, manifest)
    summary = json.loads(path.with_name("summary.json").read_text(encoding="utf-8"))
    rows = []
    for pair, _ in paired_jobs(manifest):
        iid, stratified = pair
        for nfe in sorted(iid["evaluations"].keys() & stratified["evaluations"].keys(), key=int):
            results = [json.loads(Path(job["evaluations"][nfe]["json"]["path"]).read_text(encoding="utf-8")) for job in pair]
            a, b = results
            for key in ("evaluation_seed", "evaluation_batch_size", "n_points", "euler_steps", "histogram_bins",
                        "metric_metadata", "metric_definitions"):
                if a.get(key) != b.get(key):
                    raise ValueError(f"Unmatched evaluation setting: {key}")
            metrics = {"chamfer", "leakage", "cell_mass_error", "histogram_js", "thin_region_mass_mae",
                       "gap_region_leakage", "training_seconds", "inference_seconds"}
            metrics.update(k for k in a.keys() | b.keys() if k.startswith("roi_"))
            for metric in sorted(metrics):
                left, right = a.get(metric), b.get(metric)
                if (isinstance(left, (int, float)) and not isinstance(left, bool)
                        and isinstance(right, (int, float)) and not isinstance(right, bool)):
                    rows.append({"dataset": iid["dataset"], "seed": iid["seed"], "nfe": int(nfe),
                                 "metric": metric, "iid": left, "stratified": right, "delta": right - left})
    summary.update(experiment_type=KIND, bank_audit=manifest.get("bank_audit"), paired_mass_control=rows,
                   delta_note="stratified minus IID; negative is better for errors, NOT necessarily raw ROI mass. No significance claim.")
    directory = Path(manifest["results_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    base.save_manifest(path.with_name("summary.json"), summary)
    base.save_manifest(directory / "summary.json", summary)
    with (directory / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "seed", "nfe", "metric", "iid", "stratified", "delta"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"results={portable(directory)}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "audit", "prepare", "train", "eval", "all"), default="audit")
    parser.add_argument("--manifest", help="Resume this exact mass-control study")
    parser.add_argument("--datasets", nargs="+", choices=("checkerboard", "horse"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--bank-seed", type=int)
    parser.add_argument("--bins", type=int, help="Target mass grid bins per axis (default 16); NOT MS-W2 patches")
    parser.add_argument("--device")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--nfe", nargs="+", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "eval" and not args.manifest:
        parser.error("--stage eval requires a trained --manifest")
    with base.project_directory():
        if args.manifest:
            if args.smoke or any(v is not None for v in (args.datasets, args.seeds, args.bank_seed, args.bins, args.device, args.steps)):
                parser.error("With --manifest only --stage and --nfe may change")
            path = Path(args.manifest)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("format") != base.FORMAT or manifest.get("experiment_type") != KIND:
                raise ValueError("Not a mass-control manifest; original experiments are not modified")
            if args.nfe is not None:
                manifest["evaluation_nfes"] = sorted(set(manifest["evaluation_nfes"]) | set(base._integers(args.nfe, "nfe", 1)))
                base.save_manifest(path, manifest)
        else:
            path, manifest = create_experiment(datasets=args.datasets, seeds=args.seeds,
                bank_seed=0 if args.bank_seed is None else args.bank_seed, bins=16 if args.bins is None else args.bins,
                device=args.device, steps=args.steps, smoke=args.smoke, nfe=args.nfe)
        if manifest["smoke"]:
            torch.set_num_threads(1)
            print("SMOKE ONLY: M=64 N=16 CPU two updates; not generation-quality evidence", flush=True)
        if args.stage != "plan":
            audit = audit_banks(path, manifest)
            if args.stage != "audit":
                base.prepare_jobs(path, manifest)
                verify_prepared_pairs(path, manifest, audit)
                if args.stage in ("train", "all"):
                    base.train_jobs(path, manifest)
                if args.stage in ("eval", "all"):
                    base.evaluate_jobs(path, manifest)
        write_summary(path, manifest)
        print(f"experiment_manifest={portable(path)}", flush=True)
        if args.stage in ("plan", "audit", "prepare"):
            print(f'next_command=python run_bank_mass_control.py --manifest "{portable(path)}" --stage all', flush=True)


if __name__ == "__main__":
    main()
