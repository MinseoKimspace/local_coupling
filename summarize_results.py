import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("chamfer", "leakage", "cell_mass_error", "histogram_js",
           "inference_seconds", "training_seconds")


def statistics(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    return {"mean": mean(values) if values else None,
            "std": stdev(values) if len(values) > 1 else None, "n": len(values)}


def formatted(value):
    if value["mean"] is None:
        return "undefined"
    suffix = f" ± {value['std']:.6g}" if value["std"] is not None else " (n=1)"
    return f"{value['mean']:.6g}{suffix}"


def output_directory(root, prefix):
    directory = Path(root) / f"{prefix}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def save_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)


def save_table(path, headers, rows, title):
    fig, ax = plt.subplots(figsize=(max(9, len(headers) * 2.2), 1.1 + 0.32 * len(rows)))
    ax.axis("off")
    ax.set_title(title, pad=15)
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def signature(record):
    config = {k: v for k, v in record["config"].items()
              if k not in ("seed", "checkpoint", "device", "evaluation")}
    return {"dataset": record["dataset"], "config": config,
            **{k: record.get(k) for k in (
                "coupling_details", "evaluation_seed", "evaluation_batch_size", "histogram_bins",
                "metric_definitions", "training_environment", "evaluation_environment")}}


def collect_results(root, seeds):
    groups, skipped = defaultdict(list), []
    for path in sorted(Path(root).rglob("*.json")):
        with path.open(encoding="utf-8") as file:
            record = json.load(file)
        if not isinstance(record, dict) or "euler_steps" not in record or "config" not in record:
            continue
        if not record.get("training_config_verified"):
            skipped.append({"file": str(path), "reason": "unverified checkpoint"})
            continue
        if record["config"]["seed"] not in seeds:
            continue
        record["result_file"] = str(path.resolve())
        groups[json.dumps(signature(record), sort_keys=True)].append(record)

    results = []
    for key, records in sorted(groups.items()):
        # Repeated evaluations are not additional training seeds. Never select a
        # different checkpoint for each NFE, or silently choose among retrainings.
        hashes = defaultdict(set)
        for record in records:
            hashes[record["config"]["seed"]].add(record["checkpoint_sha256"])
        if any(len(v) > 1 for v in hashes.values()):
            skipped.append({"signature": json.loads(key), "reason": "multiple checkpoints for the same training seed; separate result roots"})
            continue
        selected = {}
        for record in sorted(records, key=lambda r: (r["evaluated_at"], r["result_file"])):
            pair = (record["config"]["seed"], record["euler_steps"])
            if pair in selected:
                skipped.append({"file": selected[pair]["result_file"], "reason": "older duplicate evaluation"})
            selected[pair] = record
        rows = []
        for nfe in sorted({pair[1] for pair in selected}):
            members = [selected[(seed, nfe)] for seed in seeds if (seed, nfe) in selected]
            if len(members) != len(seeds):
                skipped.append({"signature": json.loads(key), "nfe": nfe,
                                "reason": "missing requested training seeds",
                                "present_seeds": [r["config"]["seed"] for r in members]})
                continue
            rows.append({"nfe": nfe, "seeds": seeds,
                         "metrics": {metric: statistics([r.get(metric) for r in members]) for metric in METRICS
                                     if any(metric in r for r in members)},
                         "members": [{"seed": r["config"]["seed"], "checkpoint_sha256": r["checkpoint_sha256"],
                                      "file": r["result_file"]} for r in members]})
        if rows:
            results.append({"id": hashlib.sha256(key.encode()).hexdigest()[:10],
                            "signature": json.loads(key), "rows": rows})
    return results, skipped


def render_groups(groups, directory):
    comparisons = defaultdict(list)
    for group in groups:
        spec = group["signature"]
        config = {k: v for k, v in spec["config"].items()
                  if k not in ("coupling", "num_regions", "sinkhorn_epsilon", "sinkhorn_iterations")}
        comparison = {**spec, "config": config}
        comparison.pop("coupling_details")
        comparisons[json.dumps(comparison, sort_keys=True)].append(group)
    for key, members in comparisons.items():
        prefix = hashlib.sha256(key.encode()).hexdigest()[:10]
        spec = members[0]["signature"]
        title = f"{spec['dataset']} | N={spec['config']['data']['n_points']} | training-seed mean ± sample SD"
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        table_rows = []
        for group in members:
            config = group["signature"]["config"]
            label = f"{config['coupling']} K={config.get('num_regions', 'na')}"
            for ax, metric in zip(axes.flat, METRICS):
                valid = [r for r in group["rows"] if r["metrics"].get(metric, {}).get("mean") is not None]
                if not valid:
                    continue
                xs = [r["nfe"] for r in valid]
                ys = [r["metrics"][metric]["mean"] for r in valid]
                errors = [r["metrics"][metric]["std"] or 0 for r in valid]
                if metric == "training_seconds":
                    ax.bar(label, ys[0], yerr=errors[0], capsize=3)
                    ax.tick_params(axis="x", labelsize=7, rotation=15)
                    ax.set_title("Training seconds (one value per seed)")
                    continue
                ax.errorbar(xs, ys, yerr=errors, marker="o", capsize=3, label=label)
                ax.set_xscale("log", base=2)
                if metric in ("chamfer", "histogram_js") and all(y > 0 for y in ys):
                    ax.set_yscale("log")
                ax.set_xlabel("NFE (Euler)")
                ax.set_title(metric)
                ax.grid(alpha=0.25)
            for row in group["rows"]:
                table_rows.append([label, row["nfe"], *[formatted(row["metrics"][m]) if m in row["metrics"] else "—"
                                                       for m in METRICS]])
        for ax in axes.flat:
            if not ax.has_data():
                ax.axis("off")
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(directory / f"{prefix}_curves.png", dpi=170)
        plt.close(fig)
        save_table(directory / f"{prefix}_table.png", ["Method", "NFE", *METRICS], table_rows, title)
        with (directory / f"{prefix}_table.md").open("x", encoding="utf-8") as file:
            file.write(f"{title}\n\n| " + " | ".join(["Method", "NFE", *METRICS]) + " |\n")
            file.write("|" + "---|" * (len(METRICS) + 2) + "\n")
            for row in table_rows:
                file.write("| " + " | ".join(map(str, row)) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate verified evaluation JSON over training seeds, without rerunning models.")
    parser.add_argument("root", nargs="?", default="eval_results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", default="analysis_results")
    args = parser.parse_args()
    if not Path(args.root).is_dir() or len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2:
        parser.error("Provide an existing result directory and at least two distinct training seeds")
    groups, skipped = collect_results(args.root, args.seeds)
    directory = output_directory(args.output, "seed_summary")
    save_json(directory / "summary.json", {"source": str(Path(args.root).resolve()), "seeds": args.seeds,
              "std_definition": "sample SD across training seeds, ddof=1; not SE or confidence interval",
              "policy": "complete seed groups only; latest duplicate evaluation; ambiguous retrainings excluded",
              "groups": groups, "skipped": skipped})
    render_groups(groups, directory)
    print(f"complete_groups={len(groups)} skipped_records_or_groups={len(skipped)}")
    print(f"saved={directory}")


if __name__ == "__main__":
    main()
