"""Small reports for controlled partition/coupling studies; no new renderer.

Generation panels are the unchanged PNGs written by eval.py/eval_horse.py.
"""

import csv
import json
from pathlib import Path
import statistics

from PIL import Image


VARIANTS = ("axis_axis", "axis_pca", "pca_pca")
METRICS = ("chamfer", "leakage", "histogram_js", "cell_mass_error")


def _csv(path, rows, fields):
    fields = list(dict.fromkeys([*fields, *(key for row in rows for key in row)]))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _native_panel(paths, output):
    """Concatenate existing native eval images without resampling their points."""
    images = []
    try:
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        width, height = sum(image.width for image in images), max(image.height for image in images)
        with Image.new("RGB", (width, height), "white") as panel:
            x = 0
            for image in images:
                panel.paste(image, (x, 0))
                x += image.width
            temporary = output.with_suffix(".png.tmp")
            panel.save(temporary, format="PNG")
            temporary.replace(output)
    finally:
        for image in images:
            image.close()


def write_report(directory, manifest, *, root=None):
    """Called after the runner verifies every referenced artifact hash."""
    directory = Path(directory)
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    resolve = lambda path: Path(path) if Path(path).is_absolute() else root / path
    coupling_study = manifest.get("coupling_study", False)
    variants = ("multiscale", "leaf_only", "qat") if coupling_study else VARIANTS
    bank_rows, evaluation_rows, groups = [], [], {}
    for job in manifest["jobs"]:
        identity = {"dataset": job["dataset"], "variant": job["label"], "seed": job["seed"]}
        bank = job["bank"]
        if bank:
            metadata = bank["metadata"]
            partitions = metadata.get("partition_settings", {})
            is_qat = job["label"] == "qat"
            row = {**identity, "source_partition": "quantile_aligned_tree" if is_qat else partitions.get("source_partition", "axis"),
                   "target_partition": "max_variance_mean_singletons" if is_qat else partitions.get("target_partition", "axis"),
                   "bank": bank["path"], "bank_sha256": bank["sha256"],
                   "source_sha256": metadata["endpoint_sha256"].get("source"),
                   "target_sha256": metadata["endpoint_sha256"]["target"],
                   "precompute_seconds": bank["precompute_seconds"],
                   "point_transport_cost": metadata.get("point_transport_cost"),
                   "multiscale_objective": metadata.get("multiscale_objective")}
            if coupling_study:
                row.update(implementation=metadata.get("implementation"),
                           scale_weights=json.dumps(metadata.get("scale_weights")),
                           leaf_count=metadata.get("leaf_count", metadata.get("partition_counts", [None])[-1]),
                           source_measure="continuous_gaussian" if is_qat else "finite_empirical_gaussian_bank")
            for side in ("source", "target"):
                for key, value in metadata.get("partition_diagnostics", {}).get(side, {}).items():
                    if key != "strategy":
                        row[f"{side}_{key}"] = value
            bank_rows.append(row)
        for nfe, artifacts in job["evaluations"].items():
            with resolve(artifacts["json"]["path"]).open(encoding="utf-8") as stream:
                result = json.load(stream)
            row = {**identity, "nfe": int(nfe),
                   **{name: result.get(name) for name in METRICS},
                   "training_seconds": result.get("training_seconds"),
                   "inference_seconds": result.get("inference_seconds"),
                   "json": artifacts["json"]["path"], "png": artifacts["png"]["path"]}
            row.update({key: value for key, value in result.items()
                        if key.startswith("roi_") and (value is None or isinstance(value, (int, float)))})
            evaluation_rows.append(row)
            groups.setdefault((job["dataset"], job["seed"], int(nfe)), {})[job["label"]] = row["png"]
    _csv(directory / "bank_comparison.csv", bank_rows,
         ["dataset", "variant", "seed", "source_partition", "target_partition"])
    _csv(directory / "comparison.csv", evaluation_rows, ["dataset", "variant", "seed", "nfe", *METRICS])
    if coupling_study:
        aggregates = []
        for dataset, variant, nfe in sorted({(row["dataset"], row["variant"], row["nfe"])
                                           for row in evaluation_rows}):
            selected = [row for row in evaluation_rows if
                        (row["dataset"], row["variant"], row["nfe"]) == (dataset, variant, nfe)]
            row = {"dataset": dataset, "variant": variant, "nfe": nfe, "training_seed_count": len(selected)}
            for metric in (*METRICS, "training_seconds", "inference_seconds"):
                values = [item[metric] for item in selected if item[metric] is not None]
                row[f"{metric}_mean"] = statistics.mean(values) if values else None
                # A single seed is not a zero-variance estimate.
                row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
            aggregates.append(row)
        _csv(directory / "comparison_by_seed.csv", aggregates, ["dataset", "variant", "nfe", "training_seed_count"])
    for (dataset, seed, nfe), images in groups.items():
        if all(label in images for label in variants):
            _native_panel([resolve(images[label]) for label in variants],
                          directory / f"{dataset}_seed{seed}_nfe_{nfe:03d}_comparison.png")
