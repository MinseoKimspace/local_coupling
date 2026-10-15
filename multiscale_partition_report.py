"""Small reports for the controlled partition study; no inference or new renderer.

Generation panels are the unchanged PNGs written by eval.py/eval_horse.py.
"""

import csv
import json
from pathlib import Path

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


def write_report(directory, manifest):
    """Called after the runner verifies every referenced artifact hash."""
    directory = Path(directory)
    bank_rows, evaluation_rows, groups = [], [], {}
    for job in manifest["jobs"]:
        identity = {"dataset": job["dataset"], "variant": job["label"], "seed": job["seed"]}
        bank = job["bank"]
        if bank:
            metadata = bank["metadata"]
            partitions = metadata.get("partition_settings", {})
            row = {**identity, "source_partition": partitions.get("source_partition", "axis"),
                   "target_partition": partitions.get("target_partition", "axis"),
                   "bank": bank["path"], "bank_sha256": bank["sha256"],
                   "source_sha256": metadata["endpoint_sha256"]["source"],
                   "target_sha256": metadata["endpoint_sha256"]["target"],
                   "precompute_seconds": bank["precompute_seconds"],
                   "point_transport_cost": metadata["point_transport_cost"],
                   "multiscale_objective": metadata["multiscale_objective"]}
            for side in ("source", "target"):
                for key, value in metadata.get("partition_diagnostics", {}).get(side, {}).items():
                    if key != "strategy":
                        row[f"{side}_{key}"] = value
            bank_rows.append(row)
        for nfe, artifacts in job["evaluations"].items():
            with Path(artifacts["json"]["path"]).open(encoding="utf-8") as stream:
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
    for (dataset, seed, nfe), images in groups.items():
        if all(label in images for label in VARIANTS):
            _native_panel([images[label] for label in VARIANTS],
                          directory / f"{dataset}_seed{seed}_nfe_{nfe:03d}_comparison.png")
