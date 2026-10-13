"""Read-only, paired-source comparison of saved 2D flow-matching checkpoints.

No training, OT construction, training-cache loading or Jacobian computation.
Every NFE is a fresh Euler solve from the SAME initial cloud to t=1.
"""

import argparse
import csv
import hashlib
from pathlib import Path
import re
from time import perf_counter

from matplotlib.colors import PowerNorm
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import yaml

from coupling import canonical_method
from diagnostic_data import DiagnosticData, finite_scores, tensor_sha256
from experiment import environment, load_model, synchronize
from horse_regions import HorseRegions
from metrics import chamfer_distance, checkerboard_metrics, horse_metrics
from model import PointSetTransformer
from sample import integrate_velocity
from summarize_results import output_directory, plt, save_json
from train_horse import HorsePointSetTransformer


DEFAULT_NFES = [1, 2, 4, 8, 16, 32, 128]
DATASETS = ("checkerboard", "horse")
PURPOSE = "model_comparison_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path, datasets=None):
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or raw.get("version") != 1 or set(raw) != {"version", "datasets"}:
        raise ValueError("Manifest requires version: 1 and datasets: {...}; see comparison_runs.example.yaml")
    groups = raw["datasets"]
    if not isinstance(groups, dict) or not groups or set(groups) - set(DATASETS):
        raise ValueError("Manifest datasets must be checkerboard and/or horse")
    selected = list(groups) if datasets is None else list(datasets)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(groups):
        raise ValueError("Requested datasets must be distinct and present in the manifest")
    result = {}
    for dataset in selected:
        rows = groups[dataset]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{dataset}: expected a nonempty list of checkpoints")
        entries, ids, paths = [], set(), set()
        for row in rows:
            if not isinstance(row, dict) or not {"id", "coupling", "config"} <= row.keys() \
                    or row.keys() - {"id", "label", "coupling", "config"}:
                raise ValueError(f"{dataset}: each entry needs id, coupling, config, and optional label")
            key = row["id"]
            if not isinstance(key, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", key) or key in ids:
                raise ValueError(f"{dataset}: ids must be unique lowercase identifiers")
            label = row.get("label", key)
            if not isinstance(label, str) or not label.strip() or len(label) > 60:
                raise ValueError(f"{dataset}/{key}: label must contain 1-60 characters")
            if not isinstance(row["config"], str) or not row["config"].strip():
                raise ValueError(f"{dataset}/{key}: config must be a saved run config.yaml path")
            config_path = (path.parent / row["config"]).resolve()
            if not config_path.is_file():
                raise FileNotFoundError(f"{dataset}/{key}: missing {config_path}; edit the manifest's config path")
            if config_path in paths:
                raise ValueError(f"{dataset}: duplicate config path: {config_path}")
            entries.append({"id": key, "label": label, "coupling": canonical_method(row["coupling"]),
                            "config_path": str(config_path)})
            ids.add(key)
            paths.add(config_path)
        result[dataset] = entries
    return result


def preflight(groups, allow_legacy=False):
    """Validate ALL selected checkpoints on CPU before any expensive rollout."""
    checked, notices = {}, []
    for dataset, entries in groups.items():
        checked[dataset] = []
        cls = HorsePointSetTransformer if dataset == "horse" else PointSetTransformer
        common, first_config = None, None
        hashes = set()
        for entry in entries:
            model, config, checkpoint, metadata = load_model(entry["config_path"], cls, dataset, device="cpu")
            del model
            tag = f"{dataset}/{entry['id']}"
            if config["coupling"] != entry["coupling"]:
                raise ValueError(f"{tag}: expected coupling={entry['coupling']}, got {config['coupling']}")
            verified = metadata["training_config_verified"]
            if not verified and not allow_legacy:
                raise ValueError(f"{tag}: unverified legacy checkpoint; use a saved verified run, or explicitly --allow-legacy")
            if config.get("dataset", dataset) != dataset or config["model"]["point_dim"] != 2:
                raise ValueError(f"{tag}: requires the correct dataset and point_dim=2")
            count = config["data"]["n_points"]
            grid = config["data"].get("grid_size") if dataset == "checkerboard" else None
            if isinstance(count, bool) or not isinstance(count, int) or count < 1 \
                    or (dataset == "checkerboard" and (isinstance(grid, bool) or not isinstance(grid, int) or grid < 1)):
                raise ValueError(f"{tag}: n_points and checkerboard grid_size must be positive integers")
            if config["dtype"] not in ("float32", "float64"):
                raise ValueError(f"{tag}: comparison requires float32 or float64 checkpoints")
            signature = (count, config["dtype"], grid)
            if common is not None and signature != common:
                raise ValueError(f"{dataset}: all models must have matching n_points, dtype and target grid_size")
            common = signature
            if first_config is not None:
                for key in ("model", "training", "seed"):
                    if config.get(key) != first_config.get(key):
                        notices.append(f"{tag}: {key} differs from first column; not a controlled coupling-only comparison")
                if config["data"]["batch_size"] != first_config["data"]["batch_size"]:
                    notices.append(f"{tag}: training batch_size differs from first column")
            else:
                first_config = config
            if not verified:
                notices.append(f"{tag}: LEGACY UNVERIFIED; training config/dataset cannot be established from weights")
            digest = file_sha256(checkpoint)
            if digest in hashes:
                raise ValueError(f"{dataset}: the same checkpoint was assigned to multiple columns")
            hashes.add(digest)
            checked[dataset].append({**entry, "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
                "config_sha256": file_sha256(entry["config_path"]), "config": config,
                "training_config_verified": verified, "coupling_details": metadata["coupling_details"],
                "training_environment": metadata.get("environment"),
                "training_seconds": metadata.get("training_seconds"),
                "training_source_sha256": metadata.get("source_sha256")})
            print(f"checked={tag} N={count} seed={config.get('seed')} verified={verified}", flush=True)
    return checked, notices


@torch.no_grad()
def predict_bank(model, source, nfe, batch_size):
    parameter = next(model.parameters())
    chunks = []
    synchronize(parameter.device)
    start = perf_counter()
    for begin in range(0, len(source), batch_size):
        # Clone also protects the common bank if a future model mutates its input.
        initial = source[begin:begin + batch_size].to(parameter.device).clone()
        final = integrate_velocity(model, initial, num_steps=nfe)
        if not torch.isfinite(final).all():
            raise FloatingPointError(f"Nonfinite generated points at NFE={nfe}")
        chunks.append(final.cpu())
    synchronize(parameter.device)
    return torch.cat(chunks), perf_counter() - start


def quality(points, target, dataset, config, mask, regions, histogram_bins, batch_size):
    # Bound the temporary CD matrix, even when many clouds are pooled for display.
    cd = 0.
    for begin in range(0, len(points), batch_size):
        part = points[begin:begin + batch_size]
        cd += chamfer_distance(part, target[begin:begin + batch_size]).item() * len(part)
    scores = {"chamfer": cd / len(points)}
    if dataset == "horse":
        leak, js = horse_metrics(points, mask, histogram_bins)
        scores.update(leakage=leak, histogram_js=js, **regions.score(points))
    else:
        leak, mass, js = checkerboard_metrics(points, config["data"].get("grid_size", 4), histogram_bins)
        scores.update(leakage=leak, cell_mass_error=mass, histogram_js=js)
    return finite_scores(scores)


def point_density(points, bins, extent, sigma):
    """Probability per unit area, divided by ALL points (including outside)."""
    array = np.asarray(points).reshape(-1, 2)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Density requires nonempty finite points")
    counts, _, _ = np.histogram2d(array[:, 0], array[:, 1], bins=bins, range=[extent, extent])
    outside = float(1. - counts.sum() / len(array))
    area = ((extent[1] - extent[0]) / bins) ** 2
    density = counts.T / (len(array) * area)
    if sigma > 0:
        density = gaussian_filter(density, sigma=sigma, mode="constant", cval=0.)
    return density, outside


def support(ax, dataset, config, mask):
    if dataset == "horse":
        height, width = mask.shape
        scale = max(height, width)
        ax.imshow(mask, origin="upper", cmap="Greys", vmin=0, vmax=1, alpha=.14,
                  extent=(-width / scale, width / scale, -height / scale, height / scale),
                  interpolation="nearest")
    else:
        grid = config["data"].get("grid_size", 4)
        indices = np.arange(grid)
        active = (indices[:, None] + indices[None, :]) % 2 == 0
        ax.imshow(active, origin="upper", extent=(-1, 1, -1, 1), cmap="Greys", vmin=0,
                  vmax=1, alpha=.14, interpolation="nearest")


def render_dataset(report, arrays, directory, *, view, bins, sigma, gamma, dpi):
    dataset, entries, nfes = report["dataset"], report["models"], report["nfes"]
    extent = report["plot_extent"]
    data = DiagnosticData(entries[0]["config"], dataset, "cpu", torch.float32, report["seed"])
    mask = data.mask.numpy() if data.mask is not None else None
    densities, outside = {}, {}
    for key, points in arrays.items():
        if key == "source":
            continue
        densities[key], outside[key] = point_density(points, bins, extent, sigma)
    vmax = max(float(d.max()) for d in densities.values()) or 1.
    norm = PowerNorm(gamma=gamma, vmin=0., vmax=vmax)
    lookup = {(r["method_id"], r["nfe"]): r for r in report["results"]}
    selected = report["cloud_index"]
    images = []
    for mode in (("density", "scatter") if view == "both" else (view,)):
        for rows, filename in [(nfes, f"{mode}_grid.png")] + [([n], f"{mode}_nfe_{n:03d}.png") for n in nfes]:
            fig, axes = plt.subplots(len(rows), len(entries) + 1, squeeze=False,
                                     figsize=(2.8 * (len(entries) + 1), 3.4 * len(rows) + .7))
            for row_index, nfe in enumerate(rows):
                for column in range(len(entries) + 1):
                    ax = axes[row_index, column]
                    if column == 0:
                        key, label, score = "target", "Target draw", None
                    else:
                        entry = entries[column - 1]
                        key = f"prediction_{entry['id']}_nfe_{nfe}"
                        label = entry["label"]
                        if not entry["training_config_verified"]:
                            label += " [UNVERIFIED]"
                        result = lookup[(entry["id"], nfe)]
                        score = result["metrics"] if mode == "density" else result["selected_cloud_metrics"]
                    if mode == "density":
                        ax.imshow(densities[key], origin="lower", extent=extent * 2,
                                  cmap="viridis", norm=norm, interpolation="nearest")
                        outside_value = outside[key]
                    else:
                        support(ax, dataset, entries[0]["config"], mask)
                        points = arrays[key][selected]
                        ax.scatter(points[:, 0], points[:, 1], s=7, c="#137ca8", linewidths=0)
                        _, outside_value = point_density(points, bins, extent, 0.)
                    caption = "Fixed target sample" if score is None else \
                        f"CD={score['chamfer']:.5f} | leak={100 * score['leakage']:.1f}%"
                    ax.set_title(f"{label}\n{caption}\noutside view={100 * outside_value:.1f}%", fontsize=9)
                    ax.set(xlim=extent, ylim=extent, aspect="equal", xticks=[], yticks=[])
                    if column == 0:
                        ax.set_ylabel(f"NFE {nfe}", fontsize=12)
            description = (f"pooled {report['clouds']} clouds x N={report['n_points']}" if mode == "density"
                           else f"same source cloud #{selected}, N={report['n_points']}")
            fig.suptitle(f"{dataset} | {description} | Euler to t=1 | evaluation seed={report['seed']}", fontsize=13)
            # Fixed layout leaves a separate strip for a shared colorbar.
            fig.tight_layout(h_pad=2., w_pad=.8,
                             rect=(0, 0, .955 if mode == "density" else 1, .97 if len(rows) > 1 else .88))
            if mode == "density":
                color_axis = fig.add_axes((.97, .12, .009, .65))
                colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=color_axis)
                colorbar.set_label("Probability density (shared scale)", fontsize=8)
                colorbar.ax.tick_params(labelsize=7)
            fig.savefig(directory / filename, dpi=dpi)
            plt.close(fig)
            images.append(filename)
    return {"images": images, "density_vmin": 0., "density_vmax": vmax,
            "density_gamma": gamma, "density_bins": bins, "sigma_pixels": sigma,
            "density_normalization": "all points / bin area before constant-zero Gaussian smoothing; no in-view renormalization",
            "color_scale_scope": "one shared scale across target, ALL methods and ALL NFEs within this dataset, including strips",
            "outside_view_fraction": outside}


def compare(manifest, *, datasets=None, nfes=None, clouds=64, cloud_index=0, seed=2026,
            batch_size=8, device="auto", view="both", bins=192, sigma=1., gamma=.5,
            histogram_bins=64, dpi=150, output="analysis_results", check_only=False, allow_legacy=False):
    nfes = list(DEFAULT_NFES if nfes is None else nfes)
    integer_options = {"clouds": clouds, "batch_size": batch_size, "bins": bins,
                       "histogram_bins": histogram_bins, "dpi": dpi}
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in integer_options.values()):
        raise ValueError("clouds, batch_size, bins, histogram_bins and dpi must be positive integers")
    if not nfes or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in nfes) \
            or len(set(nfes)) != len(nfes):
        raise ValueError("NFEs must be distinct positive integers")
    if isinstance(cloud_index, bool) or not isinstance(cloud_index, int) or not 0 <= cloud_index < clouds:
        raise ValueError("cloud_index must be in [0, clouds)")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if not np.isfinite([sigma, gamma]).all() or sigma < 0 or gamma <= 0 or view not in ("both", "density", "scatter"):
        raise ValueError("Require finite sigma>=0, gamma>0 and view=both/density/scatter")
    groups, notices = preflight(read_manifest(manifest, datasets), allow_legacy)
    for notice in notices:
        print(f"WARNING: {notice}", flush=True)
    if check_only:
        print("check_only=passed (no inference, training cache access or output directory)", flush=True)
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Comparison supports cpu/cuda devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    directory = output_directory(output, "model_comparison")
    top = {"format_version": 1, "manifest_path": str(Path(manifest).resolve()),
           "manifest_sha256": file_sha256(manifest), "environment": environment(device),
           "warnings": notices, "datasets": {},
           "renderer_sha256": file_sha256(__file__), "status": "running"}
    save_json(directory / "request.json", {**top, "resolved_models": groups})
    for dataset, entries in groups.items():
        dest = directory / dataset
        dest.mkdir()
        config = entries[0]["config"]
        dtype = getattr(torch, config["dtype"])
        data = DiagnosticData(config, dataset, "cpu", dtype, seed)
        source, target = data.bank(clouds, PURPOSE)
        regions = HorseRegions(data.mask) if dataset == "horse" else None
        report = {"dataset": dataset, "models": entries, "nfes": nfes, "clouds": clouds,
            "cloud_index": cloud_index, "n_points": data.n, "seed": seed, "batch_size": batch_size,
            "histogram_bins": histogram_bins, "inference_device": str(device), "dtype": config["dtype"],
            "plot_extent": [-1.2, 1.2], "source_sha256": tensor_sha256(source),
            "target_sha256": tensor_sha256(target), "draw_device": "cpu", "draw_purpose": PURPOSE,
            "source_law": "fresh iid standard Gaussian, shared byte-for-byte across methods/NFEs",
            "solver": "Euler, t=0 to t=1; independent restart for EVERY NFE, not intermediate trajectory snapshots",
            "metric_definitions": {"chamfer": "sum of directional mean squared distances against fixed independent equal-N target; then mean across clouds",
                "leakage": "invalid points / ALL points (not clipped to plot extent)",
                "histogram_js": "pooled JS to target density, nats, including outside bin",
                "cell_mass_error": "pooled checkerboard cell TV, conditioned on valid points",
                "selected_cloud_metrics": "only the prespecified cloud_index, never quality-selected"},
            "horse_rois": regions.definition() if regions is not None else None,
            "warnings": [w for w in notices if w.startswith(dataset + "/")], "results": []}
        arrays = {"source": source.numpy(), "target": target.numpy()}
        cls = HorsePointSetTransformer if dataset == "horse" else PointSetTransformer
        for entry in entries:
            if file_sha256(entry["config_path"]) != entry["config_sha256"] \
                    or file_sha256(entry["checkpoint"]) != entry["checkpoint_sha256"]:
                raise ValueError("Config/checkpoint changed after preflight; rerun with fixed files")
            model, _, _, _ = load_model(entry["config_path"], cls, dataset, device=device)
            for nfe in nfes:
                print(f"rendering={dataset}/{entry['id']} NFE={nfe} clouds={clouds}", flush=True)
                prediction, seconds = predict_bank(model, source, nfe, batch_size)
                metrics = quality(prediction, target, dataset, config, data.mask, regions, histogram_bins, batch_size)
                single = slice(cloud_index, cloud_index + 1)
                selected_metrics = quality(prediction[single], target[single], dataset, config,
                                           data.mask, regions, histogram_bins, batch_size)
                key = f"prediction_{entry['id']}_nfe_{nfe}"
                arrays[key] = prediction.numpy()
                report["results"].append({"method_id": entry["id"], "label": entry["label"], "nfe": nfe,
                    "prediction_key": key, "prediction_sha256": tensor_sha256(prediction),
                    "metrics": metrics, "selected_cloud_metrics": selected_metrics,
                    "inference_seconds": seconds,
                    "timing_scope": "batched Euler plus CPU/device transfers; no warmup, not a benchmark"})
                print(f"CD={metrics['chamfer']:.6g} leakage={100 * metrics['leakage']:.2f}%", flush=True)
            del model
        np.savez_compressed(dest / "clouds.npz", **arrays)
        with (dest / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
            metric_keys = sorted({key for row in report["results"] for key in row["metrics"]})
            writer = csv.DictWriter(stream, fieldnames=["method_id", "label", "nfe", "training_config_verified"] + metric_keys)
            writer.writeheader()
            verified = {entry["id"]: entry["training_config_verified"] for entry in entries}
            for row in report["results"]:
                writer.writerow({**{k: row[k] for k in ("method_id", "label", "nfe")},
                                 "training_config_verified": verified[row["method_id"]], **row["metrics"]})
        report["rendering"] = render_dataset(report, arrays, dest, view=view, bins=bins, sigma=sigma, gamma=gamma, dpi=dpi)
        save_json(dest / "comparison.json", report)
        top["datasets"][dataset] = f"{dataset}/comparison.json"
        print(f"saved_dataset={dest.resolve()}", flush=True)
    top["status"] = "complete"
    save_json(directory / "comparison.json", top)
    print(f"saved_comparison={directory.resolve()}", flush=True)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="YAML listing explicit saved runs/.../config.yaml paths")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS)
    parser.add_argument("--nfes", nargs="+", type=int, default=DEFAULT_NFES)
    parser.add_argument("--clouds", type=int, default=64, help="Independent N-point clouds pooled ONLY for display/metrics")
    parser.add_argument("--cloud-index", type=int, default=0, help="Same prespecified scatter cloud for every method/NFE")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=8, help="Inference microbatch size; does not change N")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--view", choices=("both", "density", "scatter"), default="both")
    parser.add_argument("--bins", type=int, default=192, help="Display histogram resolution")
    parser.add_argument("--sigma", type=float, default=1., help="Display smoothing in pixels; 0 disables smoothing")
    parser.add_argument("--gamma", type=float, default=.5, help="Shared density color PowerNorm exponent")
    parser.add_argument("--histogram-bins", type=int, default=64, help="Metric histogram resolution, separate from rendering")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--output", default="analysis_results")
    parser.add_argument("--check-only", action="store_true", help="Validate all configs/checkpoints on CPU without inference/output")
    parser.add_argument("--allow-legacy", action="store_true", help="Explicitly allow UNVERIFIED weights-only checkpoints")
    return compare(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
