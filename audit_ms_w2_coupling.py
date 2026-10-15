"""Run four read-only coupling diagnostics on verified 2D MS-W2 checkpoints.

Pass saved runs/.../config.yaml files (one or several), from the project folder.
No training or bank preparation is performed. See MSW2_COUPLING_DIAGNOSTICS.md.
"""

import argparse
from contextlib import contextmanager
import copy
from pathlib import Path

import numpy as np
import torch

import multiscale_wasserstein as msw
from audit_generation import quality_scores
from diagnostic_data import DiagnosticData, diagnostic_metadata, tensor_sha256
from experiment import load_model, read_config
from model import PointSetTransformer
from msw2_coupling_common import SpatialRegions, routing_regions, write_csv
from msw2_coupling_geometry import gaussian_routing_audit, geometry_audit
from msw2_coupling_model import fitting_audit, rollout_audit
from summarize_results import output_directory, plt, save_json
from train_horse import HorsePointSetTransformer


TIMES = (0., .25, .5, .75, .9, .95, .99, 1.)


@contextmanager
def _preserve_rng():
    state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            yield
    finally:
        np.random.set_state(state)


def _load_inputs(config_path, dataset=None, *, bank=None, device=None):
    config_path = Path(config_path)
    initial = read_config(config_path)
    if initial["coupling"] != msw.METHOD:
        raise ValueError("These four diagnostics require the original multiscale_wasserstein coupling")
    dataset = dataset or initial.get("dataset") or config_path.resolve().parent.parent.name
    if dataset not in ("checkerboard", "horse"):
        raise ValueError("Cannot infer dataset; pass --dataset checkerboard or --dataset horse")
    if initial.get("dataset", dataset) != dataset:
        raise ValueError("Requested dataset differs from saved config")
    model_class = HorsePointSetTransformer if dataset == "horse" else PointSetTransformer
    model, config, checkpoint, metadata = load_model(config_path, model_class, dataset, device=device)
    if not metadata["training_config_verified"]:
        raise ValueError("A verified checkpoint and its saved runs/.../config.yaml are required")
    expected = metadata["coupling_details"].get("cache_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Checkpoint must record the exact training bank cache_sha256")
    cache_config = copy.deepcopy(config)
    bank_path = Path(bank) if bank is not None else Path(config[msw.METHOD]["cache"])
    if not bank_path.is_file() and bank is None and not bank_path.is_absolute():
        # Original cache paths are project-relative; do not guess a different run.
        candidate = Path(__file__).resolve().parent / bank_path
        if candidate.is_file():
            bank_path = candidate
    if not bank_path.is_file():
        raise FileNotFoundError(f"Training bank not found: {bank_path}. Pass --bank with the same existing NPZ; it will be hash-verified.")
    cache_config[msw.METHOD]["cache"] = str(bank_path.resolve())
    cache_config[msw.METHOD]["cache_sha256"] = expected
    source, target, permutation, bank_metadata, digest = msw.load_cache(cache_config, dataset)
    return model, config, checkpoint, metadata, dataset, bank_path.resolve(), source, target, permutation, bank_metadata, digest


def _point_features(target_paired, spatial, geometry, routing):
    features = {"all": np.ones(len(target_paired), dtype=bool), **spatial.region_masks(target_paired)}
    crossing = geometry["point_boundary_fraction"]
    features.update(source_knn_cross_leaf=crossing > 0, source_knn_same_leaf=crossing == 0)
    thresholds = {}
    for name, values in (("velocity_roughness", geometry["point_velocity_ratio_mean"]),
                         ("gaussian_q1", routing["q1"])):
        low, high = np.quantile(values, [.25, .75])
        # Do not arbitrarily split tied/constant scores into different groups.
        features[name + "_low"] = values < low
        features[name + "_high"] = values > high
        thresholds[name] = {"q25": float(low), "q75": float(high),
                            "rule": "strictly below q25 / above q75; ties excluded from extremes"}
    return features, {"source_boundary_proxy": "A bank point with >=1 directed kNN neighbor across its finest source leaf; not geometric distance to a kd boundary",
                      "overlapping_groups": True, "thresholds": thresholds,
                      "target_regions": "Classified by the paired target endpoint, including at intermediate fitting times"}


def _point_rows(source, target_paired, levels, features, geometry, routing, fitting):
    indices = fitting["fitting_bank_indices"].ravel()
    counts = np.bincount(indices, minlength=len(source))
    means = []
    for errors in fitting["fitting_point_mse"]:
        totals = np.bincount(indices, weights=errors.ravel(), minlength=len(source))
        means.append(np.divide(totals, counts, out=np.zeros_like(totals), where=counts > 0))
    rows = []
    for i in range(len(source)):
        row = {"bank_index": i, "source_x": float(source[i, 0]), "source_y": float(source[i, 1]),
               "target_x": float(target_paired[i, 0]), "target_y": float(target_paired[i, 1]),
               "source_leaf": int(levels[-1][i]), "knn_cross_leaf_fraction": float(geometry["point_boundary_fraction"][i]),
               "velocity_jump_mean": float(geometry["point_velocity_jump_mean"][i]),
               "velocity_ratio_mean": float(geometry["point_velocity_ratio_mean"][i]),
               "fitting_draw_count": int(counts[i])}
        for key, values in routing.items():
            if key.startswith("q") and key[1:].isdigit():
                row["gaussian_" + key] = float(values[i])
        for key, values in features.items():
            if key != "all":
                row["group_" + key] = bool(values[i])
        for ti, time in enumerate(fitting["fitting_times"]):
            row[f"fm_mse_t{time:g}"] = float(means[ti][i]) if counts[i] else None
        rows.append(row)
    return rows


def _heatmap(ax, hist, bins, *, title, signed=False, limit=None):
    values = np.asarray(hist)[:-1].reshape(bins, bins).T
    if signed:
        limit = limit or max(float(np.abs(values).max()), 1e-12)
        options = dict(cmap="RdBu_r", vmin=-limit, vmax=limit)
    else:
        options = dict(cmap="viridis", vmin=0., vmax=limit)
    picture = ax.imshow(values, origin="lower", extent=(-1, 1, -1, 1), **options)
    ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    return picture


def render(payload, arrays, directory):
    """Scientific diagnostic plots; all plotted inputs are also saved in NPZ/CSV."""
    diagnostics = payload["diagnostics"]
    geometry = diagnostics["1_geometry"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    comparisons = geometry["partition_comparisons"]
    for label, color in (("same", "tab:blue"), ("cross", "tab:orange")):
        axes[0].plot([r["partition_count"] for r in comparisons],
                     [r["distance_standardized_means"]["velocity_jump"][label] for r in comparisons],
                     "o-", label=label + " source patch", color=color)
    axes[0].set(xlabel="Partition count", ylabel="Distance-stratified velocity jump")
    axes[0].legend()
    times = diagnostics["1_geometry"]["times"]
    for statistic in ("median", "p99"):
        key = "velocity_ratio_" + statistic
        if key in payload["tables"]["geometry_time_conflicts"][0]:
            axes[1].plot(times, [r[key] for r in payload["tables"]["geometry_time_conflicts"]], "o-", label=statistic)
    axes[1].set(xlabel="t", ylabel="Same-time velocity jump / distance")
    axes[1].legend()
    source = arrays["bank_source"]
    colors = arrays["geometry_point_boundary_fraction"]
    picture = axes[2].scatter(source[:, 0], source[:, 1], c=colors, s=3, cmap="viridis", vmin=0, vmax=1)
    axes[2].set(title="Source kNN cross-leaf fraction", aspect="equal", xlabel="x", ylabel="y")
    fig.colorbar(picture, ax=axes[2])
    fig.suptitle("1. Coupling geometry (descriptive)")
    fig.tight_layout()
    fig.savefig(directory / "1_coupling_geometry.png", dpi=160)
    plt.close(fig)

    rows = payload["tables"]["routing_regions"]
    names = [n for n in payload["spatial_definition"]["region_names"] if n not in ("foreground", "leakage")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for key, label in (("target_expected_mass", "Population target"), ("bank_point_mass", "Target bank")):
        selected = {r["region"]: r[key] for r in rows if r["neighbors"] == 1}
        axes[0].plot(names, [selected[n] for n in names], "o-", label=label)
    for k in payload["options"]["routing_neighbors"]:
        selected = {r["region"]: r for r in rows if r["neighbors"] == k}
        axes[0].errorbar(names, [selected[n]["gaussian_routed_proxy_mass"] for n in names],
                         yerr=[selected[n]["query_mc_se"] for n in names], fmt="o-", label=f"Routing proxy k={k} (+/- query SE)")
    axes[0].tick_params(axis="x", labelrotation=45)
    axes[0].set(ylabel="Unconditional region mass")
    axes[0].legend(fontsize=8)
    axes[1].hist(arrays["routing_q1"] * len(source), bins=50)
    axes[1].axvline(1., color="black", ls="--")
    axes[1].set(xlabel="Gaussian Voronoi mass / (1/M)", ylabel="Source bank points")
    fig.suptitle("2. Gaussian routing proxy (not the network routing rule)")
    fig.tight_layout()
    fig.savefig(directory / "2_gaussian_routing.png", dpi=160)
    plt.close(fig)

    fitting = diagnostics["3_fitting"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name in ("all", "source_knn_cross_leaf", "source_knn_same_leaf", "velocity_roughness_high", "velocity_roughness_low"):
        selected = [r for r in fitting["grouped_errors"] if r["group"] == name]
        axes[0].plot([r["time"] for r in selected], [r["mse_per_coordinate"] for r in selected], "o-", label=name)
    axes[0].set(xlabel="t", ylabel="Bank-pair velocity MSE per coordinate")
    axes[0].legend(fontsize=7)
    selected = [r for r in fitting["context_test"]["grouped_decomposition"] if r["group"] == "all"]
    for key in ("squared_bias_per_coordinate", "context_variance_per_coordinate", "mse_per_coordinate"):
        axes[1].plot([r["time"] for r in selected], [r[key] for r in selected], "o-", label=key)
    axes[1].set(xlabel="t", ylabel="Selected-anchor context decomposition")
    axes[1].legend(fontsize=7)
    fig.suptitle("3. Fitting on training-bank pairs; selected anchors are stratified")
    fig.tight_layout()
    fig.savefig(directory / "3_fitting_context.png", dpi=160)
    plt.close(fig)

    generation = diagnostics["4_generation"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes[:2], ("leakage", "histogram_js")):
        for label in ("bank", "fresh"):
            selected = [r for r in generation["endpoint_quality"] if r["input"] == label]
            ax.plot([r["nfe"] for r in selected], [r.get(metric) for r in selected], "o-", label=label)
        ax.set(xlabel="NFE (Euler)", ylabel=metric, xscale="log")
        ax.legend()
    for key, label in (("bank_vs_teacher_histogram_tv", "Bank rollout vs teacher"),
                       ("fresh_vs_teacher_histogram_tv", "Fresh rollout vs teacher"),
                       ("independent_teacher_histogram_tv", "Independent teacher control")):
        axes[2].plot([r["time"] for r in generation["path_comparison"]],
                     [r[key] for r in generation["path_comparison"]], "o-", label=label)
    axes[2].set(xlabel="t", ylabel="Wide-grid TV [-4,4] (with outside bin)")
    axes[2].legend(fontsize=7)
    fig.suptitle("4. Bank-source and fresh-source generation")
    fig.tight_layout()
    fig.savefig(directory / "4_bank_fresh_generation.png", dpi=160)
    plt.close(fig)

    bins = payload["spatial_definition"]["histogram"]["bins_per_axis"]
    teacher = arrays["maps_target_teacher"]
    bank_hist = arrays["maps_target_bank_endpoint"]
    fresh_hist = arrays["maps_target_fresh_endpoint"]
    baseline = arrays["routing_target_bank_grid"]
    routed = arrays["routing_routing_grid_k1"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, hist, title in zip(axes[0], (teacher, bank_hist, fresh_hist),
                               ("Bank target draw", "Bank-source endpoint", "Fresh-source endpoint")):
        pic = _heatmap(ax, hist, bins, title=title,
                       limit=max(teacher[:-1].max(), bank_hist[:-1].max(), fresh_hist[:-1].max()) or 1e-12)
        fig.colorbar(pic, ax=ax, fraction=.046)
    diffs = (routed - baseline, bank_hist - teacher, fresh_hist - bank_hist)
    common = max(float(np.max(np.abs(x[:-1]))) for x in diffs) or 1e-12
    for ax, diff, title in zip(axes[1], diffs,
                               ("Routing proxy minus target bank", "Bank rollout minus paired target draw", "Fresh rollout minus bank rollout")):
        pic = _heatmap(ax, diff, bins, title=title, signed=True, limit=common)
        fig.colorbar(pic, ax=ax, fraction=.046)
    fig.suptitle(f"Region mass maps; endpoint Euler NFE={generation['path_steps']}\nOutside-square mass is recorded in JSON/NPZ, not drawn in these maps")
    fig.tight_layout()
    fig.savefig(directory / "region_mass_maps.png", dpi=160)
    plt.close(fig)


def _validate_options(options, bank_size):
    for name, minimum in (("clouds", 1), ("batch_size", 1), ("neighbors", 1), ("distance_bins", 1),
                          ("routing_draws", 2), ("routing_chunk_size", 1), ("anchors", 1),
                          ("contexts", 2), ("path_steps", 1), ("grid_bins", 2), ("seed", 0)):
        value = options[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if options["neighbors"] >= bank_size:
        raise ValueError("neighbors must be less than the bank size")
    routing = options["routing_neighbors"]
    if not routing or 1 not in routing or len(set(routing)) != len(routing):
        raise ValueError("routing_neighbors must be unique and include 1 for the Voronoi proxy")
    for k in routing:
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= bank_size:
            raise ValueError("routing_neighbors must be integers in [1,M]")
    if not options["nfes"] or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in options["nfes"]):
        raise ValueError("nfes must contain positive integers")
    times = options["times"]
    if (not times or any(isinstance(t, (bool, np.bool_)) or not isinstance(t, (int, float, np.integer, np.floating))
                        or not np.isfinite(t) or not 0 <= t <= 1 for t in times)
            or len(set(times)) != len(times)):
        raise ValueError("times must be unique finite values in [0,1]")


def audit(config_path, dataset=None, *, bank=None, device=None, clouds=32, batch_size=8,
          neighbors=8, distance_bins=8, routing_draws=100000, routing_neighbors=(1, 8),
          routing_chunk_size=10000, anchors=64, contexts=8, nfes=(1, 128), path_steps=128,
          times=TIMES, grid_bins=16, seed=2026, roi_file=None, output="analysis_results",
          skip_render=False, check_only=False):
    with _preserve_rng():
        loaded = _load_inputs(config_path, dataset, bank=bank, device=device)
        model, config, checkpoint, metadata, dataset, bank_path, source, target, permutation, bank_metadata, digest = loaded
        if check_only:
            print(f"verified={dataset} bank={bank_path} bank_sha256={digest} M={len(source)} P={config[msw.METHOD]['partition_counts']}")
            return None
        options = {"clouds": clouds, "batch_size": batch_size, "neighbors": neighbors,
                   "distance_bins": distance_bins, "routing_draws": routing_draws,
                   "routing_neighbors": list(routing_neighbors), "routing_chunk_size": routing_chunk_size,
                   "anchors": anchors, "contexts": contexts, "nfes": list(nfes), "path_steps": path_steps,
                   "times": list(times), "grid_bins": grid_bins, "seed": seed}
        _validate_options(options, len(source))
        times = tuple(sorted(set([0., 1., *times])))
        options["times"] = list(times)
        parameter = next(model.parameters())
        data = DiagnosticData(config, dataset, parameter.device, parameter.dtype, seed)
        spatial = SpatialRegions(config, dataset, data.mask, grid_bins=grid_bins, roi_file=roi_file)
        paired = target[permutation]
        levels = msw.balanced_partitions(source, config[msw.METHOD]["partition_counts"])
        print("diagnostic=1/4 coupling_geometry", flush=True)
        geometry, geometry_rows, geometry_arrays = geometry_audit(source, paired, levels, neighbors=neighbors,
                                                                 times=times, distance_bins=distance_bins)
        print("diagnostic=2/4 gaussian_routing", flush=True)
        routing, routing_arrays = gaussian_routing_audit(source, paired, draws=routing_draws,
                    neighbors=tuple(routing_neighbors), seed=seed, chunk_size=routing_chunk_size)
        regional, routing_rows, routing_maps = routing_regions(source, paired, spatial, routing_arrays, routing_neighbors)
        routing["region_summary"] = regional
        routing["regions"] = routing_rows
        routing_arrays.update(routing_maps)
        features, feature_definition = _point_features(paired, spatial, geometry_arrays, routing_arrays)
        print("diagnostic=3/4 fitting_and_context", flush=True)
        fitting, fitting_rows, fitting_arrays = fitting_audit(model, source, paired, features,
                  n_points=config["data"]["n_points"], clouds=clouds, batch_size=batch_size,
                  times=times, seed=seed, anchors=anchors, contexts=contexts)
        print("diagnostic=4/4 bank_fresh_generation", flush=True)
        def score(prediction, reference):
            return quality_scores(prediction, reference, config, dataset, data.mask, spatial.horse)
        generation, generation_rows, generation_arrays = rollout_audit(model, source, paired, data, spatial,
                  config, clouds=clouds, batch_size=batch_size, nfes=nfes, path_steps=path_steps,
                  times=times, seed=seed, quality_score=score)
        tables = {**{"geometry_" + k: v for k, v in geometry_rows.items()},
                  "routing_regions": routing_rows,
                  **{"fitting_" + k: v for k, v in fitting_rows.items()},
                  **{"generation_" + k: v for k, v in generation_rows.items()}}
        tables["bank_points"] = _point_rows(source, paired, levels, features, geometry_arrays, routing_arrays, fitting_arrays)
        arrays = {"bank_source": source, "bank_target": target, "bank_permutation": permutation,
                  **{"geometry_" + k: v for k, v in geometry_arrays.items()},
                  **{"routing_" + k: v for k, v in routing_arrays.items()},
                  **{"fitting_" + k: v for k, v in fitting_arrays.items()},
                  **{"generation_" + k: v for k, v in generation_arrays.items()}}
        arrays.update(maps_target_teacher=spatial.grid_hist(generation_arrays["rollout_teacher_path"][-1]),
                      maps_target_bank_endpoint=spatial.grid_hist(generation_arrays["rollout_bank_path"][-1]),
                      maps_target_fresh_endpoint=spatial.grid_hist(generation_arrays["rollout_fresh_path"][-1]))
        for key, array in arrays.items():
            if np.asarray(array).dtype.kind not in "biuf" or not np.isfinite(array).all():
                raise FloatingPointError(f"Nonfinite/unsupported diagnostic array: {key}")
        hashes = {name: tensor_sha256(torch.as_tensor(array)) for name, array in {
                   "source_bank": source, "target_bank": target, "permutation": permutation,
                   "fitting_bank_indices": fitting_arrays["fitting_bank_indices"],
                   "rollout_bank_indices": generation_arrays["rollout_bank_indices"],
                   "rollout_fresh_source": generation_arrays["rollout_fresh_source"],
                   "rollout_target_reference": generation_arrays["rollout_target_reference"]}.items()}
        payload = {**diagnostic_metadata(config_path, config, checkpoint, metadata, dataset, parameter.device),
                   "format_version": 1, "kind": "msw2_coupling_diagnostics_v1", "complete": True,
                   "bank_path": str(bank_path), "bank_sha256": digest,
                   "bank_metadata": bank_metadata, "input_hashes": hashes, "options": options,
                   "inference_policy": {"model_mode": "eval with no_grad; standard evaluation attention backend",
                                        "mha_fastpath_enabled": torch.backends.mha.get_fastpath_enabled(),
                                        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
                                        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
                   "spatial_definition": spatial.definition(), "point_feature_definition": feature_definition,
                   "diagnostics": {"1_geometry": geometry, "2_gaussian_routing": routing,
                                   "3_fitting": fitting, "4_generation": generation},
                   "limitations": ["Read-only fixed-bank/fixed-checkpoint diagnostics; no causal or training-seed significance claims.",
                                   "No training, prior noise augmentation, bank repair, or coupling modification is performed.",
                                   "Gaussian routing is a nearest-bank proxy, not an identified rule of the network.",
                                   "Geometry edges and repeated bank points are dependent; context anchors are stratified.",
                                   "Requested off-grid path times interpolate the fixed Euler polygon; integration is not certified converged.",
                                   "Fitting and bank-source generation use seen bank points; fresh Gaussian has no assigned target velocity label."],
                   "csv_files": [name + ".csv" for name in tables], "tables": tables}
        directory = output_directory(Path(output) / dataset, checkpoint.stem + "_coupling_diagnostic")
        # The potentially large per-bank point table is CSV-only, not duplicated in JSON.
        report = {key: value for key, value in payload.items() if key != "tables"}
        for name, rows in tables.items():
            write_csv(directory / (name + ".csv"), rows)
        with (directory / "diagnostics.npz").open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        if not skip_render:
            render(payload, arrays, directory)
        report["rendered"] = not skip_render
        save_json(directory / "summary.json", report)
        print(f"saved={directory}", flush=True)
        return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="One or more saved runs/.../config.yaml files")
    parser.add_argument("--dataset", choices=["checkerboard", "horse"], help="Infer from config by default")
    parser.add_argument("--bank", help="Relocated original bank; allowed with one config, exact hash required")
    parser.add_argument("--device", help="Override inference device only (e.g. cpu or cuda)")
    for name, default in (("clouds", 32), ("batch-size", 8), ("neighbors", 8), ("distance-bins", 8),
                          ("routing-draws", 100000), ("routing-chunk-size", 10000), ("anchors", 64),
                          ("contexts", 8), ("path-steps", 128), ("grid-bins", 16), ("seed", 2026)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--routing-neighbors", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--nfes", type=int, nargs="+", default=[1, 128])
    parser.add_argument("--times", type=float, nargs="+", default=list(TIMES))
    parser.add_argument("--roi-file", help="Prespecified horse ROI definition")
    parser.add_argument("--output", default="analysis_results")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Verify checkpoint and original bank, no diagnostics/output")
    args = vars(parser.parse_args(argv))
    configs = args.pop("configs")
    if args["bank"] is not None and len(configs) != 1:
        parser.error("--bank requires exactly one config; run relocated banks separately")
    for config_path in configs:
        audit(config_path, **args)


if __name__ == "__main__":
    main()
