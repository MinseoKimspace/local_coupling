"""Join MS-W2 rollout Jacobians with quality on the EXACT same generated clouds.

No training, cache loading, coupling changes, or new Jacobian estimates. Existing
source and every recorded rollout-state hash must match before results are joined.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
import torch

from audit_generation import QUALITY_KEYS, convergence_checks, quality_scores, reference_levels
from audit_jacobian import rollout_snapshots
from diagnostic_data import DiagnosticData, diagnostic_metadata, load_verified_model, tensor_sha256
from experiment import training_signature
from horse_regions import HorseRegions
from summarize_results import output_directory, plt, save_json, statistics


def read_audit(path):
    audit = json.loads(Path(path).read_text(encoding="utf-8"))
    if audit.get("format_version") != 1 or not audit.get("training_config_verified"):
        raise ValueError("Expected a verified format_version=1 audit_jacobian.py report")
    if audit["config"]["coupling"] != "multiscale_wasserstein":
        raise ValueError("This diagnostic is MS-W2 only; cross-leaf/refined reports are not accepted")
    if not audit.get("rollout_steps") or "rollout" not in audit.get("banks", {}):
        raise ValueError("The Jacobian report needs rollout scope (rollout or both)")
    times = sorted(audit["times"])
    if (not times or times[0] != 0 or times[-1] != 1 or len(set(times)) != len(times)
            or not np.isfinite(times).all() or min(times) < 0 or max(times) > 1):
        raise ValueError("The Jacobian time grid must include 0 and 1 without duplicates")
    records = {}
    for row in audit["per_cloud"]:
        if row["scope"] != "rollout":
            continue
        key = (row["cloud_index"], row["time"])
        if key in records:
            raise ValueError("Duplicate rollout cloud/time record")
        if (not np.isfinite([row["frobenius_squared"], row["normalized_frobenius"]]).all()
                or row["frobenius_squared"] < 0 or row["normalized_frobenius"] < 0):
            raise ValueError("Invalid Jacobian norm")
        records[key] = row
    expected = {(i, t) for i in range(audit["clouds"]) for t in times}
    if not expected or set(records) != expected:
        raise ValueError("Incomplete rollout cloud/time records")
    return audit, records, times


def quartiles(rows, metric):
    """Quality-only descriptive ranking; flag ties and do not split a constant metric."""
    valid = [r for r in rows if r.get(metric) is not None and np.isfinite(r[metric])]
    ordered = sorted(valid, key=lambda r: (r[metric], r["cloud_index"]))
    result = {"metric": metric, "lower_is_better": True, "n": len(valid),
              "low_error_ids": [], "high_error_ids": [], "boundary_ties": False}
    if len(valid) < 4 or ordered[0][metric] == ordered[-1][metric]:
        return {**result, "status": "not_separable_or_fewer_than_four_valid_clouds"}
    count = max(1, len(valid) // 4)
    result.update(status="descriptive_quartiles", group_size=count,
                  low_error_ids=[r["cloud_index"] for r in ordered[:count]],
                  high_error_ids=[r["cloud_index"] for r in reversed(ordered[-count:])],
                  boundary_ties=(ordered[count - 1][metric] == ordered[count][metric]
                                 or ordered[-count][metric] == ordered[-count - 1][metric]))
    return result


def correlations(rows, quality_keys, feature_keys):
    result = []
    for metric in quality_keys:
        for feature in feature_keys:
            valid = [r for r in rows if r.get(metric) is not None and r.get(feature) is not None
                     and np.isfinite([r[metric], r[feature]]).all()]
            x, y = [r[metric] for r in valid], [r[feature] for r in valid]
            rho = (float(spearmanr(x, y).statistic)
                   if len(valid) >= 3 and len(set(x)) > 1 and len(set(y)) > 1 else None)
            result.append({"quality": metric, "jacobian_feature": feature, "n": len(valid),
                           "spearman_rho": rho})
    return result


def jacobian_features(records, index, times, dimension):
    norms = np.array([records[index, t]["normalized_frobenius"] for t in times])
    squared = np.array([records[index, t]["frobenius_squared"] / dimension for t in times])
    late = np.array(times) >= .75
    # Only integrate the observed grid; no claim about unsampled endpoint spikes.
    integral = float(np.sum(np.diff(times) * (squared[:-1] + squared[1:]) / 4))
    return {"jacobian_t1": float(norms[-1]), "jacobian_peak": float(norms.max()),
            "jacobian_peak_time": float(times[int(norms.argmax())]),
            "jacobian_late_mean": float(norms[late].mean()),
            "jacobian_energy_trapezoid": integral}


def metric_summary(rows):
    keys = sorted(set().union(*(r.keys() for r in rows)))
    return {key: statistics([r.get(key) for r in rows]) for key in keys}


def write_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render(payload, directory, source, predictions, control, mask, examples):
    """Static scientific plots, always using quality-only selections."""
    dataset, rank_nfe = payload["dataset"], payload["ranking_nfe"]
    rows = [r for r in payload["per_cloud"] if r["nfe"] == rank_nfe]
    keys = payload["ranking_metrics"]
    times = payload["jacobian_times"]
    curves = payload["jacobian_normalized_by_cloud"]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        valid = [r for r in rows if r[key] is not None]
        ax.scatter([r["jacobian_late_mean"] for r in valid], [r[key] for r in valid])
        for r in valid:
            ax.annotate(str(r["cloud_index"]), (r["jacobian_late_mean"], r[key]), fontsize=8)
        ax.set(xlabel="Late rollout Jacobian (normalized)", ylabel=key)
        ax.grid(alpha=.2)
    fig.suptitle(f"{dataset} MS-W2 | same-cloud NFE {rank_nfe} | association, NOT causality")
    fig.tight_layout()
    fig.savefig(directory / "quality_vs_jacobian.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes[0], keys):
        for i in range(len(source)):
            cloud = [r for r in payload["per_cloud"] if r["cloud_index"] == i]
            ax.plot([r["nfe"] for r in cloud], [r[key] if r[key] is not None else np.nan
                                              for r in cloud], alpha=.45, lw=1)
        values = [r[key] for r in payload["target_controls"] if r.get(key) is not None]
        if values:
            lo, hi = np.quantile(values, [.1, .9])
            ax.axhspan(lo, hi, color="gray", alpha=.18, label="Target-only 10-90%")
        ax.axvline(rank_nfe, color="black", ls=":", label="Jacobian rollout NFE")
        ax.set(xscale="log", xlabel="NFE (Euler)", ylabel=key)
        ax.legend(fontsize=8)
    fig.suptitle(f"{dataset} MS-W2 | each line is one fixed source cloud")
    fig.tight_layout()
    fig.savefig(directory / "quality_by_cloud.png", dpi=150)
    plt.close(fig)

    def background(ax):
        if dataset == "horse":
            array = mask.cpu().numpy()
            h, w = array.shape
            scale = max(h, w)
            extent = (-w / scale, w / scale, -h / scale, h / scale)
        else:
            grid = payload["config"]["data"]["grid_size"]
            array = (np.arange(grid)[:, None] + np.arange(grid)[None, :]) % 2 == 0
            extent = (-1, 1, -1, 1)
        ax.imshow(array, origin="upper", cmap="Greys", alpha=.15, extent=extent,
                  vmin=0, vmax=1, interpolation="nearest")

    gallery_nfes = sorted({min(predictions), min(predictions, key=lambda n: abs(n - 8)),
                          rank_nfe, max(predictions)})
    lookup = {(r["cloud_index"], r["nfe"]): r for r in payload["per_cloud"]}
    for group in payload["quality_groups"]:
        selected = [("low error", i) for i in group["low_error_ids"][:examples]]
        selected += [("high error", i) for i in group["high_error_ids"][:examples]]
        if not selected:
            continue
        key = group["metric"]
        fig, axes = plt.subplots(len(selected), len(gallery_nfes) + 2,
                                 figsize=(3 * (len(gallery_nfes) + 2), 2.8 * len(selected)), squeeze=False)
        points = torch.cat([predictions[n][i] for _, i in selected for n in gallery_nfes])
        limit = max(1.05, float(points.abs().max()) * 1.03)
        jacobian_limit = max(1e-6, max(max(curves[i]) for _, i in selected) * 1.05)
        for axes_row, (label, i) in zip(axes, selected):
            for ax, n in zip(axes_row, gallery_nfes):
                cloud, score = predictions[n][i].numpy(), lookup[i, n]
                background(ax)
                ax.scatter(*cloud.T, s=5)
                stable = "pass" if score["finite_refinement_checks_passed"] else "not passed"
                ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
                       title=f"#{i} {label} | NFE {n}\nCD={score['chamfer']:.4g}, leak={score['leakage']:.1%}\nrefinement: {stable}")
            ax = axes_row[-2]
            background(ax)
            ax.scatter(*control[i % len(control)].numpy().T, s=5, color="tab:green")
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal", title="Independent target-only draw")
            ax = axes_row[-1]
            ax.plot(times, curves[i], "o-", ms=3)
            ax.set(xlabel="t", ylabel="Normalized Jacobian", ylim=(0, jacobian_limit),
                   title=f"Original NFE {rank_nfe} rollout")
        fig.suptitle(f"{dataset} | quality selection: {key} at NFE {rank_nfe}; NOT Jacobian-selected")
        fig.tight_layout()
        fig.savefig(directory / f"gallery_{key}.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        for label, ids, color in (("Low error", group["low_error_ids"], "tab:blue"),
                                  ("High error", group["high_error_ids"], "tab:red")):
            array = np.array([curves[i] for i in ids])
            for curve in array:
                ax.plot(times, curve, color=color, alpha=.25, lw=.7)
            ax.plot(times, np.median(array, axis=0), "o-", color=color, label=label + " median")
        ax.set(xlabel="t", ylabel="Normalized Jacobian", title=f"{dataset}: grouped by {key} (quality only)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f"jacobian_groups_{key}.png", dpi=150)
        plt.close(fig)

    if dataset == "horse":
        regions = HorseRegions(mask)
        roi = next(r for r in regions.regions if r["name"] == "front_legs")
        group = next(g for g in payload["quality_groups"] if g["metric"] == "roi_front_legs_mass_error")
        ids = group["low_error_ids"][:examples] + group["high_error_ids"][:examples]
        if ids:
            fig, axes = plt.subplots(len(ids), len(gallery_nfes),
                                     figsize=(3 * len(gallery_nfes), 3 * len(ids)), squeeze=False)
            x0, x1, y0, y1 = roi["box"]
            for row, i in zip(axes, ids):
                for ax, n in zip(row, gallery_nfes):
                    background(ax)
                    ax.scatter(*predictions[n][i].numpy().T, s=10)
                    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="tab:blue", lw=.8)
                    score = lookup[i, n]
                    ax.set(xlim=(x0 - .04, x1 + .04), ylim=(y0 - .04, y1 + .04), aspect="equal",
                           title=f"#{i} NFE {n}\nForeground ROI count={score['front_legs_count']}")
            fig.suptitle(f"Front-leg crop | target expected count = {roi['expected_mass'] * source.shape[1]:.2f}")
            fig.tight_layout()
            fig.savefig(directory / "front_legs_zoom.png", dpi=150)
            plt.close(fig)
        regions.render(directory / "horse_rois.png")


def audit(jacobian_path, *, config_path=None, nfes=(1, 8, 128, 256, 512), reference_nfe=128,
          max_reference_nfe=512, control_clouds=64, endpoint_tolerance=1e-5,
          quality_tolerance=1e-3, examples_per_group=2, skip_render=False, output="analysis_results"):
    old, records, times = read_audit(jacobian_path)
    levels = reference_levels(reference_nfe, max_reference_nfe)
    if len(levels) < 3:
        raise ValueError("Use at least two reference doublings (e.g. 128, 256, 512)")
    if (not nfes or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in nfes)
            or max(nfes) > max_reference_nfe or old["rollout_steps"] > max_reference_nfe):
        raise ValueError("NFEs must be positive integers <= max_reference_nfe (including original rollout)")
    if (isinstance(control_clouds, bool) or not isinstance(control_clouds, int) or control_clouds < 2
            or isinstance(examples_per_group, bool) or not isinstance(examples_per_group, int)
            or examples_per_group < 1 or not np.isfinite([endpoint_tolerance, quality_tolerance]).all()
            or min(endpoint_tolerance, quality_tolerance) < 0):
        raise ValueError("Use >=2 target controls, >=1 example and finite nonnegative tolerances")
    requested = sorted(set(nfes) | set(levels) | {old["rollout_steps"]})
    path = Path(config_path or old["config_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Saved config not found: {path}. Run on the checkpoint machine or use --config RUN_CONFIG.")
    model, config, checkpoint, metadata = load_verified_model(path, old["dataset"])
    if (config["coupling"] != "multiscale_wasserstein"
            or training_signature(config) != training_signature(old["config"])):
        raise ValueError("Config does not match the original MS-W2 Jacobian audit")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != old["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA256 differs from the Jacobian audit")
    parameter = next(model.parameters())
    dataset, count, rank_nfe = old["dataset"], old["clouds"], old["rollout_steps"]
    dimension = config["data"]["n_points"] * config["model"]["point_dim"]
    if dimension != old["input_dimension"]:
        raise ValueError("Jacobian input dimension differs from the checkpoint")
    data = DiagnosticData(config, dataset, parameter.device, parameter.dtype, old["diagnostic_seed"])
    source = torch.stack([data.source(i, "jacobian_rollout").cpu() for i in range(count)])
    if tensor_sha256(source) != old["banks"]["rollout"]["source_sha256"]:
        raise ValueError("Source SHA256 mismatch. Use the original device/dtype/software, or rerun audit_jacobian.py --scope rollout.")
    # Verify ALL recorded states, not merely seed or source; never silently join stale Jacobians.
    paths = []
    for i, cloud in enumerate(source):
        snapshots = rollout_snapshots(model, cloud.to(parameter), times, rank_nfe)
        for t, state in snapshots.items():
            if tensor_sha256(state) != records[i, t]["state_sha256"]:
                raise ValueError(f"Rollout state SHA256 mismatch at cloud={i}, t={t}. "
                                 "Rerun audit_jacobian.py --scope rollout in this environment; do not join stale Jacobians.")
        paths.append(torch.stack([snapshots[t].cpu() for t in times]))
        print(f"verified_original_rollout={i + 1}/{count}", flush=True)
    paths = torch.stack(paths)
    predictions = {rank_nfe: paths[:, -1].clone()}
    for nfe in requested:
        if nfe not in predictions:
            values = []
            for i, cloud in enumerate(source):
                state = rollout_snapshots(model, cloud.to(parameter), [1.], nfe)[1.]
                values.append(state.cpu())
                print(f"quality_nfe={nfe} cloud={i + 1}/{count}", flush=True)
            predictions[nfe] = torch.stack(values)
    predictions = dict(sorted(predictions.items()))
    # Targets are fresh independent draws, shared across all NFE levels for each cloud.
    target = torch.stack([data.target(i, "ms_w2_quality_reference_v1").cpu() for i in range(count)])
    control = torch.stack([data.target(i, "ms_w2_quality_control_v1").cpu() for i in range(control_clouds)])
    control_target = torch.stack([data.target(i, "ms_w2_control_reference_v1").cpu() for i in range(control_clouds)])
    mask = data.mask.cpu() if dataset == "horse" else None
    regions = HorseRegions(mask) if dataset == "horse" else None

    def score(prediction, reference):
        return quality_scores(prediction, reference, config, dataset, mask, regions)

    features = [jacobian_features(records, i, times, dimension) for i in range(count)]
    quality, rows = {}, []
    for nfe in requested:
        quality[nfe] = score(predictions[nfe], target)
        for i in range(count):
            values = score(predictions[nfe][i:i + 1], target[i:i + 1])
            values.pop("histogram_js")  # At N=256, sparse per-cloud histograms are not a ranking metric.
            if regions:
                values["front_legs_count"] = round(values["roi_front_legs_mass"] * data.n)
            rows.append({"cloud_index": i, "nfe": nfe, "jacobian_rollout_nfe": rank_nfe,
                         **values, **features[i],
                         "paired_mse_vs_finest": float((predictions[nfe][i].double()
                                                        - predictions[max_reference_nfe][i].double()).square().mean())})
    controls = [score(control[i:i + 1], control_target[i:i + 1]) for i in range(control_clouds)]
    for item in controls:
        item.pop("histogram_js")
        if regions:
            item["front_legs_count"] = round(item["roi_front_legs_mass"] * data.n)
    refinement = []
    for i in range(count):
        checks = convergence_checks({n: predictions[n][i:i + 1] for n in levels},
                                    {r["nfe"]: r for r in rows if r["cloud_index"] == i},
                                    levels, endpoint_tolerance, quality_tolerance)
        passed = all(c["checks_passed"] for c in checks)
        refinement.append({"cloud_index": i, "checks": checks, "finite_refinement_checks_passed": passed})
        for row in rows:
            if row["cloud_index"] == i:
                row["finite_refinement_checks_passed"] = passed
    ranking_metrics = ["chamfer", "leakage", "roi_front_legs_mass_error" if regions else "cell_mass_error"]
    rank_rows = [r for r in rows if r["nfe"] == rank_nfe]
    groups = [quartiles(rank_rows, key) for key in ranking_metrics]
    payload = {
        **diagnostic_metadata(path, config, checkpoint, metadata, dataset, parameter.device),
        "format_version": 1, "diagnostic": "ms_w2_same_cloud_quality_v1",
        "jacobian_file": str(Path(jacobian_path).resolve()),
        "jacobian_file_sha256": hashlib.sha256(Path(jacobian_path).read_bytes()).hexdigest(),
        "jacobian_method": old["method"], "jacobian_probes": old["probes"],
        "jacobian_environment": old["environment"], "jacobian_times": times,
        "jacobian_rollout_nfe": rank_nfe, "ranking_nfe": rank_nfe, "clouds": count,
        "diagnostic_seed": old["diagnostic_seed"], "nfes": requested,
        "ranking_metrics": ranking_metrics, "reference_levels": levels, "reference_nfe": max_reference_nfe,
        "convergence_tolerances": {"endpoint_mse": endpoint_tolerance, "quality_absolute_change": quality_tolerance},
        "source_sha256": tensor_sha256(source), "target_sha256": tensor_sha256(target),
        "target_control_sha256": tensor_sha256(control), "control_reference_sha256": tensor_sha256(control_target),
        "all_original_rollout_state_hashes_verified": True,
        "quality_by_nfe": [{"nfe": n, "pooled_metrics": quality[n],
                            "cloud_statistics": metric_summary([{k: v for k, v in r.items() if k in QUALITY_KEYS}
                                                                 for r in rows if r["nfe"] == n])} for n in requested],
        "per_cloud": rows, "refinement_by_cloud": refinement, "quality_groups": groups,
        "correlations_at_original_rollout_nfe": correlations(
            rank_rows, ranking_metrics, ["jacobian_late_mean", "jacobian_t1", "jacobian_energy_trapezoid"]),
        "jacobian_normalized_by_cloud": [[records[i, t]["normalized_frobenius"] for t in times] for i in range(count)],
        "original_rollout_jacobian_records": list(records.values()),
        "target_controls": controls, "target_control_summary": metric_summary(controls),
        "horse_regions": regions.definition() if regions else None,
        "definitions": {
            "pairing": "Each quality row and Jacobian feature share a hash-verified source cloud and checkpoint. "
                       "Jacobian is from the ORIGINAL Euler rollout only; it is NOT remeasured at every NFE.",
            "primary_comparison": "Quality-only quartiles at the ORIGINAL Jacobian rollout NFE; lower error is better. "
                                  "Cloud IDs break ties; boundary ties are flagged. No groups for constant metrics.",
            "chamfer": "Sum of directional mean squared nearest-neighbor distances; one fresh equal-N target per cloud, fixed across NFEs.",
            "control": "Independent target-vs-target draws at the same N quantify finite-sampling variability; NOT an accuracy bound or a significance test.",
            "jacobian_late_mean": "Unweighted mean normalized Frobenius over recorded t>=0.75; not a time integral.",
            "jacobian_energy_trapezoid": "Trapezoidal integral of ||J||_F^2/(2*N*D) on the ORIGINAL finite rollout grid; not Lebesgue spatial Dirichlet energy.",
            "refinement": "Two or more Euler doublings; endpoint MSE and defined quality changes must pass absolute tolerances. Empirical stability, NOT exact convergence.",
            "quality": "Leakage uses ALL points. Horse foreground ROI mass intersects the silhouette. Checker cell TV is conditioned on valid points; undefined if none.",
        },
        "limitations": [
            "Exploratory within-checkpoint correlations, not causality or training-seed uncertainty; small cloud count and Hutchinson noise matter.",
            "Stable endpoints can still have structural/model/coupling errors. Small Jacobian alone does not imply good density or identify the error's cause.",
            "Quality-based extreme groups exaggerate quality differences by construction; inspect all-cloud scatter and target controls too.",
            "Only a finite target reference is used for per-cloud CD. Values are not paired with previous evaluation JSONs using other sources/targets.",
            "t=1 Jacobian is diagnostic: Euler does not query t=1 to obtain its endpoint.",
        ],
    }
    directory = output_directory(Path(output) / dataset, checkpoint.stem + "_same_cloud")
    save_json(directory / "same_cloud.json", payload)
    write_rows(directory / "per_cloud.csv", rows)
    write_rows(directory / "target_controls.csv", [{"control_index": i, **row} for i, row in enumerate(controls)])
    np.savez_compressed(directory / "clouds.npz", source=source.numpy(), target=target.numpy(),
                        target_control=control.numpy(), control_reference=control_target.numpy(),
                        original_rollout_times=np.array(times), original_rollout_states=paths.numpy(),
                        **{f"prediction_nfe_{n}": p.numpy() for n, p in predictions.items()})
    if not skip_render:
        render(payload, directory, source, predictions, control, mask, examples_per_group)
    print(f"same_cloud_hashes=verified; ranking_nfe={rank_nfe}; "
          f"refinement_passed={sum(r['finite_refinement_checks_passed'] for r in refinement)}/{count}")
    print(" nfe       CD       leakage   (pooled over these fixed clouds)")
    for n in requested:
        print(f"{n:4d}  {quality[n]['chamfer']:.6g}  {quality[n]['leakage']:.4%}")
    for group in groups:
        print(f"quality={group['metric']} low_error_ids={group['low_error_ids']} high_error_ids={group['high_error_ids']}")
    for item in payload["correlations_at_original_rollout_nfe"]:
        if item["jacobian_feature"] == "jacobian_late_mean":
            print(f"exploratory_spearman late_J_vs_{item['quality']}={item['spearman_rho']} n={item['n']}")
    print(f"saved={directory}", flush=True)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jacobian_path", help="Existing MS-W2 jacobian.json (rollout or both scope)")
    parser.add_argument("--config", dest="config_path", help="Matching saved run config if its path has moved")
    parser.add_argument("--nfes", type=int, nargs="+", default=[1, 8, 128, 256, 512])
    parser.add_argument("--reference-nfe", type=int, default=128)
    parser.add_argument("--max-reference-nfe", type=int, default=512)
    parser.add_argument("--control-clouds", type=int, default=64)
    parser.add_argument("--endpoint-tolerance", type=float, default=1e-5)
    parser.add_argument("--quality-tolerance", type=float, default=1e-3)
    parser.add_argument("--examples-per-group", type=int, default=2)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--output", default="analysis_results")
    return audit(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
