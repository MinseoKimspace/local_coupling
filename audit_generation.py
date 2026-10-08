"""Joint generation-quality, integration and optional FM-residual diagnostics.

All NFE levels share a fixed bank from the configured source prior and a fixed fresh target bank.
Reference refinement is an empirical check, not a proof of ODE convergence.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from coupling import CLOUD_METHODS, TG_CACHED_METHODS, coupled_points
from data import checkerboard_centers
from diagnose import BINS, NFES, TIMES, fm_errors, rollout
from diagnostic_data import (DiagnosticData, diagnostic_metadata, finite_scores,
                             load_verified_model, tensor_sha256)
from experiment import evaluation_settings, evaluation_title, synchronize
from horse_regions import HorseRegions
from metrics import chamfer_distance, checkerboard_metrics, horse_metrics
from summarize_results import formatted, output_directory, plt, save_json, save_table, statistics


QUALITY_KEYS = ("chamfer", "leakage", "histogram_js", "cell_mass_error",
                "thin_region_mass_mae", "gap_region_leakage")


def reference_levels(start, maximum):
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
           for value in (start, maximum)):
        raise ValueError("Reference NFEs must be integers")
    if start < 1 or maximum < 2 * start:
        raise ValueError("max_reference_nfe must allow at least one doubling of reference_nfe")
    levels = [start]
    while levels[-1] < maximum:
        levels.append(2 * levels[-1])
    if levels[-1] != maximum:
        raise ValueError("max_reference_nfe must equal reference_nfe times a power of two")
    return levels


def quality_scores(prediction, target, config, dataset, mask=None, regions=None):
    scores = {"chamfer": chamfer_distance(prediction, target).item()}
    bins = evaluation_settings(config)["histogram_bins"]
    if dataset == "horse":
        leakage, js = horse_metrics(prediction, mask, bins)
        scores.update(leakage=leakage, histogram_js=js)
        scores.update(regions.score(prediction))
    else:
        leakage, mass, js = checkerboard_metrics(prediction, config["data"]["grid_size"], bins)
        scores.update(leakage=leakage, cell_mass_error=mass, histogram_js=js)
    return finite_scores(scores)


@torch.no_grad()
def rollout_bank(model, noise, steps, batch_size, keep_path=False):
    parameter = next(model.parameters())
    predictions, ratios, first_path = [], [], None
    synchronize(parameter.device)
    start = perf_counter()
    for begin in range(0, len(noise), batch_size):
        part = noise[begin:begin + batch_size].to(device=parameter.device, dtype=parameter.dtype)
        prediction, ratio, path = rollout(model, part, steps, keep_path=keep_path and begin == 0)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"Nonfinite generated points at NFE={steps}")
        predictions.append(prediction.cpu())
        ratios.extend(ratio.cpu().tolist())
        if path is not None:
            first_path = path.cpu()
    synchronize(parameter.device)
    return torch.cat(predictions), ratios, first_path, perf_counter() - start


def convergence_checks(predictions, quality, levels, endpoint_tolerance, quality_tolerance):
    """Check paired endpoint errors AND stability of defined primary metrics.

    Both absolute mean-coordinate MSE and its cloud 95th percentile must pass.
    Scalar quality deltas use absolute units (nats/fractions/squared distance).
    Undefined conditioned metrics are reported and never substituted with zero.
    """
    checks = []
    for coarse, fine in zip(levels, levels[1:]):
        errors = (predictions[coarse].double() - predictions[fine].double()).square().mean((1, 2))
        values = errors.tolist()
        p95 = float(np.quantile(values, .95))
        deltas, undefined = {}, []
        for key in QUALITY_KEYS:
            if key not in quality[coarse] and key not in quality[fine]:
                continue
            left, right = quality[coarse].get(key), quality[fine].get(key)
            if left is None or right is None:
                undefined.append(key)
            else:
                deltas[key] = abs(right - left)
        mse = statistics(values)
        endpoint_passed = mse["mean"] <= endpoint_tolerance and p95 <= endpoint_tolerance
        quality_passed = bool(deltas) and all(v <= quality_tolerance for v in deltas.values())
        checks.append({"coarse_nfe": coarse, "fine_nfe": fine, "endpoint_mse": mse,
                       "endpoint_mse_p95": p95, "quality_absolute_changes": deltas,
                       "undefined_quality_metrics": undefined,
                       "endpoint_check_passed": endpoint_passed, "quality_check_passed": quality_passed,
                       "checks_passed": endpoint_passed and quality_passed})
    return checks


@torch.no_grad()
def fm_summary(model, config, data, batches, batch_size, seed):
    if batches > 0 and (config.get("anchor_flow") or {}).get("mode") == "anchor_waypoint":
        raise ValueError("Linear-path FM residuals are invalid for anchor_waypoint; use --skip-fm")
    cloud_coupling = config["coupling"] in CLOUD_METHODS
    if batch_size is None:
        batch_size = config["data"]["batch_size"] if cloud_coupling else 16
    if batch_size < 1 or cloud_coupling and batch_size != config["data"]["batch_size"]:
        raise ValueError("Cloud OT FM diagnostics require training data.batch_size; matching law must not change")
    if batches < 1:
        return None
    parameter = next(model.parameters())
    generator = torch.Generator(device=parameter.device).manual_seed(seed + 1)
    time_generator = torch.Generator(device=parameter.device).manual_seed(seed + 2)
    pair_sampler = None
    if config["coupling"] == "nsot":
        from nsot import NSOTPairSampler
        pair_sampler = NSOTPairSampler(config, data.dataset, parameter.device, parameter.dtype)
    elif config["coupling"] in TG_CACHED_METHODS:
        from tg_cache import TGCachedPairSampler
        pair_sampler = TGCachedPairSampler(config, data.dataset, parameter.device, parameter.dtype)
    centers = (checkerboard_centers(config["data"]["grid_size"], parameter.device, parameter.dtype)
               if data.dataset == "checkerboard" else None)
    errors, bins, energy = [[] for _ in TIMES], [[] for _ in BINS], []
    numpy_state = np.random.get_state()
    np.random.seed((seed + 1) % 2**32)
    try:
        for batch in range(batches):
            indices = range(batch * batch_size, (batch + 1) * batch_size)
            source = torch.stack([data.source(i, "fm") for i in indices])
            target = torch.stack([data.target(i, "fm") for i in indices])
            if pair_sampler is None:
                paired_source, paired_target = coupled_points(
                    source, target, coupling=config["coupling"], num_regions=config.get("num_regions"),
                    target_centers=centers, sinkhorn_epsilon=config.get("sinkhorn_epsilon", .1),
                    sinkhorn_iterations=config.get("sinkhorn_iterations", 100), generator=generator)
            else:
                paired_source, paired_target = pair_sampler.sample(batch_size, generator=generator)
            for j, t in enumerate(TIMES):
                residual, baseline = fm_errors(model, paired_source, paired_target,
                                               source.new_full((batch_size, 1, 1), t), config=config)
                errors[j].extend(residual.tolist())
                if j == 0:
                    energy.extend(baseline.tolist())
            for j, (lo, hi) in enumerate(BINS):
                times = torch.rand(batch_size, 1, 1, device=source.device, dtype=source.dtype,
                                   generator=time_generator) * (hi - lo) + lo
                residual, _ = fm_errors(model, paired_source, paired_target, times, config=config)
                bins[j].extend(residual.tolist())
            print(f"quality_fm_batch={batch + 1}/{batches}", flush=True)
    finally:
        np.random.set_state(numpy_state)
    if not np.isfinite(energy + sum(errors, []) + sum(bins, [])).all():
        raise FloatingPointError("Nonfinite held-out FM diagnostics")
    return {"batches": batches, "matching_batch_size": batch_size,
            "sampling_scope": ("fresh pairing draws from checkpoint's fixed training cache; NOT unseen source/target pool"
                               if pair_sampler is not None else "fresh source and target clouds"),
            "time_errors": [{"t": t, "mse": statistics(v)} for t, v in zip(TIMES, errors)],
            "time_bins": [{"interval": limits, "mse": statistics(v)} for limits, v in zip(BINS, bins)],
            "target_velocity_energy": statistics(energy),
            "definition": "raw residual under checkpoint's own coupling, NOT mean-field approximation error"}


def render(payload, directory):
    title = f"{payload['dataset']} | {evaluation_title(payload['config'])}"
    rows = payload["quality_by_nfe"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, key in zip(axes.flat, QUALITY_KEYS):
        valid = [r for r in rows if r["metrics"].get(key) is not None]
        if not valid:
            ax.axis("off")
            continue
        ax.plot([r["nfe"] for r in valid], [r["metrics"][key] for r in valid], "o-")
        ax.axvline(payload["reference_nfe"], color="gray", ls="--", lw=.8)
        ax.set_xscale("log", base=2)
        ax.set(xlabel="NFE (Euler)", ylabel=key)
        ax.grid(alpha=.2)
    fig.suptitle(title + "\nCOMMON noise and target banks; pooled quality point estimates")
    fig.tight_layout()
    fig.savefig(directory / "quality.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    errors = payload["endpoint_errors"]
    axes[0].plot([r["nfe"] for r in errors], [r["mse"]["mean"] for r in errors], "o-")
    axes[0].set_xscale("log", base=2)
    axes[0].set(xlabel="NFE", ylabel=f"Endpoint MSE vs Euler {payload['reference_nfe']}")
    checks = payload["reference_checks"]
    axes[1].plot([r["fine_nfe"] for r in checks], [r["endpoint_mse_p95"] for r in checks], "o-")
    axes[1].axhline(payload["convergence_tolerances"]["endpoint_mse"], color="red", ls="--")
    axes[1].set(xlabel="Doubled reference NFE", ylabel="Paired endpoint MSE (cloud p95)")
    fig.suptitle(title + "\nFinite refinement checks passed: " + str(payload["finite_refinement_checks_passed"]))
    fig.tight_layout()
    fig.savefig(directory / "integration.png", dpi=160)
    plt.close(fig)
    table = [[r["nfe"], *[f"{r['metrics'][key]:.6g}" if r["metrics"].get(key) is not None else "—"
                           for key in QUALITY_KEYS], formatted(errors[i]["mse"])] for i, r in enumerate(rows)]
    save_table(directory / "table.png", ["NFE", *QUALITY_KEYS, "Endpoint MSE vs reference"], table, title)
    if payload["fm"] is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        values = payload["fm"]["time_errors"]
        ax.plot([r["t"] for r in values], [r["mse"]["mean"] for r in values], "o-")
        ax.set(xlabel="t", ylabel="Held-out FM residual (per coordinate)", title=title)
        fig.tight_layout()
        fig.savefig(directory / "fm_residual.png", dpi=160)
        plt.close(fig)


def audit(config_path, dataset, *, clouds=32, batch_size=16, nfes=NFES, reference_nfe=128,
          max_reference_nfe=512, endpoint_tolerance=1e-5, quality_tolerance=1e-3,
          fm_batches=2, matching_batch_size=None, seed=2026, roi_file=None, output="analysis_results"):
    if clouds < 1 or batch_size < 1 or fm_batches < 0:
        raise ValueError("clouds/batch_size must be positive; fm_batches must be nonnegative")
    if (not nfes or any(isinstance(n, bool) or int(n) != n or n < 1 for n in nfes)
            or not np.isfinite([endpoint_tolerance, quality_tolerance]).all()
            or endpoint_tolerance < 0 or quality_tolerance < 0):
        raise ValueError("NFEs must be positive integers and tolerances finite and nonnegative")
    levels = reference_levels(reference_nfe, max_reference_nfe)
    requested = sorted(set(int(n) for n in nfes))
    if max(requested) > max_reference_nfe:
        raise ValueError("Requested NFEs must not exceed max_reference_nfe")
    model, config, checkpoint, metadata = load_verified_model(config_path, dataset)
    if fm_batches > 0 and (config.get("anchor_flow") or {}).get("mode") == "anchor_waypoint":
        raise ValueError("This audit's optional FM residual assumes a linear path, not anchor_waypoint; "
                         "add --skip-fm to evaluate generation and integration only")
    if (matching_batch_size is not None and config["coupling"] in CLOUD_METHODS
            and matching_batch_size != config["data"]["batch_size"] and fm_batches > 0):
        raise ValueError("Cloud OT diagnostics require training data.batch_size")
    parameter = next(model.parameters())
    data = DiagnosticData(config, dataset, parameter.device, parameter.dtype, seed)
    noise, target = data.bank(clouds)
    noise, target = noise.cpu(), target.cpu()
    regions = HorseRegions(data.mask, roi_file) if dataset == "horse" else None
    predictions, quality, inference_times = {}, {}, {}
    all_nfes = sorted(set(requested + levels))
    for nfe in all_nfes:
        prediction, ratios, path, seconds = rollout_bank(model, noise, nfe, batch_size,
                                                        keep_path=nfe == max_reference_nfe)
        predictions[nfe] = prediction
        quality[nfe] = quality_scores(prediction, target, config, dataset, data.mask, regions)
        inference_times[nfe] = seconds
        if nfe == max_reference_nfe:
            reference_ratios, reference_path = ratios, path
        print(f"quality_nfe={nfe} clouds={clouds} inference_seconds={seconds:.3f}", flush=True)
    checks = convergence_checks(predictions, quality, levels, endpoint_tolerance, quality_tolerance)
    # Use the finest reference regardless of apparent early convergence. A
    # single pair is reported as insufficient for two successive checks.
    checked = len(checks) >= 2 and all(r["checks_passed"] for r in checks[-2:])
    reference = predictions[max_reference_nfe]
    errors = [{"nfe": nfe, "mse": statistics(
        (predictions[nfe].double() - reference.double()).square().mean((1, 2)).tolist())}
        for nfe in all_nfes]
    fm = fm_summary(model, config, data, fm_batches, matching_batch_size, seed) if fm_batches else None
    payload = {
        **diagnostic_metadata(config_path, config, checkpoint, metadata, dataset, parameter.device),
        "format_version": 1, "diagnostic_seed": seed, "evaluation_clouds": clouds,
        "evaluation_batch_size": batch_size, "requested_nfes": requested,
        "reference_start_nfe": reference_nfe, "reference_nfe": max_reference_nfe,
        "reference_levels": levels, "reference_checks": checks,
        "finite_refinement_checks_passed": checked, "required_successive_checks": 2,
        "convergence_tolerances": {"endpoint_mse": endpoint_tolerance,
                                   "primary_quality_absolute_change": quality_tolerance},
        "noise_sha256": tensor_sha256(noise), "target_sha256": tensor_sha256(target),
        "definitions": {
            "draws": "fixed configured-source-prior and fresh target banks, indexed by cloud; independent of FM matching batch; source hashes differ when the prior changes",
            "endpoint": "same INDEXED particles vs finest finite Euler reference; NOT target correspondence or mean-field approximation error",
            "reference": "finest requested refinement, even if checks fail; two successive endpoint+quality checks, NOT a proof of exact integration",
            "quality": "Chamfer sum of directional squared-distance means, averaged across clouds; other metrics POOLED across all generated points",
            "std": "endpoint sample SD across fresh clouds; quality metrics are point estimates, NOT training-seed uncertainty",
            "tolerances": "absolute per-coordinate endpoint MSE (mean AND p95); primary quality changes in their original units; undefined conditioned TV is excluded and reported",
            "gap_vs_reference": "signed quality change vs reference is descriptive, NOT an additive error decomposition; error cancellation can make it negative",
            "fm": "separate fresh interpolation bank and original matching-batch law; raw FM residual includes conditional variance",
            "rng": "diagnostic_draw_v1; compare noise/target hashes before comparing methods",
        },
        "horse_roi_definition": regions.definition() if regions is not None else None,
        "quality_by_nfe": [{"nfe": nfe, "metrics": quality[nfe], "inference_seconds": inference_times[nfe],
                            "signed_quality_change_vs_reference": {
                                key: value - quality[max_reference_nfe][key]
                                for key, value in quality[nfe].items()
                                if value is not None and quality[max_reference_nfe].get(key) is not None}}
                           for nfe in all_nfes],
        "endpoint_errors": errors, "path_ratio": statistics(reference_ratios), "fm": fm,
        "example_trajectory": {"times": [i / max_reference_nfe for i in range(max_reference_nfe + 1)],
                               "points": reference_path.tolist()},
    }
    if config.get("anchor_flow") is not None:
        payload["initial_source_quality"] = quality_scores(noise, target, config, dataset, data.mask, regions)
        payload["definitions"]["initial_source_quality"] = (
            "no-flow prior baseline before any model step, scored against the SAME fixed target bank")
    directory = output_directory(Path(output) / dataset, checkpoint.stem + "_quality_diagnostic")
    save_json(directory / "diagnostics.json", payload)
    render(payload, directory)
    if regions is not None:
        regions.render(directory / "horse_rois.png")
    if not checked:
        print("WARNING: reference refinement checks did not pass twice; do NOT assume solver error is negligible.", flush=True)
    print(f"saved={directory}")
    return directory


def arguments(parser):
    parser.add_argument("--clouds", type=int, default=32, help="COMMON evaluation clouds, independent of matching batch size")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Inference microbatch; does not change noise/target banks")
    parser.add_argument("--nfes", type=int, nargs="+", default=list(NFES))
    parser.add_argument("--max-reference-nfe", type=int, default=512)
    parser.add_argument("--endpoint-tolerance", type=float, default=1e-5)
    parser.add_argument("--quality-tolerance", type=float, default=1e-3)
    parser.add_argument("--roi-file", default=None, help="Prespecified horse ROI JSON; identical file for all methods")
    parser.add_argument("--skip-fm", action="store_true", help="Only generation+integration; skip costly training-coupling residuals")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Verified saved runs/.../config.yaml")
    parser.add_argument("--dataset", required=True, choices=["checkerboard", "horse"])
    parser.add_argument("--batches", type=int, default=2, help="Fresh FM diagnostic batches")
    parser.add_argument("--batch-size", type=int, default=None, help="FM matching batch; cloud OT must use training batch")
    parser.add_argument("--reference-nfe", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="analysis_results")
    arguments(parser)
    args = parser.parse_args()
    audit(args.config, args.dataset, clouds=args.clouds, batch_size=args.eval_batch_size, nfes=args.nfes,
          reference_nfe=args.reference_nfe, max_reference_nfe=args.max_reference_nfe,
          endpoint_tolerance=args.endpoint_tolerance, quality_tolerance=args.quality_tolerance,
          fm_batches=0 if args.skip_fm else args.batches, matching_batch_size=args.batch_size,
          seed=args.seed, roi_file=args.roi_file, output=args.output)
