"""Read-only full-cloud spatial Jacobian diagnostics for a saved 2D velocity model.

Examples (pass the saved runs/.../config.yaml, not a training template):
    python audit_jacobian.py RUN_CONFIG --dataset horse
    python audit_jacobian.py RUN_CONFIG --dataset checkerboard --scope both --probes 16

J = d vec(v_theta(X,t)) / d vec(X), with t fixed: [N*D,N*D], INCLUDING
cross-point attention blocks. This is not d v_i/d x_i alone, a time derivative,
or the derivative of the integrated flow with respect to its initial noise.

Hutchinson: mean_r ||J^T z_r||^2 estimates ||J||_F^2 without storing J;
independent Rademacher probes satisfy E[z z^T]=I. The squared estimate is
unbiased conditional on X,t; its square root is not an unbiased norm estimate.
Exact mode sums squared norms of all output-coordinate VJPs (expensive).
Neither a sampled Frobenius estimate nor its normalization certifies a global
Lipschitz constant, and a smaller value need not mean better generation.
"""

import argparse
from contextlib import contextmanager
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from coupling import OFFLINE_METHODS, TG_CACHED_METHODS
from diagnostic_data import DiagnosticData, diagnostic_metadata, load_verified_model, tensor_sha256
from experiment import evaluation_title, synchronize
from summarize_results import output_directory, plt, save_json, statistics


TIMES = (0., .25, .5, .75, 1.)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _times(values):
    values = tuple(values)
    if not values or any(isinstance(t, bool) or not isinstance(t, (int, float))
                         or not math.isfinite(t) or not 0 <= t <= 1 for t in values):
        raise ValueError("times must be finite numbers in [0,1]")
    if len(set(values)) != len(values):
        raise ValueError("times must not contain duplicates")
    return tuple(sorted(map(float, values)))


def _rollout_indices(times, steps):
    result = {}
    for t in times:
        index = round(t * steps)
        if not math.isclose(index / steps, t, rel_tol=0, abs_tol=1e-10):
            raise ValueError("Each rollout time must lie on the Euler grid; change --times or --rollout-steps")
        result[index] = t
    return result


@contextmanager
def _math_evaluation(model):
    """Use differentiable attention, restoring every module mode and backend flag.

    PyTorch's fused Transformer inference path is not a reliable autograd path.
    Use the math SDPA backend also for rollout snapshots, for consistency.
    """
    modes = [(module, module.training) for module in model.modules()]
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        model.eval()
        torch.backends.mha.set_fastpath_enabled(False)
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        for module, training in modes:
            module.training = training


def cloud_jacobian(model, cloud, time, *, method="hutchinson", probes=8,
                   generator=None, max_exact_dim=128):
    """Measure one [N,D] cloud; no parameter gradients or input mutations.

    Deliberately evaluate one cloud at a time: never sum cross-cloud Jacobians
    or assume batch independence in a generic network. Only first derivatives
    are required. The MC standard error is for the SQUARED norm, not its root.
    """
    _positive_integer(probes, "probes")
    _positive_integer(max_exact_dim, "max_exact_dim")
    if method not in ("hutchinson", "exact"):
        raise ValueError("method must be hutchinson or exact")
    if (not isinstance(cloud, torch.Tensor) or cloud.ndim != 2 or not cloud.numel()
            or not cloud.is_floating_point() or not torch.isfinite(cloud).all()):
        raise ValueError("cloud must be a nonempty finite floating [N,D] tensor")
    time = _times((time,))[0]
    dimension = cloud.numel()
    if method == "exact" and dimension > max_exact_dim:
        raise ValueError("Exact Jacobian exceeds max_exact_dim; increase --max-exact-dim explicitly or use Hutchinson")
    if generator is None:
        generator = torch.Generator(device=cloud.device).manual_seed(0)
    count = dimension if method == "exact" else probes
    values = []
    # Input cloning is inside inference_mode(False), so this also works when
    # called from no_grad/inference_mode diagnostic code.
    with torch.inference_mode(False), torch.enable_grad(), _math_evaluation(model):
        x = cloud.detach().clone().unsqueeze(0).requires_grad_(True)
        t = x.new_full((1, 1, 1), time)
        velocity = model(x, t)
        if velocity.shape != x.shape or not torch.isfinite(velocity).all():
            raise FloatingPointError("Velocity must be finite and have the same [1,N,D] shape as the input")
        for index in range(count):
            if method == "exact":
                direction = torch.zeros_like(velocity)
                direction.reshape(-1)[index] = 1.
            else:
                direction = torch.randint(2, velocity.shape, device=x.device, generator=generator)
                direction = direction.to(dtype=x.dtype).mul_(2).sub_(1)
            gradient = None
            if velocity.requires_grad:
                gradient, = torch.autograd.grad(velocity, x, grad_outputs=direction,
                                                retain_graph=index + 1 < count,
                                                create_graph=False, allow_unused=True)
            value = x.new_zeros((), dtype=torch.float64) if gradient is None else gradient.double().square().sum()
            if not torch.isfinite(value):
                raise FloatingPointError("Nonfinite input Jacobian; report was not saved")
            values.append(value.detach())
    values = torch.stack(values)
    squared = values.sum() if method == "exact" else values.mean()
    standard_error = (0. if method == "exact" else
                      (values.std(unbiased=True) / math.sqrt(probes)).item() if probes > 1 else None)
    squared = squared.item()
    return {"frobenius_squared": squared, "frobenius": math.sqrt(squared),
            "normalized_frobenius": math.sqrt(squared / dimension),
            "mc_se_squared": standard_error, "input_dimension": dimension}


@torch.no_grad()
def rollout_snapshots(model, source, times, steps):
    """One fixed Euler grid, stopping at the last requested time; not an exact ODE."""
    _positive_integer(steps, "rollout_steps")
    times = _times(times)
    indices = _rollout_indices(times, steps)
    x = source.detach().clone().unsqueeze(0)
    snapshots = {}
    with _math_evaluation(model):
        for index in range(max(indices) + 1):
            if index in indices:
                snapshots[indices[index]] = x[0].detach().clone()
            if index != max(indices):
                t = x.new_full((1, 1, 1), index / steps)
                x = x + model(x, t) / steps
                if not torch.isfinite(x).all():
                    raise FloatingPointError("Nonfinite Euler rollout; report was not saved")
    return snapshots


def _cached_pairs(config, dataset, data, clouds):
    parameter_device, dtype = data.device, data.dtype
    if config["coupling"] in TG_CACHED_METHODS:
        from tg_cache import TGCachedPairSampler
        sampler = TGCachedPairSampler(config, dataset, parameter_device, dtype)
    elif config["coupling"] == "nsot":
        from nsot import NSOTPairSampler
        sampler = NSOTPairSampler(config, dataset, parameter_device, dtype)
    else:
        raise ValueError("coupling scope requires target_guided_cached or nsot; use rollout for other methods")
    sources, targets = [], []
    try:
        for index in range(clouds):
            generator = torch.Generator(device=parameter_device).manual_seed(data.draw_seed("jacobian_pairs", index))
            source, target = sampler.sample(1, generator=generator)
            sources.append(source[0].cpu())
            targets.append(target[0].cpu())
        details = sampler.details()
    finally:
        if hasattr(sampler, "close"):
            sampler.close()
    return torch.stack(sources), torch.stack(targets), details


def _summary(records, dimension):
    result = []
    for scope in sorted({row["scope"] for row in records}):
        for time in sorted({row["time"] for row in records if row["scope"] == scope}):
            rows = [row for row in records if row["scope"] == scope and row["time"] == time]
            mean_squared = statistics([row["frobenius_squared"] for row in rows])["mean"]
            result.append({"scope": scope, "time": time,
                           **{key: statistics([row[key] for row in rows]) for key in
                              ("frobenius_squared", "frobenius", "normalized_frobenius", "mc_se_squared")},
                           "rms_frobenius": math.sqrt(mean_squared),
                           "rms_normalized_frobenius": math.sqrt(mean_squared / dimension)})
    return result


def render(payload, directory):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for scope in sorted({row["scope"] for row in payload["summary"]}):
        rows = [row for row in payload["summary"] if row["scope"] == scope]
        times = np.array([row["time"] for row in rows])
        for axis, key in zip(axes, ("frobenius", "normalized_frobenius")):
            means = np.array([row[key]["mean"] for row in rows])
            stds = np.array([row[key]["std"] or 0. for row in rows])
            axis.plot(times, means, "o-", label=scope)
            axis.fill_between(times, np.maximum(0, means - stds), means + stds, alpha=.15)
            axis.set(xlabel="Flow time t", ylabel=key.replace("_", " "))
            axis.legend()
    figure.suptitle(f"{payload['dataset']} | {evaluation_title(payload['config'])}\n"
                   f"Full-cloud velocity Jacobian | {payload['method']} | mean +/- cloud SD")
    figure.tight_layout()
    figure.savefig(directory / "jacobian.png", dpi=170)
    plt.close(figure)


def audit(config_path, dataset, *, scope="rollout", clouds=16, times=TIMES, probes=8,
          method="hutchinson", rollout_steps=128, max_exact_dim=128, seed=2026, output="analysis_results"):
    _positive_integer(clouds, "clouds")
    _positive_integer(probes, "probes")
    _positive_integer(rollout_steps, "rollout_steps")
    _positive_integer(max_exact_dim, "max_exact_dim")
    times = _times(times)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0,2**63)")
    if scope not in ("rollout", "coupling", "both") or method not in ("hutchinson", "exact"):
        raise ValueError("Invalid scope or method")
    if scope != "coupling":
        _rollout_indices(times, rollout_steps)
    model, config, checkpoint, metadata = load_verified_model(config_path, dataset)
    parameter = next(model.parameters())
    dimension = config["data"]["n_points"] * config["model"]["point_dim"]
    if method == "exact" and dimension > max_exact_dim:
        raise ValueError("Exact Jacobian exceeds max_exact_dim; use --method hutchinson or explicitly raise --max-exact-dim")
    if scope != "rollout" and config["coupling"] not in OFFLINE_METHODS:
        raise ValueError("coupling scope requires target_guided_cached or nsot; use rollout for other methods")
    data = DiagnosticData(config, dataset, parameter.device, parameter.dtype, seed)
    synchronize(parameter.device)
    start = perf_counter()
    banks = {}
    if scope != "coupling":
        source = torch.stack([data.source(i, "jacobian_rollout").cpu() for i in range(clouds)])
        banks["rollout"] = (source, None, None)
    if scope != "rollout":
        banks["coupling"] = _cached_pairs(config, dataset, data, clouds)
    records, bank_metadata = [], {}
    for current_scope, (sources, targets, cache_details) in banks.items():
        bank_metadata[current_scope] = {"source_sha256": tensor_sha256(sources),
                                        "target_sha256": tensor_sha256(targets) if targets is not None else None,
                                        "cache_details": cache_details}
        for index, source in enumerate(sources):
            source = source.to(device=parameter.device, dtype=parameter.dtype)
            if current_scope == "rollout":
                states = rollout_snapshots(model, source, times, rollout_steps)
            else:
                target = targets[index].to(device=parameter.device, dtype=parameter.dtype)
                states = {time: (1 - time) * source + time * target for time in times}
            for time, state in states.items():
                probe_seed = data.draw_seed(f"jacobian_probe_v1:{current_scope}:{time.hex()}", index)
                generator = torch.Generator(device=parameter.device).manual_seed(probe_seed)
                values = cloud_jacobian(model, state, time, method=method, probes=probes,
                                        generator=generator, max_exact_dim=max_exact_dim)
                records.append({"scope": current_scope, "cloud_index": index, "time": time,
                                "state_sha256": tensor_sha256(state), "probe_seed": probe_seed, **values})
            print(f"jacobian_scope={current_scope} cloud={index + 1}/{clouds}", flush=True)
    synchronize(parameter.device)
    payload = {**diagnostic_metadata(config_path, config, checkpoint, metadata, dataset, parameter.device),
               "format_version": 1, "method": method, "probes": probes if method == "hutchinson" else None,
               "diagnostic_seed": seed, "clouds": clouds, "times": list(times), "scope": scope,
               "input_dimension": dimension, "jacobian_shape": [dimension, dimension],
               "rollout_steps": rollout_steps if scope != "coupling" else None,
               "attention_backend": "math SDPA; MHA/Transformer inference fastpath disabled during measurement",
               "audit_seconds": perf_counter() - start,
               "timing_scope": "bank sampling/cache loading, Euler rollouts, Jacobian VJPs and input hashes; excludes checkpoint loading and report output",
               "definitions": {
                   "jacobian": "d vec(v_theta(X,t)) / d vec(X), with fixed t; includes ALL within-cloud cross-point blocks",
                   "rollout": "fresh Gaussian source; finite Euler generated trajectory, not an exact ODE; no training cache required",
                   "coupling": "linear interpolation of fresh pairs from the checkpoint's fixed empirical training cache; NOT held-out pairs or a generated trajectory; stream caches sampled empirically, not in training order",
                   "estimator": "E_z ||J^T z||^2 = ||J||_F^2, iid Rademacher output probes; exact mode sums output-coordinate basis VJPs",
                   "frobenius": "sqrt(frobenius_squared); the Hutchinson squared estimate is unbiased, the square root generally is not",
                   "normalized_frobenius": "frobenius / sqrt(N*D); normalized sensitivity, NOT spectral norm or global Lipschitz constant",
                   "rms_frobenius": "sqrt(mean over clouds of frobenius_squared), distinct from mean of the square roots",
                   "std": "sample SD across clouds; includes MC variation; NOT training-seed uncertainty or MC standard error",
                   "mc_se_squared": "sample SD of probe squared VJP norms / sqrt(probes), conditional on each input; null for one probe, zero in exact mode",
                   "scope_warning": "Do not pool rollout and coupling distributions; low Jacobian norm alone does not establish good generation or a 1-Lipschitz model",
               },
               "banks": bank_metadata, "summary": _summary(records, dimension), "per_cloud": records}
    directory = output_directory(Path(output) / dataset, checkpoint.stem + "_jacobian")
    save_json(directory / "jacobian.json", payload)
    render(payload, directory)
    print(f"saved={directory}")
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description="Full-cloud spatial Jacobian Frobenius norm; no training or checkpoint edits.")
    parser.add_argument("config", help="Saved runs/.../config.yaml matching the checkpoint")
    parser.add_argument("--dataset", required=True, choices=("horse", "checkerboard"))
    parser.add_argument("--scope", choices=("rollout", "coupling", "both"), default="rollout")
    parser.add_argument("--clouds", type=int, default=16)
    parser.add_argument("--times", type=float, nargs="+", default=list(TIMES))
    parser.add_argument("--probes", type=int, default=8, help="VJPs per cloud/time; more probes reduce MC noise")
    parser.add_argument("--method", choices=("hutchinson", "exact"), default="hutchinson")
    parser.add_argument("--rollout-steps", type=int, default=128, help="Euler grid; rollout times must align with this grid")
    parser.add_argument("--max-exact-dim", type=int, default=128, help="Guard against accidentally expensive exact mode")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="analysis_results")
    args = vars(parser.parse_args(argv))
    return audit(args.pop("config"), args.pop("dataset"), **args)


if __name__ == "__main__":
    main()
