"""Offline, frozen-teacher ranking of valid within-patch permutations.

The teacher never learns during selection. Regression and local residual
scores depend on the current checkpoint, not on the unknown exact mean field.
The central finite-difference term estimates full-cloud Jacobian Frobenius
energy per input coordinate, not its operator norm or a Lipschitz bound.
Candidates preserve every coordinate and the patchwise bijection. Ranking
changes the finite training coupling law; it is not an inference-time layer.
"""

from collections.abc import Mapping

import numpy as np
import torch

from tg_pairing import build_source_edges, random_valid_permutation


COMPONENTS = ("regression", "local", "jacobian")
DEFAULTS = {
    "teacher_config": None,
    "neighbors": 8,
    "times": [0.25, 0.5, 0.75],
    "candidates": 8,
    "keep": 4,
    "probes": 1,
    "fd_epsilon": 1e-3,
    "weights": {"regression": 1.0, "local": 0.25, "jacobian": 0.1},
    "normalization": "candidate_mean",
    "inference_batch_size": 16,
    "selection": "score",
    "seed": 0,
}


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not minimum <= int(value) < 2**63):
        raise ValueError(f"{name} must be an integer >= {minimum} and < 2**63")
    return int(value)


def _real(value, name, minimum=0.0, strictly_positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if (not np.isfinite(number) or number < minimum
            or (strictly_positive and number == 0.0)):
        relation = ">" if strictly_positive else ">="
        raise ValueError(f"{name} must be finite and {relation} {minimum}")
    return number


def normalize_options(value):
    """Validate and copy JSON-serializable options without changing the input.

    Missing/null/False disables guidance. A mapping enables it by default.
    Enabled guidance requires an explicit teacher config; True alone therefore
    cannot select an implicit checkpoint. Disabled options still validate all
    supplied numeric settings. Partial weight mappings retain other defaults.
    """
    if value is None or value is False:
        supplied = {"enabled": False}
    elif value is True:
        supplied = {"enabled": True}
    elif isinstance(value, Mapping):
        supplied = dict(value)
    else:
        raise ValueError("tg_cache.model_guidance must be null, boolean, or a mapping")
    unknown = set(supplied) - ({"enabled"} | set(DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown model_guidance options: {', '.join(sorted(map(str, unknown)))}")
    enabled = supplied.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("model_guidance.enabled must be a boolean")
    opts = {**DEFAULTS, **supplied, "enabled": enabled}
    teacher_config = opts["teacher_config"]
    if teacher_config is not None and not isinstance(teacher_config, str):
        raise ValueError("model_guidance.teacher_config must be a string or null")
    if isinstance(teacher_config, str):
        teacher_config = teacher_config.strip()
        if not teacher_config:
            raise ValueError("model_guidance.teacher_config must be nonempty")
    if enabled and teacher_config is None:
        raise ValueError("Enabled model_guidance requires teacher_config")
    opts["teacher_config"] = teacher_config
    for key in ("neighbors", "candidates", "keep", "probes", "inference_batch_size"):
        opts[key] = _integer(opts[key], f"model_guidance.{key}", 1)
    opts["seed"] = _integer(opts["seed"], "model_guidance.seed")
    if opts["keep"] > opts["candidates"]:
        raise ValueError("model_guidance.keep must not exceed candidates")
    opts["fd_epsilon"] = _real(opts["fd_epsilon"], "model_guidance.fd_epsilon", strictly_positive=True)
    raw_times = opts["times"]
    if (not isinstance(raw_times, (list, tuple, np.ndarray))
            or (isinstance(raw_times, np.ndarray) and raw_times.ndim != 1)
            or len(raw_times) == 0):
        raise ValueError("model_guidance.times must be a nonempty sequence")
    times = [_real(item, "model_guidance.times") for item in raw_times]
    if any(not 0.0 < time < 1.0 for time in times) or len(set(times)) != len(times):
        raise ValueError("model_guidance.times must be distinct and strictly between 0 and 1")
    opts["times"] = times
    raw_weights = opts["weights"]
    if not isinstance(raw_weights, Mapping) or set(raw_weights) - set(COMPONENTS):
        raise ValueError("model_guidance.weights must contain only regression, local, jacobian")
    weights = {**DEFAULTS["weights"], **raw_weights}
    opts["weights"] = {name: _real(weights[name], f"model_guidance.weights.{name}")
                       for name in COMPONENTS}
    if not any(opts["weights"].values()):
        raise ValueError("model_guidance.weights must include at least one positive weight")
    if opts["normalization"] != "candidate_mean":
        raise ValueError("model_guidance.normalization must be candidate_mean")
    if opts["selection"] not in ("score", "random"):
        raise ValueError("model_guidance.selection must be score or random")
    return opts


def _enabled_options(value):
    opts = normalize_options(value)
    if not opts["enabled"]:
        raise ValueError("Candidate guidance requires model_guidance.enabled: true")
    return opts


def _rng(options, cloud_index, stream, *indices):
    # Disjoint candidate/probe/selection streams avoid global RNG changes and
    # keep candidate prefixes and directional probes independent of chunking.
    return np.random.default_rng(np.random.SeedSequence(
        [options["seed"], cloud_index, stream, *indices]))


def generate_candidates(source_labels, target_labels, k, options, cloud_index=0):
    """Uniform patchwise permutations, int32[C,N], deterministic per cloud.

    Changing selection or keep leaves all candidates unchanged. Increasing C
    extends the same candidate prefix rather than resampling it.
    """
    opts = _enabled_options(options)
    cloud_index = _integer(cloud_index, "cloud_index")
    source_labels = np.asarray(source_labels)
    if source_labels.ndim != 1 or len(source_labels) >= 2**31:
        raise ValueError("Candidate pools require one-dimensional labels and N < 2**31")
    candidates = [random_valid_permutation(source_labels, target_labels, k,
                                           _rng(opts, cloud_index, 0, index))
                  for index in range(opts["candidates"])]
    return np.stack(candidates).astype(np.int32, copy=False)


def _points(value, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind == "c":
            raise ValueError("Complex coordinates are not supported")
        result = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite N x D array") from error
    if result.ndim != 2 or min(result.shape) < 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty N x D array")
    return result


def _permutations(value, n):
    raw = np.asarray(value)
    if (raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] != n
            or raw.dtype.kind not in "iu" or np.any(raw < 0) or np.any(raw >= n)):
        raise ValueError("permutations must be an integer nonempty C x N array of bijections")
    expected = np.arange(n)
    if not np.all(np.sort(raw, axis=1) == expected[None, :]):
        raise ValueError("Each permutation must be a bijection of indices 0 through N-1")
    return raw.astype(np.int64, copy=False)


def _teacher_placement(model):
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Teacher must be a torch.nn.Module")
    if any(module.training for module in model.modules()):
        raise ValueError("Teacher and all its modules must already be in eval mode")
    parameters = list(model.parameters())
    if any(parameter.requires_grad for parameter in parameters):
        raise ValueError("Teacher parameters must already be frozen")
    tensors = parameters + list(model.buffers())
    floating = [tensor for tensor in tensors if tensor.is_floating_point()]
    if floating:
        reference = floating[0]
        if any(tensor.device != reference.device for tensor in floating):
            raise ValueError("Teacher floating parameters/buffers must share one device")
        return reference.device, reference.dtype
    return torch.device("cpu"), torch.float32


def _predict(model, positions, time, device, dtype):
    tensor = torch.as_tensor(positions, dtype=dtype, device=device)
    if not torch.isfinite(tensor).all().item():
        raise ValueError("Coordinates exceed the teacher's supported numerical range")
    times = torch.full((len(positions), 1, 1), time, dtype=dtype, device=device)
    output = model(tensor, times)
    if (not isinstance(output, torch.Tensor) or output.shape != tensor.shape
            or output.is_complex()):
        raise ValueError("Teacher must return a real velocity tensor with the input shape")
    output = output.detach().to(device="cpu", dtype=torch.float64).numpy()
    if not np.isfinite(output).all():
        raise ValueError("Teacher returned nonfinite velocities")
    return output


def score_candidates(model, source, target, permutations, options, *, cloud_index=0):
    """Score all candidates before score-based or matched random selection.

    Components are time averages in float64. Regression is per-coordinate MSE;
    local is edge/coordinate MSE of teacher-minus-supervision velocity
    differences on the fixed undirected source kNN union (cross-patch edges
    included). The Jacobian proxy uses the same full-cloud Rademacher probes
    for every candidate at each time. With E[r r^T]=I, its small-epsilon
    expectation is ||J||_F^2/(N D), including cross-point/context derivatives.

    Active components are divided by their candidate mean with a 1e-12 floor.
    Zero-weight components are not evaluated and report zeros. Stable score
    ranking keeps the first original candidate in ties. Random control computes
    the identical components/probes and changes only the final selection.
    This function never enables gradients or changes teacher mode/parameters;
    Torch and NumPy global random streams are preserved.
    """
    opts = _enabled_options(options)
    cloud_index = _integer(cloud_index, "cloud_index")
    source, target = _points(source, "source"), _points(target, "target")
    if source.shape != target.shape:
        raise ValueError("source and target must have the same N x D shape")
    permutations = _permutations(permutations, len(source))
    count = len(permutations)
    if opts["keep"] > count:
        raise ValueError("model_guidance.keep must not exceed the supplied candidate count")
    edges = build_source_edges(source, opts["neighbors"])
    left, right = edges.T
    device, dtype = _teacher_placement(model)
    weights = np.array([opts["weights"][name] for name in COMPONENTS], dtype=np.float64)
    components = np.zeros((count, len(COMPONENTS)), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        velocities = target[permutations] - source[None, :, :]
    if not np.isfinite(velocities).all():
        raise ValueError("Velocity coordinates exceed the supported numerical range")
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    # Standard deterministic eval teachers consume no RNG, but preserve Torch
    # state even if a custom frozen eval module performs random operations.
    with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
        for time_index, time in enumerate(opts["times"]):
            probes = []
            if weights[2] > 0.0:
                probes = [2.0 * _rng(opts, cloud_index, 1, time_index, probe_index).integers(
                    0, 2, size=source.shape).astype(np.float64) - 1.0
                    for probe_index in range(opts["probes"])]
            for start in range(0, count, opts["inference_batch_size"]):
                stop = min(start + opts["inference_batch_size"], count)
                velocity = velocities[start:stop]
                with np.errstate(over="ignore", invalid="ignore"):
                    position = source[None, :, :] + time * velocity
                if not np.isfinite(position).all():
                    raise ValueError("Interpolated coordinates exceed the supported numerical range")
                if weights[0] > 0.0 or weights[1] > 0.0:
                    predicted = _predict(model, position, time, device, dtype)
                    residual = predicted - velocity
                    if weights[0] > 0.0:
                        components[start:stop, 0] += np.mean(residual ** 2, axis=(1, 2))
                    if weights[1] > 0.0:
                        edge_residual = residual[:, left, :] - residual[:, right, :]
                        components[start:stop, 1] += np.mean(edge_residual ** 2, axis=(1, 2))
                if weights[2] > 0.0:
                    for direction in probes:
                        plus = _predict(model, position + opts["fd_epsilon"] * direction[None, :, :],
                                        time, device, dtype)
                        minus = _predict(model, position - opts["fd_epsilon"] * direction[None, :, :],
                                         time, device, dtype)
                        derivative = (plus - minus) / (2.0 * opts["fd_epsilon"])
                        components[start:stop, 2] += np.mean(derivative ** 2, axis=(1, 2)) / opts["probes"]
    components /= len(opts["times"])
    if not np.isfinite(components).all():
        raise ValueError("Guidance components exceed the supported numerical range")
    normalizers = np.ones(len(COMPONENTS), dtype=np.float64)
    active = weights > 0.0
    normalizers[active] = np.maximum(components[:, active].mean(axis=0), 1e-12)
    scores = np.sum(weights[None, :] * components / normalizers[None, :], axis=1)
    if not np.isfinite(normalizers).all() or not np.isfinite(scores).all():
        raise ValueError("Guidance scores exceed the supported numerical range")
    if opts["selection"] == "score":
        selected = np.argsort(scores, kind="stable")[:opts["keep"]]
    else:
        selected = _rng(opts, cloud_index, 2).choice(count, size=opts["keep"], replace=False)
    return {"components": components, "scores": scores, "normalizers": normalizers,
            "selected_indices": selected.astype(np.int32), "edge_count": int(len(edges))}
