"""Read-only fitting, cloud-context and bank/fresh rollout diagnostics for MS-W2.

The existing bank permutation must already have been applied to ``target_paired``.
No target labels are invented for fresh Gaussian points. Fitting draws use the
training bank with replacement and are explicitly not a held-out-bank test.
"""

from contextlib import contextmanager
import hashlib

import numpy as np
import torch

from sample import integrate_velocity


def _banks(source, target_paired):
    source = np.asarray(source)
    target = np.asarray(target_paired)
    if (source.ndim != 2 or source.shape[1] != 2 or source.shape != target.shape
            or len(source) < 2 or not np.isfinite(source).all()
            or not np.isfinite(target).all()):
        raise ValueError("Expected finite source and paired target banks of shape (M, 2), M >= 2")
    return source, target


def _times(times, endpoints=False):
    result = [float(t) for t in times]
    if (not result or not np.isfinite(result).all() or len(set(result)) != len(result)
            or min(result) < 0 or max(result) > 1):
        raise ValueError("Times must be unique finite values in [0, 1]")
    return sorted(set(result + ([0., 1.] if endpoints else [])))


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _generator(seed, purpose, index):
    encoded = f"msw2_coupling_audit_v1:{int(seed)}:{purpose}:{int(index)}".encode()
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little") % (2**63 - 1)
    return torch.Generator(device="cpu").manual_seed(value)


def indexed_bank_draws(size, clouds, n_points, seed, purpose):
    """One isolated CPU RNG stream per cloud; microbatching cannot change draws."""
    return np.stack([torch.randint(size, (n_points,), generator=_generator(seed, purpose, i)).numpy()
                     for i in range(clouds)])


def _device_dtype(model, data=None):
    item = next(model.parameters(), None)
    if item is None:
        item = next(model.buffers(), None)
    if item is not None:
        return item.device, item.dtype if item.is_floating_point() else torch.float32
    return (getattr(data, "device", torch.device("cpu")),
            getattr(data, "dtype", torch.float32))


@contextmanager
def _evaluation(model, device):
    """Restore even mixed submodule train/eval modes and global torch RNG state."""
    modes = [(module, module.training) for module in model.modules()]
    device = torch.device(device)
    cuda = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=cuda), torch.no_grad():
            model.eval()
            yield
    finally:
        for module, training in modes:
            module.training = training


def _finite_prediction(value, expected_shape=None):
    if not isinstance(value, torch.Tensor):
        raise ValueError("Model prediction must be a Torch tensor")
    if expected_shape is not None and tuple(value.shape) != tuple(expected_shape):
        raise ValueError(f"Model prediction shape {tuple(value.shape)} does not match state shape {tuple(expected_shape)}")
    if not torch.isfinite(value).all():
        raise FloatingPointError("Model produced nonfinite values during the coupling diagnostic")
    return value


def _groups(point_features, size):
    result = {"all": np.ones(size, dtype=bool)}
    for name, values in point_features.items():
        values = np.asarray(values)
        if values.shape != (size,) or values.dtype != np.bool_:
            raise ValueError(f"Point group {name!r} must be a Boolean array of bank length")
        if name == "all" and not values.all():
            raise ValueError("The all group must contain every bank point")
        result[str(name)] = values
    return result


def _stratified_anchors(groups, count, seed):
    """Round-robin nonempty groups, then fill without replacement from all points."""
    size = len(groups["all"])
    count = min(count, size)
    order = []
    for name, mask in groups.items():
        if name == "all" or not mask.any():
            continue
        ids = np.flatnonzero(mask)
        permutation = torch.randperm(len(ids), generator=_generator(seed, "anchors:" + name, 0)).numpy()
        order.append(list(ids[permutation]))
    selected, used = [], set()
    while order and len(selected) < count:
        next_order = []
        for candidates in order:
            while candidates and int(candidates[-1]) in used:
                candidates.pop()
            if candidates and len(selected) < count:
                item = int(candidates.pop())
                selected.append(item)
                used.add(item)
            if candidates:
                next_order.append(candidates)
        order = next_order
    remaining = torch.randperm(size, generator=_generator(seed, "anchors:fill", 0)).tolist()
    selected.extend(i for i in remaining if i not in used)
    return np.asarray(selected[:count], dtype=np.int64)


def _error_stats(errors, energies):
    count = int(errors.size)
    if not count:
        return {"sample_count": 0, "mse_per_coordinate": None,
                "target_velocity_energy_per_coordinate": None,
                "normalized_mse": None, "point_mse_p95": None}
    denominator = float(energies.sum())
    return {"sample_count": count, "mse_per_coordinate": float(errors.mean()),
            "target_velocity_energy_per_coordinate": float(energies.mean()),
            "normalized_mse": float(errors.sum() / denominator) if denominator > 0 else None,
            "point_mse_p95": float(np.quantile(errors, .95))}


def fitting_audit(model, source, target_paired, point_features, *, n_points=256,
                  clouds=32, batch_size=8, times=(0, .25, .5, .75, .9, .95, .99),
                  seed=2026, anchors=64, contexts=8):
    """Return summary, CSV-table rows and NumPy arrays for diagnostic 3.

    Grouped residuals use pointwise coordinate-mean squared errors, pooled over
    fixed IID bank clouds. The normalization is a ratio of summed errors and
    target energies, never an average of unstable individual ratios.

    Context rows hold one bank pair fixed at slot 0 while the other N-1 points
    are independently redrawn from the paired bank. Variances are population
    variances (ddof=0), making residual = squared bias + variance exact.
    """
    source, target = _banks(source, target_paired)
    n_points = _positive_int(n_points, "n_points", 2)
    clouds = _positive_int(clouds, "clouds")
    batch_size = _positive_int(batch_size, "batch_size")
    anchors = _positive_int(anchors, "anchors")
    contexts = _positive_int(contexts, "contexts", 2)
    times = _times(times)
    groups = _groups(point_features, len(source))
    device, dtype = _device_dtype(model)
    indices = indexed_bank_draws(len(source), clouds, n_points, seed, "fitting")
    x0, y = source[indices], target[indices]
    velocity = y.astype(np.float64) - x0.astype(np.float64)
    energies = np.mean(velocity ** 2, axis=-1)
    errors = np.empty((len(times), clouds, n_points), dtype=np.float64)
    grouped_rows = []
    anchor_ids = _stratified_anchors(groups, anchors, seed)
    context_indices = np.stack([
        torch.randint(len(source), (contexts, n_points),
                      generator=_generator(seed, "anchor-context", int(i))).numpy()
        for i in anchor_ids])
    context_indices[:, :, 0] = anchor_ids[:, None]
    context_source, context_target = source[context_indices], target[context_indices]
    anchor_predictions = np.empty((len(times), len(anchor_ids), contexts, 2), dtype=np.float64)
    with _evaluation(model, device):
        for ti, time in enumerate(times):
            for begin in range(0, clouds, batch_size):
                sl = slice(begin, begin + batch_size)
                a = torch.as_tensor(x0[sl], device=device, dtype=dtype)
                b = torch.as_tensor(y[sl], device=device, dtype=dtype)
                state = (1 - time) * a + time * b
                t = torch.full((len(a), 1, 1), time, device=device, dtype=dtype)
                prediction = _finite_prediction(model(state, t), state.shape).cpu().double().numpy()
                errors[ti, sl] = np.mean((prediction - velocity[sl]) ** 2, axis=-1)
            for name, mask in groups.items():
                selected = mask[indices]
                grouped_rows.append({"time": time, "group": name,
                                     "bank_group_size": int(mask.sum()),
                                     "cloud_count": clouds,
                                     **_error_stats(errors[ti][selected], energies[selected])})
            flat_source = context_source.reshape(-1, n_points, 2)
            flat_target = context_target.reshape(-1, n_points, 2)
            flat_prediction = anchor_predictions[ti].reshape(-1, 2)
            for begin in range(0, len(flat_source), batch_size):
                sl = slice(begin, begin + batch_size)
                a = torch.as_tensor(flat_source[sl], device=device, dtype=dtype)
                b = torch.as_tensor(flat_target[sl], device=device, dtype=dtype)
                t = torch.full((len(a), 1, 1), time, device=device, dtype=dtype)
                state = (1 - time) * a + time * b
                prediction = _finite_prediction(model(state, t), state.shape)
                flat_prediction[sl] = prediction[:, 0].cpu().double().numpy()
            print(f"MS-W2 fitting: t={time:g}, grouped residuals and anchor contexts complete", flush=True)
    anchor_velocity = target[anchor_ids].astype(np.float64) - source[anchor_ids].astype(np.float64)
    means = anchor_predictions.mean(axis=2)
    residuals = np.mean((anchor_predictions - anchor_velocity[None, :, None, :]) ** 2, axis=(2, 3))
    biases = np.mean((means - anchor_velocity[None]) ** 2, axis=-1)
    variances = np.mean((anchor_predictions - means[:, :, None, :]) ** 2, axis=(2, 3))
    context_rows, context_group_rows = [], []
    other_anchor_fraction = (context_indices[:, :, 1:] == anchor_ids[:, None, None]).mean(axis=(1, 2))
    duplicate_fraction = np.asarray([
        [(n_points - len(np.unique(ids))) / n_points for ids in draws]
        for draws in context_indices])
    for ti, time in enumerate(times):
        for ai, bank_id in enumerate(anchor_ids):
            labels = [name for name, mask in groups.items() if mask[bank_id]]
            context_rows.append({"time": time, "bank_index": int(bank_id),
                                 "groups": "|".join(labels), "context_count": contexts,
                                 "sample_count": contexts,
                                 "other_slots_equal_anchor_fraction": float(other_anchor_fraction[ai]),
                                 "mean_duplicate_slot_fraction": float(duplicate_fraction[ai].mean()),
                                 "target_velocity_x": float(anchor_velocity[ai, 0]),
                                 "target_velocity_y": float(anchor_velocity[ai, 1]),
                                 "prediction_mean_x": float(means[ti, ai, 0]),
                                 "prediction_mean_y": float(means[ti, ai, 1]),
                                 "mse_per_coordinate": float(residuals[ti, ai]),
                                 "squared_bias_per_coordinate": float(biases[ti, ai]),
                                 "context_variance_per_coordinate": float(variances[ti, ai]),
                                 "decomposition_absolute_error": float(abs(
                                     residuals[ti, ai] - biases[ti, ai] - variances[ti, ai]))})
        for name, mask in groups.items():
            selected = mask[anchor_ids]
            context_group_rows.append({"time": time, "group": name,
                                       "anchor_count": int(selected.sum()), "contexts_per_anchor": contexts,
                                       "mse_per_coordinate": float(residuals[ti, selected].mean())
                                       if selected.any() else None,
                                       "squared_bias_per_coordinate": float(biases[ti, selected].mean())
                                       if selected.any() else None,
                                       "context_variance_per_coordinate": float(variances[ti, selected].mean())
                                       if selected.any() else None})
    summary = {"scope": "fitting residuals on fixed IID draws from the training pair bank; not held-out bank",
               "fresh_source_has_velocity_labels": False, "times": times,
               "clouds": clouds, "n_points": n_points, "batch_size": batch_size,
               "seed": int(seed), "bank_index_sampling": "IID with replacement, isolated CPU stream per cloud",
               "normalization": "sum coordinate-mean squared residuals / sum coordinate-mean target velocity energies; null if zero",
               "point_p95": "95th percentile of pointwise coordinate-mean squared residuals; points are not independent replicates",
               "overlapping_groups": True,
               "group_sizes": {name: int(mask.sum()) for name, mask in groups.items()},
               "grouped_errors": grouped_rows,
               "context_test": {"anchor_count_requested": anchors, "anchor_count": len(anchor_ids),
                                "contexts_per_anchor": contexts, "fixed_slot": 0,
                                "anchor_selection": "round-robin nonempty groups then fill, no replacement; descriptive stratified set",
                                "anchor_bank_indices": anchor_ids.tolist(),
                                "group_coverage": {name: int(mask[anchor_ids].sum()) for name, mask in groups.items()},
                                "variance_denominator": "population variance, ddof=0",
                                "mean_duplicate_slot_fraction": float(duplicate_fraction.mean()),
                                "other_slots_equal_anchor_fraction": float(other_anchor_fraction.mean()),
                                "duplicates_note": "Natural IID replacement duplicates retained, including the fixed anchor in other slots",
                                "grouped_decomposition": context_group_rows,
                                "max_decomposition_absolute_error": float(np.max(np.abs(residuals - biases - variances)))}}
    rows = {"fitting_groups": grouped_rows, "context_anchors": context_rows,
            "context_groups": context_group_rows}
    arrays = {"fitting_times": np.asarray(times), "fitting_bank_indices": indices,
              "fitting_point_mse": errors, "fitting_velocity_energy": energies,
              "context_anchor_bank_indices": anchor_ids, "context_bank_indices": context_indices,
              "context_anchor_predictions": anchor_predictions,
              "context_anchor_target_velocity": anchor_velocity}
    return summary, rows, arrays


def _snapshots(model, initial, steps, times, batch_size, device, dtype):
    """Sample the fixed-step Euler polygon at exact requested times.

    An off-grid snapshot uses the step's already-computed left velocity; querying
    a time never changes the trajectory or adds another model evaluation.
    """
    snapshots = np.empty((len(times), *initial.shape), dtype=np.float64)
    dt = 1. / steps
    for begin in range(0, len(initial), batch_size):
        end = min(begin + batch_size, len(initial))
        x = torch.as_tensor(initial[begin:end], device=device, dtype=dtype)
        query = 0
        if times[0] == 0:
            snapshots[0, begin:end] = x.cpu().double().numpy()
            query = 1
        for step in range(steps):
            left, right = step / steps, (step + 1) / steps
            t = torch.full((len(x), 1, 1), left, device=device, dtype=dtype)
            velocity = _finite_prediction(model(x, t), x.shape)
            next_x = _finite_prediction(x + dt * velocity)
            while query < len(times) and times[query] <= right + 1e-13:
                time = times[query]
                state = next_x if abs(time - right) <= 1e-13 else x + (time - left) * velocity
                snapshots[query, begin:end] = _finite_prediction(state).cpu().double().numpy()
                query += 1
            x = next_x
        if query != len(times):
            raise AssertionError("Some requested rollout times were not sampled")
    return snapshots


def _endpoint(model, initial, nfe, batch_size, device, dtype):
    class ValidatedVelocity(torch.nn.Module):
        def __init__(self, velocity_model):
            super().__init__()
            self.velocity_model = velocity_model

        def forward(self, state, time):
            return _finite_prediction(self.velocity_model(state, time), state.shape)

    # integrate_velocity owns the standard Euler implementation. Validation must
    # happen at every velocity query before an incorrectly shaped tensor could
    # broadcast to the state. The outer evaluation context restores all modes.
    validated = ValidatedVelocity(model).eval()
    output = []
    for begin in range(0, len(initial), batch_size):
        value = torch.as_tensor(initial[begin:begin + batch_size], device=device, dtype=dtype)
        output.append(_finite_prediction(integrate_velocity(validated, value, num_steps=nfe),
                                         value.shape).cpu().numpy())
    return np.concatenate(output)


def _tv(left, right):
    return float(.5 * np.abs(np.asarray(left) - np.asarray(right)).sum())


def _quality(quality_score, points, reference):
    if quality_score is None:
        return {}
    result = quality_score(torch.as_tensor(points, dtype=torch.float64),
                           torch.as_tensor(reference, dtype=torch.float64))
    return {str(key): (float(value) if value is not None and np.isfinite(value) else None)
            for key, value in result.items()}


def rollout_audit(model, source, target_paired, data, spatial, config, *, clouds=32,
                  batch_size=8, nfes=(1, 128), path_steps=128,
                  times=(0, .25, .5, .75, .9, .95, .99, 1), seed=2026,
                  quality_score=None):
    """Return diagnostic 4 without a repeated solver-convergence experiment.

    ``spatial.mass`` and ``spatial.grid_hist`` consume NumPy points; ``mass``
    returns one ALL-point-normalized fraction per named region. Grid histograms
    are normalized and include a square-exterior bin. ``quality_score`` is an
    optional two-argument callback consuming CPU Torch clouds.

    Endpoint bank and fresh results use the exact same independent fresh target
    references. Additional target-only and bank-target controls quantify finite
    samples. At intermediate times compare distributions against bank teacher
    interpolation; final-mask leakage is deliberately restricted to endpoints.
    """
    source, target = _banks(source, target_paired)
    clouds = _positive_int(clouds, "clouds")
    batch_size = _positive_int(batch_size, "batch_size")
    path_steps = _positive_int(path_steps, "path_steps")
    nfes = sorted(set(_positive_int(nfe, "nfe") for nfe in nfes))
    if not nfes:
        raise ValueError("At least one endpoint NFE is required")
    times = _times(times, endpoints=True)
    n_points = _positive_int(config["data"]["n_points"], "n_points", 2)
    if hasattr(data, "n") and data.n != n_points:
        raise ValueError("DiagnosticData point count differs from the checkpoint config")
    device, dtype = _device_dtype(model, data)
    indices = indexed_bank_draws(len(source), clouds, n_points, seed, "rollout-bank")
    bank_source, bank_target = source[indices], target[indices]
    fresh_source = torch.stack([data.source(i, "coupling-rollout") for i in range(clouds)]).detach().cpu().numpy()
    reference = torch.stack([data.target(i, "coupling-rollout-reference") for i in range(clouds)]).detach().cpu().numpy()
    control = torch.stack([data.target(i, "coupling-rollout-control") for i in range(clouds)]).detach().cpu().numpy()
    if fresh_source.shape != bank_source.shape or reference.shape != bank_target.shape or control.shape != bank_target.shape:
        raise ValueError("Diagnostic sampler clouds differ from configured (clouds, n_points, 2)")
    if not all(np.isfinite(value).all() for value in (fresh_source, reference, control)):
        raise ValueError("Diagnostic samplers returned nonfinite points")
    teacher = np.stack([(1 - time) * bank_source + time * bank_target for time in times])
    replicate_indices = indexed_bank_draws(len(source), clouds, n_points, seed,
                                          "rollout-independent-teacher")
    teacher_control = np.stack([(1 - time) * source[replicate_indices]
                                + time * target[replicate_indices] for time in times])
    predictions = {}
    with _evaluation(model, device):
        bank_path = _snapshots(model, bank_source, path_steps, times, batch_size, device, dtype)
        print(f"MS-W2 rollout: bank source path complete (NFE={path_steps})", flush=True)
        fresh_path = _snapshots(model, fresh_source, path_steps, times, batch_size, device, dtype)
        print(f"MS-W2 rollout: fresh Gaussian path complete (NFE={path_steps})", flush=True)
        for nfe in nfes:
            predictions["bank", nfe] = (bank_path[-1] if nfe == path_steps
                                        else _endpoint(model, bank_source, nfe, batch_size, device, dtype))
            predictions["fresh", nfe] = (fresh_path[-1] if nfe == path_steps
                                         else _endpoint(model, fresh_source, nfe, batch_size, device, dtype))
            print(f"MS-W2 rollout: bank and fresh endpoints complete (NFE={nfe})", flush=True)
    quality_rows, cloud_rows, mass_rows, path_rows = [], [], [], []
    expected = np.asarray(spatial.expected_mass, dtype=float)
    region_names = list(spatial.region_names)
    if expected.shape != (len(region_names),):
        raise ValueError("spatial.expected_mass must match region_names")
    for label, points in (("target_only", control), ("bank_target", bank_target)):
        quality_rows.append({"input": label, "nfe": None, "clouds": clouds,
                             **_quality(quality_score, points, reference)})
        mass = np.asarray(spatial.mass(points), dtype=float)
        for name, observed, goal in zip(region_names, mass, expected):
            mass_rows.append({"input": label, "nfe": None, "region": name,
                              "observed_mass": float(observed), "expected_mass": float(goal),
                              "signed_mass_error": float(observed - goal)})
    for (label, nfe), points in predictions.items():
        quality_rows.append({"input": label, "nfe": nfe, "clouds": clouds,
                             **_quality(quality_score, points, reference)})
        for i in range(clouds):
            cloud_rows.append({"input": label, "nfe": nfe, "cloud_index": i,
                               **_quality(quality_score, points[i:i+1], reference[i:i+1])})
        mass = np.asarray(spatial.mass(points), dtype=float)
        for name, observed, goal in zip(region_names, mass, expected):
            mass_rows.append({"input": label, "nfe": nfe, "region": name,
                              "observed_mass": float(observed), "expected_mass": float(goal),
                              "signed_mass_error": float(observed - goal)})
    arrays = {"rollout_times": np.asarray(times), "rollout_bank_indices": indices,
              "rollout_bank_source": bank_source, "rollout_fresh_source": fresh_source,
              "rollout_bank_target_control": bank_target, "rollout_target_reference": reference,
              "rollout_target_control": control, "rollout_teacher_path": teacher,
              "rollout_teacher_control_indices": replicate_indices,
              "rollout_teacher_control_path": teacher_control,
              "rollout_bank_path": bank_path, "rollout_fresh_path": fresh_path}
    for (label, nfe), points in predictions.items():
        arrays[f"rollout_{label}_prediction_nfe_{nfe}"] = points
    path_hist = getattr(spatial, "path_hist", spatial.grid_hist)
    histograms = {label: np.stack([path_hist(points) for points in path])
                  for label, path in (("teacher", teacher), ("teacher_control", teacher_control),
                                      ("bank", bank_path), ("fresh", fresh_path))}
    for label, histogram in histograms.items():
        arrays[f"rollout_{label}_path_histograms"] = histogram
    for ti, time in enumerate(times):
        path_rows.append({"time": time, "path_nfe": path_steps,
                          "bank_vs_teacher_histogram_tv": _tv(histograms["bank"][ti], histograms["teacher"][ti]),
                          "fresh_vs_teacher_histogram_tv": _tv(histograms["fresh"][ti], histograms["teacher"][ti]),
                          "fresh_vs_bank_histogram_tv": _tv(histograms["fresh"][ti], histograms["bank"][ti]),
                          "independent_teacher_histogram_tv": _tv(histograms["teacher_control"][ti], histograms["teacher"][ti]),
                          "bank_vs_paired_teacher_coordinate_mse": float(np.mean((bank_path[ti] - teacher[ti]) ** 2))})
    bank_final = np.asarray(spatial.mass(bank_path[-1]), dtype=float)
    fresh_final = np.asarray(spatial.mass(fresh_path[-1]), dtype=float)
    endpoint_differences = [{"region": name, "bank_mass": float(bank), "fresh_mass": float(fresh),
                             "fresh_minus_bank": float(fresh - bank), "expected_mass": float(goal)}
                            for name, bank, fresh, goal in zip(region_names, bank_final, fresh_final, expected)]
    summary = {"scope": "fixed checkpoint bank-source versus fresh-Gaussian rollout and teacher-distribution comparison",
               "seed": int(seed), "clouds": clouds, "n_points": n_points,
               "batch_size": batch_size, "nfes": nfes, "path_steps": path_steps, "times": times,
               "bank_index_sampling": "IID with replacement, same law as training",
               "fresh_target_reference": "same isolated fresh target references for all input types and NFEs",
               "quality_precision": "CPU float64 for both predictions and references",
               "finite_sample_path_control": "Independent same-sized IID bank-pair teacher interpolant at every requested t; descriptive sample fluctuation reference, not a significance test",
               "path_histogram_scope": "spatial.path_hist when provided, otherwise spatial.grid_hist; see report spatial definition for fixed range and bins",
               "snapshot_rule": "piecewise-linear fixed-step Euler path; exact query time, no added model evaluations",
               "path_comparison": path_rows, "endpoint_quality": quality_rows,
               "endpoint_region_mass": mass_rows, "path_endpoint_fresh_minus_bank": endpoint_differences,
               "interpretation": [
                   "Intermediate comparisons are distributional; final-support leakage is not evaluated at intermediate times.",
                   "Bank teacher interpolation is a training path distribution; matching each individual pair endpoint is not required by marginal flow matching.",
                   "Bank/fresh differences are descriptive and do not by themselves prove overfitting.",
                   "No paired target velocity labels are assigned to fresh Gaussian inputs.",
                   "The same independent target reference reduces comparison noise but is not a ground-truth target correspondence."]}
    rows = {"rollout_quality": quality_rows, "rollout_per_cloud": cloud_rows,
            "rollout_region_mass": mass_rows, "rollout_paths": path_rows,
            "rollout_fresh_minus_bank": endpoint_differences}
    return summary, rows, arrays
