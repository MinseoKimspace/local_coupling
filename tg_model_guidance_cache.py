"""Prepare self-contained fine-pairing pools using a frozen verified teacher.

Original v1 Hard-bank source coordinates, unordered target clouds and coarse
labels are copied bit-for-bit. A teacher is required only for preparation;
training and generation never load it. Scores are teacher-dependent proxies,
not ground-truth mean-field error or a global Lipschitz guarantee.
"""

import copy
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import torch

from nsot import dataset_spec, file_sha256
from tg_model_guidance import generate_candidates, score_candidates


COMPONENT_NAMES = ["regression", "local", "jacobian"]


def base_config(config):
    result = copy.deepcopy(config)
    cache = result["tg_cache"]
    cache["path"] = cache.pop("base_path")
    for name in ("model_guidance", "cache_sha256", "teacher_checkpoint_sha256"):
        cache.pop(name, None)
    return result


def _teacher_identity(config, dataset, options):
    """Resolve exactly the checkpoint selected by experiment.load_model."""
    from experiment import read_config, training_signature

    config_path = Path(options["teacher_config"]).resolve()
    teacher_config = read_config(config_path)
    if teacher_config.get("dtype") != "float32":
        raise ValueError("Model guidance requires a float32 teacher checkpoint")
    if teacher_config.get("model") != config.get("model"):
        raise ValueError("Teacher and derived model settings must match")
    if teacher_config["data"]["n_points"] != config["data"]["n_points"]:
        raise ValueError("Teacher and derived n_points must match")
    if dataset_spec(teacher_config, dataset) != dataset_spec(config, dataset):
        raise ValueError("Teacher and derived dataset/grid/mask settings must match")
    checkpoint = Path(teacher_config["checkpoint"])
    if not checkpoint.is_absolute():
        relative = config_path.parent / checkpoint
        checkpoint = relative if relative.exists() else Path.cwd() / checkpoint
    checkpoint = checkpoint.resolve()
    signature = training_signature(teacher_config)
    identity = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "config": copy.deepcopy(teacher_config),
        "training_signature": signature,
        "training_signature_sha256": hashlib.sha256(json.dumps(
            signature, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "dataset_spec": dataset_spec(teacher_config, dataset),
        "n_points": teacher_config["data"]["n_points"],
        "dtype": "float32",
    }
    expected = config["tg_cache"].get("teacher_checkpoint_sha256")
    if expected is not None and expected != identity["checkpoint_sha256"]:
        raise ValueError("Teacher checkpoint SHA256 differs from the bound preparation config")
    return identity


def _load_teacher(config, dataset, identity):
    from experiment import load_model
    from model import PointSetTransformer
    if dataset == "horse":
        from train_horse import HorsePointSetTransformer
        model_class = HorsePointSetTransformer
    else:
        model_class = PointSetTransformer
    # load_model initializes a model and resets seeds; restore both CPU and all
    # CUDA generators even if the checkpoint or verification subsequently fails.
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=cuda_devices):
        model, _, checkpoint, metadata = load_model(
            identity["config_path"], model_class, dataset, device=config["device"])
    if not metadata.get("training_config_verified") or metadata.get("format_version") != 2:
        raise ValueError("Model guidance requires a verified format-2 teacher checkpoint")
    if str(checkpoint) != identity["checkpoint_path"] \
            or file_sha256(checkpoint) != identity["checkpoint_sha256"]:
        raise ValueError("Teacher checkpoint changed while loading")
    identity.update(format_version=metadata["format_version"], training_config_verified=True,
                    training_seconds=metadata.get("training_seconds"),
                    training_environment=metadata.get("environment"))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _synchronize(model):
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _close_arrays(arrays, *, flush=False):
    try:
        if flush:
            for array in arrays.values():
                array.flush()
    finally:
        for array in arrays.values():
            array._mmap.close()
        arrays.clear()


def prepare_guided(config, dataset, opts):
    from tg_cache import (_array_shapes, _cloud_count, _fingerprint, _spec,
                          load_cache, prepare, settings)

    path = Path(opts["path"])
    options = opts["model_guidance"]
    start = perf_counter()
    identity = _teacher_identity(config, dataset, options)
    if path.exists():
        _, metadata, digest = load_cache(config, dataset)
        # Unlike training-time cache loading, re-preparation must verify that
        # the external teacher still means the same checkpoint and config.
        previous = metadata.get("teacher", {})
        for key in ("checkpoint_sha256", "training_signature", "dataset_spec", "n_points", "dtype"):
            if previous.get(key) != identity[key]:
                raise ValueError(f"Model-guidance teacher changed at existing cache path: {key}")
        print(f"reused_tg_model_guidance_cache={path} sha256={digest}", flush=True)
        return path, metadata

    model = _load_teacher(config, dataset, identity)
    teacher_load_seconds = perf_counter() - start
    original = base_config(config)
    parent_prepared_now = not Path(original["tg_cache"]["path"]).exists()
    parent_path, parent_meta = prepare(original, dataset)
    if parent_meta.get("format_version") != 1:
        raise ValueError("Model guidance requires an original format-1 Hard bank, not a derivative")
    if opts["sampling"] != "bank" or parent_meta.get("sampling") != "bank":
        raise ValueError("Model guidance supports Hard bank sampling only")
    parent_digest = _fingerprint(parent_meta)
    count = _cloud_count(config, opts)
    if parent_meta["num_clouds"] != count:
        raise ValueError("Model guidance requires the original Hard bank cloud count")
    shapes = _array_shapes(count, opts)
    base_shapes = _array_shapes(count, settings(original))
    storage = sum(int(np.prod(shape)) * np.dtype(dtype).itemsize for shape, dtype in shapes.values())
    ancestor = path.resolve().parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < storage + 128 * 2**20:
        raise OSError("Insufficient disk space for the derivative TG model-guidance cache")

    variant = "model_guided_pool" if options["selection"] == "score" else "model_guided_random_control"
    print(f"TG model guidance {variant} clouds={count} K={opts['num_regions']} N={opts['n_points']} "
          f"candidates={options['candidates']} keep={options['keep']} "
          f"teacher_sha256={identity['checkpoint_sha256']} storage_GiB={storage / 2**30:.3f}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    copy_start = perf_counter()
    for name in base_shapes:
        shutil.copyfile(parent_path / f"{name}.npy", path / f"{name}.npy")
    copy_seconds = perf_counter() - copy_start
    originals, outputs = {}, {}
    component_records, selected_component_records = [], []
    score_records, selected_score_records = [], []
    guidance_seconds = 0.0
    try:
        for name in base_shapes:
            originals[name] = np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in shapes.keys() - base_shapes.keys():
            shape, dtype = shapes[name]
            outputs[name] = np.lib.format.open_memmap(path / f"{name}.npy", mode="w+",
                                                    dtype=dtype, shape=shape)
        for index in range(count):
            _synchronize(model)
            tick = perf_counter()
            candidates = generate_candidates(originals["source_labels"][index],
                                             originals["target_labels"][index],
                                             opts["num_regions"], options, cloud_index=index)
            expected_shape = (options["candidates"], opts["n_points"])
            if candidates.shape != expected_shape or not np.issubdtype(candidates.dtype, np.integer) \
                    or not np.array_equal(np.sort(candidates, axis=1),
                                          np.broadcast_to(np.arange(opts["n_points"]), candidates.shape)) \
                    or not np.array_equal(originals["target_labels"][index][candidates],
                                          np.broadcast_to(originals["source_labels"][index], candidates.shape)):
                raise RuntimeError("Model guidance candidate is not a full patchwise bijection")
            record = score_candidates(model, originals["source"][index], originals["target"][index],
                                      candidates, options, cloud_index=index)
            _synchronize(model)
            guidance_seconds += perf_counter() - tick
            components = np.asarray(record["components"], dtype=np.float64)
            scores = np.asarray(record["scores"], dtype=np.float64)
            normalizers = np.asarray(record["normalizers"], dtype=np.float64)
            indices = np.asarray(record["selected_indices"])
            if components.shape != (options["candidates"], 3) or scores.shape != (options["candidates"],) \
                    or normalizers.shape != (3,) or not np.isfinite(components).all() \
                    or not np.isfinite(scores).all() or not np.isfinite(normalizers).all() \
                    or (components < 0).any() or (scores < 0).any() or (normalizers <= 0).any():
                raise RuntimeError("Model guidance returned invalid/nonfinite score components")
            if indices.shape != (options["keep"],) or not np.issubdtype(indices.dtype, np.integer) \
                    or len(np.unique(indices)) != options["keep"] \
                    or (indices < 0).any() or (indices >= options["candidates"]).any():
                raise RuntimeError("Model guidance selected indices are invalid")
            outputs["fine_permutations"][index] = candidates[indices]
            outputs["guidance_candidates"][index] = candidates
            outputs["guidance_components"][index] = components
            outputs["guidance_scores"][index] = scores
            outputs["guidance_normalizers"][index] = normalizers
            outputs["guidance_selected_indices"][index] = indices
            component_records.append(components.mean(axis=0))
            selected_component_records.append(components[indices].mean(axis=0))
            score_records.append(float(scores.mean()))
            selected_score_records.append(float(scores[indices].mean()))
            if (index + 1) % opts["prepare_batch_size"] == 0 or index + 1 == count:
                print(f"model_guidance_prepared={index + 1}/{count} "
                      f"candidate_score={scores.mean():.6f} selected_score={scores[indices].mean():.6f}",
                      flush=True)
    finally:
        try:
            _close_arrays(outputs, flush=True)
        finally:
            _close_arrays(originals)
            del model

    all_components = np.asarray(component_records)
    selected_components = np.asarray(selected_component_records)
    all_scores = np.asarray(score_records)
    selected_scores = np.asarray(selected_score_records)
    summary = {
        "objective": "frozen_teacher_normalized_regression_plus_local_velocity_difference_plus_fd_jacobian_proxy",
        "variant": variant, "selection": options["selection"],
        "clouds": count, "candidates_per_cloud": options["candidates"],
        "kept_permutations_per_cloud": options["keep"],
        "component_names": COMPONENT_NAMES, "times": options["times"],
        "component_definitions": {
            "regression": "mean_time_point_coordinate (v_teacher(Z_t,t)-U)^2",
            "local": "mean_time_source_edge_coordinate (residual_i-residual_j)^2; residual=v_teacher-U",
            "jacobian": "mean_time_probe_coordinate ((v_teacher(Z_t+epsilon*r,t)-"
                        "v_teacher(Z_t-epsilon*r,t))/(2*epsilon))^2; shared full-cloud Rademacher r",
        },
        "jacobian_interpretation": "finite-difference full-cloud Frobenius-energy estimate per input "
                                   "coordinate, NOT operator norm; includes cross-point/context derivatives",
        "neighbors": options["neighbors"], "probes": options["probes"],
        "fd_epsilon": options["fd_epsilon"], "weights": options["weights"],
        "normalization": options["normalization"],
        "candidate_component_means": dict(zip(COMPONENT_NAMES, all_components.mean(axis=0).tolist())),
        "selected_component_means": dict(zip(COMPONENT_NAMES, selected_components.mean(axis=0).tolist())),
        "candidate_component_cloud_sd": dict(zip(COMPONENT_NAMES, all_components.std(axis=0).tolist())),
        "selected_component_cloud_sd": dict(zip(COMPONENT_NAMES, selected_components.std(axis=0).tolist())),
        "candidate_score_mean": float(all_scores.mean()),
        "selected_score_mean": float(selected_scores.mean()),
        "candidate_score_cloud_sd": float(all_scores.std()),
        "selected_score_cloud_sd": float(selected_scores.std()),
        "relative_score_reduction": float((all_scores.mean() - selected_scores.mean()) / all_scores.mean())
                                    if all_scores.mean() > 0 else None,
        "search": "rank a finite matched random patchwise candidate set; no gradient optimization or swaps",
        "teacher_frozen": True, "one_lipschitz_guaranteed": False,
        "interpretation": "raw frozen-teacher regression residual, source-neighbor residual difference and "
                          "finite-difference teacher Jacobian proxy; NOT actual conditional mean-field "
                          "approximation error, literal crossing count, global Lipschitz guarantee or generation quality",
        "coupling_change": "teacher-dependent finite permutation pool replaces fresh uniform fine pairing",
        "training_initialization": "unchanged trainer; no automatic teacher weight transfer or fine-tuning",
    }
    hashes = {name: file_sha256(path / f"{name}.npy") for name in shapes}
    if any(hashes[name] != parent_meta["array_sha256"][name] for name in base_shapes):
        raise RuntimeError("Model-guidance derivative changed original Hard-bank arrays")
    report = {"dataset": dataset, "teacher": identity, "parent_cache_sha256": parent_digest,
              "summary": summary, "model_guidance_seconds": guidance_seconds,
              "component_array": "guidance_components.npy", "component_columns": COMPONENT_NAMES,
              "combined_score_array": "guidance_scores.npy", "normalizer_array": "guidance_normalizers.npy",
              "selected_indices_array": "guidance_selected_indices.npy"}
    with (path / "model_guidance_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    metadata = {
        **_spec(config, dataset, opts), "num_clouds": count, "array_sha256": hashes,
        "implementation": "tg_offline_model_guidance_v1", "fine_pairing_variant": variant,
        "local_pairing": variant, "teacher": identity, "teacher_load_seconds": teacher_load_seconds,
        "source_coordinates": parent_meta["source_coordinates"],
        "target_partition": parent_meta["target_partition"],
        "source_assignment": parent_meta["source_assignment"], "coarse_resampling": "cached labels",
        "fine_pairing": "uniform selection from a finite offline teacher-scored patchwise pool; "
                        "NOT fresh uniform permutations",
        "marginals": parent_meta["marginals"] + "; original unordered target clouds and capacities preserved",
        "inference_source": parent_meta["inference_source"],
        "parent_cache": {"sha256": parent_digest, "metadata": parent_meta},
        "parent_cache_path_at_preparation": str(parent_path.resolve()),
        "parent_cache_prepared_now": parent_prepared_now,
        "parent_precompute_seconds": parent_meta["precompute_seconds"],
        "model_guidance_summary": summary, "model_guidance_seconds": guidance_seconds,
        "original_array_copy_seconds": copy_seconds, "precompute_seconds": perf_counter() - start,
        "precompute_timing_scope": "this invocation: teacher verification/load, parent prepare/reuse "
                                   "validation, clone, CPU graph and synchronized teacher scoring, array IO/hashing "
                                   "and report; excludes final metadata write and the teacher's own training. "
                                   "Historical parent_precompute_seconds is separate; do NOT add it again if "
                                   "parent_cache_prepared_now is true",
        "teacher_training_cost_included": False,
        "storage_bytes": storage,
        "environment": {"torch": str(torch.__version__), "numpy": np.__version__,
                        "teacher_device": str(config["device"])},
        "source_sha256": {name: file_sha256(Path(__file__).parent / name) for name in
                          ("tg_cache.py", "tg_pairing.py", "tg_model_guidance.py", "tg_model_guidance_cache.py", "coupling.py",
                           "experiment.py", "model.py", "train_horse.py")},
    }
    # Metadata last: failures cannot masquerade as a completed cache or modify
    # the original bank. Incomplete derivative directories are never reused.
    with (path / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
    print(f"saved_tg_model_guidance_cache={path} sha256={_fingerprint(metadata)} "
          f"candidate_score={summary['candidate_score_mean']:.6f} "
          f"selected_score={summary['selected_score_mean']:.6f} "
          f"guidance_seconds={guidance_seconds:.3f}", flush=True)
    return path, metadata
