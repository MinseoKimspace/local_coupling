"""Prepare derivative Hard-bank caches using only supervision conflict.

Original Gaussian coordinates, target sets and coarse labels are copied
bit-for-bit from the original hard cache. Only fine bijections change.
Preparation is CPU-only; training samples a finite permutation pool with no
online optimization. Zero swaps is the matched random-pool control.
"""

import copy
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import torch

from nsot import file_sha256
from tg_untangle import optimize_permutations


def base_config(config):
    result = copy.deepcopy(config)
    cache = result["tg_cache"]
    cache["path"] = cache.pop("base_path")
    cache.pop("untangle", None)
    cache.pop("cache_sha256", None)
    return result


def prepare_guided(config, dataset, opts):
    from tg_cache import (_array_shapes, _cloud_count, _fingerprint, _spec,
                          load_cache, prepare, settings)

    path = Path(opts["path"])
    if path.exists():
        _, metadata, digest = load_cache(config, dataset)
        print(f"reused_tg_untangle_cache={path} sha256={digest}", flush=True)
        return path, metadata

    start = perf_counter()
    original = base_config(config)
    parent_prepared_now = not Path(original["tg_cache"]["path"]).exists()
    parent_path, parent_meta = prepare(original, dataset)
    parent_digest = _fingerprint(parent_meta)
    count = _cloud_count(config, opts)
    if parent_meta["num_clouds"] != count:
        raise ValueError("Untangle requires the same cloud count as its original hard cache")
    shapes = _array_shapes(count, opts)
    base_shapes = _array_shapes(count, settings(original))
    storage = sum(int(np.prod(shape)) * np.dtype(dtype).itemsize for shape, dtype in shapes.values())
    ancestor = path.resolve().parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < storage + 128 * 2**20:
        raise OSError("Insufficient disk space for the derivative TG cache")

    options = opts["untangle"]
    variant = "conflict_optimized_pool" if options["swap_steps"] else "random_pool_control"
    print(f"TG untangle {variant} clouds={count} K={opts['num_regions']} N={opts['n_points']} "
          f"pool={options['permutations']} swaps={options['swap_steps']} "
          f"storage_GiB={storage / 2**30:.3f}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)

    copy_start = perf_counter()
    for name in base_shapes:
        shutil.copyfile(parent_path / f"{name}.npy", path / f"{name}.npy")
    copy_seconds = perf_counter() - copy_start
    originals, outputs = {}, {}
    records = []
    optimization_seconds = 0.0
    edge_count = cross_patch_edge_count = proposed_swaps = accepted_swaps = 0
    try:
        originals = {name: np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                     for name in base_shapes}
        for name in shapes.keys() - base_shapes.keys():
            shape, dtype = shapes[name]
            outputs[name] = np.lib.format.open_memmap(path / f"{name}.npy", mode="w+",
                                                    dtype=dtype, shape=shape)
        for index in range(count):
            tick = perf_counter()
            pool, stats = optimize_permutations(
                originals["source"][index], originals["target"][index],
                originals["source_labels"][index], originals["target_labels"][index],
                opts["num_regions"], options, seed=index)
            optimization_seconds += perf_counter() - tick
            if (not np.array_equal(np.sort(pool, axis=1),
                                   np.broadcast_to(np.arange(opts["n_points"]), pool.shape))
                    or not np.array_equal(originals["target_labels"][index][pool],
                                          np.broadcast_to(originals["source_labels"][index], pool.shape))):
                raise RuntimeError("Untangle optimizer did not preserve a patchwise full bijection")
            rows = stats["per_permutation"]
            scores = np.asarray([[row["initial_score"], row["final_score"]] for row in rows])
            outputs["fine_permutations"][index] = pool
            outputs["untangle_scores"][index] = scores
            outputs["untangle_accepted_swaps"][index] = [row["accepted_swaps"] for row in rows]
            records.append(scores)
            edge_count += stats["edge_count"]
            cross_patch_edge_count += stats["cross_patch_edge_count"]
            proposed_swaps += stats["proposed_swaps"]
            accepted_swaps += stats["accepted_swaps"]
            if (index + 1) % opts["prepare_batch_size"] == 0 or index + 1 == count:
                print(f"untangle_prepared={index + 1}/{count} "
                      f"conflict_before={scores[:, 0].mean():.6f} "
                      f"after={scores[:, 1].mean():.6f}", flush=True)
    finally:
        for array in outputs.values():
            array.flush()
            array._mmap.close()
        for array in originals.values():
            array._mmap.close()
        outputs.clear()
        originals.clear()

    scores = np.stack(records)
    before, after = scores[:, :, 0], scores[:, :, 1]
    mean_before, mean_after = float(before.mean()), float(after.mean())
    summary = {
        "objective": "mean_edge_time_squared_positive_supervision_lipschitz_excess",
        "definition": "mean_(t,edge) max(||delta U|| - L(t)||delta Z_t||, 0)^2",
        "graph": "fixed_undirected_source_knn_union_including_cross_patch_edges",
        "neighbors": options["neighbors"], "times": options["times"], "lipschitz": options["lipschitz"],
        "variant": variant, "clouds": count, "permutations_per_cloud": options["permutations"],
        "initial_score_mean": mean_before, "final_score_mean": mean_after,
        "initial_score_cloud_sd": float(before.mean(axis=1).std()),
        "final_score_cloud_sd": float(after.mean(axis=1).std()),
        "relative_score_reduction": (mean_before - mean_after) / mean_before if mean_before > 0 else None,
        "improved_permutations": int(np.count_nonzero(after < before)),
        "proposed_swaps": proposed_swaps, "accepted_swaps": accepted_swaps,
        "mean_edges_per_cloud": edge_count / count,
        "cross_patch_edge_fraction": cross_patch_edge_count / edge_count if edge_count else 0.0,
        "search": "bounded_greedy_same_patch_swaps; strictly improving fixed-objective moves only",
        "one_lipschitz_guaranteed": False,
        "interpretation": "sample supervision conflict proxy, NOT mean-field approximation error, "
                          "a global regression-error bound, literal crossing count, or generation quality",
    }
    hashes = {name: file_sha256(path / f"{name}.npy") for name in shapes}
    if any(hashes[name] != parent_meta["array_sha256"][name] for name in base_shapes):
        raise RuntimeError("Derivative cache changed original hard-cache arrays")

    report = {"dataset": dataset, "parent_cache_sha256": parent_digest,
              "summary": summary, "untangle_optimization_seconds": optimization_seconds,
              "per_cloud_score_file": "untangle_scores.npy", "score_columns": ["initial", "final"]}
    with (path / "untangle_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)

    metadata = {
        **_spec(config, dataset, opts), "num_clouds": count, "array_sha256": hashes,
        "implementation": "tg_offline_untangle_v1", "fine_pairing_variant": variant,
        "local_pairing": variant,
        "source_coordinates": parent_meta["source_coordinates"],
        "target_partition": parent_meta["target_partition"],
        "source_assignment": parent_meta["source_assignment"], "coarse_resampling": "cached labels",
        "fine_pairing": "uniform selection from a finite offline patchwise permutation pool; "
                        "NOT fresh uniform permutations",
        "marginals": parent_meta["marginals"] + "; original unordered target clouds and capacities preserved",
        "inference_source": parent_meta["inference_source"],
        "parent_cache": {"sha256": parent_digest, "metadata": parent_meta},
        "parent_cache_path_at_preparation": str(parent_path.resolve()),
        "parent_cache_prepared_now": parent_prepared_now,
        "parent_precompute_seconds": parent_meta["precompute_seconds"],
        "untangle_summary": summary, "untangle_optimization_seconds": optimization_seconds,
        "original_array_copy_seconds": copy_seconds, "precompute_seconds": perf_counter() - start,
        "precompute_timing_scope": "this invocation: parent prepare/reuse validation, clone, conflict search, "
                                   "array IO/hashing and report; excludes final metadata write. Historical "
                                   "parent_precompute_seconds is separate; do NOT add it again if "
                                   "parent_cache_prepared_now is true",
        "storage_bytes": storage,
        "environment": {"torch": str(torch.__version__), "numpy": np.__version__},
        "source_sha256": {name: file_sha256(Path(__file__).parent / name) for name in
                          ("tg_cache.py", "tg_untangle.py", "tg_untangle_cache.py", "coupling.py")},
    }
    # Metadata is written last. Failed preparation leaves an incomplete NEW
    # derivative, never an apparently valid cache and never modified originals.
    with (path / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
    print(f"saved_tg_untangle_cache={path} sha256={_fingerprint(metadata)} "
          f"conflict_before={mean_before:.6f} after={mean_after:.6f} "
          f"optimization_seconds={optimization_seconds:.3f}", flush=True)
    return path, metadata
