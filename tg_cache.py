"""Offline 2D hard TG, with fresh random within-patch bijections.

bank: reuse a finite paired-cloud bank with cached exact coarse assignments.
stream: prepare the complete training stream; each cloud is used once.
Neither mode requires online OT. By default fine pairing is resampled on every
visit. An opt-in untangle experiment samples a separately prepared finite
permutation pool, without changing the original bank or source coordinates.
Inference always starts from fresh iid standard Gaussian, without the cache.
"""

import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

from coupling import assign_regions, balanced_target_partition
from nsot import dataset_spec, file_sha256
from tg_untangle import normalize_options


METHODS = {"target_guided_cached"}
FORMAT_VERSION = 1
UNTANGLE_FORMAT_VERSION = 2


def random_fine_permutation(source_labels, target_labels, k, rng):
    """Uniform random bijections per patch; source indices are unchanged."""
    permutation = np.empty(len(source_labels), dtype=np.int64)
    for patch in range(k):
        source = np.flatnonzero(source_labels == patch)
        target = np.flatnonzero(target_labels == patch)
        if len(source) != len(target):
            raise ValueError("Source and target patch counts differ")
        permutation[source] = rng.permutation(target)
    return permutation


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value < 2**63:
        raise ValueError(f"{name} must be an integer >= {minimum} and < 2**63")
    return value


def settings(config):
    method = config.get("coupling")
    if method not in METHODS:
        raise ValueError("TG cache requires target_guided_cached")
    value = config.get("tg_cache", {})
    n = _integer(config["data"]["n_points"], "n_points")
    _integer(config["data"]["batch_size"], "batch_size")
    _integer(config["training"]["num_steps"], "num_steps")
    k = _integer(config.get("num_regions"), "num_regions")
    if k > n or config["model"]["point_dim"] != 2 or config.get("dtype") != "float32":
        raise ValueError("TG cache supports float32 2D with 1 <= K <= N")
    sampling = value.get("sampling", "bank")
    if sampling not in ("bank", "stream"):
        raise ValueError("tg_cache.sampling must be bank or stream")
    count = value.get("num_clouds")
    if count is not None:
        count = _integer(count, "tg_cache.num_clouds")
    elif sampling != "stream":
        raise ValueError("bank sampling requires tg_cache.num_clouds")
    if not isinstance(value.get("path"), str) or not value["path"]:
        raise ValueError("tg_cache.path must name a cache directory")
    workers = _integer(value.get("num_workers", 0), "tg_cache.num_workers", 0)
    result = {"method": method, "path": value["path"], "sampling": sampling,
              "num_clouds": count, "cache_seed": _integer(value.get("seed", 0), "tg_cache.seed", 0),
              "prepare_batch_size": _integer(value.get("prepare_batch_size", 64), "prepare_batch_size"),
              "num_workers": workers, "n_points": n, "num_regions": k}
    untangle = normalize_options(value.get("untangle"))
    if untangle["enabled"]:
        if untangle["neighbors"] >= n:
            raise ValueError("untangle.neighbors must be smaller than n_points")
        if not isinstance(value.get("base_path"), str) or not value["base_path"]:
            raise ValueError("Untangle requires tg_cache.base_path naming the original hard cache")
        if Path(value["base_path"]).resolve() == Path(value["path"]).resolve():
            raise ValueError("Untangle must use a NEW cache path, different from base_path")
        result.update(untangle=untangle, base_path=value["base_path"])
    return result


def _spec(config, dataset, opts):
    result = {"format_version": FORMAT_VERSION, "method": opts["method"],
            "sampling": opts["sampling"], "configured_num_clouds": opts["num_clouds"],
            "cache_seed": opts["cache_seed"], "prepare_batch_size": opts["prepare_batch_size"],
            "n_points": opts["n_points"], "num_regions": opts["num_regions"],
            "dataset_spec": dataset_spec(config, dataset)}
    if "untangle" in opts:
        result.update(format_version=UNTANGLE_FORMAT_VERSION, untangle=opts["untangle"])
    return result


def _cloud_count(config, opts):
    count = opts["num_clouds"]
    required = config["data"]["batch_size"] * config["training"]["num_steps"]
    if count is None:
        count = required
    if opts["sampling"] == "stream" and count < required:
        raise ValueError("stream needs num_clouds >= batch_size * num_steps (or num_clouds: null)")
    return count


def _array_shapes(count, opts):
    n, k = opts["n_points"], opts["num_regions"]
    shapes = {"source": ((count, n, 2), np.float32), "target": ((count, n, 2), np.float32),
              "target_labels": ((count, n), np.int32), "capacities": ((count, k), np.int32)}
    shapes["source_labels"] = ((count, n), np.int32)
    if "untangle" in opts:
        p = opts["untangle"]["permutations"]
        shapes.update(fine_permutations=((count, p, n), np.int32),
                      untangle_scores=((count, p, 2), np.float64),
                      untangle_accepted_swaps=((count, p), np.int32))
    return shapes


def _fingerprint(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def load_cache(config, dataset):
    opts = settings(config)
    path = Path(opts["path"])
    if not (path / "metadata.json").is_file():
        raise FileNotFoundError(f"TG cache missing/incomplete: {path}; run python prepare_tg.py CONFIG --dataset {dataset}")
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    for key, value in _spec(config, dataset, opts).items():
        if metadata.get(key) != value:
            raise ValueError(f"TG cache/config mismatch: {key}")
    count = _integer(metadata.get("num_clouds"), "cached num_clouds")
    if opts["num_clouds"] is not None and count != opts["num_clouds"]:
        raise ValueError("TG cache cloud count mismatch")
    if opts["sampling"] == "stream" and count < config["data"]["batch_size"] * config["training"]["num_steps"]:
        raise ValueError("Training steps exceed the prepared TG stream; prepare a new, longer cache")
    shapes = _array_shapes(count, opts)
    if set(metadata.get("array_sha256", {})) != set(shapes):
        raise ValueError("TG cache has missing/extra arrays")
    for name, (shape, dtype) in shapes.items():
        file = path / f"{name}.npy"
        if file_sha256(file) != metadata["array_sha256"][name]:
            raise ValueError(f"TG cache array SHA256 mismatch: {name}")
        array = np.load(file, mmap_mode="r", allow_pickle=False)
        try:
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"TG cache shape/dtype mismatch: {name}")
        finally:
            array._mmap.close()
    if "untangle" in opts:
        parent = metadata.get("parent_cache", {})
        parent_meta = parent.get("metadata", {})
        if parent.get("sha256") != _fingerprint(parent_meta):
            raise ValueError("Untangle parent cache fingerprint mismatch")
        base_opts = {k: v for k, v in opts.items() if k not in ("untangle", "base_path")}
        for key, value in _spec(config, dataset, base_opts).items():
            if parent_meta.get(key) != value:
                raise ValueError(f"Untangle parent cache/config mismatch: {key}")
        if parent_meta.get("num_clouds") != count:
            raise ValueError("Untangle parent cache cloud count mismatch")
        for name in _array_shapes(count, base_opts):
            if parent_meta.get("array_sha256", {}).get(name) != metadata["array_sha256"][name]:
                raise ValueError(f"Untangle changed original cached array: {name}")
    digest = _fingerprint(metadata)
    if config["tg_cache"].get("cache_sha256", digest) != digest:
        raise ValueError("TG cache differs from the checkpoint training cache")
    return path, metadata, digest


def prepare(config, dataset):
    """Exclusive, memory-mapped cache creation; existing caches are never overwritten."""
    opts = settings(config)
    if "untangle" in opts:
        from tg_untangle_cache import prepare_guided
        return prepare_guided(config, dataset, opts)
    spec = _spec(config, dataset, opts)
    path = Path(opts["path"])
    if path.exists():
        _, metadata, digest = load_cache(config, dataset)
        print(f"reused_tg_cache={path} sha256={digest}", flush=True)
        return path, metadata
    count = _cloud_count(config, opts)
    shapes = _array_shapes(count, opts)
    storage = sum(int(np.prod(shape)) * np.dtype(dtype).itemsize for shape, dtype in shapes.values())
    print(f"TG prepare {opts['method']} {opts['sampling']} clouds={count} K={opts['num_regions']} "
          f"N={opts['n_points']} storage_GiB={storage / 2**30:.3f}", flush=True)
    ancestor = path.resolve().parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < storage + 128 * 2**20:
        raise OSError("Insufficient disk space for this TG cache; choose another drive/path or a smaller bank")
    start = perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    arrays = {name: np.lib.format.open_memmap(path / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
              for name, (shape, dtype) in shapes.items()}
    draw_seconds = partition_seconds = assignment_seconds = 0.
    threads = torch.get_num_threads()
    try:
        # Small CPU N x K solves; avoid large thread-pool overhead. Restore the
        # caller's thread count/RNG, and never touch its CUDA RNG.
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(opts["cache_seed"]).get_state())
            if dataset == "horse":
                from train_horse import load_horse_mask, sample_horse
                mask = load_horse_mask("cpu", torch.float32)
            else:
                from data import sample_checkerboard
            for begin in range(0, count, opts["prepare_batch_size"]):
                end = min(begin + opts["prepare_batch_size"], count)
                tick = perf_counter()
                source = torch.randn(end - begin, opts["n_points"], 2, device="cpu", dtype=torch.float32)
                target = (sample_horse(mask, end - begin, opts["n_points"]) if dataset == "horse"
                          else sample_checkerboard(end - begin, opts["n_points"], "cpu", torch.float32,
                                                   config["data"]["grid_size"]))
                draw_seconds += perf_counter() - tick
                tick = perf_counter()
                _, labels, centers, capacities = balanced_target_partition(
                    target, opts["num_regions"], solver="exact_batched")
                partition_seconds += perf_counter() - tick
                arrays["source"][begin:end], arrays["target"][begin:end] = source.numpy(), target.numpy()
                arrays["target_labels"][begin:end] = labels.numpy()
                arrays["capacities"][begin:end] = capacities.numpy()
                tick = perf_counter()
                arrays["source_labels"][begin:end] = assign_regions(
                    source, centers, capacities, solver="exact_batched").numpy()
                assignment_seconds += perf_counter() - tick
                print(f"tg_prepared={end}/{count}", flush=True)
    finally:
        torch.set_num_threads(threads)
        for array in arrays.values():
            array.flush()
            array._mmap.close()
        arrays.clear()
    hashes = {name: file_sha256(path / f"{name}.npy") for name in shapes}
    metadata = {**spec, "num_clouds": count, "array_sha256": hashes,
                "implementation": "tg_offline_v1", "source_coordinates": "unmodified iid standard Gaussian draws",
                "target_partition": "balanced FPS, exact POT transportation; actual patch centroids",
                "source_assignment": "exact POT",
                "coarse_resampling": "cached labels",
                "fine_pairing": "fresh uniform random bijection per patch per visit",
                "marginals": "finite empirical paired-cloud bank" if opts["sampling"] == "bank" else "pre-drawn iid cloud stream; each cloud used at most once",
                "inference_source": "fresh iid standard Gaussian; no cache, anchors or patches",
                "draw_seconds": draw_seconds, "target_partition_seconds": partition_seconds,
                "source_assignment_seconds": assignment_seconds,
                "precompute_seconds": perf_counter() - start,
                "precompute_timing_scope": "draws, solves, array IO and array hashing; excludes final metadata write",
                "storage_bytes": storage,
                "environment": {"torch": str(torch.__version__), "numpy": np.__version__},
                "source_sha256": {name: file_sha256(Path(__file__).parent / name)
                                  for name in ("tg_cache.py", "coupling.py", "data.py", "train_horse.py")}}
    # Written last: interruption leaves an explicitly incomplete cache, never a
    # apparently valid one. No overwrite/automatic deletion of existing data.
    with (path / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
    print(f"saved_tg_cache={path} sha256={_fingerprint(metadata)} "
          f"precompute_seconds={metadata['precompute_seconds']:.3f}", flush=True)
    return path, metadata


class _CloudDataset(Dataset):
    """Open read-only memmaps separately in each Windows-spawned worker."""

    def __init__(self, path, metadata, seed):
        self.path, self.metadata = str(path), metadata
        self.arrays = None
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.metadata["num_clouds"]

    def draw(self, index, rng):
        if self.arrays is None:
            self.arrays = {name: np.load(Path(self.path) / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                           for name in self.metadata["array_sha256"]}
        arrays = self.arrays
        source_labels = arrays["source_labels"][index]
        if "fine_permutations" in arrays:
            pool = arrays["fine_permutations"][index]
            permutation = pool[int(rng.integers(len(pool)))]
        else:
            permutation = random_fine_permutation(source_labels, arrays["target_labels"][index],
                                                 self.metadata["num_regions"], rng)
        # Copies make writable CPU tensors without modifying read-only caches.
        return torch.from_numpy(arrays["source"][index].copy()), torch.from_numpy(arrays["target"][index][permutation].copy())

    def __getitem__(self, index):
        return self.draw(index, self.rng)

    def close(self):
        if self.arrays is not None:
            for array in self.arrays.values():
                array._mmap.close()
            self.arrays = None


def _worker_seed(worker_id):
    dataset = torch.utils.data.get_worker_info().dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset
    dataset.rng = np.random.default_rng(torch.initial_seed())


class TGCachedPairSampler:
    def __init__(self, config, dataset, device, dtype, *, training=False):
        start = perf_counter()
        opts = settings(config)
        self.path, self.metadata, self.cache_sha256 = load_cache(config, dataset)
        self.device, self.dtype = torch.device(device), dtype
        self.dataset = _CloudDataset(self.path, self.metadata, config["seed"] + 1)
        self.iterator, self.loader = None, None
        if training:
            batch = config["data"]["batch_size"]
            count = batch * config["training"]["num_steps"]
            generator = torch.Generator().manual_seed(config["seed"] + 1)
            if opts["sampling"] == "bank":
                sampler = RandomSampler(self.dataset, replacement=True, num_samples=count, generator=generator)
                dataset_value = self.dataset
            else:
                sampler, dataset_value = None, Subset(self.dataset, range(count))
            self.loader = DataLoader(dataset_value, batch_size=batch, sampler=sampler,
                                     num_workers=opts["num_workers"], worker_init_fn=_worker_seed,
                                     generator=generator, pin_memory=self.device.type == "cuda",
                                     persistent_workers=False, drop_last=True)
            self.iterator = iter(self.loader)
        self.setup_seconds = perf_counter() - start

    def sample(self, batch_size, *, generator=None):
        _integer(batch_size, "batch_size")
        if self.iterator is not None:
            if batch_size != self.loader.batch_size:
                raise ValueError("TG training cache requires configured data.batch_size")
            try:
                source, target = next(self.iterator)
            except StopIteration as error:
                raise RuntimeError("Prepared TG training stream exhausted; no automatic reuse") from error
        else:
            # Diagnostics use the cache's empirical law, not a hypothetical
            # fresh-X conditional mean field. A supplied RNG controls all draws.
            rng = self.dataset.rng
            if generator is not None:
                seed = torch.randint(0, 2**63 - 1, (), device=generator.device, generator=generator).item()
                rng = np.random.default_rng(seed)
            pairs = [self.dataset.draw(int(index), rng)
                     for index in rng.integers(len(self.dataset), size=batch_size)]
            source, target = (torch.stack(values) for values in zip(*pairs))
        return (source.to(device=self.device, dtype=self.dtype, non_blocking=True),
                target.to(device=self.device, dtype=self.dtype, non_blocking=True))

    def close(self):
        # DataLoader workers are scoped to this run, including exceptional exits.
        self.iterator = None
        self.loader = None
        self.dataset.close()

    def details(self):
        meta = self.metadata
        result = {"cache_sha256": self.cache_sha256, "cache_seed": meta["cache_seed"],
                "cache_clouds": meta["num_clouds"], "cache_sampling": meta["sampling"],
                "cache_setup_seconds": self.setup_seconds, "precompute_seconds": meta["precompute_seconds"],
                "precompute_timing_scope": meta["precompute_timing_scope"],
                "timing_scope": "training_seconds is online loop only; report precompute and cache setup separately",
                "marginals": meta["marginals"], "coarse_resampling": meta["coarse_resampling"],
                "fine_pairing": meta["fine_pairing"],
                "cached_source_sha256": meta["array_sha256"]["source"],
                "cached_target_sha256": meta["array_sha256"]["target"],
                "cached_target_partition_sha256": meta["array_sha256"]["target_labels"],
                "precompute_source_sha256": meta["source_sha256"]}
        if "untangle" in meta:
            result.update(implementation=meta["implementation"],
                          local_pairing=meta["local_pairing"],
                          fine_pairing_variant=meta["fine_pairing_variant"],
                          fine_pairing_objective=meta["untangle_summary"]["objective"],
                          untangle=meta["untangle"], untangle_summary=meta["untangle_summary"],
                          untangle_optimization_seconds=meta["untangle_optimization_seconds"],
                          parent_cache_sha256=meta["parent_cache"]["sha256"],
                          parent_precompute_seconds=meta["parent_precompute_seconds"],
                          parent_cache_prepared_now=meta["parent_cache_prepared_now"],
                          cached_permutations_sha256=meta["array_sha256"]["fine_permutations"],
                          one_lipschitz_guaranteed=False)
        return result
