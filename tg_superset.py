"""Dense TG coarse geometry shared by fresh, without-replacement subsets.

No fine permutation is cached. Choose N source indices uniformly, then choose
the SAME NUMBER of target indices per patch, uniformly without replacement.
Because dense source/target capacities agree, the induced target subset is
also uniform over all N-subsets (multivariate hypergeometric count law).
This preserves finite-pool subset marginals, NOT exact population iid laws.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from coupling import assign_regions, balanced_target_partition
from data import validate_gaussian_source
from nsot import dataset_spec, file_sha256, source_moments
from tg_cache import _fingerprint, _integer, hybrid_source, validate_beta


IMPLEMENTATION = "tg_dense_superset_v1"


def settings(config):
    if config.get("coupling") != "target_guided_superset":
        raise ValueError("Dense TG requires coupling: target_guided_superset")
    value = config.get("tg_superset", {})
    if not isinstance(value, dict):
        raise ValueError("tg_superset must be a mapping")
    unknown = value.keys() - {"path", "superset_size", "num_supersets", "seed", "beta", "cache_sha256"}
    if unknown:
        raise ValueError("Unsupported tg_superset settings: " + ", ".join(sorted(map(str, unknown))))
    size = _integer(value.get("superset_size"), "superset_size")
    pools = _integer(value.get("num_supersets"), "num_supersets")
    n = _integer(config["data"]["n_points"], "n_points")
    k = _integer(config.get("num_regions"), "num_regions")
    _integer(config["data"]["batch_size"], "batch_size")
    _integer(config["training"]["num_steps"], "num_steps")
    if not k <= n <= size <= np.iinfo(np.int32).max:
        raise ValueError("Dense TG requires 1 <= K <= N <= M <= int32 max")
    if config["model"]["point_dim"] != 2 or config.get("dtype") != "float32":
        raise ValueError("Dense TG supports float32 2D only")
    if not isinstance(value.get("path"), str) or not value["path"]:
        raise ValueError("tg_superset.path must name a cache directory")
    validate_gaussian_source(config)
    return {"path": value["path"], "superset_size": size, "num_supersets": pools,
            "cache_seed": _integer(value.get("seed", 0), "tg_superset.seed", 0),
            "beta": validate_beta(value.get("beta", 0.0)), "n_points": n, "num_regions": k}


def _spec(config, dataset, opts):
    # N and beta do not affect dense geometry; paired ablations reuse the cache.
    return {"format_version": 1, "implementation": IMPLEMENTATION,
            "superset_size": opts["superset_size"], "num_supersets": opts["num_supersets"],
            "num_regions": opts["num_regions"], "cache_seed": opts["cache_seed"],
            "dataset_spec": dataset_spec(config, dataset)}


def _shapes(opts):
    r, m, k = opts["num_supersets"], opts["superset_size"], opts["num_regions"]
    return {"source": ((r, m, 2), np.float32), "target": ((r, m, 2), np.float32),
            "source_labels": ((r, m), np.int32), "target_labels": ((r, m), np.int32),
            "capacities": ((r, k), np.int32)}


def load_cache(config, dataset):
    opts = settings(config)
    path = Path(opts["path"])
    if not (path / "metadata.json").is_file():
        raise FileNotFoundError(f"Dense TG cache missing/incomplete: {path}; run prepare_tg_superset.py")
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    for key, expected in _spec(config, dataset, opts).items():
        if meta.get(key) != expected:
            raise ValueError(f"Dense TG cache/config mismatch: {key}")
    shapes = _shapes(opts)
    if set(meta.get("array_sha256", {})) != set(shapes):
        raise ValueError("Dense TG cache has missing/extra arrays")
    arrays = {}
    for name, (shape, dtype) in shapes.items():
        file = path / f"{name}.npy"
        if file_sha256(file) != meta["array_sha256"][name]:
            raise ValueError(f"Dense TG cache array SHA256 mismatch: {name}")
        array = np.load(file, allow_pickle=False)
        if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
            raise ValueError(f"Dense TG cache shape/dtype/values mismatch: {name}")
        arrays[name] = array
    k = opts["num_regions"]
    expected_caps = np.full(k, opts["superset_size"] // k)
    expected_caps[:opts["superset_size"] % k] += 1
    for pool in range(opts["num_supersets"]):
        if not np.array_equal(arrays["capacities"][pool], expected_caps):
            raise ValueError("Invalid dense TG capacities")
        for name in ("source_labels", "target_labels"):
            labels = arrays[name][pool]
            if np.any(labels < 0) or np.any(labels >= k) \
                    or not np.array_equal(np.bincount(labels, minlength=k), expected_caps):
                raise ValueError("Invalid dense TG patch labels/capacities")
    digest = _fingerprint(meta)
    if config["tg_superset"].get("cache_sha256", digest) != digest:
        raise ValueError("Dense TG cache differs from the checkpoint training cache")
    return arrays, meta, digest


def prepare(config, dataset):
    """Same FPS/exact coarse TG as baseline, applied at M rather than N."""
    opts = settings(config)
    spec, path = _spec(config, dataset, opts), Path(opts["path"])
    if path.exists():
        _, meta, digest = load_cache(config, dataset)
        print(f"reused_tg_superset={path} sha256={digest}", flush=True)
        return path, meta
    shapes = _shapes(opts)
    start = perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    arrays = {name: np.lib.format.open_memmap(path / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
              for name, (shape, dtype) in shapes.items()}
    draw_seconds = partition_seconds = assignment_seconds = 0.0
    moments = []
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(opts["cache_seed"]).get_state())
            if dataset == "horse":
                from train_horse import load_horse_mask, sample_horse
                mask = load_horse_mask("cpu", torch.float32)
            else:
                from data import sample_checkerboard
            for pool in range(opts["num_supersets"]):
                tick = perf_counter()
                m = opts["superset_size"]
                source = torch.randn(1, m, 2, device="cpu", dtype=torch.float32)
                target = (sample_horse(mask, 1, m) if dataset == "horse" else
                          sample_checkerboard(1, m, "cpu", torch.float32, config["data"]["grid_size"]))
                draw_seconds += perf_counter() - tick
                tick = perf_counter()
                _, labels, centers, capacities = balanced_target_partition(
                    target, opts["num_regions"], solver="exact_batched")
                partition_seconds += perf_counter() - tick
                tick = perf_counter()
                source_labels = assign_regions(source, centers, capacities, solver="exact_batched")
                assignment_seconds += perf_counter() - tick
                for name, value in (("source", source), ("target", target), ("source_labels", source_labels),
                                    ("target_labels", labels), ("capacities", capacities)):
                    arrays[name][pool] = value[0].numpy()
                moments.append(source_moments(source[0].numpy()))
                print(f"tg_superset_prepared={pool + 1}/{opts['num_supersets']} M={m} K={opts['num_regions']}", flush=True)
    finally:
        torch.set_num_threads(threads)
        for array in arrays.values():
            array.flush()
            array._mmap.close()
        arrays.clear()
    meta = {**spec, "array_sha256": {name: file_sha256(path / f"{name}.npy") for name in shapes},
            "source_coordinates": "unmodified iid standard Gaussian draws in each dense pool",
            "target_coordinates": "unchanged project sampler; no centroid replacement or whitening",
            "fine_pairing": "fresh random bijection; NO stored fine permutation",
            "subset_sampling": "uniform without-replacement source; hypergeometric matched target counts",
            "marginals": "uniform subsets of fixed finite pools; approximate, not exact iid population clouds",
            "source_moments_by_pool": moments, "draw_seconds": draw_seconds,
            "target_partition_seconds": partition_seconds, "source_assignment_seconds": assignment_seconds,
            "precompute_seconds": perf_counter() - start,
            "precompute_timing_scope": "draws, M-by-K coarse solves, array IO and hashing; excludes metadata write",
            "storage_bytes": sum(int(np.prod(shape)) * np.dtype(dtype).itemsize for shape, dtype in shapes.values()),
            "source_sha256": {name: file_sha256(Path(__file__).parent / name)
                              for name in ("tg_superset.py", "tg_cache.py", "coupling.py", "data.py", "train_horse.py")}}
    with (path / "metadata.json").open("x", encoding="utf-8") as stream:
        json.dump(meta, stream, indent=2, allow_nan=False)
    print(f"saved_tg_superset={path} precompute_seconds={meta['precompute_seconds']:.3f}", flush=True)
    return path, meta


def subset_indices(source_labels, target_groups, n, rng):
    """Uniform target subsets, random fine bijection, no fixed per-patch quota.

    If dense group sizes are c_k, source counts have probability
    prod_k C(c_k,n_k)/C(M,N). Choosing target subsets with probability
    prod_k 1/C(c_k,n_k) gives 1/C(M,N) for every target N-subset.
    Ordered uniform target draws simultaneously randomize the fine bijection.
    """
    source_index = rng.choice(len(source_labels), size=n, replace=False)
    labels = source_labels[source_index]
    target_index = np.empty(n, dtype=np.int64)
    for patch, group in enumerate(target_groups):
        positions = np.flatnonzero(labels == patch)
        target_index[positions] = rng.choice(group, size=len(positions), replace=False)
    return source_index, target_index


class TGSupersetPairSampler:
    def __init__(self, config, dataset, device, dtype, *, training=False):
        start = perf_counter()
        opts = settings(config)
        self.arrays, self.metadata, self.cache_sha256 = load_cache(config, dataset)
        self.device, self.dtype = torch.device(device), dtype
        self.beta, self.n_points = opts["beta"], opts["n_points"]
        self.training = training
        # Separate geometry and jitter RNGs: beta ablations see identical subsets.
        self.rng = np.random.default_rng(config["seed"] + 1)
        self.target_groups = [[np.flatnonzero(labels == k) for k in range(opts["num_regions"])]
                              for labels in self.arrays["target_labels"]]
        self.setup_seconds = perf_counter() - start

    @torch.no_grad()
    def sample(self, batch_size, *, generator=None):
        _integer(batch_size, "batch_size")
        rng = self.rng
        if not self.training and generator is not None:
            seed = torch.randint(0, 2**63 - 1, (), device=generator.device, generator=generator).item()
            rng = np.random.default_rng(seed)
        source, target = [], []
        for pool in rng.integers(len(self.target_groups), size=batch_size):
            s, t = subset_indices(self.arrays["source_labels"][pool], self.target_groups[pool], self.n_points, rng)
            source.append(self.arrays["source"][pool, s])
            target.append(self.arrays["target"][pool, t])
        source = torch.as_tensor(np.stack(source), device=self.device, dtype=self.dtype)
        target = torch.as_tensor(np.stack(target), device=self.device, dtype=self.dtype)
        return hybrid_source(source, self.beta, generator=generator), target

    def close(self):
        pass  # Small dense pools are ordinary arrays, not open memmaps/workers.

    def details(self):
        return {"implementation": IMPLEMENTATION, "cache_sha256": self.cache_sha256,
                "beta": self.beta, "num_supersets": self.metadata["num_supersets"],
                "superset_size": self.metadata["superset_size"], "cache_seed": self.metadata["cache_seed"],
                "cache_setup_seconds": self.setup_seconds, "precompute_seconds": self.metadata["precompute_seconds"],
                "precompute_timing_scope": self.metadata["precompute_timing_scope"],
                "timing_scope": "training_seconds includes subset sampling, fresh bijections and perturbation; excludes cache setup/precompute",
                "fine_pairing": self.metadata["fine_pairing"], "subset_sampling": self.metadata["subset_sampling"],
                "source_coordinates": "original draws" if self.beta == 0 else "Gaussian hybrid; source coordinates changed",
                "marginals": (self.metadata["marginals"] if self.beta == 0 else
                              "target: uniform finite-pool subsets; source: scaled subsets plus independent Gaussian; approximate population clouds"),
                "marginals_before_hybrid": self.metadata["marginals"],
                "array_sha256": self.metadata["array_sha256"],
                "cached_source_moments_by_pool": self.metadata["source_moments_by_pool"],
                "precompute_source_sha256": self.metadata["source_sha256"]}
