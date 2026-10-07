"""Paper-based 2D NSOT: exact superset OT, iid pair subsampling, hybrid noise.

Reference: Hui et al., ICLR 2025, https://arxiv.org/abs/2502.12456
Sections 3.3--3.4, Appendix A.1.2, and the M<=10,000 exact-OT setting.
This is NOT author code or the main 100K approximate-gradient-flow experiment.
Keep the original Gaussian coordinates; exact OT only permutes the target.
Fixed finite caches approximate, rather than exactly equal, population marginals.
Inference must still start from fresh iid standard Gaussian, without this cache.
"""

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch


FORMAT_VERSION = 1
SOLVER = "scipy_linear_sum_assignment"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def settings(config):
    if config.get("coupling") != "nsot":
        raise ValueError("NSOT preparation requires coupling: nsot")
    value = config.get("nsot", {})
    size, seed, beta = value.get("superset_size"), value.get("cache_seed"), value.get("beta")
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 10000:
        raise ValueError("Exact NSOT superset_size must be an integer in [1,10000]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("NSOT cache_seed must be an integer in [0,2**63)")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) \
            or not 0 <= beta <= 1:
        raise ValueError("NSOT beta must be finite and in [0,1]")
    if not value.get("cache") or value.get("solver", SOLVER) != SOLVER:
        raise ValueError("NSOT requires a cache path and scipy_linear_sum_assignment solver")
    if config["model"]["point_dim"] != 2 or not 1 <= config["data"]["n_points"] <= size:
        raise ValueError("2D NSOT requires point_dim=2 and 1 <= n_points <= superset_size")
    return {"superset_size": size, "cache_seed": seed, "beta": float(beta),
            "cache": str(value["cache"]), "solver": SOLVER}


def dataset_spec(config, dataset):
    if dataset == "checkerboard":
        grid = config["data"]["grid_size"]
        if isinstance(grid, bool) or not isinstance(grid, int) or grid < 1:
            raise ValueError("grid_size must be a positive integer")
        return {"dataset": dataset, "point_dim": 2, "grid_size": grid}
    if dataset == "horse":
        from skimage.data import horse
        mask = np.ascontiguousarray(~horse())
        return {"dataset": dataset, "point_dim": 2, "mask_shape": list(mask.shape),
                "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest()}
    raise ValueError("NSOT supports checkerboard and horse only")


def draw_supersets(config, dataset):
    """Isolated CPU float32 draws using the unmodified project target samplers."""
    opts = settings(config)
    count = opts["superset_size"]
    # Reset CPU RNG only, without advancing/restoring any CUDA generator.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(opts["cache_seed"]).get_state())
        source = torch.randn(count, 2, device="cpu", dtype=torch.float32)
        if dataset == "checkerboard":
            from data import sample_checkerboard
            target = sample_checkerboard(1, count, "cpu", torch.float32,
                                         config["data"]["grid_size"])[0]
        elif dataset == "horse":
            from train_horse import load_horse_mask, sample_horse
            target = sample_horse(load_horse_mask("cpu", torch.float32), 1, count)[0]
        else:
            raise ValueError("Expected checkerboard or horse")
    return source.numpy(), target.numpy()


def exact_superset_permutation(source, target):
    """Exact linear assignment; same OT objective as the paper's Hungarian solve.

    SciPy uses a modified Jonker-Volgenant algorithm. The dense float64 matrix
    needs approximately 8*M*M bytes; no M*M*D tensor is materialized.
    """
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 \
            or not len(source) or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Expected matching finite nonempty [M,2] arrays")
    cost = cdist(source, target, metric="sqeuclidean")
    rows, columns = linear_sum_assignment(cost)
    permutation = np.empty(len(source), dtype=np.int64)
    permutation[rows] = columns
    return permutation, float(cost[rows, columns].mean())


def source_moments(source):
    value = np.asarray(source, dtype=np.float64)
    centered = value - value.mean(0)
    return {"mean": value.mean(0).tolist(), "covariance": (centered.T @ centered / len(value)).tolist()}


def load_cache(config, dataset):
    opts = settings(config)
    path = Path(opts["cache"])
    if not path.is_file():
        raise FileNotFoundError(f"NSOT cache missing: {path}; run python prepare_nsot.py CONFIG --dataset {dataset}")
    cache_hash = file_sha256(path)
    expected_hash = config["nsot"].get("cache_sha256")
    if expected_hash is not None and expected_hash != cache_hash:
        raise ValueError("NSOT cache SHA256 differs from the checkpoint training cache")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        source, target, permutation = archive["source"], archive["target"], archive["permutation"]
    expected = {"format_version": FORMAT_VERSION, "solver": SOLVER,
                "superset_size": opts["superset_size"], "cache_seed": opts["cache_seed"],
                "dataset_spec": dataset_spec(config, dataset)}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"NSOT cache/config mismatch: {key}")
    count = opts["superset_size"]
    if source.shape != (count, 2) or target.shape != source.shape \
            or source.dtype != np.float32 or target.dtype != np.float32 \
            or not np.isfinite(source).all() or not np.isfinite(target).all() \
            or permutation.shape != (count,) or not np.issubdtype(permutation.dtype, np.integer) \
            or not np.array_equal(np.sort(permutation), np.arange(count)):
        raise ValueError("NSOT cache contains invalid points or a non-bijective permutation")
    return source, target, permutation, metadata, cache_hash


def prepare(config, dataset):
    """Prepare once; existing files are validated/reused, NEVER overwritten."""
    opts = settings(config)
    spec = dataset_spec(config, dataset)
    path = Path(opts["cache"])
    if path.exists():
        _, _, _, metadata, digest = load_cache(config, dataset)
        print(f"reused_nsot_cache={path} sha256={digest}", flush=True)
        return path, metadata
    start = perf_counter()
    source, target = draw_supersets(config, dataset)
    size = opts["superset_size"]
    print(f"NSOT exact OT M={size}; dense float64 cost={8 * size * size / 2**20:.1f} MiB CPU RAM", flush=True)
    permutation, optimal_cost = exact_superset_permutation(source, target)
    metadata = {
        "format_version": FORMAT_VERSION, "solver": SOLVER,
        "implementation": "paper_based_2d_exact_superset_v1",
        "paper": "https://arxiv.org/abs/2502.12456",
        "paper_setting": "exact superset OT for M<=10000, NOT main M=100000 gradient-flow results",
        "superset_size": size, "cache_seed": opts["cache_seed"], "dataset_spec": spec,
        "cost_before": float(np.square(source.astype(np.float64) - target).sum(1).mean()),
        "cost_after": optimal_cost, "source_moments": source_moments(source),
        "precompute_seconds": perf_counter() - start,
        "precompute_timing_scope": "sampling and exact OT compute; excludes cache file IO and training-time loading",
        "environment": {"torch": str(torch.__version__), "numpy": np.__version__},
        "source_sha256": {name: file_sha256(Path(__file__).parent / name)
                          for name in ("nsot.py", "data.py", "train_horse.py")},
        "marginals": "fixed finite empirical supersets; Gaussian/target population marginals are approximate",
        "target_coordinates": "unchanged project orientation/scale; no centering, rotation, whitening or clipping",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental destruction of an existing cache.
    # If interrupted during writing, the partial file fails load_cache validation.
    with path.open("xb") as stream:
        np.savez_compressed(stream, source=source, target=target, permutation=permutation,
                            metadata=np.array(json.dumps(metadata, allow_nan=False)))
    digest = file_sha256(path)
    print(f"saved_nsot_cache={path} sha256={digest}", flush=True)
    print(f"precompute_seconds={metadata['precompute_seconds']:.3f} "
          f"cost_before={metadata['cost_before']:.6f} cost_after={optimal_cost:.6f}", flush=True)
    return path, metadata


class NSOTPairSampler:
    def __init__(self, config, dataset, device, dtype):
        opts = settings(config)
        source, target, permutation, self.metadata, self.cache_sha256 = load_cache(config, dataset)
        self.source = torch.as_tensor(source, device=device, dtype=dtype)
        self.target = torch.as_tensor(target[permutation], device=device, dtype=dtype)
        self.beta, self.n_points = opts["beta"], config["data"]["n_points"]

    @torch.no_grad()
    def sample(self, batch_size, *, generator=None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        # Independent draws WITH replacement: Appendix A.1.2's product law.
        index = torch.randint(len(self.source), (batch_size, self.n_points),
                              device=self.source.device, generator=generator)
        source, target = self.source[index], self.target[index]
        noise = torch.randn(source.shape, device=source.device, dtype=source.dtype, generator=generator)
        return math.sqrt(1 - self.beta) * source + math.sqrt(self.beta) * noise, target

    def details(self):
        return {"beta": self.beta, "superset_size": len(self.source), "cache_sha256": self.cache_sha256,
                "cache_seed": self.metadata["cache_seed"], "precompute_seconds": self.metadata["precompute_seconds"],
                "timing_scope": "training_seconds is online loop only; precompute_seconds excludes cache file IO/loading",
                "cost_before": self.metadata["cost_before"], "cost_after": self.metadata["cost_after"],
                "source_moments": self.metadata["source_moments"], "dataset_spec": self.metadata["dataset_spec"],
                "precompute_source_sha256": self.metadata["source_sha256"]}
