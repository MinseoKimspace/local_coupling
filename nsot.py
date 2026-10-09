"""Paper-based 2D NSOT: exact superset OT, iid pair subsampling, hybrid noise.

Reference: Hui et al., ICLR 2025, https://arxiv.org/abs/2502.12456
Sections 3.3--3.4, Appendix A.1.2, and the M<=10,000 exact-OT setting.
This is NOT author code or the main 100K approximate-gradient-flow experiment.
Keep the original Gaussian coordinates; exact OT only permutes the target.
Fixed finite caches approximate, rather than exactly equal, population marginals.
Baseline inference starts from fresh iid standard Gaussian, without this cache.
Optional anchor_prior uses a GMM BEFORE OT and a component-centered hybrid
kernel; its population GMM is preserved, not the original NSOT Gaussian prior.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import torch

import anchor_flow

FORMAT_VERSION = 1
PRIOR_FORMAT_VERSION = 2
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
    if not isinstance(value, dict):
        raise ValueError("nsot must be a mapping")
    unknown = value.keys() - {"cache", "superset_size", "cache_seed", "beta", "solver", "cache_sha256"}
    if unknown:
        raise ValueError("Unsupported nsot settings: " + ", ".join(sorted(map(str, unknown))))
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
    anchor_flow.settings(config)
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
    if anchor_flow.settings(config) is not None:
        source, target, _, _, _ = _draw_prior_supersets(config, dataset)
        return source, target
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


def _draw_prior_supersets(config, dataset):
    """New experimental draws with separately isolated source/target streams.

    The original Gaussian baseline's draw order above is deliberately untouched.
    This variant does not promise its target superset equals that old stream.
    The GMM draw uses sample_source's exact categorical-then-Gaussian ordering.
    """
    opts = settings(config)
    source_config = copy.deepcopy(config)
    tick = perf_counter()
    centers = anchor_flow.fit_prior_centers(source_config, dataset)
    anchor_flow.bind_prior_centers(source_config, centers)
    fit_seconds = perf_counter() - tick
    source_config["data"]["n_points"] = opts["superset_size"]
    source_generator = torch.Generator(device="cpu").manual_seed(opts["cache_seed"])
    source, components = anchor_flow.sample_source_with_components(
        source_config, 1, device="cpu", dtype=torch.float32, generator=source_generator)
    target_seed = int.from_bytes(hashlib.sha256(
        f"nsot_anchor_prior_target_v1:{opts['cache_seed']}".encode()).digest()[:8], "little") % (2**63 - 1)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(target_seed).get_state())
        if dataset == "checkerboard":
            from data import sample_checkerboard
            target = sample_checkerboard(1, opts["superset_size"], "cpu", torch.float32,
                                         config["data"]["grid_size"])[0]
        elif dataset == "horse":
            from train_horse import load_horse_mask, sample_horse
            target = sample_horse(load_horse_mask("cpu", torch.float32), 1, opts["superset_size"])[0]
        else:
            raise ValueError("Expected checkerboard or horse")
    return source[0].numpy(), target.numpy(), components[0].numpy(), centers, {
        "prior_fit_seconds": fit_seconds, "source_draw_seed": opts["cache_seed"],
        "target_draw_seed": target_seed,
        "draw_rng": "isolated CPU source generator and separate hashed target seed; independent of prior fitting"}


def _component_sha256(components):
    return hashlib.sha256(np.ascontiguousarray(components).tobytes()).hexdigest()


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


def _load_cache(config, dataset):
    opts = settings(config)
    prior_spec = anchor_flow.cache_spec(config)
    path = Path(opts["cache"])
    if not path.is_file():
        raise FileNotFoundError(f"NSOT cache missing: {path}; run python prepare_nsot.py CONFIG --dataset {dataset}")
    cache_hash = file_sha256(path)
    expected_hash = config["nsot"].get("cache_sha256")
    if expected_hash is not None and expected_hash != cache_hash:
        raise ValueError("NSOT cache SHA256 differs from the checkpoint training cache")
    with np.load(path, allow_pickle=False) as archive:
        expected_arrays = {"source", "target", "permutation", "metadata"}
        if prior_spec is not None:
            expected_arrays.add("source_components")
        if set(archive.files) != expected_arrays:
            raise ValueError("NSOT cache has missing/extra arrays or a different source prior")
        metadata = json.loads(str(archive["metadata"].item()))
        source, target, permutation = archive["source"], archive["target"], archive["permutation"]
        components = archive["source_components"] if prior_spec is not None else None
    expected = {"format_version": PRIOR_FORMAT_VERSION if prior_spec is not None else FORMAT_VERSION,
                "solver": SOLVER,
                "superset_size": opts["superset_size"], "cache_seed": opts["cache_seed"],
                "dataset_spec": dataset_spec(config, dataset)}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"NSOT cache/config mismatch: {key}")
    if metadata.get("anchor_prior") != prior_spec:
        raise ValueError("NSOT cache/config mismatch: anchor_prior")
    if prior_spec is None and "prior_centers" in metadata:
        raise ValueError("A Gaussian NSOT cache must not contain prior_centers")
    count = opts["superset_size"]
    if source.shape != (count, 2) or target.shape != source.shape \
            or source.dtype != np.float32 or target.dtype != np.float32 \
            or not np.isfinite(source).all() or not np.isfinite(target).all() \
            or permutation.shape != (count,) or not np.issubdtype(permutation.dtype, np.integer) \
            or not np.array_equal(np.sort(permutation), np.arange(count)):
        raise ValueError("NSOT cache contains invalid points or a non-bijective permutation")
    if prior_spec is not None:
        k = config["num_regions"]
        if (components.shape != (count,) or components.dtype != np.int64
                or (components < 0).any() or (components >= k).any()):
            raise ValueError("NSOT anchor prior requires original int64 source_components in [0,K)")
        if metadata.get("source_components_sha256") != _component_sha256(components):
            raise ValueError("NSOT cache source_components SHA256 mismatch")
        # Bind only after every stored array and prior setting has validated.
        # This rejects replacing an explicitly saved center configuration.
        anchor_flow.bind_prior_centers(config, metadata.get("prior_centers"))
    return source, target, permutation, metadata, cache_hash, components


def load_cache(config, dataset):
    """Validate and load a cache; preserve the original five-value public API.

    Anchor priors additionally bind the exact stored centers into config for
    checkpoint saving. Components remain a binary array, not a huge JSON list.
    """
    return _load_cache(config, dataset)[:5]




def prepare(config, dataset):
    """Prepare once; existing files are validated/reused, NEVER overwritten."""
    opts = settings(config)
    prior_spec = anchor_flow.cache_spec(config)
    spec = dataset_spec(config, dataset)
    path = Path(opts["cache"])
    if path.exists():
        _, _, _, metadata, digest = load_cache(config, dataset)
        print(f"reused_nsot_cache={path} sha256={digest}", flush=True)
        return path, metadata
    start = perf_counter()
    components, centers, prior_draw = None, None, {}
    if prior_spec is None:
        source, target = draw_supersets(config, dataset)
    else:
        source, target, components, centers, prior_draw = _draw_prior_supersets(config, dataset)
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
    if prior_spec is not None:
        resolved = copy.deepcopy(config)
        anchor_flow.bind_prior_centers(resolved, centers)
        metadata.update(
            format_version=PRIOR_FORMAT_VERSION,
            implementation="nsot_anchor_prior_component_hybrid_v1",
            paper_setting="experimental anchor-GMM/component-centered extension; NOT the original NSOT prior/kernel",
            anchor_prior=prior_spec, prior_centers=centers,
            source_components_sha256=_component_sha256(components),
            source_component_counts=np.bincount(components, minlength=config["num_regions"]).tolist(),
            component_label_origin="sampled categorical BEFORE exact OT; never reassigned by nearest centroid",
            source_moments_scope="cached GMM source superset BEFORE component-centered hybrid jitter",
            transport_cost_scope="full source/target superset squared cost BEFORE training-time hybrid jitter",
            marginals="finite point-superset plus component-centered jitter approximates the saved population GMM",
            source_prior=anchor_flow.experiment_details(resolved), **prior_draw)
        metadata["source_sha256"]["anchor_flow.py"] = file_sha256(Path(__file__).parent / "anchor_flow.py")
        if prior_spec.get("center_fit") == "gmm_em":
            metadata.update(
                implementation="nsot_fixed_isotropic_gmm_em_v1",
                paper_setting="experimental equal-weight fixed-sigma GMM-EM/component-centered extension; NOT original NSOT",
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental destruction of an existing cache.
    # If interrupted during writing, the partial file fails load_cache validation.
    with path.open("xb") as stream:
        arrays = {"source": source, "target": target, "permutation": permutation,
                  "metadata": np.array(json.dumps(metadata, allow_nan=False))}
        if components is not None:
            arrays["source_components"] = components
        np.savez_compressed(stream, **arrays)
    digest = file_sha256(path)
    print(f"saved_nsot_cache={path} sha256={digest}", flush=True)
    print(f"precompute_seconds={metadata['precompute_seconds']:.3f} "
          f"cost_before={metadata['cost_before']:.6f} cost_after={optimal_cost:.6f}", flush=True)
    return path, metadata


def component_centered_hybrid(source, centers, *, sigma, beta, noise):
    """Population-GMM-preserving noise, centered on original source labels.

    Configuration scalars are checked without synchronizing the GPU. Tensor
    shape/device/dtype checks only: finite cache validation happens at setup.
    For iid residual/noise N(0,I), each component keeps its mean and sigma²I.
    Finite cached residuals approximate this statement rather than equal it.
    """
    if (isinstance(beta, bool) or not isinstance(beta, (int, float))
            or not math.isfinite(beta) or not 0 <= beta <= 1
            or isinstance(sigma, bool) or not isinstance(sigma, (int, float))
            or not math.isfinite(sigma) or sigma <= 0):
        raise ValueError("Centered hybrid requires finite beta in [0,1] and positive sigma")
    if (not isinstance(source, torch.Tensor) or not source.is_floating_point()
            or not isinstance(centers, torch.Tensor) or not isinstance(noise, torch.Tensor)
            or centers.shape != source.shape or noise.shape != source.shape
            or centers.device != source.device or noise.device != source.device
            or centers.dtype != source.dtype or noise.dtype != source.dtype):
        raise ValueError("Centered hybrid source, centers and noise must share shape/device/dtype")
    return centers + math.sqrt(1 - beta) * (source - centers) + sigma * math.sqrt(beta) * noise


class NSOTPairSampler:
    def __init__(self, config, dataset, device, dtype):
        opts = settings(config)
        source, target, permutation, self.metadata, self.cache_sha256, components = _load_cache(config, dataset)
        self.source = torch.as_tensor(source, device=device, dtype=dtype)
        self.target = torch.as_tensor(target[permutation], device=device, dtype=dtype)
        self.beta, self.n_points = opts["beta"], config["data"]["n_points"]
        flow = anchor_flow.settings(config)
        self.source_components, self.prior_centers, self.sigma = None, None, None
        if flow is not None:
            self.source_components = torch.as_tensor(components, device=device, dtype=torch.long)
            self.prior_centers = torch.tensor(flow["centers"], device=device, dtype=dtype)
            self.sigma = flow["sigma"]
        self.flow_details = anchor_flow.experiment_details(config)

    @torch.no_grad()
    def sample(self, batch_size, *, generator=None):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        # Independent draws WITH replacement: Appendix A.1.2's product law.
        index = torch.randint(len(self.source), (batch_size, self.n_points),
                              device=self.source.device, generator=generator)
        source, target = self.source[index], self.target[index]
        noise = torch.randn(source.shape, device=source.device, dtype=source.dtype, generator=generator)
        if self.source_components is not None:
            components = self.source_components[index]
            centers = self.prior_centers[components]
            paired_source = component_centered_hybrid(source, centers, sigma=self.sigma,
                                                      beta=self.beta, noise=noise)
            return paired_source, target
        return math.sqrt(1 - self.beta) * source + math.sqrt(self.beta) * noise, target

    def details(self):
        result = {"beta": self.beta, "superset_size": len(self.source), "cache_sha256": self.cache_sha256,
                "cache_seed": self.metadata["cache_seed"], "precompute_seconds": self.metadata["precompute_seconds"],
                "timing_scope": "training_seconds is online loop only; precompute_seconds excludes cache file IO/loading",
                "cost_before": self.metadata["cost_before"], "cost_after": self.metadata["cost_after"],
                "source_moments": self.metadata["source_moments"], "dataset_spec": self.metadata["dataset_spec"],
                "precompute_source_sha256": self.metadata["source_sha256"]}
        result.update(self.flow_details)
        if self.source_components is not None:
            result.update(prior_fit_seconds=self.metadata["prior_fit_seconds"],
                          source_components_sha256=self.metadata["source_components_sha256"],
                          source_component_counts=self.metadata["source_component_counts"],
                          source_moments_scope=self.metadata["source_moments_scope"],
                          transport_cost_scope=self.metadata["transport_cost_scope"],
                          source_draw_seed=self.metadata["source_draw_seed"],
                          target_draw_seed=self.metadata["target_draw_seed"],
                          draw_rng=self.metadata["draw_rng"])
        return result
