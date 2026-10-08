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
import anchor_conditioning
import nsot_directional

FORMAT_VERSION = 1
PRIOR_FORMAT_VERSION = 2
SOLVER = "scipy_linear_sum_assignment"
DIRECTIONAL_IMPLEMENTATION = "nsot_anchor_prior_directional_hybrid_v1"


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
    anchor_flow.settings(config)
    nsot_directional.settings(config)
    anchor_conditioning.settings(config)
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


def _directional_parameters(config):
    options = nsot_directional.settings(config)
    return {key: value for key, value in options.items()
            if key not in ("artifact", "artifact_sha256")}


def _directional_identity(config, dataset, pair_cache_sha256):
    """Bind a small fitted kernel to the unchanged, separately hashed OT bank."""
    return {
        "format_version": 1, "implementation": DIRECTIONAL_IMPLEMENTATION,
        "dataset": dataset, "pair_cache_sha256": pair_cache_sha256,
        "anchor_prior": anchor_flow.cache_spec(config),
        "prior_centers": config["anchor_flow"]["centers"],
        "beta": float(config["nsot"]["beta"]),
        "settings": _directional_parameters(config),
    }


def _reject_nonfinite_json(value):
    raise ValueError(f"Nonfinite JSON constant in directional artifact: {value}")


def _load_directional_artifact(config, dataset, pair_cache_sha256):
    options = nsot_directional.settings(config)
    path = Path(options["artifact"])
    if not path.is_file():
        raise FileNotFoundError(f"Directional NSOT artifact missing: {path}; "
                                "run python prepare_nsot.py CONFIG --dataset " + dataset)
    digest = file_sha256(path)
    if options.get("artifact_sha256") not in (None, digest):
        raise ValueError("Directional artifact SHA256 differs from the checkpoint training artifact")
    try:
        with path.open(encoding="utf-8") as stream:
            artifact = json.load(stream, parse_constant=_reject_nonfinite_json)
    except (ValueError, UnicodeError) as error:
        raise ValueError("Invalid directional NSOT artifact JSON") from error
    if not isinstance(artifact, dict):
        raise ValueError("Directional NSOT artifact must be a mapping")
    identity = _directional_identity(config, dataset, pair_cache_sha256)
    required = set(identity) | {"matrices", "components", "fit_seconds", "fit_timing_scope",
                                "fit_source_sha256", "prior_preservation", "target_coordinates", "quality_guarantee"}
    if set(artifact) != required:
        raise ValueError("Directional artifact has missing/extra fields")
    for key, value in identity.items():
        if artifact.get(key) != value:
            raise ValueError(f"Directional artifact/config mismatch: {key}")
    matrices = np.asarray(artifact.get("matrices"), dtype=np.float64)
    k = config["num_regions"]
    if matrices.shape != (k, 2, 2):
        raise ValueError("Directional artifact requires one 2x2 matrix per original component")
    nsot_directional.factors(matrices)  # PSD/range/finiteness checks, CPU setup only.
    if not np.allclose(np.trace(matrices, axis1=1, axis2=2), 2 * identity["beta"],
                       rtol=0, atol=1e-8):
        raise ValueError("Directional artifact changes the scalar hybrid noise budget")
    reports = artifact.get("components")
    if not isinstance(reports, list) or len(reports) != k:
        raise ValueError("Directional artifact requires a fit report for every component")
    count_total = 0
    for component, report in enumerate(reports):
        if not isinstance(report, dict) or report.get("component") != component:
            raise ValueError("Directional component report has an invalid component ID")
        count = report.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Directional component report has an invalid sample count")
        count_total += count
        if not isinstance(report.get("fallback"), bool) or not isinstance(report.get("reason"), str):
            raise ValueError("Directional component report has no fallback status/reason")
        eigenvalues = np.asarray(report.get("beta_eigenvalues"), dtype=np.float64)
        if eigenvalues.shape != (2,) or not np.isfinite(eigenvalues).all() \
                or not np.allclose(eigenvalues, np.linalg.eigvalsh(matrices[component]), rtol=0, atol=1e-8):
            raise ValueError("Directional component report eigenvalues differ from its matrix")
    if count_total != config["nsot"]["superset_size"]:
        raise ValueError("Directional component counts differ from the pair cache size")
    seconds = artifact.get("fit_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) \
            or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Invalid directional fitting time")
    if not isinstance(artifact["fit_source_sha256"], dict) \
            or set(artifact["fit_source_sha256"]) != {"nsot.py", "nsot_directional.py"}:
        raise ValueError("Invalid directional fitting provenance")
    for digest_value in artifact["fit_source_sha256"].values():
        if not isinstance(digest_value, str) or len(digest_value) != 64 \
                or any(character not in "0123456789abcdef" for character in digest_value):
            raise ValueError("Invalid directional fitting source SHA256")
    if not isinstance(artifact["fit_timing_scope"], str) or not artifact["fit_timing_scope"]:
        raise ValueError("Invalid directional fitting timing scope")
    if artifact["quality_guarantee"] is not False:
        raise ValueError("Directional artifacts must not claim a quality guarantee")
    if any(not isinstance(artifact[key], str) or not artifact[key]
           for key in ("prior_preservation", "target_coordinates")):
        raise ValueError("Invalid directional prior/target scope description")
    # Includes nested fit reports. Reject overflows such as JSON 1e999 before training/save.
    try:
        json.dumps(artifact, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Directional artifact contains nonfinite or nonserializable metadata") from error
    return artifact, digest


def prepare_directional(config, dataset):
    """Fit/validate a small sidecar without changing or re-solving the OT bank."""
    options = nsot_directional.settings(config)
    if options is None:
        raise ValueError("prepare_directional requires nsot.directional_hybrid")
    resolved = copy.deepcopy(config)
    source, target, permutation, _, digest, components = _load_cache(resolved, dataset)
    path = Path(options["artifact"])
    if path.resolve() == Path(config["nsot"]["cache"]).resolve():
        raise ValueError("Directional artifact must not replace the OT pair cache")
    if path.exists():
        artifact, artifact_hash = _load_directional_artifact(resolved, dataset, digest)
        print(f"reused_directional_artifact={path} sha256={artifact_hash}", flush=True)
        return path, artifact
    if options.get("artifact_sha256") is not None:
        raise FileNotFoundError("Pinned directional artifact is missing; do not recreate a trained artifact")
    tick = perf_counter()
    fitted = nsot_directional.fit(source, target[permutation], components,
                                 resolved["anchor_flow"]["centers"],
                                 sigma=resolved["anchor_flow"]["sigma"],
                                 beta=resolved["nsot"]["beta"], options=options)
    fit_seconds = perf_counter() - tick
    artifact = {
        **_directional_identity(resolved, dataset, digest), **fitted,
        "fit_seconds": fit_seconds,
        "fit_timing_scope": "component statistics and ridge/PCA fitting only; excludes parent OT and all file IO/loading",
        "fit_source_sha256": {name: file_sha256(Path(__file__).parent / name)
                              for name in ("nsot.py", "nsot_directional.py")},
        "prior_preservation": "exact for fresh population Gaussian residuals and independent noise; fixed finite cache is approximate",
        "target_coordinates": "unchanged cached OT targets; no oversampling, deformation or re-assignment",
        "quality_guarantee": False,
    }
    # Validate numerical output before exclusive creation. Never overwrite an old fit.
    nsot_directional.factors(fitted["matrices"])
    encoded = json.dumps(artifact, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(f"saved_directional_artifact={path} sha256={file_sha256(path)} "
          f"fit_seconds={fit_seconds:.6f}", flush=True)
    return path, artifact


def prepare(config, dataset):
    """Prepare once; existing files are validated/reused, NEVER overwritten."""
    opts = settings(config)
    prior_spec = anchor_flow.cache_spec(config)
    spec = dataset_spec(config, dataset)
    path = Path(opts["cache"])
    if path.exists():
        _, _, _, metadata, digest = load_cache(config, dataset)
        print(f"reused_nsot_cache={path} sha256={digest}", flush=True)
        if nsot_directional.settings(config) is not None:
            prepare_directional(config, dataset)
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
    if nsot_directional.settings(config) is not None:
        prepare_directional(config, dataset)
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
        self.directional_shrink, self.directional_refresh = None, None
        self.directional_metadata, self.artifact_sha256 = None, None
        self.directional_artifact_path = None
        self.directional_setup_seconds = None
        if nsot_directional.settings(config) is not None:
            tick = perf_counter()
            self.directional_artifact_path = nsot_directional.settings(config)["artifact"]
            self.directional_metadata, self.artifact_sha256 = _load_directional_artifact(
                config, dataset, self.cache_sha256)
            shrink, refresh = nsot_directional.factors(self.directional_metadata["matrices"])
            self.directional_shrink = torch.as_tensor(shrink, device=device, dtype=dtype)
            self.directional_refresh = torch.as_tensor(refresh, device=device, dtype=dtype)
            config["nsot"]["directional_hybrid"]["artifact_sha256"] = self.artifact_sha256
            self.directional_setup_seconds = perf_counter() - tick
        self.flow_details = anchor_flow.experiment_details(config)

    @torch.no_grad()
    def sample(self, batch_size, *, generator=None, return_components=False):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(return_components, bool):
            raise ValueError("return_components must be a bool")
        if return_components and self.source_components is None:
            raise ValueError("Original source components require an anchor-prior NSOT bank")
        # Independent draws WITH replacement: Appendix A.1.2's product law.
        index = torch.randint(len(self.source), (batch_size, self.n_points),
                              device=self.source.device, generator=generator)
        source, target = self.source[index], self.target[index]
        noise = torch.randn(source.shape, device=source.device, dtype=source.dtype, generator=generator)
        if self.source_components is not None:
            components = self.source_components[index]
            centers = self.prior_centers[components]
            if self.directional_shrink is not None:
                paired_source = nsot_directional.apply(
                    source, centers, sigma=self.sigma,
                    shrink=self.directional_shrink[components],
                    refresh=self.directional_refresh[components], noise=noise)
            else:
                paired_source = component_centered_hybrid(source, centers, sigma=self.sigma,
                                                          beta=self.beta, noise=noise)
            if return_components:
                return paired_source, target, components
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
        if self.directional_metadata is not None:
            result["directional_hybrid"] = {
                "artifact": self.directional_artifact_path,
                "artifact_sha256": self.artifact_sha256,
                "pair_cache_sha256": self.cache_sha256,
                "settings": self.directional_metadata["settings"],
                "matrices": self.directional_metadata["matrices"],
                "components": self.directional_metadata["components"],
                "fit_source_sha256": self.directional_metadata["fit_source_sha256"],
                "conditioning": "original sampled component labels select fixed kernels; labels are NOT model inputs",
                "noise_budget": "trace(S_k)=2*beta for every component, including isotropic fallbacks",
                "online_work": "two small fixed matrix-vector products per sampled point; no solves or network Jacobians",
            }
            result.update(directional_precompute_seconds=self.directional_metadata["fit_seconds"],
                          directional_precompute_timing_scope=self.directional_metadata["fit_timing_scope"],
                          directional_setup_seconds=self.directional_setup_seconds,
                          directional_setup_timing_scope="artifact read/validation and factor placement; no explicit CUDA synchronization")
        return result
