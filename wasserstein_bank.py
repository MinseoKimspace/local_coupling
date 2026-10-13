"""Offline selection of complete coupling banks on fixed 2D supersets.

All candidates preserve the same empirical endpoints and shape frequencies.
Wasserstein subset scores and local velocity variograms are hypotheses/proxies,
not conditional-variance or neural approximation guarantees.
"""

import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr
import torch

from data import sample_checkerboard, validate_gaussian_source
from nsot import dataset_spec as original_dataset_spec, file_sha256


METHOD = "wasserstein_bank"
SELECTORS = ("random", "transport", "displacement", "joint", "path_joint", "multiscale", "variance")
SHAPES = ("disk", "ring", "two_disks", "cross")
DEFAULT_CANDIDATES = [
    {"kind": "random"}, {"kind": "exact"},
    {"kind": "projection", "angle": 0.0},
    {"kind": "projection", "angle": math.pi / 4},
    {"kind": "projection", "angle": math.pi / 2},
    {"kind": "perturbed_exact", "noise_std": 0.05},
    {"kind": "perturbed_exact", "noise_std": 0.2},
    {"kind": "perturbed_exact", "noise_std": 0.5},
]
DEFAULTS = {
    "superset_size": 2048, "cache_seed": 0, "backend": "exact",
    "pool_per_shape": 256, "subset_size": 32, "scales": [8, 32, 128],
    "repetitions": 8, "projections": 32, "velocity_weight": 1.0,
    "alpha_values": [0.25, 0.5, 1.0, 2.0, 4.0],
    "joint_times": [0.0], "variance_times": [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
    "bandwidths": [0.05, 0.1, 0.2], "variance_bandwidth": 0.1,
    "max_cost_mib": 512, "candidates": DEFAULT_CANDIDATES,
}


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def settings(config):
    validate_gaussian_source(config)
    if config.get("coupling") != METHOD or config["model"]["point_dim"] != 2:
        raise ValueError("wasserstein_bank currently supports 2D point-set models only")
    raw = config.get("wasserstein_bank", {})
    unknown = raw.keys() - (DEFAULTS.keys() | {"cache", "cache_sha256", "selector", "candidate_id"})
    if unknown:
        raise ValueError("Unknown wasserstein_bank settings: " + ", ".join(sorted(unknown)))
    opts = {**DEFAULTS, **raw}
    for key in ("superset_size", "pool_per_shape", "subset_size", "repetitions", "projections"):
        _positive_int(opts[key], key, 2 if key in ("superset_size", "pool_per_shape") else 1)
    _positive_int(opts["cache_seed"], "cache_seed", 0)
    if opts["cache_seed"] >= 2**63 - 100003:
        raise ValueError("cache_seed must leave room for the independent holdout seed")
    _positive_int(config["data"]["n_points"], "n_points")
    if config["data"]["n_points"] > opts["superset_size"]:
        raise ValueError("n_points exceeds superset_size")
    if opts["pool_per_shape"] > opts["superset_size"]:
        raise ValueError("pool_per_shape exceeds superset_size")
    if opts["backend"] not in ("exact", "sliced"):
        raise ValueError("backend must be exact or sliced")
    if opts.get("selector", "joint") not in SELECTORS:
        raise ValueError("Unknown bank selector")
    for key in ("velocity_weight", "variance_bandwidth", "max_cost_mib"):
        if isinstance(opts[key], bool) or not math.isfinite(opts[key]) or opts[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("joint_times", "variance_times", "bandwidths", "scales", "alpha_values"):
        values = opts[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{key} must be a nonempty list of distinct numbers")
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in values):
            raise ValueError(f"Invalid {key}")
    for size in opts["scales"]:
        _positive_int(size, "scales")
    if any(not 0 <= t <= 1 for t in opts["joint_times"] + opts["variance_times"]):
        raise ValueError("Path times must lie in [0,1]")
    if len(opts["variance_times"]) < 2 or opts["variance_times"] != sorted(opts["variance_times"]):
        raise ValueError("variance_times must contain at least two increasing times")
    if any(h <= 0 for h in opts["bandwidths"]) or opts["variance_bandwidth"] not in opts["bandwidths"]:
        raise ValueError("Positive bandwidths must include variance_bandwidth")
    if any(alpha <= 0 for alpha in opts["alpha_values"]):
        raise ValueError("alpha_values must be positive")
    candidates = opts["candidates"]
    if not isinstance(candidates, list) or len(candidates) < 2 or candidates[0] != {"kind": "random"}:
        raise ValueError("Provide >=2 candidates; candidate 0 must be random for scale normalization")
    for candidate in candidates:
        kind = candidate.get("kind")
        allowed = {"random": {"kind"}, "exact": {"kind"},
                   "projection": {"kind", "angle"}, "perturbed_exact": {"kind", "noise_std"}}
        if kind not in allowed or set(candidate) != allowed[kind]:
            raise ValueError(f"Invalid candidate specification: {candidate}")
        for key in set(candidate) - {"kind"}:
            value = candidate[key]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError("Candidate parameters must be finite numbers")
        if kind == "perturbed_exact" and candidate["noise_std"] <= 0:
            raise ValueError("noise_std must be positive")
    if "candidate_id" in opts:
        _positive_int(opts["candidate_id"], "candidate_id", 0)
        if opts["candidate_id"] >= len(candidates):
            raise ValueError("candidate_id exceeds available candidates")
    dataset = config.get("dataset", "checkerboard")
    spec = dataset_spec(config, dataset)
    pool_size = opts["pool_per_shape"] * len(spec.get("shapes", [dataset]))
    if max(opts["scales"] + [opts["subset_size"]]) >= pool_size:
        raise ValueError("Subset/scale size must be smaller than the complete scoring pool (otherwise score is identically zero)")
    if any(c["kind"] in ("exact", "perturbed_exact") for c in candidates):
        if 8 * opts["superset_size"] ** 2 / 2**20 > opts["max_cost_mib"]:
            raise ValueError("Dense candidate assignment exceeds max_cost_mib; reduce superset_size")
    return opts


def dataset_spec(config, dataset):
    if dataset in ("checkerboard", "horse"):
        return original_dataset_spec(config, dataset)
    if dataset != "mixture2d":
        raise ValueError("dataset must be checkerboard, horse or mixture2d")
    shapes = config["data"].get("shapes", ["disk", "ring", "two_disks"])
    if not isinstance(shapes, list) or len(shapes) < 2 or len(set(shapes)) != len(shapes) or any(s not in SHAPES for s in shapes):
        raise ValueError("mixture2d requires >=2 distinct supported shapes")
    return {"dataset": dataset, "point_dim": 2, "shapes": shapes,
            "shape_weights": [1.0 / len(shapes)] * len(shapes), "sampler_version": 1}


def target_points(config, dataset, shape, count, rng):
    if dataset in ("checkerboard", "horse"):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(rng.integers(0, 2**63 - 1)))
            if dataset == "checkerboard":
                return sample_checkerboard(1, count, "cpu", torch.float32, config["data"]["grid_size"])[0].numpy()
            from train_horse import load_horse_mask, sample_horse
            return sample_horse(load_horse_mask("cpu", torch.float32), 1, count)[0].numpy()
    if shape == "cross":
        points = rng.uniform(-1, 1, (count, 2)) * [0.9, 0.08]
        vertical = rng.random(count) < 0.5
        points[vertical] = points[vertical, ::-1]
    else:
        angle = rng.uniform(0, 2 * math.pi, count)
        low, high = (0.55, 0.9) if shape == "ring" else (0.0, 0.28 if shape == "two_disks" else 0.85)
        radius = np.sqrt(rng.uniform(low**2, high**2, count))
        points = radius[:, None] * np.stack((np.cos(angle), np.sin(angle)), 1)
        if shape == "two_disks":
            points[:, 0] += np.where(rng.random(count) < 0.5, -0.55, 0.55)
    return points.astype(np.float32)


def draw_supersets(config, dataset, seed):
    opts, spec = settings(config), dataset_spec(config, dataset)
    rng = np.random.default_rng(seed)
    source = rng.standard_normal((opts["superset_size"], 2)).astype(np.float32)
    targets = np.stack([target_points(config, dataset, shape, len(source), rng)
                        for shape in spec.get("shapes", [dataset])])
    return source, targets


def candidate_permutation(source, target, candidate, rng):
    """Change assignment costs only; return original source/target indices."""
    kind, n = candidate["kind"], len(source)
    if kind == "random":
        return rng.permutation(n)
    if kind == "projection":
        direction = np.array([math.cos(candidate["angle"]), math.sin(candidate["angle"])])
        order = np.empty(n, dtype=np.int64)
        order[np.argsort(source @ direction, kind="stable")] = np.argsort(target @ direction, kind="stable")
        return order
    cost_source = source.astype(np.float64)
    if kind == "perturbed_exact":
        cost_source = cost_source + candidate["noise_std"] * rng.standard_normal(source.shape)
    rows, cols = linear_sum_assignment(cdist(cost_source, target, "sqeuclidean"))
    order = np.empty(n, dtype=np.int64)
    order[rows] = cols
    return order


def wasserstein_squared(a, b, backend="exact", directions=None):
    """Exact empirical W2^2, or explicitly labelled sliced W2^2."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if backend == "exact":
        value, log = ot.emd2(np.full(len(a), 1 / len(a)), np.full(len(b), 1 / len(b)),
                             cdist(a, b, "sqeuclidean"), numItermax=1000000,
                             check_marginals=True, log=True)
        if log.get("warning") or not np.isfinite(value):
            raise RuntimeError(f"Exact Wasserstein scoring did not converge: {log.get('warning')}")
        return float(value)
    if backend != "sliced" or directions is None:
        raise ValueError("Sliced Wasserstein requires fixed shared directions")
    # POT handles unequal numbers of uniformly weighted observations.
    return float(ot.sliced_wasserstein_distance(a, b, projections=directions) ** 2)


def velocity_variogram(position, velocity, bandwidth):
    """Row-normalized local pairwise variogram, per coordinate; not MMSE.

    Self-pairs excluded. Finite bandwidth includes variation of the mean field.
    This conditions on point position, not the model's entire input cloud.
    """
    distances = cdist(position, position, "sqeuclidean")
    logits = -distances / (2 * bandwidth**2)
    np.fill_diagonal(logits, -np.inf)
    logits -= logits.max(1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(1, keepdims=True)
    differences = cdist(velocity, velocity, "sqeuclidean")
    nearest = distances + np.eye(len(distances)) * 1e30
    return {"value": float((weights * differences).sum(1).mean() / (2 * velocity.shape[1])),
            "effective_neighbors": float((1 / np.square(weights).sum(1)).mean()),
            "coverage_3h": float((nearest.min(1) <= (3 * bandwidth)**2).mean())}


def _correlation(x, y, control=None):
    x, y = np.asarray(x), np.asarray(y)
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    if control is None:
        return float(spearmanr(x, y).statistic)
    design = np.stack((np.ones(len(x)), rankdata(control)), 1)
    left = rankdata(x) - design @ np.linalg.lstsq(design, rankdata(x), rcond=None)[0]
    right = rankdata(y) - design @ np.linalg.lstsq(design, rankdata(y), rcond=None)[0]
    if np.linalg.norm(left) < 1e-10 or np.linalg.norm(right) < 1e-10:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def integrated_curve(validation_by_time, value_key="mse_per_coordinate"):
    """Duration-normalized trapezoidal integral on a strictly increasing time grid.

    The result approximates a uniform-in-time expectation on the covered interval;
    it is not an unweighted mean of samples on a nonuniform time grid.
    """
    if len(validation_by_time) < 2:
        raise ValueError("A time integral requires at least two curve points")
    times = np.asarray([row["time"] for row in validation_by_time], dtype=np.float64)
    values = np.asarray([row[value_key] for row in validation_by_time], dtype=np.float64)
    if not np.isfinite(times).all() or not np.isfinite(values).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Curve values must be finite and times strictly increasing")
    return float(np.sum(np.diff(times) * (values[:-1] + values[1:]) / 2) / (times[-1] - times[0]))


def fixed_bank_t0_bayes_mse(source, velocities):
    """Exact t=0 per-coordinate Bayes risk for the current fixed-bank sampler.

    Shape is uniform and independent of iid point-index draws, and all shapes share
    the same source bank. Source-cloud context therefore cannot reveal shape. This
    is an endpoint result, not a positive-time or population learnability estimate.
    Repeated identical source coordinates are pooled rather than treated as IDs.
    """
    source = np.asarray(source, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    if source.ndim != 2 or velocities.ndim != 3 or velocities.shape[1:] != source.shape:
        raise ValueError("Expected source [M,D] and velocities [S,M,D]")
    if not len(source) or not len(velocities) or not np.isfinite(source).all() or not np.isfinite(velocities).all():
        raise ValueError("Source and velocities must be nonempty and finite")
    _, inverse, counts = np.unique(source, axis=0, return_inverse=True, return_counts=True)
    conditional_mean = np.zeros((len(counts), source.shape[1]), dtype=np.float64)
    np.add.at(conditional_mean, inverse, velocities.sum(axis=0))
    conditional_mean /= (len(velocities) * counts[:, None])
    return float(np.square(velocities - conditional_mean[inverse][None]).mean())


def score_banks(source, targets, permutations, opts, seed):
    """Use common scoring indices/subsets/directions for every candidate."""
    rng = np.random.default_rng(seed)
    n_shapes, m = targets.shape[:2]
    indices = rng.choice(m, opts["pool_per_shape"], replace=False)
    x = np.tile(source[indices], (n_shapes, 1)).astype(np.float64)
    scales = sorted(set(opts["scales"] + [opts["subset_size"]]))
    subsets = {s: [rng.choice(len(x), s, replace=False) for _ in range(opts["repetitions"])] for s in scales}
    directions = {}
    for dimension in (2, 4):
        projection = rng.standard_normal((dimension, opts["projections"]))
        directions[dimension] = projection / np.linalg.norm(projection, axis=0, keepdims=True)
    rows = []
    for candidate_id, mappings in enumerate(permutations):
        full_u = targets[np.arange(n_shapes)[:, None], mappings].astype(np.float64) - source.astype(np.float64)[None]
        u = full_u[:, indices].reshape(-1, source.shape[-1])
        def consistency(features, size):
            return float(np.mean([wasserstein_squared(features[part], features, opts["backend"],
                                                        directions[features.shape[1]]) for part in subsets[size]]))
        displacement = consistency(u, opts["subset_size"])
        joint_cache = {}
        def joint_score(size, t, alpha):
            key = (size, float(t), float(alpha))
            if key not in joint_cache:
                joint_cache[key] = consistency(np.concatenate((x + t * u, alpha * u), 1), size)
            return joint_cache[key]
        joint_scales = {}
        for size in scales:
            joint_scales[str(size)] = float(np.mean([
                joint_score(size, t, opts["velocity_weight"])
                for t in opts["joint_times"]]))
        alpha_audit = {str(float(alpha)): joint_score(opts["subset_size"], 0.0, alpha)
                       for alpha in opts["alpha_values"]}
        path_joint = [{"time": t, "score": joint_score(opts["subset_size"], t, opts["velocity_weight"])}
                      for t in opts["variance_times"]]
        path = []
        for t in opts["variance_times"]:
            state = x + t * u
            bandwidth_results = {}
            for h in opts["bandwidths"]:
                global_value = velocity_variogram(state, u, h)
                # Also report within-shape values, without claiming a variance decomposition.
                per_shape = [velocity_variogram(state[s*len(indices):(s+1)*len(indices)],
                                               u[s*len(indices):(s+1)*len(indices)], h)["value"]
                             for s in range(n_shapes)]
                bandwidth_results[str(h)] = {**global_value, "within_shape_mean": float(np.mean(per_shape))}
            path.append({"time": t, "bandwidths": bandwidth_results})
        integral = integrated_curve([{"time": p["time"], "value": p["bandwidths"][str(opts["variance_bandwidth"])]["value"]}
                                     for p in path], "value")
        mean_displacement = full_u.mean(axis=(0, 1))
        covariance_trace = float(np.square(full_u - mean_displacement).sum(-1).mean())
        rows.append({"candidate": candidate_id, "specification": opts["candidates"][candidate_id],
                     "transport": float(np.square(full_u).sum(-1).mean()),
                     "mean_displacement": mean_displacement.tolist(),
                     "displacement_covariance_trace": covariance_trace,
                     "t0_fixed_bank_bayes_mse_per_coordinate": fixed_bank_t0_bayes_mse(source, full_u),
                     "t0_floor_informative": bool(n_shapes > 1 or len(np.unique(source, axis=0)) < len(source)),
                     "displacement": displacement, "joint": joint_scales[str(opts["subset_size"])],
                     "joint_t0_by_alpha": alpha_audit,
                     "path_joint": integrated_curve(path_joint, "score"), "path_joint_by_time": path_joint,
                     "joint_by_scale": joint_scales, "variance": integral, "variance_path": path})
    # Same random-candidate reference for all candidates; no per-candidate normalization.
    reference = {str(s): max(rows[0]["joint_by_scale"][str(s)], 1e-12) for s in opts["scales"]}
    for row in rows:
        row["multiscale"] = float(np.mean([row["joint_by_scale"][str(s)] / reference[str(s)] for s in opts["scales"]]))
    return {"candidates": rows, "scale_reference_random": reference,
            "t0_floor_scope": "exact t=0, fixed finite bank, shared source, uniform shape sampling independent of point-index draws; not a positive-time or population floor",
            "t0_floor_single_shape_note": "zero for every candidate with one shape and distinct source coordinates, including a random finite lookup; not informative for learnability ranking",
            "path_joint_definition": "duration-normalized trapezoidal integral over variance_times, approximating uniform-time joint subset W2 squared",
            "alpha_audit_definition": "t=0 joint subset W2 squared with shared pool/subset/projection draws; exact B(alpha)/alpha^2 approaches displacement score at large alpha",
            "variogram_bandwidth_note": "finite-sample kernel bandwidth sensitivity; a small-bandwidth plateau does not identify irreducible ambiguity, and h or h^2 scaling requires additional dense-sample regularity assumptions",
            "correlations": {key: {"spearman_with_variogram": _correlation([r[key] for r in rows], [r["variance"] for r in rows]),
                                    "spearman_with_transport": _correlation([r[key] for r in rows], [r["transport"] for r in rows]),
                                    "partial_rank_controlling_transport": _correlation([r[key] for r in rows], [r["variance"] for r in rows], [r["transport"] for r in rows])}
                             for key in ("displacement", "joint", "path_joint", "multiscale")},
            "alpha_rank_correlations": {str(float(alpha)): {
                "spearman_with_transport": _correlation([r["joint_t0_by_alpha"][str(float(alpha))] for r in rows], [r["transport"] for r in rows]),
                "spearman_with_joint": _correlation([r["joint_t0_by_alpha"][str(float(alpha))] for r in rows], [r["joint"] for r in rows])}
                for alpha in opts["alpha_values"]}}


def select_candidates(report, seed):
    rows = report["candidates"]
    rng = np.random.default_rng(seed)
    return {selector: (int(rng.integers(len(rows))) if selector == "random"
                       else int(min(rows, key=lambda row: (row[selector], row["candidate"]))["candidate"]))
            for selector in SELECTORS}


def cache_spec(config, dataset):
    opts = settings(config)
    return {"format_version": 2, "implementation": "complete_wasserstein_bank_2d_v2",
            "dataset": dataset_spec(config, dataset),
            "settings": {key: opts[key] for key in DEFAULTS}}


def load_cache(config, dataset):
    opts = settings(config)
    path = Path(opts["cache"])
    if not path.is_file():
        raise FileNotFoundError(f"Wasserstein bank missing: {path}; run run_wasserstein_bank.py --stage prepare")
    digest = file_sha256(path)
    if opts.get("cache_sha256") not in (None, digest):
        raise ValueError("Wasserstein bank differs from checkpoint training cache")
    with np.load(path, allow_pickle=False) as archive:
        expected = {"source", "targets", "permutations", "validation_source", "validation_targets", "validation_permutations", "metadata"}
        if set(archive.files) != expected:
            raise ValueError("Invalid Wasserstein bank arrays")
        arrays = {key: archive[key] for key in expected - {"metadata"}}
        metadata = json.loads(str(archive["metadata"].item()))
    if metadata["spec"] != cache_spec(config, dataset):
        raise ValueError("Wasserstein bank/config mismatch; use a new cache path")
    count, shapes = opts["superset_size"], len(metadata["spec"]["dataset"].get("shapes", [dataset]))
    for prefix in ("", "validation_"):
        source, targets, mappings = (arrays[prefix + key] for key in ("source", "targets", "permutations"))
        if source.shape != (count, 2) or targets.shape != (shapes, count, 2) or source.dtype != np.float32 or targets.dtype != np.float32:
            raise ValueError("Invalid bank coordinates")
        if not np.isfinite(source).all() or not np.isfinite(targets).all():
            raise ValueError("Nonfinite bank coordinates")
        if mappings.shape != (len(opts["candidates"]), shapes, count) or not np.issubdtype(mappings.dtype, np.integer):
            raise ValueError("Invalid bank permutation shape/dtype")
        if not np.array_equal(np.sort(mappings, axis=-1), np.broadcast_to(np.arange(count), mappings.shape)):
            raise ValueError("Bank contains a non-bijective candidate")
    for selector, chosen in metadata["selected"].items():
        if selector not in SELECTORS or not isinstance(chosen, int) or not 0 <= chosen < len(opts["candidates"]):
            raise ValueError("Invalid selected bank index")
    if set(metadata["selected"]) != set(SELECTORS):
        raise ValueError("Missing selector results")
    return arrays, metadata, digest


def prepare(config, dataset):
    opts = settings(config)
    path = Path(opts["cache"])
    if path.exists():
        _, metadata, _ = load_cache(config, dataset)
        print(f"reused_wasserstein_bank={path}", flush=True)
        return path, metadata
    start = perf_counter()
    arrays, reports = {}, {}
    for prefix, offset in (("", 0), ("validation_", 100003)):
        source, targets = draw_supersets(config, dataset, opts["cache_seed"] + offset)
        mappings = []
        for index, candidate in enumerate(opts["candidates"]):
            print(f"bank_split={prefix or 'selection_'} candidate={index} {candidate}", flush=True)
            # Re-create one common noise stream at each amplitude so an ordered
            # cost-noise sweep changes its magnitude, not its realization.
            candidate_seed = 1500 if candidate["kind"] == "perturbed_exact" else 1000 + index
            rng = np.random.default_rng(opts["cache_seed"] + offset + candidate_seed)
            mappings.append(np.stack([candidate_permutation(source, target, candidate, rng) for target in targets]))
        mappings = np.stack(mappings)
        arrays.update({prefix + "source": source, prefix + "targets": targets, prefix + "permutations": mappings})
        reports[prefix or "selection"] = score_banks(source, targets, mappings, opts, opts["cache_seed"] + offset + 2000)
    selected = select_candidates(reports["selection"], opts["cache_seed"] + 3000)
    metadata = {"spec": cache_spec(config, dataset), "selected": selected, "reports": reports,
                "precompute_seconds": perf_counter() - start,
                "score_backend": "exact_empirical_w2_squared" if opts["backend"] == "exact" else "sliced_w2_squared",
                "variance_definition": "row-normalized point-position local velocity variogram per coordinate; self-pairs excluded; not full-cloud conditional variance",
                "holdout_definition": "independent source and target supersets; same shape modes and candidate families; no holdout-based selection",
                "marginals": "all candidates share unchanged finite empirical endpoints and uniform shape weights; population marginals are approximate",
                "multiscale_definition": "equal mean of joint subset scores divided by the same split's random-candidate scale reference",
                "cost_noise_definition": "perturbed_exact amplitudes share the same Gaussian cost-noise draws within each split; original endpoints remain unchanged",
                "source_sha256": {name: file_sha256(Path(__file__).parent / name) for name in ("wasserstein_bank.py", "data.py", "train_horse.py")}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays, metadata=np.array(json.dumps(metadata, allow_nan=False)))
    print(f"saved_wasserstein_bank={path} selected={selected} precompute_seconds={metadata['precompute_seconds']:.3f}", flush=True)
    return path, metadata


class WassersteinBankSampler:
    def __init__(self, config, dataset, device, dtype):
        start = perf_counter()
        self.config, self.dataset = config, dataset
        self.opts = settings(config)
        arrays, self.metadata, self.cache_sha256 = load_cache(config, dataset)
        self.device, self.dtype = torch.device(device), dtype
        self.selector = "candidate" if "candidate_id" in self.opts else self.opts.get("selector", "joint")
        self.selected = self.opts["candidate_id"] if self.selector == "candidate" else self.metadata["selected"][self.selector]
        self.arrays = {key: torch.as_tensor(value, device=device, dtype=(torch.long if "permutations" in key else dtype))
                       for key, value in arrays.items()}
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.setup_seconds = perf_counter() - start

    @torch.no_grad()
    def sample(self, batch_size, *, generator=None, validation=False):
        _positive_int(batch_size, "batch_size")
        prefix = "validation_" if validation else ""
        source, targets, mappings = (self.arrays[prefix + key] for key in ("source", "targets", "permutations"))
        shapes = torch.randint(len(targets), (batch_size,), device=self.device, generator=generator)
        index = torch.randint(len(source), (batch_size, self.config["data"]["n_points"]), device=self.device, generator=generator)
        destination = mappings[self.selected, shapes[:, None], index]
        return source[index], targets[shapes[:, None], destination]

    def validation_curve(self, model, batch_size, seed):
        # Isolated RNG + eval mode: validation cannot change training draws/dropout.
        generator = torch.Generator(device=self.device).manual_seed(seed)
        source, target = self.sample(batch_size, generator=generator, validation=True)
        was_training = model.training
        model.eval()
        result = []
        with torch.no_grad():
            for t in self.opts["variance_times"]:
                time = source.new_full((batch_size, 1, 1), t)
                loss = (model(source + t * (target - source), time) - (target - source)).square().mean().item()
                result.append({"time": t, "mse_per_coordinate": loss})
        model.train(was_training)
        return result

    def details(self):
        return {"cache_sha256": self.cache_sha256, "selector": self.selector,
                "selection_mode": "explicit_candidate" if self.selector == "candidate" else "score_selector",
                "selected_candidate": self.selected, "candidate_specification": self.opts["candidates"][self.selected],
                "precompute_seconds": self.metadata["precompute_seconds"],
                "cache_setup_seconds": self.setup_seconds,
                "timing_scope": "training_seconds is online training including scheduled validation; precomputation reported separately",
                "score_backend": self.metadata["score_backend"], "marginals": self.metadata["marginals"],
                "validation": self.metadata["holdout_definition"], "variance_definition": self.metadata["variance_definition"]}
