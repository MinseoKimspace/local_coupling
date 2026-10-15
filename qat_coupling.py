"""Paper-based 2D Quantile AlignTree coupling, with the project's backbone.

Reference: https://arxiv.org/html/2609.03443v1, Appendix B, Algorithms 1--2.
The tree splits the target on its maximum-variance coordinate at its mean,
down to singleton leaves. Identical target coordinates form one terminal leaf
instead of an impossible geometric split. Conditional Gaussian quantiles mirror
the target split masses. Sampling remains continuous: this is NOT a finite
Gaussian bank, extra noise added to bank points, or a local OT assignment.

Only the coupling is reproduced here, not the paper's full training experiment.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from data import validate_gaussian_source
from nsot import dataset_spec, file_sha256

METHOD = "qat"
FORMAT_VERSION = 1
IMPLEMENTATION = "qat_max_variance_mean_singleton_cdf_v1"
PAPER = "https://arxiv.org/html/2609.03443v1"
SPLIT_STRATEGY = "maximum empirical variance axis; target mean threshold; <= left, > right"
TARGET_COORDINATES = "unchanged original sampled points; no barycentric projection"


def _integer(value, name, low=1, high=2**63 - 1):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}]")
    return value


def settings(config):
    if config.get("coupling") != METHOD:
        raise ValueError("QAT preparation requires coupling: qat")
    value = config.get(METHOD, {})
    if not isinstance(value, dict):
        raise ValueError("qat must be a mapping")
    unknown = value.keys() - {"cache", "superset_size", "cache_seed", "cache_sha256"}
    if unknown:
        raise ValueError("Unsupported qat settings: " + ", ".join(sorted(map(str, unknown))))
    size = _integer(value.get("superset_size", 10000), "QAT superset_size", high=10000)
    seed = _integer(value.get("cache_seed", 0), "QAT cache_seed", low=0)
    cache = value.get("cache")
    if not isinstance(cache, (str, Path)) or not str(cache).strip() or str(cache) == "auto":
        raise ValueError("QAT requires a concrete cache path")
    if config["model"]["point_dim"] != 2:
        raise ValueError("2D QAT requires point_dim=2")
    _integer(config["data"]["n_points"], "QAT n_points", high=size)
    _integer(config.get("seed", 0), "QAT training seed", low=0)
    digest = value.get("cache_sha256")
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                               or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("QAT cache_sha256 must be a lowercase SHA256 hex string")
    validate_gaussian_source(config)
    return {"cache": str(cache), "superset_size": size, "cache_seed": seed}


def _array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def build_quantile_tree(target):
    """Return per-target-row (CDF lower, CDF upper, leaf labels, diagnostics).

    Bounds are float64 in Gaussian-CDF space, not target coordinates. The
    product of interval widths equals that leaf's target count divided by M.
    Target row order and coordinates are never changed. A stack avoids Python's
    recursion-depth limit on unusually unbalanced mean-split trees.
    """
    target = np.asarray(target)
    if target.ndim != 2 or target.shape[1] != 2 or not len(target) \
            or target.dtype.kind not in "fiu" or not np.isfinite(target).all():
        raise ValueError("QAT requires finite nonempty real [M,2] target points")
    points = target.astype(np.float64)
    count = len(points)
    lower, upper = np.empty((count, 2), dtype=np.float64), np.empty((count, 2), dtype=np.float64)
    labels = np.empty(count, dtype=np.int64)
    pending = [(np.arange(count), np.zeros(2), np.ones(2), 0)]
    leaves, duplicates, max_depth, max_size = 0, 0, 0, 0
    while pending:
        indices, lo, hi, depth = pending.pop()
        block = points[indices]
        max_depth = max(max_depth, depth)
        if len(indices) == 1 or np.all(block == block[0]):
            lower[indices], upper[indices], labels[indices] = lo, hi, leaves
            leaves += 1
            duplicates += int(len(indices) > 1)
            max_size = max(max_size, len(indices))
            continue
        means = block.mean(axis=0)
        variances = np.square(block - means).mean(axis=0)
        if not np.isfinite(means).all() or not np.isfinite(variances).all():
            raise ValueError("QAT target magnitude exceeds float64 tree arithmetic")
        axis = int(np.argmax(variances))  # Deterministic first-axis tie break.
        left_mask = block[:, axis] <= means[axis]
        left, right = indices[left_mask], indices[~left_mask]
        if not len(left) or not len(right):
            raise ValueError("QAT mean split cannot separate distinct target coordinates at float64 precision")
        cut = lo[axis] + (len(left) / len(indices)) * (hi[axis] - lo[axis])
        if not lo[axis] < cut < hi[axis]:
            raise ValueError("QAT source quantile interval exhausted float64 precision")
        left_hi, right_lo = hi.copy(), lo.copy()
        left_hi[axis], right_lo[axis] = cut, cut
        # Push right first to make leaf IDs follow deterministic left-first order.
        pending.append((right, right_lo, hi.copy(), depth + 1))
        pending.append((left, lo.copy(), left_hi, depth + 1))
    mass = np.prod(upper - lower, axis=1)
    expected = np.bincount(labels)[labels] / count
    if not np.allclose(mass, expected, rtol=1e-10, atol=1e-14):
        raise ValueError("QAT source/target leaf mass consistency check failed")
    return lower, upper, labels, {
        "leaf_count": leaves, "singleton_leaf_count": leaves - duplicates,
        "duplicate_group_count": duplicates, "max_leaf_size": max_size,
        "max_tree_depth": max_depth, "split_count": leaves - 1,
        "leaf_mass_max_abs_error": float(np.max(np.abs(mass - expected))),
    }


def draw_target(config, dataset):
    """Draw byte-identical targets to MS-W2 at the same M/cache seed.

    The old endpoint-drawing helper consumes its Gaussian draw before the target
    draw. That temporary Gaussian array is discarded, never cached or reused by
    QAT. This preserves the existing baseline's target sampling RNG exactly.
    """
    from nsot import SOLVER, draw_supersets
    opts = settings(config)
    baseline = copy.deepcopy(config)
    baseline["coupling"] = "nsot"
    baseline["nsot"] = {**opts, "beta": 0.0, "solver": SOLVER}
    return draw_supersets(baseline, dataset)[1]


def cache_spec(config, dataset):
    opts = settings(config)
    return {"format_version": FORMAT_VERSION, "implementation": IMPLEMENTATION,
            "superset_size": opts["superset_size"], "cache_seed": opts["cache_seed"],
            "dataset_spec": dataset_spec(config, dataset), "split_strategy": SPLIT_STRATEGY,
            "termination": "singleton or identical-coordinate target group",
            "source_distribution": "continuous standard Gaussian; no finite source bank",
            "target_coordinates": TARGET_COORDINATES}


def load_cache(config, dataset):
    """Return (target, cdf_lower, cdf_upper, leaf_labels, metadata, file SHA256)."""
    opts = settings(config)
    path = Path(opts["cache"])
    if not path.is_file():
        raise FileNotFoundError(f"QAT cache missing: {path}; prepare the coupling study first")
    digest = file_sha256(path)
    expected_hash = config[METHOD].get("cache_sha256")
    if expected_hash is not None and expected_hash != digest:
        raise ValueError("QAT cache SHA256 differs from the checkpoint training cache")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"target", "cdf_lower", "cdf_upper", "leaf_labels", "metadata"}:
            raise ValueError("QAT cache has missing/extra arrays; no finite source bank is supported")
        target, lower, upper, labels = (archive[k] for k in
                                        ("target", "cdf_lower", "cdf_upper", "leaf_labels"))
        metadata = json.loads(str(archive["metadata"].item()))
    if not isinstance(metadata, dict):
        raise ValueError("QAT metadata must be a mapping")
    for key, value in cache_spec(config, dataset).items():
        if metadata.get(key) != value:
            raise ValueError(f"QAT cache/config mismatch: {key}")
    size = opts["superset_size"]
    if target.shape != (size, 2) or target.dtype != np.float32 or not np.isfinite(target).all() \
            or lower.shape != target.shape or upper.shape != target.shape \
            or lower.dtype != np.float64 or upper.dtype != np.float64 \
            or not np.isfinite(lower).all() or not np.isfinite(upper).all() \
            or np.any(lower < 0) or np.any(upper > 1) or np.any(lower >= upper) \
            or labels.shape != (size,) or labels.dtype != np.int64:
        raise ValueError("QAT cache contains invalid target points or source CDF bounds/labels")
    if metadata.get("endpoint_sha256") != {"target": _array_hash(target)}:
        raise ValueError("QAT target hash does not match stored points")
    if metadata.get("tree_sha256") != {"cdf_lower": _array_hash(lower), "cdf_upper": _array_hash(upper),
                                       "leaf_labels": _array_hash(labels)}:
        raise ValueError("QAT tree hashes do not match stored arrays")
    # Validate the actual coupling as well as file hashes: a valid-shaped but
    # unrelated tree must not silently change which source region targets use.
    expected_lower, expected_upper, expected_labels, diagnostics = build_quantile_tree(target)
    if not np.array_equal(lower, expected_lower) or not np.array_equal(upper, expected_upper) \
            or not np.array_equal(labels, expected_labels):
        raise ValueError("QAT tree does not match target mean/variance construction")
    for key, value in diagnostics.items():
        if metadata.get(key) != value:
            raise ValueError(f"QAT tree diagnostics mismatch: {key}")
    for name in ("precompute_seconds", "sampling_seconds", "tree_seconds"):
        seconds = metadata.get(name)
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) \
                or not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f"QAT cache contains invalid {name}")
    return target, lower, upper, labels, metadata, digest


def prepare(config, dataset):
    """Create exclusively, or validate/reuse an existing cache without overwrite."""
    opts, spec = settings(config), cache_spec(config, dataset)
    path = Path(opts["cache"])
    if path.exists():
        *_, metadata, digest = load_cache(config, dataset)
        print(f"reused_qat_cache={path} sha256={digest}", flush=True)
        return path, metadata
    start = perf_counter()
    target = draw_target(config, dataset)
    sampling_seconds = perf_counter() - start
    tree_start = perf_counter()
    lower, upper, labels, diagnostics = build_quantile_tree(target)
    tree_seconds = perf_counter() - tree_start
    metadata = {
        **spec, **diagnostics, "paper": PAPER,
        "paper_setting": "Appendix B Algorithms 1-2 coupling; project PointSetTransformer training, not full paper reproduction",
        "endpoint_sha256": {"target": _array_hash(target)},
        "tree_sha256": {"cdf_lower": _array_hash(lower), "cdf_upper": _array_hash(upper),
                        "leaf_labels": _array_hash(labels)},
        "precompute_seconds": perf_counter() - start, "sampling_seconds": sampling_seconds,
        "tree_seconds": tree_seconds,
        "precompute_timing_scope": "shared endpoint draw and QAT tree construction; excludes cache IO/loading and training",
        "marginals": "continuous standard Gaussian source and uniform empirical target; floating-point inverse-CDF sampling",
        "source_sha256": {name: file_sha256(Path(__file__).parent / name) for name in
                          ("qat_coupling.py", "nsot.py", "data.py", "train_horse.py")},
        "environment": {"torch": str(torch.__version__), "numpy": np.__version__},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, target=target, cdf_lower=lower, cdf_upper=upper,
                            leaf_labels=labels, metadata=np.array(json.dumps(metadata, allow_nan=False)))
    print(f"saved_qat_cache={path} sha256={file_sha256(path)} "
          f"leaves={diagnostics['leaf_count']} precompute_seconds={metadata['precompute_seconds']:.3f}", flush=True)
    return path, metadata


class QATPairSampler:
    def __init__(self, config, dataset, device, dtype):
        settings(config)
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("QAT supports float32 or float64 model coordinates")
        target, lower, upper, _, self.metadata, self.cache_sha256 = load_cache(config, dataset)
        self.target = torch.as_tensor(target, device=device, dtype=dtype)
        # Bounds and inverse normal CDF deliberately stay float64 even when the
        # network uses float32, so small/tail intervals do not collapse.
        self.cdf_lower = torch.as_tensor(lower, device=device, dtype=torch.float64)
        self.cdf_upper = torch.as_tensor(upper, device=device, dtype=torch.float64)
        self.n_points = config["data"]["n_points"]
        self.source_seed = (config.get("seed", 0) + 2) % 2**63
        self.source_generator = torch.Generator(device=device).manual_seed(self.source_seed)

    @torch.no_grad()
    def sample(self, batch_size, *, generator=None):
        _integer(batch_size, "batch_size")
        indices = torch.randint(len(self.target), (batch_size, self.n_points),
                                device=self.target.device, generator=generator)
        lower, upper = self.cdf_lower[indices], self.cdf_upper[indices]
        uniform = torch.rand(lower.shape, device=lower.device, dtype=torch.float64,
                             generator=self.source_generator)
        # Avoid the PRNG's representable 0 and rounding to a CDF boundary. These
        # are machine-precision endpoint guards, not a user-tuned tail cutoff.
        eps = torch.finfo(torch.float64).eps
        uniform.clamp_(min=eps / 2, max=1 - eps / 2)
        quantile = lower + uniform * (upper - lower)
        quantile = torch.maximum(torch.nextafter(lower, upper),
                                 torch.minimum(quantile, torch.nextafter(upper, lower)))
        source = torch.special.ndtri(quantile).to(dtype=self.target.dtype)
        return source, self.target[indices]

    def details(self):
        return {**self.metadata, "cache_sha256": self.cache_sha256,
                "plan_sampling": "iid target rows with replacement; fresh continuous conditional Gaussian for every pair",
                "source_sampling_seed": self.source_seed,
                "source_rng": "independent generator; does not advance pair-index or time RNG",
                "inference_source": "fresh iid standard Gaussian; no tree, bank or target labels"}
