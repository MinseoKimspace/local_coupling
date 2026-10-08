"""Experimental 2D anchor priors and anchor-waypoint flow paths.

These are deliberately separate from baseline Hard Bank TG.  The anchor-prior
variant changes the *joint cloud prior*: every point independently chooses one
of K fixed, training-derived Gaussian components.  It does not impose fixed
component counts, whiten the mixture, or use an evaluation target's anchors.

The waypoint variant keeps standard-Gaussian endpoints and uses an exact
quartic interpolant through a noisy, assigned patch centroid at t=1/2.
Its derivative, not the straight-path displacement, is the FM label.  A
smooth bump preserves the straight path's endpoint velocities. The path can
still bend and overshoot (endpoint coefficients can be negative); this is an
experiment, not a low-NFE or quality guarantee.
"""

import math
from numbers import Real

import torch


MODES = {"anchor_prior", "anchor_waypoint"}
_COMMON_KEYS = {"mode", "sigma"}
_PRIOR_KEYS = {"reference_points", "seed", "centers"}


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value < 2**63:
        raise ValueError(f"{name} must be an integer >= {minimum} and < 2**63")
    return value


def _sigma(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError("anchor_flow.sigma must be positive and finite")
    return float(value)


def _centers(value, k):
    """Validate configuration data on CPU, never in the per-update path."""
    if not isinstance(value, (list, tuple)):
        raise ValueError("anchor_flow.centers must be a K x 2 list")
    try:
        points = torch.tensor(value, device="cpu", dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("anchor_flow.centers must be a finite K x 2 list") from error
    if points.shape != (k, 2) or not torch.isfinite(points).all().item():
        raise ValueError("anchor_flow.centers must be finite and have exactly [num_regions, 2] shape")
    # Reject values that would overflow the experiment's required float32.
    if not torch.isfinite(points.float()).all().item():
        raise ValueError("anchor_flow.centers must be representable as float32")
    return points.tolist()


def settings(config):
    """Validate and normalize the optional top-level experimental settings.

    Invoke at setup/config-read time.  Resolved centers are optional before
    cache preparation, but mandatory when sampling an anchor prior.
    """
    value = config.get("anchor_flow")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("anchor_flow must be a mapping")
    mode = value.get("mode")
    if mode not in MODES:
        raise ValueError("anchor_flow.mode must be anchor_prior or anchor_waypoint")
    allowed = _COMMON_KEYS | (_PRIOR_KEYS if mode == "anchor_prior" else set())
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unsupported anchor_flow settings: {', '.join(sorted(map(str, unknown)))}")
    if config.get("coupling") != "target_guided_cached":
        raise ValueError("anchor_flow experiments require target_guided_cached coupling")
    n = _integer(config["data"]["n_points"], "n_points")
    k = _integer(config.get("num_regions"), "num_regions")
    if (k > n or config["model"]["point_dim"] != 2 or config.get("dtype") != "float32"):
        raise ValueError("anchor_flow experiments require float32 2D with 1 <= K <= N")
    result = {"mode": mode, "sigma": _sigma(value.get("sigma"))}
    if mode == "anchor_prior":
        result.update(reference_points=_integer(value.get("reference_points", 4096),
                                                "anchor_flow.reference_points", k),
                      seed=_integer(value.get("seed", 0), "anchor_flow.seed", 0))
        if "centers" in value:
            result["centers"] = _centers(value["centers"], k)
    return result


def cache_spec(config):
    """Stable input hyperparameters; resolved centers are stored separately."""
    opts = settings(config)
    return None if opts is None else {key: value for key, value in opts.items() if key != "centers"}


@torch.no_grad()
def fit_prior_centers(config, dataset):
    """Fit K fixed centroids using only the original training distribution.

    A fresh CPU reference sample is partitioned with the existing one-pass
    balanced FPS/exact-OT rule, followed by actual patch means, NOT refinement.
    Existing resolved centers are returned unchanged after validation.  Caller
    RNG and CPU thread count are restored, including on failure; CUDA RNG is
    never touched.
    """
    opts = settings(config)
    if opts is None or opts["mode"] != "anchor_prior":
        raise ValueError("fit_prior_centers requires anchor_prior mode")
    if dataset not in ("checkerboard", "horse"):
        raise ValueError("anchor priors support checkerboard or horse training data")
    if "centers" in opts:
        return opts["centers"]
    from coupling import balanced_target_partition

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator(device="cpu").manual_seed(opts["seed"]).get_state())
            if dataset == "horse":
                from train_horse import load_horse_mask, sample_horse
                reference = sample_horse(load_horse_mask("cpu", torch.float32),
                                         1, opts["reference_points"])
            else:
                from data import sample_checkerboard
                reference = sample_checkerboard(1, opts["reference_points"], "cpu", torch.float32,
                                                config["data"]["grid_size"])
            _, _, centers, _ = balanced_target_partition(reference, config["num_regions"],
                                                         solver="exact_batched")
            return _centers(centers[0].tolist(), config["num_regions"])
    finally:
        torch.set_num_threads(threads)


def bind_prior_centers(config, centers):
    """Embed the exact training centers into a config/checkpoint snapshot.

    Never silently replace explicitly configured or previously bound centers.
    This mutates only config['anchor_flow']; no cache or file is written.
    """
    opts = settings(config)
    if opts is None or opts["mode"] != "anchor_prior":
        raise ValueError("bind_prior_centers requires anchor_prior mode")
    resolved = _centers(centers, config["num_regions"])
    if "centers" in opts and opts["centers"] != resolved:
        raise ValueError("anchor_flow.centers differ from the resolved training prior")
    config["anchor_flow"] = {**opts, "centers": resolved}
    return resolved


def sample_source(config, batch_size, *, device, dtype, generator=None):
    """Draw fresh source clouds, including the exact old draw when absent.

    Prior mode uses independent categorical(1/K) component draws for all B*N
    points.  It does not condition on target labels or enforce balanced counts.
    Reusing a finite cache remains an empirical approximation to this law.
    """
    opts = settings(config)
    _integer(batch_size, "batch_size")
    n = config["data"]["n_points"]
    if opts is None or opts["mode"] == "anchor_waypoint":
        return torch.randn(batch_size, n, config["model"]["point_dim"],
                           device=device, dtype=dtype, generator=generator)
    if dtype != torch.float32:
        raise ValueError("anchor_prior sampling requires float32")
    if "centers" not in opts:
        raise ValueError("anchor_prior requires resolved training centers; prepare/load its TG cache first")
    centers = torch.tensor(opts["centers"], device=device, dtype=dtype)
    labels = torch.randint(config["num_regions"], (batch_size, n), device=device, generator=generator)
    noise = torch.randn(batch_size, n, 2, device=device, dtype=dtype, generator=generator)
    return centers[labels] + opts["sigma"] * noise


def _cloud_tensor(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[-1] != 2 or min(value.shape) < 1:
        raise ValueError(f"{name} must be a nonempty [B, N, 2] tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")


def path_and_velocity(source, target, t, *, waypoint=None):
    """Return conditional path and its exact time derivative.

    Tensor validation here is shape/device/dtype only: no per-update GPU
    synchronization or .item() finite checks.  Training owns finite-loss checks.
    A waypoint is sampled ONCE for a path, not freshly at different time probes.
    """
    _cloud_tensor(source, "source")
    if (not isinstance(target, torch.Tensor) or target.shape != source.shape
            or target.device != source.device or target.dtype != source.dtype):
        raise ValueError("source and target must share [B, N, 2] shape, device and dtype")
    if isinstance(t, torch.Tensor):
        if t.device != source.device or t.dtype != source.dtype:
            raise ValueError("t must share the source device and dtype")
        if t.ndim and (t.ndim != 3 or t.shape[-1] != 1
                       or t.shape[0] not in (1, source.shape[0])
                       or t.shape[1] not in (1, source.shape[1])):
            raise ValueError("t must be scalar or broadcastable [B, 1, 1]/[B, N, 1]")
    elif isinstance(t, bool) or not isinstance(t, Real) or not math.isfinite(t):
        raise ValueError("t must be a finite number or a time tensor")
    point = (1 - t) * source + t * target
    velocity = target - source
    if waypoint is not None:
        if (not isinstance(waypoint, torch.Tensor) or waypoint.shape != source.shape
                or waypoint.device != source.device or waypoint.dtype != source.dtype):
            raise ValueError("waypoint must share the source shape, device and dtype")
        offset = waypoint - 0.5 * (source + target)
        # b(0)=b(1)=b'(0)=b'(1)=0, b(1/2)=1. Unlike the
        # quadratic bump this does not arbitrarily triple the initial
        # pairing-averaged velocity, a confound for one-step Euler sampling.
        bump = 16 * t**2 * (1 - t)**2
        bump_derivative = 32 * t * (1 - t) * (1 - 2 * t)
        point = point + bump * offset
        velocity = velocity + bump_derivative * offset
    return point, velocity


def make_waypoint(config, patch_centers, *, generator=None):
    """Noisy actual patch centroids, already aligned with SOURCE point order.

    Configuration is fully validated at setup.  This hot-path helper avoids
    repeating center validation or GPU synchronization on each update.
    """
    value = config.get("anchor_flow")
    if not isinstance(value, dict) or value.get("mode") != "anchor_waypoint":
        raise ValueError("make_waypoint requires anchor_waypoint mode")
    sigma = _sigma(value.get("sigma"))
    _cloud_tensor(patch_centers, "patch_centers")
    noise = torch.randn(patch_centers.shape, device=patch_centers.device,
                        dtype=patch_centers.dtype, generator=generator)
    return patch_centers + sigma * noise


def experiment_details(config):
    """Accurate JSON metadata that overrides baseline Gaussian/linear claims."""
    opts = settings(config)
    if opts is None:
        return {}
    shared = {"anchor_flow": opts, "anchor_flow_implementation": "anchor_flow_2d_v1",
              "target_endpoint": "original sampled target cloud; no target deformation",
              "local_pairing": "fresh uniform random within-patch bijection",
              "quality_guarantee": False,
              "finite_bank_note": "reused cloud bank is an empirical approximation to the declared source law"}
    if opts["mode"] == "anchor_prior":
        shared.update(
            source_coordinates="iid uniform-K Gaussian mixture around fixed training-derived centroids; no whitening",
            inference_source="fresh iid points from the SAME saved fixed-centroid mixture; no evaluation target access",
            source_prior="product over N points of (1/K) sum_k Normal(c_k, sigma^2 I_2)",
            prior_component_sampling="pointwise iid categorical; counts are random, not fixed per component",
            prior_center_fit="original training reference sample; one balanced FPS/exact-OT partition; actual means; no refinement",
            inference_anchors="saved training prior centers, not target-cloud-specific centers",
            path="x_t=(1-t)*source+t*paired_target",
            conditional_velocity="paired_target-source",
            limitation="changed prior; not a coupling-only comparison; fixed 2D shape experiment, not a multi-shape anchor generator",
        )
    else:
        shared.update(
            source_coordinates="unmodified iid standard Gaussian draws",
            inference_source="fresh iid standard Gaussian; no cache, anchors or patches",
            source_prior="product over N points of Normal(0, I_2)",
            waypoint="assigned actual target-patch centroid + sigma * iid Normal(0, I_2)",
            waypoint_sampling="fresh once per training pair visit; fixed when evaluating its conditional path",
            path="x_t=(1-t)*source+t*paired_target+16*t^2*(1-t)^2*(waypoint-(source+paired_target)/2)",
            conditional_velocity="paired_target-source+32*t*(1-t)*(1-2*t)*(waypoint-(source+paired_target)/2)",
            endpoint_velocity="same paired_target-source at t=0 and t=1; zero bump derivative at both endpoints",
            waypoint_time=0.5,
            limitation="quartic interpolant can bend/overshoot and have negative endpoint coefficients; low-NFE benefit is unknown",
        )
    return shared


def variant_suffix(config):
    opts = settings(config)
    return "" if opts is None else opts["mode"]
