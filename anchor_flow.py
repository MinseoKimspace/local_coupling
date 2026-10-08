"""Experimental fixed, training-derived 2D anchor Gaussian-mixture priors.

These are deliberately separate from baseline Hard Bank TG and Gaussian NSOT. The anchor-prior
variant changes the *joint cloud prior*: every point independently chooses one
of K fixed, training-derived Gaussian components.  It does not impose fixed
component counts, whiten the mixture, or use an evaluation target's anchors.

Both training and inference use the same saved centers and pointwise iid
mixture law. Neither target-cloud-specific inference anchors nor whitening
are used. A finite reused cache approximates this population prior.
"""

import math
from numbers import Real

import torch


MODES = {"anchor_prior"}
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
        raise ValueError("anchor_flow.mode must be anchor_prior")
    allowed = _COMMON_KEYS | _PRIOR_KEYS
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"Unsupported anchor_flow settings: {', '.join(sorted(map(str, unknown)))}")
    if config.get("coupling") not in ("target_guided_cached", "nsot"):
        raise ValueError("anchor_flow experiments require target_guided_cached or nsot coupling")
    n = _integer(config["data"]["n_points"], "n_points")
    k = _integer(config.get("num_regions"), "num_regions")
    if (k > n or config["model"]["point_dim"] != 2 or config.get("dtype") != "float32"):
        raise ValueError("anchor_flow experiments require float32 2D with 1 <= K <= N")
    result = {"mode": mode, "sigma": _sigma(value.get("sigma"))}
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
    if opts is None:
        return torch.randn(batch_size, n, config["model"]["point_dim"],
                           device=device, dtype=dtype, generator=generator)
    return sample_source_with_components(config, batch_size, device=device, dtype=dtype,
                                         generator=generator)[0]


def sample_source_with_components(config, batch_size, *, device, dtype, generator=None):
    """Draw the same iid GMM as sample_source, retaining original labels.

    NSOT records these labels BEFORE OT. Centered hybrid noise must use the
    originally drawn component, not a nearest-center classification afterward.
    Component labels are drawn first, then the Gaussian tensor, exactly as in
    the existing anchor-prior sampler.
    """
    opts = settings(config)
    _integer(batch_size, "batch_size")
    if opts is None or dtype != torch.float32:
        raise ValueError("sample_source_with_components requires float32 anchor_prior settings")
    if "centers" not in opts:
        raise ValueError("anchor_prior requires resolved training centers; prepare/load its coupling cache first")
    n = config["data"]["n_points"]
    centers = torch.tensor(opts["centers"], device=device, dtype=dtype)
    labels = torch.randint(config["num_regions"], (batch_size, n), device=device, generator=generator)
    noise = torch.randn(batch_size, n, 2, device=device, dtype=dtype, generator=generator)
    return centers[labels] + opts["sigma"] * noise, labels


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
    if config["coupling"] == "nsot":
        shared.update(
            implementation="nsot_anchor_prior_component_hybrid_v1",
            target_endpoint="original undeformed target-superset points; iid cached-pair subsampling with replacement",
            local_pairing="fixed full-superset exact point OT; iid pair subsampling with replacement; no TG patches",
            finite_bank_note="finite point superset with component-centered jitter approximates the declared population GMM",
            source_coordinates="fixed training-derived anchor GMM before exact OT; component-centered hybrid noise during training",
            anchor_flow_implementation="nsot_anchor_prior_component_hybrid_v1",
            conditional_velocity="paired_target-component_centered_hybrid_source",
            hybrid="c[original_component]+sqrt(1-beta)*(cached_source-c[original_component])+sigma*sqrt(beta)*fresh_gaussian",
            beta_one="residual is fully refreshed within the original component; source/target still share coarse component information, NOT independent coupling",
            paper_variant="experimental fixed anchor-GMM source plus component-centered hybrid; not the original NSOT Gaussian kernel",
            limitation="changed prior and changed hybrid kernel; experimental NSOT extension, not original-NSOT marginal or quality guarantees",
        )
        if config.get("nsot", {}).get("directional_hybrid") is not None:
            shared.update(
                implementation="nsot_anchor_prior_directional_hybrid_v1",
                anchor_flow_implementation="nsot_anchor_prior_directional_hybrid_v1",
                source_coordinates="same fixed anchor GMM and exact OT bank; fixed per-component directional stationary hybrid during training",
                conditional_velocity="paired_target-actual_directional_hybrid_source",
                hybrid="c[k]+sqrt(I-S[k])@(cached_source-c[k])+sigma*sqrt(S[k])@fresh_gaussian",
                paper_variant="experimental same-anchor-prior directional hybrid; not original NSOT and not model conditioning",
                limitation="population prior preservation does not guarantee finite-cache marginals, learned output density or thin-structure quality",
            )
    return shared


def variant_suffix(config):
    opts = settings(config)
    if opts is None:
        return ""
    suffix = opts["mode"]
    if config.get("nsot", {}).get("directional_hybrid") is not None:
        suffix += "_directional"
    return suffix
