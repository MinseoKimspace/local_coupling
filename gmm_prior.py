"""Ordinary full-covariance GMM priors fitted to training-only 2D samples.

This comparison changes the prior, not merely the coupling: EM learns all
component weights, means, and full covariance matrices.  It does not balance
component counts, whiten source coordinates, or inspect evaluation targets.
The saved parameters, rather than a fitted sklearn object, define inference.
"""

import hashlib
import math
from numbers import Real
import warnings

import torch


MODE = "gmm_prior"
_RESOLVED_KEYS = {"weights", "means", "covariances"}
_FIT_KEYS = {"reference_points", "seed", "n_init", "max_iter", "tol", "reg_covar"}


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value < 2**63:
        raise ValueError(f"{name} must be an integer >= {minimum} and < 2**63")
    return value


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _tensor(value, shape, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"anchor_flow.{name} must be a finite list with shape {shape}")
    try:
        result = torch.tensor(value, device="cpu", dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"anchor_flow.{name} must be a finite list with shape {shape}") from error
    if result.shape != shape or not torch.isfinite(result).all().item():
        raise ValueError(f"anchor_flow.{name} must be finite and have exactly {shape} shape")
    if not torch.isfinite(result.float()).all().item():
        raise ValueError(f"anchor_flow.{name} must be representable as float32")
    return result


def _parameters(parameters, k):
    if not isinstance(parameters, dict) or set(parameters) != _RESOLVED_KEYS:
        raise ValueError("GMM parameters require exactly weights, means, and covariances together")
    weights = _tensor(parameters["weights"], (k,), "weights")
    if (weights <= 0).any().item() or (weights.float() <= 0).any().item():
        raise ValueError("anchor_flow.weights must be strictly positive, including in float32")
    total = weights.sum().item()
    if not math.isclose(total, 1., rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("anchor_flow.weights must be normalized to sum to 1")
    means = _tensor(parameters["means"], (k, 2), "means")
    covariances = _tensor(parameters["covariances"], (k, 2, 2), "covariances")
    if not torch.allclose(covariances, covariances.transpose(-1, -2), rtol=1e-6, atol=1e-10):
        raise ValueError("anchor_flow.covariances must be symmetric")
    # Canonicalize negligible roundoff in symmetric covariance matrices only.
    covariances = .5 * (covariances + covariances.transpose(-1, -2))
    for covariance_dtype in (torch.float64, torch.float32):
        _, info = torch.linalg.cholesky_ex(covariances.to(covariance_dtype))
        if (info != 0).any().item():
            raise ValueError("anchor_flow.covariances must be positive definite in float64 and float32")
    # Once normalized, retain machine-roundoff differences from 1 so repeated
    # config validation/binding is idempotent (important for cache identity).
    if abs(total - 1.) > 1e-12:
        weights = weights / total
    return {"weights": weights.tolist(), "means": means.tolist(),
            "covariances": covariances.tolist()}


def settings(config):
    """Normalize this prior's settings, returning None for other prior modes."""
    value = config.get("anchor_flow")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("anchor_flow must be a mapping")
    if value.get("mode") != MODE:
        return None
    unknown = value.keys() - ({"mode"} | _FIT_KEYS | _RESOLVED_KEYS)
    if unknown:
        raise ValueError(f"Unsupported gmm_prior settings: {', '.join(sorted(map(str, unknown)))}")
    if config.get("coupling") != "nsot":
        raise ValueError("gmm_prior requires nsot coupling")
    n = _integer(config["data"]["n_points"], "n_points")
    k = _integer(config.get("num_regions"), "num_regions")
    if k > n or config["model"]["point_dim"] != 2 or config.get("dtype") != "float32":
        raise ValueError("gmm_prior requires float32 2D with 1 <= K <= N")
    result = {"mode": MODE,
              "reference_points": _integer(value.get("reference_points", 4096),
                                           "anchor_flow.reference_points", max(2, k)),
              "seed": _integer(value.get("seed", 0), "anchor_flow.seed", 0),
              "n_init": _integer(value.get("n_init", 5), "anchor_flow.n_init"),
              "max_iter": _integer(value.get("max_iter", 300), "anchor_flow.max_iter"),
              "tol": _positive(value.get("tol", 1e-4), "anchor_flow.tol"),
              "reg_covar": _positive(value.get("reg_covar", 1e-6), "anchor_flow.reg_covar")}
    present = value.keys() & _RESOLVED_KEYS
    if present and present != _RESOLVED_KEYS:
        raise ValueError("gmm_prior requires weights, means, and covariances together")
    if present:
        result.update(_parameters({key: value[key] for key in _RESOLVED_KEYS}, k))
    return result


def cache_spec(config):
    """Stable fit inputs; cache metadata stores the resolved parameters."""
    opts = settings(config)
    return None if opts is None else {key: value for key, value in opts.items() if key not in _RESOLVED_KEYS}


@torch.no_grad()
def fit(config, dataset):
    """Fit ordinary EM to the SAME seeded CPU training reference as anchors.

    sklearn is a preparation-only dependency: resolved configs can sample and
    run inference without importing it.  Caller RNG and thread settings are
    restored on success and failure.  A nonconverged best restart is rejected,
    not silently saved as a valid trained prior.
    """
    opts = settings(config)
    if opts is None:
        raise ValueError("GMM fitting requires gmm_prior mode")
    if dataset not in ("checkerboard", "horse"):
        raise ValueError("GMM priors support checkerboard or horse training data")
    if _RESOLVED_KEYS <= opts.keys():
        return {key: opts[key] for key in _RESOLVED_KEYS}, {
            "parameters_reused": True, "fit_performed": False}
    try:
        import sklearn
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.mixture import GaussianMixture
        from threadpoolctl import threadpool_limits
    except ImportError as error:
        raise ImportError("Preparing gmm_prior requires scikit-learn; install requirements.txt first") from error
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
        reference_array = reference[0].contiguous().numpy()
        reference_hash = hashlib.sha256(reference_array.tobytes()).hexdigest()
        points = reference_array.astype("float64")
        model = GaussianMixture(n_components=config["num_regions"], covariance_type="full",
                                n_init=opts["n_init"], max_iter=opts["max_iter"], tol=opts["tol"],
                                reg_covar=opts["reg_covar"], init_params="kmeans",
                                random_state=opts["seed"] % 2**32)
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(points)
            if not model.converged_:
                raise RuntimeError("gmm_prior EM best restart did not converge; increase max_iter "
                                   "or inspect the training reference before preparing its NSOT cache")
            average_log_likelihood = float(model.score(points))
            aic, bic = float(model.aic(points)), float(model.bic(points))
        parameters = _parameters({"weights": model.weights_.tolist(), "means": model.means_.tolist(),
                                  "covariances": model.covariances_.tolist()}, config["num_regions"])
        details = {"parameters_reused": False, "fit_performed": True,
                   "implementation": "sklearn.mixture.GaussianMixture", "sklearn_version": sklearn.__version__,
                   "covariance_type": "full", "init_params": "kmeans", "learned_weights": True,
                   "learned_means": True, "learned_covariances": True,
                   "dataset": dataset, "training_reference_points": opts["reference_points"],
                   "training_reference_seed": opts["seed"], "training_reference_sha256": reference_hash,
                   "n_init": opts["n_init"], "max_iter": opts["max_iter"], "tol": opts["tol"],
                   "reg_covar": opts["reg_covar"], "converged": bool(model.converged_),
                   "n_iter": int(model.n_iter_), "lower_bound": float(model.lower_bound_),
                   "average_log_likelihood": average_log_likelihood,
                   "total_log_likelihood": average_log_likelihood * len(points), "aic": aic, "bic": bic}
        if not all(math.isfinite(details[key]) for key in (
                "lower_bound", "average_log_likelihood", "total_log_likelihood", "aic", "bic")):
            raise RuntimeError("gmm_prior EM returned nonfinite fit diagnostics")
        return parameters, details
    finally:
        torch.set_num_threads(threads)


def bind(config, parameters):
    """Bind fitted parameters, refusing to replace an existing resolved prior."""
    opts = settings(config)
    if opts is None:
        raise ValueError("Binding GMM parameters requires gmm_prior mode")
    resolved = _parameters(parameters, config["num_regions"])
    if _RESOLVED_KEYS <= opts.keys() and any(opts[key] != resolved[key] for key in _RESOLVED_KEYS):
        raise ValueError("gmm_prior parameters differ from the resolved training prior")
    config["anchor_flow"] = {**opts, **resolved}
    return resolved


def sample_source_with_components(config, batch_size, *, device, dtype, generator=None):
    """Draw pointwise iid learned-weight/full-covariance source and labels."""
    opts = settings(config)
    _integer(batch_size, "batch_size")
    if opts is None or dtype != torch.float32:
        raise ValueError("GMM sampling requires float32 gmm_prior settings")
    if not _RESOLVED_KEYS <= opts.keys():
        raise ValueError("gmm_prior requires resolved training parameters; prepare/load its NSOT cache first")
    n = config["data"]["n_points"]
    weights = torch.tensor(opts["weights"], device=device, dtype=dtype)
    means = torch.tensor(opts["means"], device=device, dtype=dtype)
    factors = torch.linalg.cholesky(torch.tensor(opts["covariances"], device=device, dtype=dtype))
    labels = torch.multinomial(weights, batch_size * n, replacement=True, generator=generator).reshape(batch_size, n)
    noise = torch.randn(batch_size, n, 2, device=device, dtype=dtype, generator=generator)
    residuals = torch.matmul(factors[labels], noise.unsqueeze(-1)).squeeze(-1)
    return means[labels] + residuals, labels


def experiment_details(config):
    """Metadata explicitly distinguishing this prior from anchors/original NSOT."""
    opts = settings(config)
    if opts is None:
        return {}
    return {"anchor_flow": opts, "anchor_flow_implementation": "general_gmm_prior_nsot_v1",
            "implementation": "general_gmm_prior_nsot_component_hybrid_v1",
            "source_prior": "product over N points of sum_k pi_k Normal(mu_k, Sigma_k)",
            "source_coordinates": "training-fitted weighted full-covariance GMM; no whitening",
            "prior_component_sampling": "pointwise iid categorical learned weights; random component counts",
            "prior_center_fit": "ordinary EM learns weights, means, and full covariance matrices from training reference only",
            "inference_source": "fresh iid points from SAME saved GMM parameters; no evaluation target access or refit",
            "inference_anchors": "saved training GMM parameters, not target-cloud-specific centers",
            "local_pairing": "fixed full-superset exact point OT; iid pair subsampling with replacement; no TG patches",
            "target_endpoint": "original undeformed target-superset points",
            "path": "x_t=(1-t)*source+t*paired_target",
            "conditional_velocity": "paired_target-component_centered_hybrid_source",
            "hybrid": "mu[original_component]+sqrt(1-beta)*(cached_source-mu[original_component])+sqrt(beta)*chol(Sigma[original_component])*fresh_gaussian",
            "beta_one": "residual refreshed within original component; coarse source/target association remains, NOT independent coupling",
            "finite_bank_note": "finite reused point superset approximates the declared population GMM",
            "paper_variant": "experimental general-GMM source plus covariance-preserving component hybrid; not original Gaussian NSOT",
            "quality_guarantee": False,
            "limitation": "prior comparison, not coupling-only comparison; fixed 2D training shape; no 1-NFE quality guarantee"}
