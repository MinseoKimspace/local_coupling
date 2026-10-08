"""Optional fixed, prior-preserving directional noise for 2D anchor NSOT.

For component k, this module fits a small linear map from the original source
to its OT-paired target.  Noise is reduced along the source direction whose
image changes the thin target's normal coordinate, and increased orthogonally.
The total noise budget is unchanged.  No neural-network Jacobian is computed.

The kernel is c + sqrt(I-S)(z-c) + sigma*sqrt(S)*epsilon.  With fixed S, an
isotropic component N(c, sigma^2 I) is invariant at the population level.
Fitting S on a finite reused cache does NOT make that cache's marginal exact.
Matrices must be fixed per component, not recomputed from the current point.
"""

import math
from numbers import Real
import re

import numpy as np
import torch

import anchor_flow


_KEYS = {"artifact", "strength", "ridge", "min_points",
         "min_target_anisotropy", "min_normal_r2", "artifact_sha256"}
_MATRIX_TOLERANCE = 1e-8


def _real(value, name, *, lower=None, upper=None, strictly_positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if strictly_positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if (lower is not None and value < lower) or (upper is not None and value > upper):
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return value


def _options(value):
    if not isinstance(value, dict):
        raise ValueError("nsot.directional_hybrid must be a mapping")
    unknown = value.keys() - _KEYS
    if unknown:
        raise ValueError("Unsupported nsot.directional_hybrid settings: "
                         + ", ".join(sorted(map(str, unknown))))
    artifact = value.get("artifact")
    if not isinstance(artifact, str) or not artifact.strip():
        raise ValueError("nsot.directional_hybrid.artifact must be a nonempty path string")
    result = {"artifact": artifact,
              "strength": _real(value.get("strength", 0.9), "directional strength", lower=0, upper=1),
              "ridge": _real(value.get("ridge", 0.001), "directional ridge", strictly_positive=True),
              "min_target_anisotropy": _real(value.get("min_target_anisotropy", 1.5),
                                             "directional min_target_anisotropy", lower=1),
              "min_normal_r2": _real(value.get("min_normal_r2", 0.25),
                                     "directional min_normal_r2", lower=0, upper=1)}
    minimum = value.get("min_points", 32)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 3:
        raise ValueError("directional min_points must be an integer >= 3")
    result["min_points"] = minimum
    if "artifact_sha256" in value:
        digest = value["artifact_sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError("directional artifact_sha256 must contain exactly 64 hexadecimal digits")
        result["artifact_sha256"] = digest.lower()
    return result


def settings(config):
    """Return normalized opt-in options, or None without the optional block.

    Baseline NSOT receives no new defaults and follows its existing path.  The
    original nsot.beta remains the shared mean refresh budget; it is not copied
    into this block or overridden by directional settings.
    """
    nsot = config.get("nsot", {})
    if not isinstance(nsot, dict):
        raise ValueError("nsot must be a mapping")
    if "directional_hybrid" not in nsot:
        return None
    if config.get("coupling") != "nsot":
        raise ValueError("directional_hybrid requires coupling: nsot")
    prior = anchor_flow.settings(config)
    if prior is None or prior["mode"] != "anchor_prior":
        raise ValueError("directional_hybrid requires anchor_flow.mode: anchor_prior")
    if config.get("dtype") != "float32" or config.get("model", {}).get("point_dim") != 2:
        raise ValueError("directional_hybrid requires float32 2D experiments")
    _real(nsot.get("beta"), "NSOT beta", lower=0, upper=1)
    return _options(nsot["directional_hybrid"])


def _points(value, name, *, nonempty=True):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite [N,2] array") from error
    if result.ndim != 2 or result.shape[1] != 2 or (nonempty and not len(result)) \
            or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {'nonempty ' if nonempty else ''}[N,2] array")
    return result


def fit(source, paired_target, components, centers, *, sigma, beta, options):
    """Fit one constant S per original Gaussian component on CPU float64.

    Inputs are never modified.  Target covariance selects its narrow PCA
    normal n.  A centered ridge regression J pulls that direction back to the
    source as a=J.T@n.  The normal's in-sample regression R^2 is a conservative
    fit gate, not a generalization guarantee.  Isotropic/unstable/small groups
    retain exactly the original S=beta*I kernel.

    Returns JSON-compatible matrices and per-component diagnostic reports.
    Paths and hashes in options are validation/provenance only; no files are
    read or written here.
    """
    opts = _options(options)
    _real(sigma, "sigma", strictly_positive=True)
    beta = _real(beta, "NSOT beta", lower=0, upper=1)
    source = _points(source, "source")
    target = _points(paired_target, "paired_target")
    centers = _points(centers, "centers")
    labels = np.asarray(components)
    if target.shape != source.shape or labels.shape != (len(source),) \
            or not np.issubdtype(labels.dtype, np.integer) \
            or (labels < 0).any() or (labels >= len(centers)).any():
        raise ValueError("paired_target must match source and components must be integers [N] in [0,K)")
    identity = np.eye(2, dtype=np.float64)
    delta = opts["strength"] * min(beta, 1 - beta)
    matrices, reports = [], []

    for component in range(len(centers)):
        mask = labels == component
        count = int(mask.sum())
        matrix = beta * identity
        report = {"component": component, "count": count, "J": None,
                  "source_mean": None, "target_mean": None,
                  "source_eigenvalues": None, "target_eigenvalues": None,
                  "target_anisotropy": None, "normal_r2": None,
                  "normal_source_direction": None, "beta_eigenvalues": [beta, beta],
                  "fallback": True, "reason": "insufficient_points"}
        if count >= opts["min_points"]:
            x, y = source[mask], target[mask]
            with np.errstate(over="ignore", invalid="ignore"):
                x_mean, y_mean = x.mean(0), y.mean(0)
                x, y = x - x_mean, y - y_mean
                cxx, cyy, cyx = x.T @ x / count, y.T @ y / count, y.T @ x / count
            if np.isfinite(x_mean).all() and np.isfinite(y_mean).all():
                report["source_mean"], report["target_mean"] = x_mean.tolist(), y_mean.tolist()
            if not all(np.isfinite(covariance).all() for covariance in (cxx, cyy, cyx)):
                report["reason"] = "nonfinite_statistics"
            else:
                source_eigs = np.linalg.eigvalsh(cxx)
                target_eigs, target_vectors = np.linalg.eigh(cyy)
                if np.isfinite(source_eigs).all() and np.isfinite(target_eigs).all():
                    report["source_eigenvalues"] = source_eigs.tolist()
                    report["target_eigenvalues"] = target_eigs.tolist()
                if not np.isfinite(source_eigs).all() or not np.isfinite(target_eigs).all():
                    report["reason"] = "nonfinite_statistics"
                elif source_eigs[-1] <= 0 or source_eigs[0] <= source_eigs[-1] * 1e-10:
                    report["reason"] = "degenerate_source"
                elif target_eigs[-1] <= 0 or target_eigs[0] <= target_eigs[-1] * 1e-10:
                    report["reason"] = "degenerate_target"
                else:
                    anisotropy = float(target_eigs[-1] / target_eigs[0])
                    report["target_anisotropy"] = anisotropy
                    scale = float(np.trace(cxx) / 2)
                    # solve(...).T is Cyx @ inverse(...), without an explicit inverse.
                    jacobian = np.linalg.solve(cxx + opts["ridge"] * scale * identity, cyx.T).T
                    normal = target_vectors[:, 0]
                    sensitive = jacobian.T @ normal
                    normal_residual = y @ normal - x @ sensitive
                    normal_r2 = float(1 - np.mean(normal_residual**2) / target_eigs[0])
                    report["J"] = jacobian.tolist()
                    report["normal_r2"] = normal_r2
                    sensitive_norm = float(np.linalg.norm(sensitive))
                    if not np.isfinite(jacobian).all() or not math.isfinite(normal_r2) \
                            or not math.isfinite(sensitive_norm):
                        report["J"], report["normal_r2"] = None, None
                        report["reason"] = "nonfinite_statistics"
                    elif anisotropy < opts["min_target_anisotropy"]:
                        report["reason"] = "target_near_isotropic"
                    elif normal_r2 < opts["min_normal_r2"]:
                        report["reason"] = "poor_normal_fit"
                    elif sensitive_norm <= max(float(np.linalg.norm(jacobian)), 1.0) * 1e-12:
                        report["reason"] = "degenerate_sensitive_direction"
                    elif delta == 0:
                        report["reason"] = "zero_directional_budget"
                    else:
                        direction = sensitive / sensitive_norm
                        matrix = ((beta + delta) * identity
                                  - 2 * delta * np.outer(direction, direction))
                        # Symmetrization avoids serialization of numerical asymmetry.
                        matrix = (matrix + matrix.T) / 2
                        report.update(normal_source_direction=direction.tolist(),
                                      beta_eigenvalues=[beta - delta, beta + delta],
                                      fallback=False, reason="directional")
        matrices.append(matrix.tolist())
        reports.append(report)
    return {"matrices": matrices, "components": reports}


def factors(matrices):
    """Validate [...,2,2] kernels and return sqrt(I-S), sqrt(S) on CPU.

    Tiny boundary roundoff is clipped only after the PSD and upper-bound
    checks; genuinely invalid or asymmetric matrices are rejected.  The
    returned symmetric factors satisfy A A.T + B B.T = I to float64 precision.
    """
    try:
        value = np.asarray(matrices, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("directional matrices must be finite [...,2,2] arrays") from error
    if value.ndim < 2 or value.shape[-2:] != (2, 2) or not np.isfinite(value).all():
        raise ValueError("directional matrices must be finite [...,2,2] arrays")
    transpose = np.swapaxes(value, -1, -2)
    if not np.allclose(value, transpose, rtol=0, atol=_MATRIX_TOLERANCE):
        raise ValueError("directional matrices must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh((value + transpose) / 2)
    if (eigenvalues < -_MATRIX_TOLERANCE).any() or (eigenvalues > 1 + _MATRIX_TOLERANCE).any():
        raise ValueError("directional matrices must satisfy 0 <= S <= I")
    eigenvalues = np.clip(eigenvalues, 0, 1)
    shrink = (eigenvectors * np.sqrt(1 - eigenvalues)[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)
    refresh = (eigenvectors * np.sqrt(eigenvalues)[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)
    return shrink, refresh


def apply(source, centers, *, sigma, shrink, refresh, noise):
    """Apply already-validated fixed factors using torch batch matrix products.

    Vectors have shape [...,2], factors [...,2,2], with broadcastable leading
    dimensions.  All tensors share device and floating dtype.  This hot path
    intentionally performs no finite-value or spectrum checks/GPU sync; those
    belong in factors() once, before transferring the fixed factors to device.
    """
    sigma = _real(sigma, "sigma", strictly_positive=True)
    tensors = (source, centers, shrink, refresh, noise)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise ValueError("directional apply requires torch tensors")
    if any(value.ndim < 1 or value.shape[-1] != 2 for value in (source, centers, noise)) \
            or any(value.ndim < 2 or value.shape[-2:] != (2, 2) for value in (shrink, refresh)):
        raise ValueError("directional vectors must have shape [...,2] and factors [...,2,2]")
    if not source.is_floating_point() or any(value.dtype != source.dtype or value.device != source.device
                                           for value in tensors[1:]):
        raise ValueError("directional tensors must share a floating dtype and device")
    try:
        torch.broadcast_shapes(source.shape[:-1], centers.shape[:-1], noise.shape[:-1],
                               shrink.shape[:-2], refresh.shape[:-2])
    except RuntimeError as error:
        raise ValueError("directional tensor leading dimensions must be broadcastable") from error
    residual = torch.matmul(shrink, (source - centers).unsqueeze(-1)).squeeze(-1)
    fresh = torch.matmul(refresh, noise.unsqueeze(-1)).squeeze(-1)
    return centers + residual + sigma * fresh
