"""Generic patchwise bijections and source-neighbor graphs for TG guidance.

These helpers only sample valid correspondences or construct a fixed graph;
they do not define a coupling objective or optimize permutations. Graph
construction is dense and intended for small experimental point clouds.
"""

import numpy as np


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not minimum <= int(value) < 2**63):
        raise ValueError(f"{name} must be an integer >= {minimum} and < 2**63")
    return int(value)


def _points(value, name):
    try:
        raw = np.asarray(value)
        if raw.dtype.kind == "c":
            raise ValueError("Complex coordinates are not supported")
        result = raw.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite N x D array") from error
    if result.ndim != 2 or min(result.shape) < 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite nonempty N x D array")
    return result


def _labels(value, name, n, k):
    labels = np.asarray(value)
    if (labels.shape != (n,) or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels >= k)):
        raise ValueError(f"{name} must be an integer length-N array in [0, K)")
    return labels.astype(np.int64, copy=False)


def random_valid_permutation(source_labels, target_labels, k, rng):
    """Uniform bijection per patch, without using NumPy's global RNG."""
    k = _integer(k, "k", 1)
    source_labels = np.asarray(source_labels)
    if source_labels.ndim != 1 or len(source_labels) < 1 or k > len(source_labels):
        raise ValueError("Patch labels require a nonempty cloud and 1 <= K <= N")
    n = len(source_labels)
    source_labels = _labels(source_labels, "source_labels", n, k)
    target_labels = _labels(target_labels, "target_labels", n, k)
    source_groups = [np.flatnonzero(source_labels == patch) for patch in range(k)]
    target_groups = [np.flatnonzero(target_labels == patch) for patch in range(k)]
    if any(len(a) != len(b) for a, b in zip(source_groups, target_groups)):
        raise ValueError("Source and target patch counts differ")
    permutation = np.empty(n, dtype=np.int32)
    for source, target in zip(source_groups, target_groups):
        permutation[source] = rng.permutation(target)
    return permutation


def build_source_edges(source, neighbors):
    """Fixed undirected kNN union, unique lexicographic int64 edges.

    Cross-patch edges are retained. Equal-distance ties follow source index
    order, so graph construction is deterministic for an indexed cloud.
    """
    source = _points(source, "source")
    neighbors = _integer(neighbors, "neighbors", 1)
    n = len(source)
    if neighbors >= n:
        raise ValueError("neighbors must be smaller than the cloud's point count")
    with np.errstate(over="ignore", invalid="ignore"):
        differences = source[:, None, :] - source[None, :, :]
        distance_squared = np.einsum("ijd,ijd->ij", differences, differences)
    if not np.isfinite(distance_squared).all():
        raise ValueError("Source coordinates exceed the supported numerical range")
    np.fill_diagonal(distance_squared, np.inf)
    nearest = np.argsort(distance_squared, axis=1, kind="stable")[:, :neighbors]
    edges = np.stack((np.repeat(np.arange(n), neighbors), nearest.reshape(-1)), axis=1)
    edges.sort(axis=1)
    return np.unique(edges, axis=0).astype(np.int64, copy=False)
