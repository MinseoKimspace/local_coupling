"""Offline target-only proportional stratification; no source perturbation.

The fixed grid is independent of MS-W2's data-dependent equal-mass patches.
Horse grid edges follow native pixel boundaries, so probabilities are exact
foreground areas under the original continuous, within-pixel target sampler.
"""

import numpy as np

IMPLEMENTATION = "target_foreground_grid_largest_remainder_v1"


def sampling_settings(raw=None):
    raw = {"mode": "iid"} if raw is None else raw
    if not isinstance(raw, dict) or raw.keys() - {"mode", "bins"}:
        raise ValueError("target_sampling must contain only mode and optional bins")
    mode = raw.get("mode", "iid")
    if mode == "iid" and "bins" not in raw:
        return {"mode": "iid"}
    if mode != "stratified":
        raise ValueError("target_sampling mode must be iid (without bins) or stratified")
    bins = raw.get("bins", 16)
    if isinstance(bins, bool) or not isinstance(bins, int) or not 1 <= bins <= 128:
        raise ValueError("target_sampling bins must be an integer in [1,128]")
    return {"mode": mode, "bins": bins}


def largest_remainder(probabilities, size, rng):
    """Sum-preserving floor/ceil allocation; seeded random tie-breaking only."""
    p = np.asarray(probabilities, dtype=np.float64)
    if (p.ndim != 1 or not len(p) or not np.isfinite(p).all() or (p < 0).any()
            or not np.isclose(p.sum(), 1.0)
            or isinstance(size, bool) or not isinstance(size, int) or size < 1):
        raise ValueError("Expected probabilities summing to one and positive integer size")
    desired = size * p / p.sum()
    counts = np.floor(desired).astype(np.int64)
    remainder = size - int(counts.sum())
    # Do not favor top-left spatial bins when fractional parts are tied.
    ties = rng.random(len(p))
    order = np.lexsort((ties, -(desired - counts)))
    counts[order[:remainder]] += 1
    if counts.sum() != size or (counts[p == 0] != 0).any():
        raise RuntimeError("Invalid proportional allocation")
    return counts


class TargetStrata:
    def __init__(self, config, dataset, bins=16):
        sampling_settings({"mode": "stratified", "bins": bins})
        self.dataset, self.bins = dataset, bins
        if dataset == "checkerboard":
            grid = config["data"]["grid_size"]
            if isinstance(grid, bool) or not isinstance(grid, int) or grid < 1 or bins % grid:
                raise ValueError("For checkerboard, bins must be a multiple of data.grid_size")
            indices = np.arange(grid)
            active = (indices[:, None] + indices[None, :]) % 2 == 0
            self.mask = np.repeat(np.repeat(active, bins // grid, axis=0), bins // grid, axis=1)
        elif dataset == "horse":
            from skimage.data import horse
            self.mask = ~horse()
        else:
            raise ValueError("Expected checkerboard or horse")
        self.height, self.width = self.mask.shape
        self.scale = float(max(self.height, self.width))
        if bins > min(self.mask.shape):
            raise ValueError("More strata than target pixels")
        self.row_edges = np.linspace(0, self.height, bins + 1, dtype=np.int64)
        self.col_edges = np.linspace(0, self.width, bins + 1, dtype=np.int64)
        rows, cols = np.nonzero(self.mask)
        labels = self._pixel_labels(rows, cols)
        self.pixels = [np.column_stack((rows[labels == i], cols[labels == i])) for i in range(bins**2)]
        areas = np.array([len(p) for p in self.pixels], dtype=np.float64)
        self.probabilities = areas / areas.sum()

    def _pixel_labels(self, rows, cols):
        r = np.searchsorted(self.row_edges[1:], rows, side="right")
        c = np.searchsorted(self.col_edges[1:], cols, side="right")
        return r * self.bins + c

    def labels(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Expected finite [M,2] coordinates")
        px = points[:, 0] * self.scale / 2 + self.width / 2
        py = self.height / 2 - points[:, 1] * self.scale / 2
        inside = (px >= 0) & (px < self.width) & (py >= 0) & (py < self.height)
        labels = np.full(len(points), -1, dtype=np.int64)
        ids = np.flatnonzero(inside)
        rows, cols = np.floor(py[ids]).astype(int), np.floor(px[ids]).astype(int)
        foreground = self.mask[rows, cols]
        labels[ids[foreground]] = self._pixel_labels(rows[foreground], cols[foreground])
        return labels

    def sample(self, size, seed):
        # Independent local RNG: no global NumPy/Torch training RNG is advanced.
        rng = np.random.default_rng(seed)
        counts = largest_remainder(self.probabilities, size, rng)
        points = np.empty((size, 2), dtype=np.float32)
        offset = 0
        for region, count in enumerate(counts):
            if count == 0:
                continue
            pixels = self.pixels[region]
            selected = pixels[rng.integers(len(pixels), size=int(count))]
            # Draw coordinates inside original target pixels, as sample_horse
            # does. This is target sampling, NOT additive noise on bank points.
            uv = rng.random((count, 2))
            block = np.column_stack(((selected[:, 1] + uv[:, 0] - self.width / 2) * 2 / self.scale,
                                     (self.height / 2 - selected[:, 0] - uv[:, 1]) * 2 / self.scale))
            block = block.astype(np.float32)
            # Extremely rare float32 rounding can land on another pixel/bin.
            # Resample those coordinates, rather than clipping/projecting points.
            invalid = self.labels(block) != region
            while invalid.any():
                uv = rng.random((int(invalid.sum()), 2))
                chosen = selected[invalid]
                block[invalid, 0] = (chosen[:, 1] + uv[:, 0] - self.width / 2) * 2 / self.scale
                block[invalid, 1] = (self.height / 2 - chosen[:, 0] - uv[:, 1]) * 2 / self.scale
                invalid = self.labels(block) != region
            points[offset:offset + count] = block
            offset += count
        points = points[rng.permutation(size)]
        observed = np.bincount(self.labels(points), minlength=self.bins**2)
        if not np.array_equal(observed, counts):
            raise RuntimeError("Sampled target does not satisfy its mass allocation")
        return points

    def report(self, points):
        labels = self.labels(points)
        counts = np.bincount(labels[labels >= 0], minlength=self.bins**2)
        mass = counts / len(labels)
        outside = float((labels < 0).mean())
        errors = mass - self.probabilities
        return {"bins": self.bins, "size": len(labels), "counts": counts.tolist(),
                "target_probabilities": self.probabilities.tolist(),
                "expected_counts": (len(labels) * self.probabilities).tolist(),
                "row_edges_pixels": self.row_edges.tolist(), "column_edges_pixels": self.col_edges.tolist(),
                "mask_shape": list(self.mask.shape), "leakage": outside,
                "region_mass_tv": float((np.abs(errors).sum() + outside) / 2),
                "max_abs_region_mass_error": float(np.abs(errors).max()),
                "max_abs_count_error": float(np.abs(counts - len(labels) * self.probabilities).max()),
                "definition": "pooled foreground-region probabilities over ALL points, plus one outside bin"}
