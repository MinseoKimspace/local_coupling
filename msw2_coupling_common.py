"""Fixed spatial measurements for read-only MS-W2 coupling diagnostics."""

import csv
import json
from pathlib import Path

import numpy as np

from horse_regions import HorseRegions


def as_points(points):
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    array = np.asarray(points, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] != 2 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Expected nonempty finite [...,2] points")
    return array.reshape(-1, 2)


class SpatialRegions:
    """Unconditional fixed-region masses and a partitioning histogram.

    Semantic regions can overlap. Only grid_hist (including its outside bin)
    is a probability partition. Coordinates and mask area match metrics.py.
    """

    def __init__(self, config, dataset, mask=None, *, grid_bins=16, roi_file=None):
        if isinstance(grid_bins, bool) or not isinstance(grid_bins, int) or grid_bins < 2:
            raise ValueError("grid_bins must be an integer >=2")
        self.dataset, self.grid_bins = dataset, grid_bins
        self.grid_size = int(config["data"].get("grid_size", 4))
        if dataset == "checkerboard":
            indices = np.arange(self.grid_size)
            self.mask = (indices[:, None] + indices[None, :]) % 2 == 0
            self.horse = None
        elif dataset == "horse":
            if mask is None:
                raise ValueError("Horse spatial measurements require the training mask")
            self.mask = (mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)).astype(bool)
            self.horse = HorseRegions(self.mask, roi_file)
        else:
            raise ValueError("Expected checkerboard or horse")
        self.region_names = ["foreground", "leakage"]
        expected = [1., 0.]
        if self.horse is None:
            for row, col in np.argwhere(self.mask):
                self.region_names.append(f"cell_{row}_{col}")
                expected.append(1. / self.mask.sum())
        else:
            for roi in self.horse.regions:
                self.region_names.append(roi["name"])
                expected.append(roi["expected_mass"])
        self.expected_mass = np.asarray(expected, dtype=np.float64)
        self.expected_grid_mass = self._expected_grid(grid_bins)
        self.path_grid_bins = 32
        self.path_grid_range = (-4., 4.)

    def support(self, points):
        array = as_points(points)
        if self.horse is None:
            inside = (np.abs(array) <= 1).all(1)
            bounded = np.clip(array, -1., 1.)
            col = np.floor((bounded[:, 0] + 1) * self.grid_size / 2).astype(np.int64).clip(0, self.grid_size - 1)
            row = np.floor((1 - bounded[:, 1]) * self.grid_size / 2).astype(np.int64).clip(0, self.grid_size - 1)
            return inside & ((row + col) % 2 == 0)
        height, width = self.mask.shape
        scale = max(height, width)
        px = array[:, 0] * scale / 2 + width / 2
        py = height / 2 - array[:, 1] * scale / 2
        valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        result = np.zeros(len(array), dtype=bool)
        result[valid] = self.mask[np.floor(py[valid]).astype(int), np.floor(px[valid]).astype(int)]
        return result

    def region_masks(self, points):
        array = as_points(points)
        foreground = self.support(array)
        result = {"foreground": foreground, "leakage": ~foreground}
        if self.horse is None:
            bounded = np.clip(array, -1., 1.)
            col = np.floor((bounded[:, 0] + 1) * self.grid_size / 2).astype(np.int64).clip(0, self.grid_size - 1)
            row = np.floor((1 - bounded[:, 1]) * self.grid_size / 2).astype(np.int64).clip(0, self.grid_size - 1)
            for r, c in np.argwhere(self.mask):
                result[f"cell_{r}_{c}"] = foreground & (row == r) & (col == c)
        else:
            for roi in self.horse.regions:
                x0, x1, y0, y1 = roi["box"]
                crop = (array[:, 0] >= x0) & (array[:, 0] < x1) & (array[:, 1] >= y0) & (array[:, 1] < y1)
                result[roi["name"]] = crop & (foreground if roi["kind"] == "foreground" else ~foreground)
        return result

    def mass(self, points):
        masks = self.region_masks(points)
        return np.asarray([masks[name].mean() for name in self.region_names])

    def grid_labels(self, points):
        array = as_points(points)
        inside = (np.abs(array) <= 1).all(1)
        bins = self.grid_bins
        indices = np.floor((np.clip(array, -1., 1.) + 1) * bins / 2).astype(np.int64).clip(0, bins - 1)
        return np.where(inside, indices[:, 0] * bins + indices[:, 1], bins * bins)

    def grid_hist(self, points):
        labels = self.grid_labels(points)
        return np.bincount(labels, minlength=self.grid_bins ** 2 + 1).astype(np.float64) / len(labels)

    def path_hist(self, points):
        """One common wide grid for Gaussian/intermediate/endpoint path comparisons."""
        array = as_points(points)
        low, high = self.path_grid_range
        counts, _, _ = np.histogram2d(array[:, 0], array[:, 1], bins=self.path_grid_bins,
                                      range=[[low, high], [low, high]])
        return np.append(counts.ravel(), len(array) - counts.sum()) / len(array)

    def _expected_grid(self, bins):
        height, width = self.mask.shape
        scale = max(height, width)
        edges = np.linspace(-1., 1., bins + 1)
        overlaps = []
        for size in (width, height):
            pixels = (np.arange(size + 1) - size / 2) * 2 / scale
            overlaps.append(np.maximum(0., np.minimum(edges[1:, None], pixels[None, 1:])
                                       - np.maximum(edges[:-1, None], pixels[None, :-1])))
        area = (overlaps[0] @ self.mask[::-1].T @ overlaps[1].T).ravel()
        return np.append(area / area.sum(), 0.)

    def definition(self):
        return {"region_names": self.region_names, "expected_mass": self.expected_mass.tolist(),
                "region_denominator": "ALL points, including leakage; semantic regions may overlap",
                "histogram": {"bins_per_axis": self.grid_bins, "range": [-1., 1.],
                              "outside_bin": self.grid_bins ** 2,
                              "flatten_order": "x_bin * bins + y_bin",
                              "expected_mass": self.expected_grid_mass.tolist()},
                "path_histogram": {"bins_per_axis": self.path_grid_bins,
                                   "range": list(self.path_grid_range),
                                   "outside_bin": self.path_grid_bins ** 2,
                                   "scope": "Same fixed wide grid for every t and all teacher/bank/fresh paths; distinct from endpoint quality histogram"},
                "horse_roi_definition": self.horse.definition() if self.horse is not None else None}


def write_csv(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (list, dict, tuple))
                             else value for key, value in row.items()})


def routing_regions(source, target_paired, spatial, routing_arrays, neighbors):
    """Aggregate the proxy weights; MC SE refers ONLY to Gaussian queries."""
    masks = spatial.region_masks(target_paired)
    labels = spatial.grid_labels(target_paired)
    rows, arrays, summary = [], {}, {}
    indices = routing_arrays["query_neighbor_indices"]
    for k in neighbors:
        q = routing_arrays[f"q{k}"]
        observed = []
        for index, name in enumerate(spatial.region_names):
            selected = masks[name]
            query_contributions = selected[indices[:, :k]].mean(1)
            routed = float(q[selected].sum())
            baseline = float(selected.mean())
            se = float(query_contributions.std(ddof=1) / np.sqrt(len(indices)))
            expected = float(spatial.expected_mass[index])
            rows.append({"neighbors": k, "region": name, "target_expected_mass": expected,
                         "bank_point_mass": baseline, "gaussian_routed_proxy_mass": routed,
                         "routing_minus_bank": routed - baseline, "routing_minus_target": routed - expected,
                         "query_mc_se": se, "query_count": len(indices)})
            observed.append(routed)
        hist = np.bincount(labels, weights=q, minlength=spatial.grid_bins ** 2 + 1)
        bank_hist = spatial.grid_hist(target_paired)
        arrays[f"routing_grid_k{k}"] = hist
        arrays[f"routing_region_mass_k{k}"] = np.asarray(observed)
        summary[str(k)] = {"grid_tv_vs_empirical_target_bank": float(.5 * np.abs(hist - bank_hist).sum()),
                           "grid_tv_vs_population_target": float(.5 * np.abs(hist - spatial.expected_grid_mass).sum()),
                           "weight_sum": float(q.sum())}
    arrays["target_bank_grid"] = spatial.grid_hist(target_paired)
    arrays["population_target_grid"] = spatial.expected_grid_mass
    return summary, rows, arrays
