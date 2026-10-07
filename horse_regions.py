"""Prespecified horse foreground mass and background-gap diagnostics."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from metrics import _points_numpy


DEFAULT_ROIS = Path(__file__).with_name("horse_rois.json")


class HorseRegions:
    def __init__(self, mask, roi_path=None):
        if hasattr(mask, "detach"):
            mask = mask.detach().cpu().numpy()
        self.mask = np.asarray(mask, dtype=bool)
        if self.mask.ndim != 2 or not self.mask.any():
            raise ValueError("Horse ROI metrics require a nonempty 2D foreground mask")
        self.path = Path(roi_path) if roi_path is not None else DEFAULT_ROIS
        raw = self.path.read_bytes()
        spec = json.loads(raw)
        if spec.get("version") != 1 or not isinstance(spec.get("regions"), list) or not spec["regions"]:
            raise ValueError("ROI JSON needs version=1 and a nonempty regions list")
        self.regions, names = [], set()
        for item in spec["regions"]:
            name, kind = item.get("name"), item.get("kind")
            box = np.asarray(item.get("box", []), dtype=float)
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names:
                raise ValueError("ROI names must be unique lowercase identifiers")
            if (kind not in ("foreground", "background") or box.shape != (4,)
                    or not np.isfinite(box).all() or box[0] >= box[1] or box[2] >= box[3]
                    or np.any(np.abs(box) > 1)):
                raise ValueError("ROI box must be finite [xmin,xmax,ymin,ymax] within [-1,1]")
            names.add(name)
            expected = self.foreground_mass(box) if kind == "foreground" else 0.0
            if kind == "foreground" and expected <= 0:
                raise ValueError(f"Foreground ROI {name} has no target foreground area")
            self.regions.append({"name": name, "kind": kind, "box": box.tolist(),
                                 "expected_mass": expected})
        if {r["kind"] for r in self.regions} != {"foreground", "background"}:
            raise ValueError("Specify at least one foreground crop and one background gap")
        self.file_hash = hashlib.sha256(raw).hexdigest()

    def foreground_mass(self, box):
        height, width = self.mask.shape
        scale = float(max(height, width))
        x_edges = (np.arange(width + 1) - width / 2) * 2 / scale
        y_top = (height / 2 - np.arange(height)) * 2 / scale
        pixel = 2 / scale
        overlap_x = np.maximum(0, np.minimum(x_edges[1:], box[1]) - np.maximum(x_edges[:-1], box[0])) / pixel
        overlap_y = np.maximum(0, np.minimum(y_top, box[3]) - np.maximum(y_top - pixel, box[2])) / pixel
        # Exact partial-pixel area for continuous uniform jitter, not center tests.
        return float(overlap_y @ self.mask @ overlap_x / self.mask.sum())

    def definition(self):
        return {
            "version": 1, "roi_file_sha256": self.file_hash,
            "mask_sha256": hashlib.sha256(self.mask.tobytes()).hexdigest(),
            "mask_shape": list(self.mask.shape), "regions": self.regions,
            "coordinates": "box=[xmin,xmax,ymin,ymax]; original unrotated training coordinates",
            "denominator": "ALL generated points, including leakage; never conditioned on foreground",
            "expected_mass": "exact foreground pixel area intersected with crop / total mask area; background=0",
            "thin_region_mass_mae": "mean absolute mass error across fixed foreground crops; crops may overlap",
            "gap_region_leakage": "fraction in the UNION of background gaps; no overlap double counting",
            "scope": "fixed spatial crops, NOT anatomical segmentation or a proof of sharpness",
        }

    def score(self, points):
        array = _points_numpy(points)
        height, width = self.mask.shape
        scale = float(max(height, width))
        px = array[:, 0] * scale / 2 + width / 2
        py = height / 2 - array[:, 1] * scale / 2
        inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        foreground = np.zeros(len(array), dtype=bool)
        foreground[inside] = self.mask[np.floor(py[inside]).astype(int), np.floor(px[inside]).astype(int)]
        scores, mass_errors = {}, []
        gap_union = np.zeros(len(array), dtype=bool)
        for roi in self.regions:
            lo_x, hi_x, lo_y, hi_y = roi["box"]
            crop = ((array[:, 0] >= lo_x) & (array[:, 0] < hi_x)
                    & (array[:, 1] >= lo_y) & (array[:, 1] < hi_y))
            selected = crop & (foreground if roi["kind"] == "foreground" else ~foreground)
            observed = float(selected.mean())
            prefix = "roi_" + roi["name"]
            if roi["kind"] == "foreground":
                delta = observed - roi["expected_mass"]
                scores.update({prefix + "_mass": observed, prefix + "_mass_error": abs(delta),
                               prefix + "_mass_deficit": max(-delta, 0.0),
                               prefix + "_mass_excess": max(delta, 0.0)})
                mass_errors.append(abs(delta))
            else:
                scores[prefix + "_leakage"] = observed
                gap_union |= selected
        scores["thin_region_mass_mae"] = float(np.mean(mass_errors))
        scores["gap_region_leakage"] = float(gap_union.mean())
        return scores

    def render(self, output_path):
        from summarize_results import plt
        from matplotlib.patches import Rectangle
        height, width = self.mask.shape
        scale = float(max(height, width))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(self.mask, cmap="Greys", origin="upper",
                  extent=(-width / scale, width / scale, -height / scale, height / scale))
        for i, roi in enumerate(self.regions, 1):
            x0, x1, y0, y1 = roi["box"]
            color = "tab:blue" if roi["kind"] == "foreground" else "tab:red"
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color=color,
                                   lw=1.5, label=f"{i}: {roi['name']} ({roi['kind']})"))
            ax.text(x0, y1, str(i), color=color, fontsize=10, va="bottom")
        ax.set(xlim=(-1.04, 1.04), ylim=(-.86, .86), aspect="equal",
               title="Fixed horse ROIs | foreground crops / background-only gaps")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
