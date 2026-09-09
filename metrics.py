import numpy as np
import torch
from scipy.spatial.distance import jensenshannon


def chamfer_distance(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(prediction, target).square()
    forward = distances.min(dim=2).values.mean(dim=1)
    backward = distances.min(dim=1).values.mean(dim=1)
    return (forward + backward).mean()


def _points_numpy(points):
    if points.shape[-1] != 2:
        raise ValueError("Silhouette metrics require 2D points")
    points = points.detach().cpu().double().numpy().reshape(-1, 2)
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("Expected nonempty, finite points")
    return points


def histogram_js(points, mask, bins):
    prediction, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=bins,
                                     range=[[-1.0, 1.0], [-1.0, 1.0]])
    prediction = np.append(prediction.ravel(), len(points) - prediction.sum()) / len(points)
    height, width = mask.shape
    scale = float(max(height, width))
    edges = np.linspace(-1.0, 1.0, bins + 1)
    # Exact foreground area, including pixels/cells straddling histogram bins.
    overlaps = []
    for size in (width, height):
        pixels = (np.arange(size + 1) - size / 2.0) * 2.0 / scale
        overlaps.append(np.maximum(0.0, np.minimum(edges[1:, None], pixels[None, 1:])
                                    - np.maximum(edges[:-1, None], pixels[None, :-1])))
    target = (overlaps[0] @ mask[::-1].T @ overlaps[1].T).ravel()
    target = np.append(target / target.sum(), 0.0)
    # SciPy returns sqrt(JS); square it to retain divergence in natural-log units.
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
    return float(jensenshannon(prediction, target, base=np.e) ** 2)


def checkerboard_metrics(points, grid_size, histogram_bins=64):
    array = _points_numpy(points)
    points = points.reshape(-1, 2)
    x, y = points[:, 0], points[:, 1]
    inside = (x >= -1) & (x <= 1) & (y >= -1) & (y <= 1)
    cell_size = 2.0 / grid_size
    cols = ((x + 1) / cell_size).floor().long().clamp(0, grid_size - 1)
    rows = ((1 - y) / cell_size).floor().long().clamp(0, grid_size - 1)
    valid = inside & ((rows + cols) % 2 == 0)
    leakage = 1.0 - valid.float().mean().item()
    indices = torch.arange(grid_size, device=points.device)
    active = (indices[:, None] + indices[None, :]) % 2 == 0
    counts = torch.bincount((rows * grid_size + cols)[valid], minlength=grid_size ** 2)
    counts = counts.reshape(grid_size, grid_size)[active].to(points.dtype)
    mass_error = (0.5 * (counts / counts.sum() - 1.0 / counts.numel()).abs().sum().item()
                  if counts.sum() > 0 else float("nan"))
    js = histogram_js(array, active.cpu().numpy().astype(np.float64), histogram_bins)
    return leakage, mass_error, js


def horse_metrics(points, mask, histogram_bins=64):
    points = _points_numpy(points)
    mask = mask.detach().cpu().double().numpy()
    height, width = mask.shape
    scale = float(max(height, width))
    px = points[:, 0] * scale / 2.0 + width / 2.0
    py = height / 2.0 - points[:, 1] * scale / 2.0
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    valid = np.zeros(len(points), dtype=bool)
    valid[inside] = mask[np.floor(py[inside]).astype(int), np.floor(px[inside]).astype(int)] > 0
    return float(1.0 - valid.mean()), histogram_js(points, mask, histogram_bins)
