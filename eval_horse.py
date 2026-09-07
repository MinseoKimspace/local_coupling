import sys
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import rel_entr
import torch
import yaml

from sample import integrate_velocity
from train_horse import HorsePointSetTransformer, load_horse_mask, sample_horse


def chamfer_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    distances = torch.cdist(prediction, target).square()
    prediction_to_target = distances.min(dim=2).values.mean(dim=1)
    target_to_prediction = distances.min(dim=1).values.mean(dim=1)
    return (prediction_to_target + target_to_prediction).mean()


def point_density(points: torch.Tensor) -> np.ndarray:
    points = points.reshape(-1, 2).numpy()
    density, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=256,
        range=[[-1.1, 1.1], [-1.1, 1.1]],
    )
    return gaussian_filter(density.T, sigma=1.0)


def horse_metrics(
    points: torch.Tensor,
    mask: torch.Tensor,
    histogram_bins: int = 64,
) -> tuple[float, float]:
    points = points.detach().cpu().double().numpy().reshape(-1, 2)
    mask = mask.detach().cpu().double().numpy()
    if points.shape[0] == 0 or not np.isfinite(points).all():
        raise ValueError("Expected nonempty, finite generated points")
    height, width = mask.shape
    scale = float(max(height, width))
    pixel_x = points[:, 0] * scale / 2.0 + width / 2.0
    pixel_y = height / 2.0 - points[:, 1] * scale / 2.0
    inside = (pixel_x >= 0) & (pixel_x < width) & (pixel_y >= 0) & (pixel_y < height)
    valid = np.zeros(points.shape[0], dtype=bool)
    cols = np.floor(pixel_x[inside]).astype(int)
    rows = np.floor(pixel_y[inside]).astype(int)
    valid[inside] = mask[rows, cols] > 0
    leakage = 1.0 - valid.mean()

    prediction, _, _ = np.histogram2d(
        points[:, 0], points[:, 1], bins=histogram_bins,
        range=[[-1.0, 1.0], [-1.0, 1.0]],
    )
    outside_count = points.shape[0] - prediction.sum()
    prediction = np.append(prediction.ravel(), outside_count) / points.shape[0]

    # Integrate foreground pixel area into bins, including partial pixel overlaps.
    edges = np.linspace(-1.0, 1.0, histogram_bins + 1)
    x_edges = (np.arange(width + 1) - width / 2.0) * 2.0 / scale
    y_edges = (np.arange(height + 1) - height / 2.0) * 2.0 / scale
    x_overlap = np.maximum(
        0.0, np.minimum(edges[1:, None], x_edges[None, 1:])
        - np.maximum(edges[:-1, None], x_edges[None, :-1]),
    )
    y_overlap = np.maximum(
        0.0, np.minimum(edges[1:, None], y_edges[None, 1:])
        - np.maximum(edges[:-1, None], y_edges[None, :-1]),
    )
    target = (x_overlap @ mask[::-1].T @ y_overlap.T).ravel()
    target = np.append(target / target.sum(), 0.0)
    mixture = 0.5 * (prediction + target)
    js = 0.5 * (rel_entr(prediction, mixture).sum() + rel_entr(target, mixture).sum())
    return float(leakage), float(js)


def render_comparison(
    target: torch.Tensor,
    prediction: torch.Tensor,
    title: str,
    output_path: str,
) -> None:
    densities = [point_density(target), point_density(prediction)]
    values = np.concatenate([density.ravel() for density in densities])
    vmax = np.percentile(values[values > 0], 99.5)
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
    figure, axes = plt.subplots(1, 2, figsize=(8, 4), facecolor="black")
    figure.suptitle(f"Horse silhouette - {title}", color="white", fontsize=16)

    for axis, density, name in zip(axes, densities, ("Target", "Generated")):
        axis.imshow(
            density,
            origin="lower",
            extent=(-1.1, 1.1, -1.1, 1.1),
            cmap="viridis",
            norm=norm,
            interpolation="bilinear",
        )
        axis.set_title(name, color="white", fontsize=14)
        axis.set_facecolor("black")
        axis.set_aspect("equal")
        axis.set_axis_off()

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output_path, dpi=200, facecolor="black")
    plt.close(figure)


def main(config_path: str = "horse_independent.yaml") -> None:
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    torch.manual_seed(1)
    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])
    data_config = config["data"]
    run_name = Path(config["checkpoint"]).stem
    evaluation_batch_size = data_config["batch_size"] * 4

    model = HorsePointSetTransformer(**config["model"]).to(device=device, dtype=dtype)
    state_dict = torch.load(
        config["checkpoint"],
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    x_noise = torch.randn(
        evaluation_batch_size,
        data_config["n_points"],
        2,
        device=device,
        dtype=dtype,
    )
    warmup_t = torch.zeros(
        evaluation_batch_size,
        1,
        1,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        for _ in range(10):
            model(x_noise, warmup_t)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_start = perf_counter()
    prediction = integrate_velocity(model, x_noise, num_steps=100)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = perf_counter() - inference_start

    mask = load_horse_mask(device, dtype)
    target = sample_horse(
        mask,
        evaluation_batch_size,
        data_config["n_points"],
    )
    score = chamfer_distance(prediction, target)
    leakage, histogram_js = horse_metrics(prediction, mask)
    output_path = f"horse_{run_name}.png"
    render_comparison(
        target.cpu(),
        prediction.cpu(),
        run_name,
        output_path,
    )

    print(f"chamfer={score.item():.6f}")
    print(f"leakage={leakage:.6f}")
    print(f"histogram_js={histogram_js:.6f}")
    print(f"inference_seconds={inference_seconds:.6f}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
