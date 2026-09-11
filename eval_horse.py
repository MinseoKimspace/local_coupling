import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from experiment import (evaluation_settings, evaluation_title, load_model,
                        sample_for_evaluation, save_evaluation)
from metrics import chamfer_distance, horse_metrics
from train_horse import HorsePointSetTransformer, load_horse_mask, sample_horse


def point_density(points: torch.Tensor) -> np.ndarray:
    points = points.reshape(-1, 2).numpy()
    density, _, _ = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=256,
        range=[[-1.1, 1.1], [-1.1, 1.1]],
    )
    return gaussian_filter(density.T, sigma=1.0)


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


def main(config_path="horse_experiments/horse_independent_n256_seed0.yaml", num_steps=100):
    steps = int(num_steps)
    if steps < 1:
        raise ValueError("num_steps must be positive")
    model, config, checkpoint, metadata = load_model(config_path, HorsePointSetTransformer, "horse")
    settings, data = evaluation_settings(config), config["data"]
    print(f"nfe={steps}")
    _, prediction, seconds = sample_for_evaluation(model, config, steps)
    mask = load_horse_mask(prediction.device, prediction.dtype)
    target = sample_horse(mask, settings["batch_size"], data["n_points"])
    leakage, js = horse_metrics(prediction, mask, settings["histogram_bins"])
    scores = {"chamfer": chamfer_distance(prediction, target).item(), "leakage": leakage, "histogram_js": js}
    output = save_evaluation(config_path, config, checkpoint, metadata, "horse", steps, seconds, scores)
    render_comparison(target.cpu(), prediction.cpu(), f"{evaluation_title(config)}\nNFE: {steps} (Euler)", output)
    print(f"saved={output}")
    return output


if __name__ == "__main__":
    main(*sys.argv[1:])
