import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from data import sample_checkerboard
from experiment import (evaluation_settings, evaluation_title, load_model,
                        sample_for_evaluation, save_evaluation)
from metrics import chamfer_distance, checkerboard_metrics
from model import PointSetTransformer


def sample_snapshots(
    model: torch.nn.Module,
    x_noise: torch.Tensor,
    num_steps: int,
    snapshot_steps: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    x = x_noise
    dt = 1.0 / num_steps
    snapshots = {}
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for step in range(1, num_steps + 1):
            t = torch.full(
                (x.shape[0], 1, 1),
                (step - 1) * dt,
                device=x.device,
                dtype=x.dtype,
            )
            x = x + dt * model(x, t)

            if step in snapshot_steps:
                snapshots[step] = x.cpu()

    model.train(was_training)
    return snapshots


def render_density(
    snapshots: list[torch.Tensor],
    times: tuple[float, ...],
    title: str,
    output_path: str,
) -> None:
    limit = 1.2
    densities = []

    for points in snapshots:
        points = points.reshape(-1, 2).numpy()
        density, _, _ = np.histogram2d(
            points[:, 0],
            points[:, 1],
            bins=192,
            range=[[-limit, limit], [-limit, limit]],
        )
        density = gaussian_filter(density.T, sigma=1.2)
        density /= max(density.sum(), 1.0)
        densities.append(density)

    values = np.concatenate([density.ravel() for density in densities])
    positive = values[values > 0]
    vmax = np.percentile(positive, 99.5) if positive.size else 1.0
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
    figure, axes = plt.subplots(1, len(times), figsize=(9, 3.4))
    figure.text(
        0.5,
        0.97,
        f"coupling: {title}",
        ha="center",
        va="top",
        fontsize=16,
    )

    for axis, density, time in zip(np.atleast_1d(axes), densities, times):
        axis.imshow(
            density,
            origin="lower",
            extent=(-limit, limit, -limit, limit),
            cmap="viridis",
            norm=norm,
            interpolation="bilinear",
        )
        axis.set_title(f"t = {time:.2f}", fontsize=18)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_axis_off()

    figure.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.04,
        top=0.70,
        wspace=0.12,
    )
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main(config_path="checkerboard_experiments/independent.yaml", num_steps=100, *, render=True):
    steps = int(num_steps)
    if steps < 1:
        raise ValueError("num_steps must be positive")
    model, config, checkpoint, metadata = load_model(config_path, PointSetTransformer, "checkerboard")
    settings, data = evaluation_settings(config), config["data"]
    print(f"nfe={steps}")
    noise, prediction, seconds = sample_for_evaluation(model, config, steps)
    target = sample_checkerboard(settings["batch_size"], data["n_points"], prediction.device,
                                 prediction.dtype, data["grid_size"])
    leakage, mass, js = checkerboard_metrics(prediction, data["grid_size"], settings["histogram_bins"])
    scores = {"chamfer": chamfer_distance(prediction, target).item(), "leakage": leakage,
              "cell_mass_error": mass, "histogram_js": js}
    output = save_evaluation(config_path, config, checkpoint, metadata, "checkerboard",
                             steps, seconds, scores, render=render)
    if render:
        snapshot_steps = tuple(sorted({round(t * steps) for t in (0.78, 0.89, 1.0)}))
        times = tuple(step / steps for step in snapshot_steps)
        snapshots = sample_snapshots(model, noise, steps, snapshot_steps)
        render_density([snapshots[i] for i in snapshot_steps], times,
                       f"{evaluation_title(config)}\nNFE: {steps} (Euler)", output)
        print(f"saved={output}")
    return output


if __name__ == "__main__":
    main(*sys.argv[1:])
