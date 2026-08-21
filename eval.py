import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import yaml

from data import sample_checkerboard
from model import PointSetTransformer


def chamfer_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    distances = torch.cdist(prediction, target).square()
    prediction_to_target = distances.min(dim=2).values.mean(dim=1)
    target_to_prediction = distances.min(dim=1).values.mean(dim=1)
    return (prediction_to_target + target_to_prediction).mean()


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
        density /= density.sum()
        densities.append(density)

    values = np.concatenate([density.ravel() for density in densities])
    vmax = np.percentile(values[values > 0], 99.5)
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
    figure, axes = plt.subplots(1, len(times), figsize=(9, 3))
    figure.suptitle(f"coupling: {title}", fontsize=16)

    for axis, density, time in zip(axes, densities, times):
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

    figure.tight_layout(rect=(0, 0, 1, 0.92), pad=0.6)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main(config_path: str = "independent.yaml") -> None:
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    torch.manual_seed(1)
    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])
    data_config = config["data"]
    coupling = config["coupling"]
    integration_steps = 100
    snapshot_times = (0.78, 0.89, 1.00)
    snapshot_steps = tuple(round(time * integration_steps) for time in snapshot_times)
    evaluation_batch_size = data_config["batch_size"] * 4

    model = PointSetTransformer(**config["model"]).to(device=device, dtype=dtype)
    state_dict = torch.load(
        config["checkpoint"],
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    x_noise = torch.randn(
        evaluation_batch_size,
        data_config["n_points"],
        2,
        device=device,
        dtype=dtype,
    )
    snapshots = sample_snapshots(
        model,
        x_noise,
        integration_steps,
        snapshot_steps,
    )
    target = sample_checkerboard(
        evaluation_batch_size,
        data_config["n_points"],
        device=device,
        dtype=dtype,
        grid_size=data_config["grid_size"],
    )

    prediction = snapshots[integration_steps].to(device)
    score = chamfer_distance(prediction, target)
    output_path = f"density_{coupling}.png"
    render_density(
        [snapshots[step] for step in snapshot_steps],
        snapshot_times,
        coupling,
        output_path,
    )
    print(f"chamfer={score.item():.6f}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
