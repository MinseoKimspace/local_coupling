import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import yaml

from coupling import apply_coupling
from data import checkerboard_centers, sample_checkerboard
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


def conditional_velocity_variance_proxy(
    model: torch.nn.Module,
    x_data: torch.Tensor,
    x_noise: torch.Tensor,
    times: tuple[float, ...],
) -> dict[float, float]:
    velocity = x_data - x_noise
    values = {}
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for time in times:
            t = torch.full(
                (x_data.shape[0], 1, 1),
                time,
                device=x_data.device,
                dtype=x_data.dtype,
            )
            x_t = (1.0 - t) * x_noise + t * x_data
            prediction = model(x_t, t)
            values[time] = (prediction - velocity).square().mean().item()

    model.train(was_training)
    return values


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
    figure, axes = plt.subplots(1, len(times), figsize=(9, 3.4))
    figure.text(
        0.5,
        0.97,
        f"coupling: {title}",
        ha="center",
        va="top",
        fontsize=16,
    )

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

    figure.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.04,
        top=0.78,
        wspace=0.12,
    )
    figure.savefig(output_path, dpi=200)
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

    torch.manual_seed(config["seed"] + 1000)
    variance_target = sample_checkerboard(
        evaluation_batch_size,
        data_config["n_points"],
        device=device,
        dtype=dtype,
        grid_size=data_config["grid_size"],
    )
    variance_noise = torch.randn_like(variance_target)
    target_centers = checkerboard_centers(
        data_config["grid_size"],
        device,
        dtype,
    )
    coupling_generator = torch.Generator(device=device)
    coupling_generator.manual_seed(config["seed"] + 1001)
    variance_target = apply_coupling(
        variance_noise,
        variance_target,
        method=coupling,
        num_regions=config.get("num_regions"),
        target_centers=target_centers,
        sinkhorn_epsilon=config.get("sinkhorn_epsilon", 0.1),
        sinkhorn_iterations=config.get("sinkhorn_iterations", 100),
        generator=coupling_generator,
    )
    variance_times = (0.1, 0.3, 0.5, 0.7, 0.9)
    variance_values = conditional_velocity_variance_proxy(
        model,
        variance_target,
        variance_noise,
        variance_times,
    )

    print(f"chamfer={score.item():.6f}")
    for time, value in variance_values.items():
        print(f"velocity_variance_proxy_t={time:.1f} value={value:.6f}")
    mean_variance = sum(variance_values.values()) / len(variance_values)
    print(f"velocity_variance_proxy_mean={mean_variance:.6f}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
