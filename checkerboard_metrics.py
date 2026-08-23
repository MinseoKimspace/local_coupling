import sys

import numpy as np
import torch
import yaml

from model import PointSetTransformer
from sample import integrate_velocity


def checkerboard_metrics(
    points: torch.Tensor,
    grid_size: int,
    histogram_bins: int = 64,
) -> tuple[float, float, float]:
    points = points.reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    inside = (
        (x >= -1.0)
        & (x <= 1.0)
        & (y >= -1.0)
        & (y <= 1.0)
    )

    cell_size = 2.0 / grid_size
    cols = ((x + 1.0) / cell_size).floor().long().clamp(0, grid_size - 1)
    rows = ((1.0 - y) / cell_size).floor().long().clamp(0, grid_size - 1)
    active = (rows + cols) % 2 == 0
    valid = inside & active
    leakage = 1.0 - valid.float().mean()

    active_cells = torch.arange(
        grid_size * grid_size,
        device=points.device,
    ).reshape(grid_size, grid_size)
    active_cells = active_cells[
        (torch.arange(grid_size, device=points.device).unsqueeze(1)
         + torch.arange(grid_size, device=points.device).unsqueeze(0)) % 2 == 0
    ]
    flat_cells = rows * grid_size + cols
    counts = torch.stack(
        [(flat_cells[valid] == cell).sum() for cell in active_cells]
    ).to(points.dtype)
    if counts.sum() == 0:
        mass_error = torch.tensor(float("nan"), device=points.device)
    else:
        mass = counts / counts.sum()
        mass_error = 0.5 * (mass - 1.0 / active_cells.numel()).abs().sum()

    points_numpy = points.detach().cpu().numpy()
    prediction, _, _ = np.histogram2d(
        points_numpy[:, 0],
        points_numpy[:, 1],
        bins=histogram_bins,
        range=[[-1.0, 1.0], [-1.0, 1.0]],
    )
    outside_count = points_numpy.shape[0] - prediction.sum()
    prediction = np.append(prediction.ravel(), outside_count)
    prediction /= prediction.sum()

    edges = np.linspace(-1.0, 1.0, histogram_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    grid_x, grid_y = np.meshgrid(centers, centers, indexing="ij")
    target_cols = np.floor((grid_x + 1.0) / cell_size).astype(int)
    target_rows = np.floor((1.0 - grid_y) / cell_size).astype(int)
    target_active = (target_rows + target_cols) % 2 == 0
    target = target_active.astype(np.float64).ravel()
    target /= target.sum()
    target = np.append(target, 0.0)

    mixture = 0.5 * (prediction + target)
    prediction_mask = prediction > 0
    target_mask = target > 0
    js = 0.5 * np.sum(
        prediction[prediction_mask]
        * np.log(prediction[prediction_mask] / mixture[prediction_mask])
    )
    js += 0.5 * np.sum(
        target[target_mask]
        * np.log(target[target_mask] / mixture[target_mask])
    )

    return leakage.item(), mass_error.item(), float(js)


def main(config_path: str = "independent.yaml") -> None:
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    torch.manual_seed(1)
    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])
    data_config = config["data"]

    model = PointSetTransformer(**config["model"]).to(device=device, dtype=dtype)
    state_dict = torch.load(
        config["checkpoint"],
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    x_noise = torch.randn(
        data_config["batch_size"] * 4,
        data_config["n_points"],
        2,
        device=device,
        dtype=dtype,
    )
    prediction = integrate_velocity(model, x_noise, num_steps=100)
    leakage, mass_error, histogram_js = checkerboard_metrics(
        prediction,
        data_config["grid_size"],
    )

    print(f"coupling={config['coupling']}")
    print(f"leakage={leakage:.6f}")
    print(f"cell_mass_error={mass_error:.6f}")
    print(f"histogram_js={histogram_js:.6f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
