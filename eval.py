import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from data import sample_checkerboard
from model import PointSetTransformer
from sample import integrate_velocity


def chamfer_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    distances = torch.cdist(prediction, target).square()
    prediction_to_target = distances.min(dim=2).values.mean(dim=1)
    target_to_prediction = distances.min(dim=1).values.mean(dim=1)
    return (prediction_to_target + target_to_prediction).mean()


def render_points(
    x_noise: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
    output_path: str,
) -> None:
    point_sets = [x_noise[0], prediction[0], target[0]]
    titles = ["Noise", "Generated", "Target"]
    limits = [3.0, 1.2, 1.2]
    figure, axes = plt.subplots(1, 3, figsize=(9, 3))

    for axis, points, title, limit in zip(axes, point_sets, titles, limits):
        points = points.detach().cpu()
        axis.scatter(points[:, 0], points[:, 1], s=5)
        axis.set_title(title)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal")

    figure.tight_layout()
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

    model = PointSetTransformer(**config["model"]).to(device=device, dtype=dtype)
    state_dict = torch.load(
        config["checkpoint"],
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    x_noise = torch.randn(
        data_config["batch_size"],
        data_config["n_points"],
        2,
        device=device,
        dtype=dtype,
    )
    prediction = integrate_velocity(
        model,
        x_noise,
        num_steps=integration_steps,
    )
    target = sample_checkerboard(
        data_config["batch_size"],
        data_config["n_points"],
        device=device,
        dtype=dtype,
        grid_size=data_config["grid_size"],
    )

    score = chamfer_distance(prediction, target)
    output_path = f"samples_{coupling}.png"
    render_points(x_noise, prediction, target, output_path)
    print(f"chamfer={score.item():.6f}")
    print(f"saved={output_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
