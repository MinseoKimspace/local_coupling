import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from sample import integrate_velocity
from train_mnist import MNISTPointSetTransformer


def render_samples(points: torch.Tensor, title: str, output_path: str) -> None:
    figure, axes = plt.subplots(4, 4, figsize=(8, 8), facecolor="black")
    figure.suptitle(f"MNIST - {title}", fontsize=16, color="white")

    for axis, sample in zip(axes.flat, points):
        sample = sample.numpy()
        axis.set_facecolor("black")
        axis.scatter(sample[:, 0], sample[:, 1], s=8, c="white", linewidths=0)
        axis.set_xlim(-1.2, 1.2)
        axis.set_ylim(-1.2, 1.2)
        axis.set_aspect("equal")
        axis.set_axis_off()

    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(output_path, dpi=200, facecolor="black")
    plt.close(figure)


def main(config_path: str = "mnist_independent.yaml") -> None:
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    torch.manual_seed(1)
    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])
    run_name = Path(config["checkpoint"]).stem

    model = MNISTPointSetTransformer(**config["model"]).to(device=device, dtype=dtype)
    state_dict = torch.load(
        config["checkpoint"],
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    x_noise = torch.randn(
        16,
        config["data"]["n_points"],
        2,
        device=device,
        dtype=dtype,
    )
    samples = integrate_velocity(
        model,
        x_noise,
        num_steps=100,
    ).cpu()
    output_path = f"mnist_{run_name}.png"
    render_samples(samples, run_name, output_path)
    print(f"saved={output_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
