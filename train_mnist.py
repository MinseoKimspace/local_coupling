import sys

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import yaml

from model import PointSetTransformer
from train import train_step


def images_to_points(images: torch.Tensor, n_points: int) -> torch.Tensor:
    batch_size, _, height, width = images.shape
    weights = images.flatten(1)
    pixel_indices = torch.multinomial(weights, n_points, replacement=True)
    rows = torch.div(pixel_indices, width, rounding_mode="floor")
    columns = pixel_indices % width
    jitter = torch.rand(
        batch_size,
        n_points,
        2,
        device=images.device,
        dtype=images.dtype,
    )
    x = (columns.to(images.dtype) + jitter[..., 0]) / width * 2.0 - 1.0
    y = 1.0 - (rows.to(images.dtype) + jitter[..., 1]) / height * 2.0
    return torch.stack([x, y], dim=-1)


def repeat(loader: DataLoader):
    while True:
        yield from loader


def main(config_path: str = "mnist_independent.yaml") -> None:
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])
    data_config = config["data"]
    training_config = config["training"]

    dataset = MNIST(
        root=data_config["root"],
        train=True,
        download=True,
        transform=ToTensor(),
    )
    loader = DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=data_config["num_workers"],
        pin_memory=device.type == "cuda",
    )
    batches = repeat(loader)

    model = PointSetTransformer(**config["model"]).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    coupling_generator = torch.Generator(device=device)
    coupling_generator.manual_seed(config["seed"] + 1)
    model.train()

    for step in range(1, training_config["num_steps"] + 1):
        images, _ = next(batches)
        images = images.to(device=device, dtype=dtype, non_blocking=True)
        x_data = images_to_points(images, data_config["n_points"])
        loss = train_step(
            model,
            optimizer,
            x_data,
            coupling=config["coupling"],
            num_regions=config.get("num_regions"),
            coupling_generator=coupling_generator,
        )

        if step == 1 or step % training_config["log_every"] == 0:
            print(f"step={step} loss={loss.item():.6f}")

    torch.save(model.state_dict(), config["checkpoint"])


if __name__ == "__main__":
    main(*sys.argv[1:])
