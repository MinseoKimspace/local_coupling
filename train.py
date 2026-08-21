import sys

import torch
from torch import nn
from torch.optim import Optimizer
import torch.nn.functional as F
import yaml

from coupling import (
    global_hungarian_permutation,
    regional_permutation,
    strict_target_guided_balanced_permutation,
    strict_target_guided_local_permutation,
    strict_target_guided_permutation,
    target_guided_permutation,
)
from data import checkerboard_centers, sample_checkerboard
from model import PointSetTransformer


def sample_time(
    batch_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    eps: float = 0.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if not 0.0 <= eps < 0.5:
        raise ValueError("eps must satisfy 0 <= eps < 0.5")

    return torch.rand(
        batch_size,
        1,
        1,
        device=device,
        dtype=dtype,
        generator=generator,
    ) * (1.0 - 2.0 * eps) + eps


def linear_path(
    x_data: torch.Tensor,
    x_noise: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    return (1.0 - t) * x_noise + t * x_data


def flow_matching_loss(
    model: nn.Module,
    x_data: torch.Tensor,
    x_noise: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    x_t = linear_path(x_data, x_noise, t)
    target = x_data - x_noise
    prediction = model(x_t, t)
    return F.mse_loss(prediction, target)


def train_step(
    model: nn.Module,
    optimizer: Optimizer,
    x_data: torch.Tensor,
    *,
    coupling: str,
    num_regions: int | None = None,
    target_centers: torch.Tensor | None = None,
    coupling_generator: torch.Generator | None = None,
) -> torch.Tensor:
    x_noise = torch.randn(
        x_data.shape,
        device=x_data.device,
        dtype=x_data.dtype,
    )

    if coupling == "regional":
        if num_regions is None:
            raise ValueError("num_regions is required for regional coupling")

        permutation = regional_permutation(
            x_noise,
            x_data,
            num_regions=num_regions,
            generator=coupling_generator,
        )
        permutation = permutation.unsqueeze(-1).expand(-1, -1, x_data.shape[-1])
        x_data = torch.gather(x_data, dim=1, index=permutation)
    elif coupling == "target_guided":
        if num_regions is None:
            raise ValueError("num_regions is required for target-guided coupling")

        permutation = target_guided_permutation(
            x_noise,
            x_data,
            num_regions=num_regions,
            generator=coupling_generator,
        )
        permutation = permutation.unsqueeze(-1).expand(-1, -1, x_data.shape[-1])
        x_data = torch.gather(x_data, dim=1, index=permutation)
    elif coupling == "target_guided_strict":
        if target_centers is None:
            raise ValueError("target_centers is required for strict target-guided coupling")

        permutation = strict_target_guided_permutation(
            x_noise,
            x_data,
            target_centers,
            generator=coupling_generator,
        )
        permutation = permutation.unsqueeze(-1).expand(-1, -1, x_data.shape[-1])
        x_data = torch.gather(x_data, dim=1, index=permutation)
    elif coupling == "target_guided_strict_local":
        if target_centers is None:
            raise ValueError("target_centers is required for strict local coupling")

        permutation = strict_target_guided_local_permutation(
            x_noise,
            x_data,
            target_centers,
        )
        permutation = permutation.unsqueeze(-1).expand(-1, -1, x_data.shape[-1])
        x_data = torch.gather(x_data, dim=1, index=permutation)
    elif coupling == "target_guided_strict_balanced":
        if target_centers is None:
            raise ValueError("target_centers is required for strict balanced coupling")

        permutation = strict_target_guided_balanced_permutation(
            x_noise,
            x_data,
            target_centers,
        )
        permutation = permutation.unsqueeze(-1).expand(-1, -1, x_data.shape[-1])
        x_data = torch.gather(x_data, dim=1, index=permutation)
    elif coupling == "global_hungarian":
        permutation = global_hungarian_permutation(x_noise, x_data)
        permutation = permutation.unsqueeze(-1).expand(-1, -1, x_data.shape[-1])
        x_data = torch.gather(x_data, dim=1, index=permutation)
    elif coupling != "independent":
        raise ValueError("unknown coupling")

    t = sample_time(
        x_data.shape[0],
        device=x_data.device,
        dtype=x_data.dtype,
    )

    optimizer.zero_grad(set_to_none=True)
    loss = flow_matching_loss(model, x_data, x_noise, t)
    loss.backward()
    optimizer.step()
    return loss.detach()


def main(config_path: str = "independent.yaml") -> None:
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    coupling = config["coupling"]
    coupling_generator = torch.Generator(device=device)
    coupling_generator.manual_seed(config["seed"] + 1)
    target_centers = checkerboard_centers(
        data_config["grid_size"],
        device,
        dtype,
    )

    model = PointSetTransformer(**model_config).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    model.train()

    for step in range(1, training_config["num_steps"] + 1):
        x_data = sample_checkerboard(
            data_config["batch_size"],
            data_config["n_points"],
            device=device,
            dtype=dtype,
            grid_size=data_config["grid_size"],
        )
        loss = train_step(
            model,
            optimizer,
            x_data,
            coupling=coupling,
            num_regions=config.get("num_regions"),
            target_centers=target_centers,
            coupling_generator=coupling_generator,
        )

        if step == 1 or step % training_config["log_every"] == 0:
            print(f"step={step} loss={loss.item():.6f}")

    torch.save(model.state_dict(), config["checkpoint"])


if __name__ == "__main__":
    main(*sys.argv[1:])
