import sys

import torch
import yaml

from coupling import (
    geometry_aware_hungarian_permutation,
    geometry_aware_sinkhorn_permutation,
    global_hungarian_permutation,
    regional_permutation,
    strict_target_guided_balanced_permutation,
    strict_target_guided_local_permutation,
    strict_target_guided_permutation,
    target_guided_permutation,
    target_guided_sinkhorn_permutation,
)
from data import checkerboard_centers, sample_checkerboard
from model import PointSetTransformer


def apply_coupling(
    source: torch.Tensor,
    target: torch.Tensor,
    config: dict,
    target_centers: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    method = config["coupling"]
    num_regions = config.get("num_regions")

    if method == "independent":
        return target
    if method == "regional":
        permutation = regional_permutation(
            source,
            target,
            num_regions=num_regions,
            generator=generator,
        )
    elif method == "target_guided":
        permutation = target_guided_permutation(
            source,
            target,
            num_regions=num_regions,
            generator=generator,
        )
    elif method == "target_guided_sinkhorn":
        permutation = target_guided_sinkhorn_permutation(
            source,
            target,
            num_regions=num_regions,
            epsilon=config.get("sinkhorn_epsilon", 0.1),
            num_iterations=config.get("sinkhorn_iterations", 100),
            generator=generator,
        )
    elif method == "geometry_aware_sinkhorn":
        permutation = geometry_aware_sinkhorn_permutation(
            source,
            target,
            num_regions=num_regions,
            epsilon=config.get("sinkhorn_epsilon", 0.1),
            num_iterations=config.get("sinkhorn_iterations", 100),
            generator=generator,
        )
    elif method == "geometry_aware_hungarian":
        permutation = geometry_aware_hungarian_permutation(
            source,
            target,
            num_regions=num_regions,
            generator=generator,
        )
    elif method == "target_guided_strict":
        permutation = strict_target_guided_permutation(
            source,
            target,
            target_centers,
            generator=generator,
        )
    elif method == "target_guided_strict_local":
        permutation = strict_target_guided_local_permutation(
            source,
            target,
            target_centers,
        )
    elif method == "target_guided_strict_balanced":
        permutation = strict_target_guided_balanced_permutation(
            source,
            target,
            target_centers,
        )
    elif method == "global_hungarian":
        permutation = global_hungarian_permutation(source, target)
    else:
        raise ValueError("unknown coupling")

    permutation = permutation.unsqueeze(-1).expand(-1, -1, target.shape[-1])
    return torch.gather(target, dim=1, index=permutation)


def measure(
    model: torch.nn.Module,
    x_data: torch.Tensor,
    x_noise: torch.Tensor,
    times: tuple[float, ...],
) -> dict[float, float]:
    velocity = x_data - x_noise
    values = {}
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

    return values


def main(config_path: str = "independent.yaml") -> None:
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    torch.manual_seed(config["seed"] + 1000)
    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])
    data_config = config["data"]
    batch_size = data_config["batch_size"] * 4

    model = PointSetTransformer(**config["model"]).to(device=device, dtype=dtype)
    state_dict = torch.load(
        config["checkpoint"],
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    x_data = sample_checkerboard(
        batch_size,
        data_config["n_points"],
        device=device,
        dtype=dtype,
        grid_size=data_config["grid_size"],
    )
    x_noise = torch.randn_like(x_data)
    target_centers = checkerboard_centers(
        data_config["grid_size"],
        device,
        dtype,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(config["seed"] + 1001)
    x_data = apply_coupling(
        x_noise,
        x_data,
        config,
        target_centers,
        generator,
    )

    values = measure(model, x_data, x_noise, (0.1, 0.3, 0.5, 0.7, 0.9))
    for time, value in values.items():
        print(f"t={time:.1f} variance_proxy={value:.6f}")
    print(f"mean={sum(values.values()) / len(values):.6f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
