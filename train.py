import sys
from time import perf_counter

import torch
import torch.nn.functional as F

from coupling import coupling_permutation
from data import checkerboard_centers, sample_checkerboard
from experiment import read_config, save_training, synchronize
from model import PointSetTransformer
from source_randomization import randomization_settings


def sample_time(batch_size, *, device, dtype, eps=0.0, generator=None):
    if not 0.0 <= eps < 0.5:
        raise ValueError("eps must satisfy 0 <= eps < 0.5")
    return torch.rand(batch_size, 1, 1, device=device, dtype=dtype, generator=generator) * (1 - 2 * eps) + eps


def linear_path(x_data, x_noise, t):
    return (1.0 - t) * x_noise + t * x_data


def flow_matching_loss(model, x_data, x_noise, t):
    return F.mse_loss(model(linear_path(x_data, x_noise, t), t), x_data - x_noise)


def train_step(model, optimizer, x_data, *, coupling, num_regions=None, target_centers=None,
               sinkhorn_epsilon=0.1, sinkhorn_iterations=100, coupling_generator=None,
               source_randomization=None, assignment_generator=None, coupling_diagnostics=None):
    x_noise = torch.randn(x_data.shape, device=x_data.device, dtype=x_data.dtype)
    permutation = coupling_permutation(
        x_noise, x_data, coupling=coupling, num_regions=num_regions, target_centers=target_centers,
        sinkhorn_epsilon=sinkhorn_epsilon, sinkhorn_iterations=sinkhorn_iterations,
        generator=coupling_generator,
        source_randomization=source_randomization, assignment_generator=assignment_generator,
        coupling_diagnostics=coupling_diagnostics,
    )
    if permutation is not None:
        x_data = x_data.gather(1, permutation.unsqueeze(-1).expand(-1, -1, x_data.shape[-1]))
    t = sample_time(x_data.shape[0], device=x_data.device, dtype=x_data.dtype)
    optimizer.zero_grad(set_to_none=True)
    loss = flow_matching_loss(model, x_data, x_noise, t)
    loss.backward()
    optimizer.step()
    return loss.detach()


def train_model(model, config, sample_batch, *, dataset, config_path, target_centers=None):
    randomized = config["coupling"] == "target_guided_randomized"
    if randomized:
        config = {**config, "source_randomization": randomization_settings(config.get("source_randomization"))}
    training = config["training"]
    if training["num_steps"] < 1 or training["log_every"] < 1:
        raise ValueError("num_steps and log_every must be positive")
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                 weight_decay=training["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(config["seed"] + 1)
    assignment_generator = torch.Generator(device="cpu").manual_seed(config["seed"] + 3)
    history = []
    model.train()
    synchronize(device)
    start = perf_counter()
    for step in range(1, training["num_steps"] + 1):
        log_step = step == 1 or step % training["log_every"] == 0
        diagnostics = {} if randomized and log_step else None
        loss = train_step(
            model, optimizer, sample_batch(), coupling=config["coupling"],
            num_regions=config.get("num_regions"), target_centers=target_centers,
            sinkhorn_epsilon=config.get("sinkhorn_epsilon", 0.1),
            sinkhorn_iterations=config.get("sinkhorn_iterations", 100), coupling_generator=generator,
            source_randomization=config.get("source_randomization"), assignment_generator=assignment_generator,
            coupling_diagnostics=diagnostics,
        )
        if log_step:
            print(f"step={step} loss={loss.item():.6f}")
            if diagnostics is not None:
                history.append({"step": step, **diagnostics})
    synchronize(device)
    seconds = perf_counter() - start
    print(f"training_seconds={seconds:.3f}")
    return save_training(model, config, dataset, config_path, seconds, loss.item(), coupling_diagnostics=history or None)


def main(config_path="checkerboard_experiments/independent.yaml"):
    config = read_config(config_path)
    torch.manual_seed(config["seed"])
    device, dtype = torch.device(config["device"]), getattr(torch, config["dtype"])
    data = config["data"]
    centers = checkerboard_centers(data["grid_size"], device, dtype)
    model = PointSetTransformer(**config["model"]).to(device=device, dtype=dtype)
    return train_model(
        model, config,
        lambda: sample_checkerboard(data["batch_size"], data["n_points"], device, dtype, data["grid_size"]),
        dataset="checkerboard", config_path=config_path, target_centers=centers,
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
