import argparse
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from coupling import OFFLINE_METHODS, TG_CACHED_METHODS, coupled_points
from data import checkerboard_centers, sample_checkerboard, validate_gaussian_source
from experiment import read_config, save_training, synchronize
from model import PointSetTransformer


def sample_time(batch_size, *, device, dtype, eps=0.0, generator=None):
    if not 0.0 <= eps < 0.5:
        raise ValueError("eps must satisfy 0 <= eps < 0.5")
    return torch.rand(batch_size, 1, 1, device=device, dtype=dtype, generator=generator) * (1 - 2 * eps) + eps


def linear_path(x_data, x_noise, t):
    return (1.0 - t) * x_noise + t * x_data


def flow_matching_loss(model, x_data, x_noise, t):
    return F.mse_loss(model(linear_path(x_data, x_noise, t), t), x_data - x_noise)


def coupled_flow_matching_loss(model, x_data, *, coupling, num_regions=None, target_centers=None,
                               sinkhorn_epsilon=0.1, sinkhorn_iterations=100, coupling_generator=None,
                               paired_noise=None):
    if paired_noise is None:
        x_noise = torch.randn(x_data.shape, device=x_data.device, dtype=x_data.dtype)
        x_noise, x_data = coupled_points(
            x_noise, x_data, coupling=coupling, num_regions=num_regions, target_centers=target_centers,
            sinkhorn_epsilon=sinkhorn_epsilon, sinkhorn_iterations=sinkhorn_iterations,
            generator=coupling_generator,
        )
    else:
        if coupling not in OFFLINE_METHODS or paired_noise.shape != x_data.shape \
                or paired_noise.device != x_data.device or paired_noise.dtype != x_data.dtype:
            raise ValueError("Precomputed noise requires an offline coupling and matching [B,N,D] tensors")
        x_noise = paired_noise
    t = sample_time(x_data.shape[0], device=x_data.device, dtype=x_data.dtype)
    return flow_matching_loss(model, x_data, x_noise, t)


def train_step(model, optimizer, x_data, *, coupling, num_regions=None, target_centers=None,
               sinkhorn_epsilon=0.1, sinkhorn_iterations=100, coupling_generator=None, paired_noise=None):
    loss = coupled_flow_matching_loss(
        model, x_data, coupling=coupling, num_regions=num_regions, target_centers=target_centers,
        sinkhorn_epsilon=sinkhorn_epsilon, sinkhorn_iterations=sinkhorn_iterations,
        coupling_generator=coupling_generator, paired_noise=paired_noise,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach()


def train_model(model, config, sample_batch, *, dataset, config_path, target_centers=None):
    validate_gaussian_source(config)
    training = config["training"]
    if training["num_steps"] < 1 or training["log_every"] < 1:
        raise ValueError("num_steps and log_every must be positive")
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                 weight_decay=training["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(config["seed"] + 1)
    np.random.seed(config["seed"] + 1)  # Original upstream OT samplers use NumPy.
    pair_sampler = None
    if config["coupling"] == "nsot":
        from nsot import NSOTPairSampler
        pair_sampler = NSOTPairSampler(config, dataset, device, next(model.parameters()).dtype)
        config["nsot"]["cache_sha256"] = pair_sampler.cache_sha256
        print(f"nsot_cache_sha256={pair_sampler.cache_sha256} beta={pair_sampler.beta}", flush=True)
    elif config["coupling"] in TG_CACHED_METHODS:
        from tg_cache import TGCachedPairSampler
        pair_sampler = TGCachedPairSampler(config, dataset, device, next(model.parameters()).dtype, training=True)
        config["tg_cache"]["cache_sha256"] = pair_sampler.cache_sha256
        print(f"tg_cache_sha256={pair_sampler.cache_sha256} "
              f"sampling={pair_sampler.metadata['sampling']} clouds={pair_sampler.metadata['num_clouds']} "
              f"fine={pair_sampler.metadata.get('fine_pairing_mode', 'random')}", flush=True)
    model.train()
    synchronize(device)
    start = perf_counter()
    try:
        for step in range(1, training["num_steps"] + 1):
            log_step = step == 1 or step % training["log_every"] == 0
            paired_noise = None
            if pair_sampler is None:
                target = sample_batch()
            else:
                paired_noise, target = pair_sampler.sample(config["data"]["batch_size"], generator=generator)
            loss = train_step(
                model, optimizer, target, coupling=config["coupling"],
                num_regions=config.get("num_regions"), target_centers=target_centers,
                sinkhorn_epsilon=config.get("sinkhorn_epsilon", 0.1),
                sinkhorn_iterations=config.get("sinkhorn_iterations", 100), coupling_generator=generator,
                paired_noise=paired_noise,
            )
            if log_step:
                value = loss.item()
                timing = (f" seconds/update={(perf_counter() - start) / step:.4f}"
                          if config["coupling"] in TG_CACHED_METHODS else "")
                print(f"step={step} loss={value:.6f}{timing}")
    finally:
        if config["coupling"] in TG_CACHED_METHODS and pair_sampler is not None:
            pair_sampler.close()
    synchronize(device)
    seconds = perf_counter() - start
    print(f"training_seconds={seconds:.3f}")
    return save_training(model, config, dataset, config_path, seconds, loss.item(),
                         coupling_metadata=pair_sampler.details() if pair_sampler is not None else None)


def read_training_config(config_path, *, seed=None, steps=None):
    config = read_config(config_path)
    if seed is not None:
        config["seed"] = seed
    if steps is not None:
        if steps < 1:
            raise ValueError("steps must be positive")
        config["training"]["num_steps"] = steps
    return config


def training_arguments(default_config):
    parser = argparse.ArgumentParser(description="Train with the configured coupling and standard Gaussian source.")
    parser.add_argument("config_path", nargs="?", default=default_config)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help="Override update count; saved in the run config")
    return vars(parser.parse_args())


def main(config_path="checkerboard_experiments/independent.yaml", *, seed=None, steps=None):
    config = read_training_config(config_path, seed=seed, steps=steps)
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
    main(**training_arguments("checkerboard_experiments/independent.yaml"))
