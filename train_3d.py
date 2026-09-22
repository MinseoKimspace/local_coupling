"""Single-GPU PSF first-stage FM with this repository's coupling; no EMD."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch
import yaml

from coupling import coupling_info
from experiment import environment, synchronize
from prepare_psf import ROOT, provenance
from psf_adapter import PSFVelocity, shapenet_dataset
from train import train_step


def main(config_path, dataroot=None, steps=None):
    with Path(config_path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config = copy.deepcopy(config)
    if dataroot is not None:
        config["data"]["root"] = dataroot
    if steps is not None:
        config["training"]["num_steps"] = steps
    if config["coupling"] not in ("independent", "target_guided", "target_guided_exact_optimized"):
        raise ValueError("Initial PSF comparison supports independent or exact TG only.")
    data, training = config["data"], config["training"]
    if min(training["num_steps"], training["log_every"], training["save_every"], data["batch_size"]) < 1:
        raise ValueError("Steps, batch_size, log_every and save_every must be positive.")
    if not 1 <= config["num_regions"] <= data["n_points"]:
        raise ValueError("Expected 1 <= num_regions <= n_points")
    psf = provenance()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PSF training.")
    device = torch.device("cuda")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = shapenet_dataset(data["root"], data["category"], data["n_points"])
    if len(dataset) < data["batch_size"]:
        raise ValueError("Training split is smaller than batch_size; reduce batch_size.")
    # workers=0 avoids Windows spawn/import ambiguity and keeps subsampling reproducible.
    loader = torch.utils.data.DataLoader(dataset, batch_size=data["batch_size"], shuffle=True,
        num_workers=0, drop_last=True, pin_memory=True,
        generator=torch.Generator().manual_seed(seed + 2))
    model = PSFVelocity(**config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"],
                                 betas=(0.5, 0.999), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=training["lr_gamma"])
    pairing_rng = torch.Generator(device=device).manual_seed(seed + 1)
    normalization = {"mean": dataset.all_points_mean.tolist(), "std": dataset.all_points_std.tolist()}
    now = datetime.now(timezone.utc)
    name = f"{data['category']}_{config['coupling']}_k{config['num_regions']}_n{data['n_points']}_seed{seed}"
    run = ROOT / "runs" / "psf3d" / f"{name}_{now:%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    run.mkdir(parents=True)
    with (run / "config.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    metadata = {
        "format": "psf_tg_v1", "config": config, "source_config": str(Path(config_path).resolve()),
        "psf": psf, "normalization": normalization, "environment": environment(device),
        "coupling_details": coupling_info(config["coupling"]), "training_shapes": len(dataset),
        "training_shape_ids": dataset.all_cate_mids,
        "optimizer": "Adam(beta1=0.5,beta2=0.999,weight_decay=0)",
        "schedule": "ExponentialLR after each completed data epoch", "stage": "first_stage_fm_no_reflow_no_distillation_no_ema",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("train_3d.py", "psf_adapter.py", "coupling.py", "train.py", "sample.py")},
    }
    print(f"run={run}", flush=True)
    model.train()
    iterator, history = iter(loader), []
    synchronize(device)
    start = perf_counter()
    for step in range(1, training["num_steps"] + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            scheduler.step()
            iterator = iter(loader)
            batch = next(iterator)
        target = batch["train_points"].to(device, non_blocking=True)
        loss = train_step(model, optimizer, target, coupling=config["coupling"],
                          num_regions=config["num_regions"], coupling_generator=pairing_rng)
        value = loss.item()
        if not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        if step == 1 or step % training["log_every"] == 0 or step == training["num_steps"]:
            history.append({"step": step, "loss": value, "learning_rate": optimizer.param_groups[0]["lr"]})
            print(f"step={step} loss={value:.6f}", flush=True)
        if step % training["save_every"] == 0 or step == training["num_steps"]:
            synchronize(device)
            record = {**metadata, "step": step, "training_seconds": perf_counter() - start,
                      "final_loss": value, "history": history}
            # A unique run directory; atomic replacement only of this run's latest checkpoint.
            temporary = run / "checkpoint.tmp"
            torch.save({**record, "model_state_dict": model.state_dict()}, temporary)
            temporary.replace(run / "checkpoint.pt")
            with (run / "training.json").open("w", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2, allow_nan=False)
    print(f"checkpoint={run / 'checkpoint.pt'}")
    print(f'eval_command=python eval_3d.py "{run / "checkpoint.pt"}"')
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path")
    parser.add_argument("--dataroot")
    parser.add_argument("--steps", type=int, help="Override optimizer steps, e.g. 100 for a smoke run")
    main(**vars(parser.parse_args()))
