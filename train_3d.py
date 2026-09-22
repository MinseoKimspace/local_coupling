"""Single-GPU PSF first-stage FM with this repository's coupling; no EMD."""
import argparse
import copy
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import random
import shutil
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch

from prepare_psf import ROOT, provenance
from psf_adapter import PSFVelocity, shapenet_dataset


class ResumableBatchSampler:
    """Shuffled full microbatches, no prefetch; save the next position exactly."""
    def __init__(self, size, batch_size, seed):
        if not 1 <= batch_size <= size:
            raise ValueError("Expected 1 <= microbatch size <= training split size")
        self.size, self.batch_size = size, batch_size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.empty(0, dtype=torch.long)
        self.position = 0

    def __iter__(self):
        while True:
            if self.position + self.batch_size > len(self.order):
                self.order = torch.randperm(self.size, generator=self.generator)
                self.position = 0
            indices = self.order[self.position:self.position + self.batch_size].tolist()
            self.position += self.batch_size
            yield indices

    def state_dict(self):
        return {"order": self.order.clone(), "position": self.position,
                "generator": self.generator.get_state()}

    def load_state_dict(self, state):
        order, position = state["order"], state["position"]
        if (len(order) != self.size or not torch.equal(order.sort().values, torch.arange(self.size))
                or not 0 <= position <= self.size or position % self.batch_size):
            raise ValueError("Invalid saved data order/position")
        self.order, self.position = order.clone(), position
        self.generator.set_state(state["generator"])


def accumulated_update(model, optimizer, batches, accumulation_steps, loss_fn):
    """Equal-size microbatches; one Adam update of their mean gradient."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    optimizer.zero_grad(set_to_none=True)
    total, batch_size = None, None
    for _ in range(accumulation_steps):
        target = next(batches)
        if batch_size is not None and len(target) != batch_size:
            raise ValueError("Gradient accumulation requires equal microbatch sizes")
        batch_size = len(target)
        loss = loss_fn(model, target)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite training loss; optimizer update was not applied")
        (loss / accumulation_steps).backward()
        total = loss.detach() if total is None else total + loss.detach()
    optimizer.step()
    return (total / accumulation_steps).item()


def rng_state(pairing_rng):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            "torch": torch.get_rng_state(), "pairing": pairing_rng.get_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state, pairing_rng):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(state["torch"])
    pairing_rng.set_state(state["pairing"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def resume_signature(config):
    config = copy.deepcopy(config)
    config["data"].pop("root", None)  # May move the same dataset to another drive.
    for key in ("num_steps", "log_every", "save_every", "checkpoint_steps"):
        config["training"].pop(key, None)
    return config


def main(config_path, dataroot=None, steps=None, resume=None):
    import yaml
    from coupling import coupling_info
    from experiment import environment, synchronize
    from train import coupled_flow_matching_loss

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
    for value in (training["num_steps"], training["accumulation_steps"], training["log_every"],
                  training["save_every"], data["batch_size"], *training.get("checkpoint_steps", [])):
        if type(value) is not int or value < 1:
            raise ValueError("Steps, batch_size and logging/checkpoint intervals must be positive integers.")
    if not np.isfinite(training["learning_rate"]) or training["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if "lr_gamma" in training or "lr_decay_every" in training:
        raise ValueError("This runner uses constant LR; remove obsolete scheduler settings")
    if not 1 <= config["num_regions"] <= data["n_points"]:
        raise ValueError("Expected 1 <= num_regions <= n_points")
    psf = provenance()
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in ("train_3d.py", "psf_adapter.py", "coupling.py", "train.py", "sample.py")}
    previous = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if previous is not None:
        if previous.get("format") != "psf_tg_v2":
            raise ValueError("Resume requires a v2 checkpoint with optimizer and RNG states; v1 is evaluation-only")
        if resume_signature(previous["config"]) != resume_signature(config):
            raise ValueError("Resume config differs from training settings (including microbatch and accumulation)")
        if previous["psf"] != psf or previous["source_sha256"] != source_hashes:
            raise ValueError("Resume requires unchanged PSF/bridge/training/coupling source code")
        if training["num_steps"] <= previous["step"]:
            raise ValueError("--steps / num_steps is the TOTAL optimizer update target; it must exceed the saved step")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PSF training.")
    device = torch.device("cuda")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = shapenet_dataset(data["root"], data["category"], data["n_points"])
    # No workers/prefetch: sampler state always identifies the NEXT microbatch.
    sampler = ResumableBatchSampler(len(dataset), data["batch_size"], seed + 2)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True,
        generator=torch.Generator().manual_seed(seed + 4))
    model = PSFVelocity(**config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"],
                                 betas=(0.5, 0.999), weight_decay=0)
    pairing_rng = torch.Generator(device=device).manual_seed(seed + 1)
    normalization = {"mean": dataset.all_points_mean.tolist(), "std": dataset.all_points_std.tolist()}
    first_step, elapsed, history = 1, 0.0, []
    if previous is not None:
        if (previous["training_shape_ids"] != dataset.all_cate_mids
                or previous["normalization"] != normalization):
            raise ValueError("Resume dataset shape IDs or normalization changed")
        model.load_state_dict(previous["model_state_dict"])
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        sampler.load_state_dict(previous["sampler_state"])
        first_step = previous["step"] + 1
        elapsed, history = previous["training_seconds"], previous["history"]
        restore_rng(previous["rng_state"], pairing_rng)
        del previous
    now = datetime.now(timezone.utc)
    name = f"{data['category']}_{config['coupling']}_k{config['num_regions']}_n{data['n_points']}_seed{seed}"
    run = ROOT / "runs" / "psf3d" / f"{name}_{now:%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    run.mkdir(parents=True)
    with (run / "config.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    metadata = {
        "format": "psf_tg_v2", "config": config, "source_config": str(Path(config_path).resolve()),
        "psf": psf, "normalization": normalization, "environment": environment(device),
        "coupling_details": coupling_info(config["coupling"]), "training_shapes": len(dataset),
        "training_shape_ids": dataset.all_cate_mids,
        "optimizer": "Adam(beta1=0.5,beta2=0.999,weight_decay=0)",
        "schedule": "constant LR; no scheduler", "stage": "first_stage_fm_no_reflow_no_distillation_no_ema",
        "ema": False, "precision": "float32_no_amp", "step_definition": "one optimizer update after accumulation",
        "effective_batch_size": data["batch_size"] * training["accumulation_steps"],
        "resumed_from": str(Path(resume).resolve()) if resume else None,
        "source_sha256": source_hashes,
    }
    print(f"run={run}", flush=True)
    print(f"microbatch={data['batch_size']} accumulation={training['accumulation_steps']} "
          f"effective_batch={metadata['effective_batch_size']} updates={training['num_steps']} "
          f"constant_lr={training['learning_rate']} ema=False", flush=True)
    model.train()
    batches = (batch["train_points"].to(device, non_blocking=True) for batch in loader)
    loss_fn = partial(coupled_flow_matching_loss, coupling=config["coupling"],
                      num_regions=config["num_regions"], coupling_generator=pairing_rng)
    milestones = set(training.get("checkpoint_steps", []))
    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    for step in range(first_step, training["num_steps"] + 1):
        value = accumulated_update(model, optimizer, batches, training["accumulation_steps"], loss_fn)
        if step == first_step or step % training["log_every"] == 0 or step == training["num_steps"]:
            seconds_per_update = (perf_counter() - start) / (step - first_step + 1)
            history.append({"step": step, "loss": value, "learning_rate": optimizer.param_groups[0]["lr"],
                            "mean_seconds_per_update": seconds_per_update,
                            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30})
            print(f"step={step} loss={value:.6f} seconds/update={seconds_per_update:.3f} "
                  f"peak_VRAM_GiB={history[-1]['peak_allocated_gib']:.2f}", flush=True)
        if step % training["save_every"] == 0 or step in milestones or step == training["num_steps"]:
            synchronize(device)
            record = {**metadata, "step": step, "training_seconds": elapsed + perf_counter() - start,
                      "shape_presentations": step * metadata["effective_batch_size"],
                      "final_loss": value, "history": history}
            # A unique run directory; atomic replacement only of this run's latest checkpoint.
            temporary = run / "checkpoint.tmp"
            torch.save({**record, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "sampler_state": sampler.state_dict(),
                        "rng_state": rng_state(pairing_rng)}, temporary)
            temporary.replace(run / "checkpoint.pt")
            if step in milestones:
                shutil.copyfile(run / "checkpoint.pt", run / f"checkpoint_step_{step:06d}.pt")
            with (run / "training.json").open("w", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2, allow_nan=False)
    print(f"checkpoint={run / 'checkpoint.pt'}")
    print(f'eval_command=python eval_3d.py "{run / "checkpoint.pt"}"')
    return run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path")
    parser.add_argument("--dataroot")
    parser.add_argument("--steps", type=int, help="Total optimizer update target, e.g. 10 for a smoke run")
    parser.add_argument("--resume", help="Resume a v2 checkpoint into a NEW run directory; --steps is the total target")
    main(**vars(parser.parse_args()))
