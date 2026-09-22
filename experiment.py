import copy
import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import ot
import scipy
import torch
import yaml

from coupling import canonical_method, coupling_info
from sample import integrate_velocity
from source_randomization import randomization_settings


def read_config(path):
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["coupling"] = canonical_method(config["coupling"])
    if config["coupling"] == "target_guided_randomized":
        config["source_randomization"] = randomization_settings(config.get("source_randomization"))
    return config


def environment(device="cpu"):
    device = torch.device(device)
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "scipy": scipy.__version__, "POT": ot.__version__,
            "cuda": torch.version.cuda, "platform": platform.platform(), "cpu": platform.processor(),
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def training_signature(config):
    values = {k: v for k, v in config.items() if k not in ("checkpoint", "device", "evaluation")}
    values["coupling"] = canonical_method(values["coupling"])
    return values


def save_training(model, config, dataset, config_path, seconds, loss, *, coupling_diagnostics=None):
    if not np.isfinite(loss):
        raise FloatingPointError("Final training loss is nonfinite; checkpoint was not saved")
    now = datetime.now(timezone.utc)
    method = canonical_method(config["coupling"])
    name = f"{dataset}_{method}_k{config.get('num_regions', 'na')}_n{config['data']['n_points']}_seed{config['seed']}"
    run_dir = Path("runs") / dataset / f"{name}_{now:%Y%m%dT%H%M%S%fZ}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot = copy.deepcopy(config)
    snapshot["checkpoint"] = Path(config["checkpoint"]).name
    metadata = {
        "format_version": 2, "dataset": dataset, "config": snapshot,
        "source_config": str(Path(config_path).resolve()),
        "trained_at": now.isoformat(), "coupling_details": coupling_info(method),
        "training_seconds": seconds, "final_loss": loss, "environment": environment(next(model.parameters()).device),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    if coupling_diagnostics is not None:
        metadata["coupling_diagnostics"] = {
            "sampling": "Training batches at step 1 and log_every, not every update; included in training_seconds.",
            "records": coupling_diagnostics,
        }
    checkpoint = run_dir / snapshot["checkpoint"]
    torch.save({**metadata, "model_state_dict": model.state_dict()}, checkpoint)
    with (run_dir / "config.yaml").open("x", encoding="utf-8") as file:
        yaml.safe_dump(snapshot, file, sort_keys=False)
    with (run_dir / "training.json").open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, allow_nan=False)
    eval_script = "eval_horse.py" if dataset == "horse" else "eval.py"
    print(f"checkpoint={checkpoint}")
    print(f"eval_command=python {eval_script} {run_dir / 'config.yaml'}")
    return run_dir


def load_model(config_path, model_class, dataset):
    config = read_config(config_path)
    checkpoint = Path(config["checkpoint"])
    if not checkpoint.is_absolute():
        relative = Path(config_path).resolve().parent / checkpoint
        checkpoint = relative if relative.exists() else Path.cwd() / checkpoint
    checkpoint = checkpoint.resolve()
    device = torch.device(config["device"])
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    verified = isinstance(payload, dict) and "model_state_dict" in payload
    if verified:
        if payload.get("format_version") != 2 or payload["dataset"] != dataset:
            raise ValueError("Checkpoint format or dataset does not match this evaluator")
        expected, actual = training_signature(payload["config"]), training_signature(config)
        changed = [key for key in expected.keys() | actual.keys() if expected.get(key) != actual.get(key)]
        if changed:
            raise ValueError(f"YAML differs from checkpoint training settings: {', '.join(sorted(changed))}")
        weights = payload["model_state_dict"]
        metadata = {k: v for k, v in payload.items() if k != "model_state_dict"}
    else:
        warnings.warn("Legacy weights-only checkpoint: training settings and solver are UNVERIFIED. "
                      "For a newly trained run, evaluate its runs/.../config.yaml instead.", stacklevel=2)
        weights, metadata = payload, {"coupling_details": {"implementation": "legacy_unverified"}}
    metadata["training_config_verified"] = verified
    torch.manual_seed(config.get("evaluation", {}).get("seed", 1))
    model = model_class(**config["model"]).to(device=device, dtype=getattr(torch, config["dtype"]))
    model.load_state_dict(weights)
    model.eval()
    print(f"checkpoint={checkpoint}")
    return model, config, checkpoint, metadata


def evaluation_settings(config):
    settings = {"seed": 1, "batch_size": config["data"]["batch_size"] * 4, "histogram_bins": 64}
    settings.update(config.get("evaluation", {}))
    if settings["batch_size"] < 1 or settings["histogram_bins"] < 1:
        raise ValueError("Evaluation batch size and histogram bins must be positive")
    return settings


def sample_for_evaluation(model, config, steps):
    parameter = next(model.parameters())
    batch = evaluation_settings(config)["batch_size"]
    noise = torch.randn(batch, config["data"]["n_points"], config["model"]["point_dim"],
                        device=parameter.device, dtype=parameter.dtype)
    time = parameter.new_zeros(batch, 1, 1)
    with torch.no_grad():
        for _ in range(10):
            model(noise, time)
    synchronize(parameter.device)
    start = perf_counter()
    prediction = integrate_velocity(model, noise, num_steps=steps)
    synchronize(parameter.device)
    return noise, prediction, perf_counter() - start


def save_evaluation(config_path, config, checkpoint, metadata, dataset, steps, seconds, scores, *, render=True):
    now = datetime.now(timezone.utc)
    settings = evaluation_settings(config)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output_dir = Path("eval_results") / dataset / f"{checkpoint.stem}_{checkpoint_hash[:12]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"nfe_{steps:03d}_{now:%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    image_path = output_dir / f"{name}.png"
    results = {
        "dataset": dataset, "evaluated_at": now.isoformat(),
        "config_path": str(Path(config_path).resolve()), "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "config": config, "coupling": config["coupling"],
        "training_config_verified": metadata["training_config_verified"],
        "coupling_details": metadata["coupling_details"],
        "training_seconds": metadata.get("training_seconds"),
        "training_environment": metadata.get("environment"), "evaluation_environment": environment(config["device"]),
        "evaluation_seed": settings["seed"], "evaluation_batch_size": settings["batch_size"],
        "n_points": config["data"]["n_points"], "total_points": settings["batch_size"] * config["data"]["n_points"],
        "euler_steps": steps, "histogram_bins": settings["histogram_bins"], "inference_seconds": seconds,
        "metric_definitions": {"chamfer": "sum of directional mean squared distances; then mean over clouds",
                               "leakage": "invalid points / all pooled points",
                               "cell_mass_error": "TV over valid checkerboard cells, conditioned on valid points",
                               "histogram_js": "pooled JS divergence in nats, with an outside bin"},
        "image": str(image_path.resolve()) if render else None,
        **scores,
    }
    # Undefined conditional TV (no valid points) is recorded as null, not NaN.
    if "cell_mass_error" in results and not np.isfinite(results["cell_mass_error"]):
        results["cell_mass_error"] = None
    json_path = output_dir / f"{name}.json"
    with json_path.open("x", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)
    for key, value in scores.items():
        print(f"{key}={value:.6f}")
    print(f"inference_seconds={seconds:.6f}")
    print(f"saved_json={json_path}")
    return image_path if render else json_path


def evaluation_title(config):
    return (f"{config['coupling']} | K={config.get('num_regions', 'na')} | "
            f"N={config['data']['n_points']} | seed={config['seed']}")
