"""Optional time-only FM loss weights; uniform time sampling is unchanged."""

import math
from numbers import Integral, Real

import torch


def settings(value=None):
    if value is None:
        return {"mode": "uniform"}
    if not isinstance(value, dict):
        raise ValueError("training.time_weighting must be a mapping")
    mode = value.get("mode", "uniform")
    if mode not in ("uniform", "late"):
        raise ValueError("training.time_weighting.mode must be uniform or late")
    allowed = {"mode"} if mode == "uniform" else {"mode", "threshold", "multiplier"}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown time_weighting settings: {sorted(unknown)}")
    if mode == "uniform":
        return {"mode": mode}
    threshold, multiplier = value.get("threshold", 0.75), value.get("multiplier", 2.0)
    for name, number in (("threshold", threshold), ("multiplier", multiplier)):
        if isinstance(number, bool) or not isinstance(number, Real) or not math.isfinite(number):
            raise ValueError(f"time_weighting.{name} must be a finite number")
    if not 0 < threshold < 1 or multiplier < 1:
        raise ValueError("Late weighting requires 0 < threshold < 1 and multiplier >= 1")
    return {"mode": mode, "threshold": float(threshold), "multiplier": float(multiplier)}


def validate_training_options(training):
    settings(training.get("time_weighting"))
    if "time_seed" in training:
        seed = training["time_seed"]
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2 ** 63:
            raise ValueError("training.time_seed must be an integer in [0, 2**63)")


def weights(t, config=None):
    """Weights normalized by E[w(t)] for t ~ Uniform[0,1], not by minibatch."""
    config = settings(config)
    if config["mode"] == "uniform":
        return torch.ones_like(t)
    threshold, multiplier = config["threshold"], config["multiplier"]
    normalizer = threshold + multiplier * (1.0 - threshold)
    return (1.0 + (multiplier - 1.0) * (t >= threshold).to(t.dtype)) / normalizer
