"""Precompute hard TG on checkerboard/horse, without training.

Use the SAME experiment YAML here and in train.py/train_horse.py.
Existing caches are validated/reused, never overwritten. Interruptions leave
an incomplete directory; choose a new path rather than silently reusing it.

Defaults are bank mode, 4096 clouds, K=8, N=256, seed=0. Hard YAMLs have
the same model, optimizer, batch and updates as the corresponding online TG.

For a full single-use stream, set tg_cache.sampling: stream, num_clouds: null,
and a NEW tg_cache.path in a COPY of the YAML. At B=64 and 10000 updates this
prepares 640000 clouds (~3.68 GiB per baseline cache). Both modes cache coarse
labels offline and draw fine pairing and time afresh in training. Optional
path-affine experiments additionally store hard grouping tables and target
fine-group labels. Scoring is offline; fine bijections remain fresh during
training. See TG_LOCAL.md for the comparison runner and cost/quality limits.

Generation uses the existing eval.py/eval_horse.py and saved run config.yaml.
audit_generation.py supports these caches; FM residuals use their training
bank and are NOT unseen-target tests. audit_mean_field.py intentionally does
NOT claim to estimate a population conditional field for a finite cloud bank.
Report precompute_seconds, cache_setup_seconds and training_seconds separately.

Alternatively pass --sampling stream. The CLI derives a separate _stream
cache, writes its effective config.yaml there WITHOUT changing your input,
and prints the training command for that generated configuration.
"""

import argparse
from pathlib import Path

import yaml

from experiment import read_config
from tg_cache import prepare


def main(config_path, dataset, *, sampling=None):
    config = read_config(config_path)
    if sampling is not None:
        previous = config["tg_cache"].get("sampling", "bank")
        if sampling != previous:
            config["tg_cache"]["sampling"] = sampling
            config["tg_cache"]["num_clouds"] = None if sampling == "stream" else 4096
            config["tg_cache"]["path"] += "_" + sampling
            config["tg_cache"].pop("cache_sha256", None)
    path, metadata = prepare(config, dataset)
    effective = Path(config_path)
    if sampling is not None:
        effective = path / "config.yaml"
        if effective.exists():
            if yaml.safe_load(effective.read_text(encoding="utf-8")) != config:
                raise ValueError("Derived cache config already exists with different settings; choose a new cache path")
        else:
            with effective.open("x", encoding="utf-8") as stream:
                yaml.safe_dump(config, stream, sort_keys=False)
    trainer = "train_horse.py" if dataset == "horse" else "train.py"
    print(f'train_command=python {trainer} "{effective}"', flush=True)
    return path, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--dataset", required=True, choices=["checkerboard", "horse"])
    parser.add_argument("--sampling", choices=["bank", "stream"], default=None,
                        help="Derive a separate cache/config without editing the input YAML")
    args = parser.parse_args()
    main(args.config, args.dataset, sampling=args.sampling)
