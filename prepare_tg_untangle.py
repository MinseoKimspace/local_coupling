"""Prepare optimized fine permutations or the matched zero-swap control.

Use a NEW untangle YAML; the original hard cache is validated or created at
tg_cache.base_path and never overwritten. --control uses identical initial
permutation RNG seeds but zero swaps, in a separate _control cache. Both
variants write an effective config.yaml and print their training command.

Examples (PowerShell or a shell, at the repository root):
  python prepare_tg_untangle.py checkerboard_experiments/target_guided_cached_untangle_k8_n256_seed0.yaml --dataset checkerboard
  python prepare_tg_untangle.py horse_experiments/horse_target_guided_cached_untangle_k8_n256_seed0.yaml --dataset horse
  python train.py coupling_cache/checkerboard_tg_untangle_k8_n256_seed0/config.yaml
  python train_horse.py coupling_cache/horse_tg_untangle_k8_n256_seed0/config.yaml

Repeat preparation with --control and train its printed _control/config.yaml
to isolate score optimization from restricting pairing to a finite pool.
After training, evaluate the SAVED runs/.../config.yaml (printed by training):
  python eval.py runs/checkerboard/RUN/config.yaml 8
  python eval_horse.py runs/horse/RUN/config.yaml 8
Repeat at NFE 1, 2, 4, 8, 16, 32, 64, 128. Generation is unchanged.

Defaults in the supplied YAMLs: K8/N256/seed0, original 4096-cloud bank,
source-kNN8, t=.05/.15/.30, L=1, four independent initial permutations and
64 proposed within-patch swaps per permutation. Change swap_steps for a
stronger search, using a NEW path. The full N x N source graph construction
is intended for these small 2D clouds, NOT 150000 points. L=1 is a score
budget, not a global model Lipschitz constraint. Accepted moves cannot raise
the fixed score, but lower score does not guarantee better generation.

untangle_report.json and untangle_scores.npy record before/after conflict;
metadata/checkpoints separately record search, preparation and training time.
Do not use the old analytic uniform-fine mean-field audit for this variant.
"""

import argparse
from pathlib import Path

import yaml

from experiment import read_config
from tg_cache import prepare, settings


def main(config_path, dataset, *, control=False):
    config = read_config(config_path)
    opts = settings(config)
    if "untangle" not in opts:
        raise ValueError("This command requires an enabled tg_cache.untangle experiment")
    config["tg_cache"]["untangle"] = opts["untangle"]
    if control:
        config["tg_cache"]["untangle"]["swap_steps"] = 0
        config["tg_cache"]["path"] += "_control"
        config["tg_cache"].pop("cache_sha256", None)
        checkpoint = Path(config["checkpoint"])
        config["checkpoint"] = str(checkpoint.with_name(checkpoint.stem + "_control" + checkpoint.suffix))
    path, metadata = prepare(config, dataset)
    effective = path / "config.yaml"
    if effective.exists():
        if yaml.safe_load(effective.read_text(encoding="utf-8")) != config:
            raise ValueError("Derived config already exists with different settings; choose a new cache path")
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
    parser.add_argument("--control", action="store_true", help="Matched finite random pool with zero swaps")
    args = parser.parse_args()
    main(args.config, args.dataset, control=args.control)
