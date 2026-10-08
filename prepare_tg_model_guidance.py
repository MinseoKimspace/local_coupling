"""Prepare frozen-teacher pairing pools without changing original Hard banks.

Use a verified runs/.../config.yaml as --teacher-config. The teacher must
match the dataset, model dimensions, N and checkerboard grid; its coupling
may differ. A score-selected pool and --control use identical candidate and
finite-difference probe seeds. The control keeps random candidate indices.
Neither training nor evaluation requires the teacher after preparation.

Examples:
  python prepare_tg_model_guidance.py checkerboard_experiments/target_guided_cached_model_guided_k8_n256_seed0.yaml --dataset checkerboard --teacher-config runs/checkerboard/EXISTING_HARD_RUN/config.yaml
  python prepare_tg_model_guidance.py horse_experiments/horse_target_guided_cached_model_guided_k8_n256_seed0.yaml --dataset horse --teacher-config runs/horse/EXISTING_HARD_RUN/config.yaml

Repeat with --control to create a separate matched random-pool cache.
Train the printed effective config, starting from scratch as usual. Evaluate
the new run's saved config.yaml with eval.py/eval_horse.py and the NFE.
Reported preparation time excludes training the existing teacher; charge
that separately for total-budget comparisons. Scores are teacher-dependent
proxies, not measured conditional mean-field error or generation quality.
"""

import argparse
from pathlib import Path

import yaml

from experiment import read_config
from tg_cache import prepare, settings


def main(config_path, dataset, *, teacher_config, control=False):
    config = read_config(config_path)
    value = config.get("tg_cache", {}).get("model_guidance")
    if not isinstance(value, dict) or not value.get("enabled", True):
        raise ValueError("This command requires enabled tg_cache.model_guidance")
    value["teacher_config"] = str(Path(teacher_config).resolve())
    config["tg_cache"]["model_guidance"] = settings(config)["model_guidance"]
    if control:
        config["tg_cache"]["model_guidance"]["selection"] = "random"
        config["tg_cache"]["path"] += "_control"
        config["tg_cache"].pop("cache_sha256", None)
        checkpoint = Path(config["checkpoint"])
        config["checkpoint"] = str(checkpoint.with_name(checkpoint.stem + "_control" + checkpoint.suffix))
    path, metadata = prepare(config, dataset)
    config["tg_cache"]["teacher_checkpoint_sha256"] = metadata["teacher"]["checkpoint_sha256"]
    effective = path / "config.yaml"
    if effective.exists():
        if yaml.safe_load(effective.read_text(encoding="utf-8")) != config:
            raise ValueError("Derived config exists with different settings; choose a new cache path")
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
    parser.add_argument("--teacher-config", required=True, help="Verified frozen runs/.../config.yaml")
    parser.add_argument("--control", action="store_true", help="Same scored candidate pool, random selection")
    args = parser.parse_args()
    main(args.config, args.dataset, teacher_config=args.teacher_config, control=args.control)
