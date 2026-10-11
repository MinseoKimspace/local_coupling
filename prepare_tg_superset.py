"""Prepare dense TG pools only; does not train or overwrite existing caches."""

import argparse

from experiment import read_config
from tg_superset import prepare


def main(config_path, dataset):
    result = prepare(read_config(config_path), dataset)
    trainer = "train_horse.py" if dataset == "horse" else "train.py"
    print(f'train_command=python {trainer} "{config_path}"', flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--dataset", required=True, choices=("checkerboard", "horse"))
    args = parser.parse_args()
    main(args.config, args.dataset)
