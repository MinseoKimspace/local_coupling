"""Prepare 2D exact-superset NSOT, once per dataset and configured source prior.

python prepare_nsot.py checkerboard_experiments/nsot.yaml --dataset checkerboard
python prepare_nsot.py horse_experiments/horse_nsot_n256_seed0.yaml --dataset horse

Experimental fixed anchor-GMM prior with component-centered preserving noise:
python prepare_nsot.py checkerboard_experiments/nsot_anchor_prior_k8_n256_seed0.yaml --dataset checkerboard
python prepare_nsot.py horse_experiments/horse_nsot_anchor_prior_k8_n256_seed0.yaml --dataset horse

Then use the existing train.py/train_horse.py and eval.py/eval_horse.py commands.
The experimental GMM is drawn BEFORE fresh exact OT. Original component labels
and fixed centers are cached; checkpoint configs embed the same centers for
cache-free inference. Changing the prior requires a NEW cache path. Gaussian
baseline caches remain format 1; anchor-prior caches use separate format 2.
No GeomLoss, KeOps, custom CUDA extension, author-code claim or 100K approximation.
"""

import argparse

from experiment import read_config
from nsot import prepare


def main(config_path, dataset):
    return prepare(read_config(config_path), dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="NSOT experiment YAML; identical YAML is used for training")
    parser.add_argument("--dataset", required=True, choices=["checkerboard", "horse"])
    args = parser.parse_args()
    main(args.config, args.dataset)
