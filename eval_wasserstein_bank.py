"""Fresh-Gaussian inference; evaluates multi-shape 2D cloud distributions."""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiment import (evaluation_settings, evaluation_title, load_model,
                        sample_for_evaluation, save_evaluation)
from model import PointSetTransformer
from wasserstein_bank import dataset_spec, target_points


@torch.no_grad()
def cd_matrix(x, y):
    result = torch.empty(len(x), len(y), device=x.device)
    for i in range(len(x)):
        for j in range(0, len(y), 4):
            distance = torch.cdist(x[i:i+1], y[j:j+4]).square()
            result[i, j:j+4] = distance.amin(-1).mean(-1) + distance.amin(-2).mean(-1)
    return result


@torch.no_grad()
def distribution_metrics(generated, reference):
    if min(len(generated), len(reference)) < 2:
        raise ValueError("Cloud-distribution evaluation requires >=2 clouds")
    cross, gg, rr = cd_matrix(generated, reference), cd_matrix(generated, generated), cd_matrix(reference, reference)
    gg.fill_diagonal_(float("inf"))
    rr.fill_diagonal_(float("inf"))
    distances = torch.cat((torch.cat((gg, cross), 1), torch.cat((cross.T, rr), 1)), 0)
    labels = torch.arange(len(distances), device=distances.device) >= len(generated)
    predicted = labels[distances.argmin(1)]
    one_nna = ((predicted[:len(generated)] == labels[:len(generated)]).float().mean()
               + (predicted[len(generated):] == labels[len(generated):]).float().mean()) / 2
    return {"mmd_cd": cross.amin(0).mean().item(),
            "cov_cd": cross.argmin(1).unique().numel() / len(reference),
            "one_nna_cd": one_nna.item()}


def main(config_path, num_steps=64, *, render=True):
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    model, config, checkpoint, metadata = load_model(config_path, PointSetTransformer, "mixture2d")
    opts = evaluation_settings(config)
    spec = dataset_spec(config, "mixture2d")
    if opts["batch_size"] < 2 * len(spec["shapes"]):
        raise ValueError("Use >=2 reference clouds per shape for mixture evaluation")
    _, generated, seconds = sample_for_evaluation(model, config, num_steps)
    rng = np.random.default_rng(opts["seed"] + 100003)
    # Stratified fresh references, independent of every cached target point.
    shape_ids = np.arange(opts["batch_size"]) % len(spec["shapes"])
    reference = torch.as_tensor(np.stack([target_points(config, "mixture2d", spec["shapes"][s],
                                                        config["data"]["n_points"], rng) for s in shape_ids]),
                                device=generated.device, dtype=generated.dtype)
    scores = distribution_metrics(generated, reference)
    cross = cd_matrix(generated, reference)
    predicted_shapes = np.asarray(shape_ids)[cross.argmin(1).cpu().numpy()]
    # Nearest-reference assignments are a proxy, not a calibrated shape classifier.
    counts = np.bincount(predicted_shapes, minlength=len(spec["shapes"])) / len(generated)
    scores["shape_mass_tv_proxy"] = float(np.abs(counts - 1 / len(counts)).sum() / 2)
    output = save_evaluation(config_path, config, checkpoint, metadata, "mixture2d", num_steps, seconds,
                             scores, render=render,
                             metric_metadata={"dataset_spec": spec, "predicted_shape_mass": counts.tolist(),
                                              "shape_assignment": "nearest fresh reference by CD; proxy",
                                              "mmd_cd": "reference-to-generated minimum mean CD; Minimum Matching Distance",
                                              "cov_cd": "fraction of reference clouds selected by generated nearest neighbors",
                                              "one_nna_cd": "balanced leave-one-out two-sample NN accuracy; ideal approximately 0.5"})
    if render:
        fig, axes = plt.subplots(2, 6, figsize=(12, 4.5))
        for j in range(6):
            for row, clouds, name in ((0, reference, "Reference"), (1, generated, "Generated")):
                cloud = clouds[j % len(clouds)].cpu().numpy()
                axes[row, j].scatter(cloud[:, 0], cloud[:, 1], s=3)
                axes[row, j].set(xlim=(-1.3, 1.3), ylim=(-1.3, 1.3), aspect="equal", title=name)
        fig.suptitle(f"{evaluation_title(config)} | NFE={num_steps}")
        fig.tight_layout()
        fig.savefig(output, dpi=160)
        plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("num_steps", nargs="?", type=int, default=64)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    main(args.config, args.num_steps, render=not args.no_render)
