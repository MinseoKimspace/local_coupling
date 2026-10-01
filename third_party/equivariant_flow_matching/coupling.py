"""Permutation-only adaptation of the Equivariant Flow Matching supplement.

See README.md and original_coupling.txt for provenance and the removed rotations.
"""

import numpy as np
import ot as pot
import torch
from scipy.optimize import linear_sum_assignment


@torch.no_grad()
def sample_permutation_plan(source, target):
    """Return coupled [B, N, D] clouds, sampling the cloud plan with replacement."""
    if source.ndim != 3 or source.shape != target.shape or min(source.shape) < 1:
        raise ValueError("Source and target must have the same nonempty [B, N, D] shape")
    if (source.device != target.device or source.dtype != target.dtype
            or not source.is_floating_point()):
        raise ValueError("Source and target must share a floating dtype and device")

    # The original notebook solves on CPU. Adapt tensor placement at this boundary.
    work_dtype = source.dtype if source.dtype in (torch.float32, torch.float64) else torch.float32
    x0 = source.detach().to(device="cpu", dtype=work_dtype)
    x1 = target.detach().to(device="cpu", dtype=work_dtype)
    batchsize, n_particles, n_dimensions = x0.shape

    # Resample x0, x1 according to transport matrix
    a1, b1 = pot.unif(x0.size()[0]), pot.unif(x1.size()[0])
    M = torch.zeros(batchsize, batchsize)
    for i in range(batchsize):
        points2_reordered = []
        points1 = x0[i].reshape(n_particles, n_dimensions)
        for j in range(batchsize):
            points2 = x1[j].reshape(n_particles, n_dimensions)
            distances = torch.cdist(points1, points2)
            row_idx, col_idx = linear_sum_assignment(distances**2)
            points2_reordered.append(points2[col_idx].unsqueeze(0))

        # Adaptation: retain the matched target coordinates without SVD alignment.
        M[i] = torch.cdist(
            points1.reshape(1, -1),
            torch.cat(points2_reordered, dim=0).reshape(batchsize, -1),
        )[0]

    M = M**2
    if M.max() > 0:
        M = M / M.max()
    pi = pot.emd(a1, b1, M.detach().cpu().numpy())
    # Sample random interpolations on pi; numpy's default is replacement=True.
    p = pi.flatten()
    p = p / p.sum()
    choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batchsize)
    i, j = np.divmod(choices, pi.shape[1])
    x0 = x0[i]
    x1 = x1[j]
    for i in range(batchsize):
        points1 = x0[i].reshape(n_particles, n_dimensions)
        points2 = x1[i].reshape(n_particles, n_dimensions)
        distances = torch.cdist(points1, points2)
        row_idx, col_idx = linear_sum_assignment(distances**2)
        # Adaptation: apply only the selected pair's permutation.
        x1[i] = points2[col_idx]

    return (
        x0.to(device=source.device, dtype=source.dtype),
        x1.to(device=target.device, dtype=target.dtype),
    )
