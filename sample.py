import torch

from torch import nn

def integrate_velocity(
    model: nn.Module,
    x_noise: torch.Tensor,
    *,
    num_steps: int,
) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    x = x_noise
    dt = 1.0 / num_steps
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for step in range(num_steps):
            t = torch.full(
                (x.shape[0], 1, 1),
                step * dt,
                device=x.device,
                dtype=x.dtype,
            )
            x = x + dt * model(x, t)

    model.train(was_training)
    return x
