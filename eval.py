import torch

from data import sample_checkerboard
from model import PointSetTransformer
from sample import integrate_velocity


def chamfer_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    distances = torch.cdist(prediction, target).square()
    prediction_to_target = distances.min(dim=2).values.mean(dim=1)
    target_to_prediction = distances.min(dim=1).values.mean(dim=1)
    return (prediction_to_target + target_to_prediction).mean()


def main() -> None:
    torch.manual_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    coupling = "independent"
    batch_size = 32
    n_points = 256
    integration_steps = 100

    model = PointSetTransformer().to(device=device, dtype=dtype)
    state_dict = torch.load(
        f"model_{coupling}.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    x_noise = torch.randn(
        batch_size,
        n_points,
        2,
        device=device,
        dtype=dtype,
    )
    prediction = integrate_velocity(
        model,
        x_noise,
        num_steps=integration_steps,
    )
    target = sample_checkerboard(
        batch_size,
        n_points,
        device=device,
        dtype=dtype,
    )

    score = chamfer_distance(prediction, target)
    print(f"chamfer={score.item():.6f}")


if __name__ == "__main__":
    main()
