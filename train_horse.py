import math
import sys

import torch
from skimage.data import horse
from torch import nn

from experiment import read_config
from train import train_model


class HorsePointSetTransformer(nn.Module):
    def __init__(
        self,
        *,
        point_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(point_dim, d_model)
        frequencies = 1000.0 * torch.exp(
            -math.log(10000.0) * torch.linspace(0.0, 1.0, d_model // 2)
        )
        self.register_buffer("time_frequencies", frequencies)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_proj = nn.Linear(d_model, point_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        angles = t[:, 0, 0].unsqueeze(-1) * self.time_frequencies.unsqueeze(0)
        time_embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        time_embedding = self.time_proj(time_embedding).unsqueeze(1)
        h = self.input_proj(x_t) + time_embedding
        h = self.encoder(h)
        return self.output_proj(h)


def load_horse_mask(
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(~horse(), device=device, dtype=dtype)


def sample_horse(
    mask: torch.Tensor,
    batch_size: int,
    n_points: int,
) -> torch.Tensor:
    height, width = mask.shape
    indices = torch.multinomial(
        mask.flatten(),
        batch_size * n_points,
        replacement=True,
    ).reshape(batch_size, n_points)
    rows = torch.div(indices, width, rounding_mode="floor")
    columns = indices % width
    jitter = torch.rand(
        batch_size,
        n_points,
        2,
        device=mask.device,
        dtype=mask.dtype,
    )
    scale = float(max(height, width))
    x = (columns.to(mask.dtype) + jitter[..., 0] - width / 2.0) * 2.0 / scale
    y = (height / 2.0 - rows.to(mask.dtype) - jitter[..., 1]) * 2.0 / scale
    return torch.stack([x, y], dim=-1)


def main(config_path: str = "horse_experiments/horse_independent_n256_seed0.yaml"):
    config = read_config(config_path)
    torch.manual_seed(config["seed"])
    device, dtype = torch.device(config["device"]), getattr(torch, config["dtype"])
    data = config["data"]
    mask = load_horse_mask(device, dtype)
    model = HorsePointSetTransformer(**config["model"]).to(device=device, dtype=dtype)
    return train_model(
        model, config, lambda: sample_horse(mask, data["batch_size"], data["n_points"]),
        dataset="horse", config_path=config_path,
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
