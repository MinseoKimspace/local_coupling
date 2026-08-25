import math
import sys
from time import perf_counter

import torch
from skimage.data import horse
from torch import nn
import yaml

from train import train_step


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


def main(config_path: str = "horse_independent.yaml") -> None:
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])
    data_config = config["data"]
    training_config = config["training"]
    mask = load_horse_mask(device, dtype)

    model = HorsePointSetTransformer(**config["model"]).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    coupling_generator = torch.Generator(device=device)
    coupling_generator.manual_seed(config["seed"] + 1)
    model.train()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_start = perf_counter()

    for step in range(1, training_config["num_steps"] + 1):
        x_data = sample_horse(
            mask,
            data_config["batch_size"],
            data_config["n_points"],
        )
        loss = train_step(
            model,
            optimizer,
            x_data,
            coupling=config["coupling"],
            num_regions=config.get("num_regions"),
            coupling_generator=coupling_generator,
        )

        if step == 1 or step % training_config["log_every"] == 0:
            print(f"step={step} loss={loss.item():.6f}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = perf_counter() - training_start
    torch.save(model.state_dict(), config["checkpoint"])
    print(f"training_seconds={training_seconds:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
