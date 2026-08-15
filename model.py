import torch
from torch import nn


class PointSetTransformer(nn.Module):
    def __init__(
        self,
        *,
        point_dim: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(point_dim + 1, d_model)

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
        t = t.expand(-1, x_t.shape[1], -1)
        h = self.input_proj(torch.cat([x_t, t], dim=-1))
        h = self.encoder(h)
        return self.output_proj(h)
