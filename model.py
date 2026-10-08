import torch
from torch import nn

import anchor_conditioning


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
        anchor_id_count: int = 0,
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
        # Construct last, without random initialization, for an exact backbone/RNG control.
        self.anchor_id_count = anchor_conditioning.embedding_count(anchor_id_count)
        self.anchor_embedding = anchor_conditioning.make_embedding(self.anchor_id_count, d_model)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                component_ids: torch.Tensor | None = None) -> torch.Tensor:
        t = t.expand(-1, x_t.shape[1], -1)
        h = self.input_proj(torch.cat([x_t, t], dim=-1))
        embedding = anchor_conditioning.embedding_features(self.anchor_embedding, component_ids, x_t)
        if embedding is not None:
            h = h + embedding
        h = self.encoder(h)
        return self.output_proj(h)
