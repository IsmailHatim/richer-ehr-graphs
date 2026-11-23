"""SimpleGCN: a graph-level mortality classifier over patient graphs."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 — torch convention
from torch import Tensor, nn
from torch_geometric.nn import GCNConv, global_mean_pool

from foundationmed.dataset._dataset import FeatureSpec
from foundationmed.models._encoders import HeteroNodeEncoder, homogenize
from foundationmed.models._registry import register_model


class GCNCore(nn.Module):
    """Homogeneous GCN encoder + graph-level readout. Operates on raw tensors."""

    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList(
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, batch: Tensor | None = None
    ) -> Tensor:
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        pooled = global_mean_pool(x, batch)
        return self.head(pooled).squeeze(-1)  # (num_graphs,)


@register_model("gcn")
class SimpleGCN(nn.Module):
    """Heterogeneous-input, homogeneous-message-passing mortality classifier."""

    def __init__(
        self,
        spec: FeatureSpec,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder = HeteroNodeEncoder(spec, hidden_dim)
        self.core = GCNCore(hidden_dim, num_layers, dropout)

    def forward(self, data) -> Tensor:
        x, edge_index, batch = self.encode_homogeneous(data)
        return self.core(x, edge_index, batch)

    def encode_homogeneous(self, data) -> tuple[Tensor, Tensor, Tensor]:
        """Return the homogenized (x, edge_index, batch) for a graph or batch."""
        encoded = self.encoder(data)
        return homogenize(encoded, data)
