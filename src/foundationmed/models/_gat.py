"""Edge-aware GAT model: carry edge features through homogenization into GATv2."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 — torch convention
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

from foundationmed.dataset._dataset import FeatureSpec
from foundationmed.graph._schema import EDGE_DIM, reverse_attr, structural_attr
from foundationmed.models._encoders import NODE_ORDER, HeteroNodeEncoder
from foundationmed.models._registry import register_model


def homogenize_with_edges(
    encoded: dict[str, Tensor], data
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Flatten a HeteroData to (x, edge_index, edge_attr, batch), bidirectional.

    Structural edge types (no edge_attr) get the constant structural attribute;
    trajectory/coupling edges carry their own. Reverse edges flip Deltav on trajectory.
    """
    device = encoded["visit"].device
    xs: list[Tensor] = []
    batches: list[Tensor] = []
    offset: dict[str, int] = {}
    cursor = 0
    for ntype in NODE_ORDER:
        h = encoded[ntype]
        offset[ntype] = cursor
        xs.append(h)
        store_batch = getattr(data[ntype], "batch", None)
        if store_batch is None:
            store_batch = torch.zeros(h.size(0), dtype=torch.long, device=device)
        batches.append(store_batch)
        cursor += h.size(0)

    x = torch.cat(xs, dim=0)
    batch = torch.cat(batches, dim=0)

    idx_segs: list[Tensor] = []
    attr_segs: list[Tensor] = []
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = data[edge_type].edge_index.to(device)
        if edge_index.numel() == 0:
            continue
        attr = getattr(data[edge_type], "edge_attr", None)
        if attr is None:
            attr = structural_attr(edge_index.size(1), device=device)
        attr = attr.to(device)
        src = edge_index[0] + offset[src_type]
        dst = edge_index[1] + offset[dst_type]
        idx_segs.append(torch.stack([src, dst]))
        attr_segs.append(attr)
        idx_segs.append(torch.stack([dst, src]))  # reverse
        attr_segs.append(reverse_attr(attr))

    if idx_segs:
        edge_index = torch.cat(idx_segs, dim=1)
        edge_attr = torch.cat(attr_segs, dim=0)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_attr = torch.zeros(0, EDGE_DIM, dtype=torch.float, device=device)
    return x, edge_index, edge_attr, batch


class GATCore(nn.Module):
    """Edge-aware GATv2 stack + graph-level readout over homogenized tensors."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.5,
        edge_dim: int = EDGE_DIM,
    ):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList(
            GATv2Conv(
                hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=edge_dim
            )
            for _ in range(num_layers)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        return self.head(global_mean_pool(x, batch)).squeeze(-1)


@register_model("gat")
class GATModel(nn.Module):
    """Heterogeneous-input, edge-aware GATv2 mortality classifier."""

    def __init__(
        self,
        spec: FeatureSpec,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.encoder = HeteroNodeEncoder(spec, hidden_dim)
        self.core = GATCore(hidden_dim, num_layers, heads, dropout)

    def forward(self, data) -> Tensor:
        encoded = self.encoder(data)
        x, edge_index, edge_attr, batch = homogenize_with_edges(encoded, data)
        return self.core(x, edge_index, edge_attr, batch)
