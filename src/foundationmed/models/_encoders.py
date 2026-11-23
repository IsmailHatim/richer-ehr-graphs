"""Per-node-type input encoders for heterogeneous patient graphs."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from foundationmed.dataset._dataset import FeatureSpec

# For reproducibility
NODE_ORDER: tuple[str, ...] = ("visit", "diagnosis", "treatment", "medication", "lab")
EVENT_TYPES: tuple[str, ...] = ("diagnosis", "treatment", "medication", "lab")


class HeteroNodeEncoder(nn.Module):
    """Encode each node type of a (possibly batched) HeteroData to hidden_dim."""

    def __init__(self, spec: FeatureSpec, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.code_emb = nn.ModuleDict(
            {
                etype: nn.Embedding(max(spec.vocab_sizes.get(etype, 0), 1), hidden_dim)
                for etype in EVENT_TYPES
            }
        )
        self.lab_value = nn.Linear(1, hidden_dim)

        # General extra-node-feature rail: project the columns a component
        self.extra_dims: dict[str, int] = {
            etype: dim
            for etype, dim in spec.extra_node_dims.items()
            if etype in EVENT_TYPES and dim > 0
        }
        self.extra_proj = nn.ModuleDict(
            {
                etype: nn.Linear(dim, hidden_dim)
                for etype, dim in self.extra_dims.items()
            }
        )

        self.gender_emb = nn.Embedding(spec.n_genders, hidden_dim)
        self.ethnicity_emb = nn.Embedding(spec.n_ethnicities, hidden_dim)
        self.visit_proj = nn.Linear(spec.apache_dim + spec.demo_cont_dim, hidden_dim)

    def forward(self, data) -> dict[str, Tensor]:
        device = data["visit"].x.device

        out: dict[str, Tensor] = {}
        out["visit"] = self._encode_visits(data["visit"])

        for etype in EVENT_TYPES:
            store = data[etype]
            if store.num_nodes == 0:
                out[etype] = torch.zeros(0, self.hidden_dim, device=device)
                continue
            codes = store.x[:, 0].long()
            h = self.code_emb[etype](codes)
            if etype == "lab":
                h = h + self.lab_value(store.x[:, 1:2])
            dim = self.extra_dims.get(etype, 0)
            if dim > 0:
                h = h + self.extra_proj[etype](store.x[:, -dim:])
            out[etype] = h

        return out

    def _encode_visits(self, store) -> Tensor:
        cont = torch.stack([store.age / 100.0, store.duration_h / 24.0], dim=1)
        projected = self.visit_proj(torch.cat([store.x, cont], dim=1))
        return (
            projected
            + self.gender_emb(store.gender)
            + self.ethnicity_emb(store.ethnicity)
        )


def homogenize(encoded: dict[str, Tensor], data) -> tuple[Tensor, Tensor, Tensor]:
    """Flatten encoded heterogeneous nodes into a single homogeneous graph."""
    device = encoded["visit"].device

    xs: list[Tensor] = []
    batches: list[Tensor] = []
    offset: dict[str, int] = {}
    cursor = 0
    for ntype in NODE_ORDER:
        h = encoded[ntype]
        offset[ntype] = cursor
        xs.append(h)
        store = data[ntype]
        store_batch = getattr(store, "batch", None)
        if store_batch is None:
            store_batch = torch.zeros(h.size(0), dtype=torch.long, device=device)
        batches.append(store_batch)
        cursor += h.size(0)

    x = torch.cat(xs, dim=0)
    batch = torch.cat(batches, dim=0)

    edge_segments: list[Tensor] = []
    for edge_type in data.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = data[edge_type].edge_index
        if edge_index.numel() == 0:
            continue
        src = edge_index[0] + offset[src_type]
        dst = edge_index[1] + offset[dst_type]
        edge_segments.append(torch.stack([src, dst]))
        edge_segments.append(torch.stack([dst, src]))  # reverse edge

    if edge_segments:
        edge_index = torch.cat(edge_segments, dim=1)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

    return x, edge_index, batch
