"""Shared backbones and helpers for HeteroData."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 — torch convention
from torch import Tensor, nn
from torch_geometric.nn import global_mean_pool

from foundationmed.dataset._dataset import FeatureSpec
from foundationmed.graph._schema import reverse_attr, structural_attr
from foundationmed.models._encoders import NODE_ORDER, HeteroNodeEncoder

HETERO_NODE_TYPES: list[str] = list(NODE_ORDER)

_FORWARD_EDGE_TYPES: tuple[tuple[str, str, str], ...] = (
    ("visit", "has_diagnosis", "diagnosis"),
    ("visit", "has_treatment", "treatment"),
    ("visit", "has_medication", "medication"),
    ("visit", "has_lab", "lab"),
    ("visit", "next", "visit"),
    ("lab", "trajectory", "lab"),
    ("lab", "couples", "lab"),
)

HETERO_EDGE_TYPES: list[tuple[str, str, str]] = [
    *_FORWARD_EDGE_TYPES,
    *((dst, f"rev_{rel}", src) for src, rel, dst in _FORWARD_EDGE_TYPES),
]

HETERO_METADATA: tuple[list[str], list[tuple[str, str, str]]] = (
    HETERO_NODE_TYPES,
    HETERO_EDGE_TYPES,
)

_FORWARD_SET = frozenset(_FORWARD_EDGE_TYPES)


def bidirectional_edge_index_dict(data) -> dict[tuple[str, str, str], Tensor]:
    """Present canonical relations plus their rev_* mirrors, empty ones dropped."""
    out: dict[tuple[str, str, str], Tensor] = {}
    for edge_type in data.edge_types:
        if edge_type not in _FORWARD_SET:
            continue
        edge_index = data[edge_type].edge_index
        if edge_index.numel() == 0:
            continue
        src, rel, dst = edge_type
        out[edge_type] = edge_index
        out[(dst, f"rev_{rel}", src)] = edge_index.flip(0)
    return out


def bidirectional_edge_attr_dict(
    data,
) -> tuple[dict[tuple[str, str, str], Tensor], dict[tuple[str, str, str], Tensor]]:
    """Like bidirectional_edge_index_dict, but also returns per-relation edge_attr.

    Relations without a stored edge_attr get the constant structural attribute;
    rev_* mirrors get reverse_attr (flips Deltav on trajectory rows).
    """
    edge_index_dict: dict[tuple[str, str, str], Tensor] = {}
    edge_attr_dict: dict[tuple[str, str, str], Tensor] = {}
    for edge_type in data.edge_types:
        if edge_type not in _FORWARD_SET:
            continue
        edge_index = data[edge_type].edge_index
        if edge_index.numel() == 0:
            continue
        src, rel, dst = edge_type
        attr = getattr(data[edge_type], "edge_attr", None)
        if attr is None:
            attr = structural_attr(edge_index.size(1), device=edge_index.device)
        edge_index_dict[edge_type] = edge_index
        edge_attr_dict[edge_type] = attr
        rev = (dst, f"rev_{rel}", src)
        edge_index_dict[rev] = edge_index.flip(0)
        edge_attr_dict[rev] = reverse_attr(attr)
    return edge_index_dict, edge_attr_dict


class HeteroBackbone(nn.Module):
    """Shared encode to typed conv stack to visit readout; subclasses set convs."""

    convs: nn.ModuleList

    def __init__(self, spec: FeatureSpec, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = HeteroNodeEncoder(spec, hidden_dim)
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _edges(self, data):
        """Edge inputs passed to each conv. Base: edge_index_dict (types only)."""
        return bidirectional_edge_index_dict(data)

    def _has_edges(self, edges) -> bool:
        return bool(edges)

    def _call_conv(self, conv, x_dict, edges):
        return conv(x_dict, edges)

    def forward(self, data) -> Tensor:
        x_dict = self.encoder(data)
        edges = self._edges(data)
        if self._has_edges(edges):
            for conv in self.convs:
                out = self._call_conv(conv, x_dict, edges)
                # identity skip: node types receiving no messages keep their
                # previous embedding so they survive to the next layer
                x_dict = {
                    ntype: h if (h := out.get(ntype)) is not None else x_dict[ntype]
                    for ntype in x_dict
                }
                x_dict = {
                    ntype: F.dropout(F.relu(h), p=self.dropout, training=self.training)
                    for ntype, h in x_dict.items()
                }
        visit = x_dict["visit"]
        batch = getattr(data["visit"], "batch", None)
        if batch is None:
            batch = visit.new_zeros(visit.size(0), dtype=torch.long)
        return self.head(global_mean_pool(visit, batch)).squeeze(-1)
