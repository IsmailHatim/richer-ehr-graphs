"""Edge-aware heterogeneous baseline."""

from __future__ import annotations

from torch import nn
from torch_geometric.nn import GATv2Conv, HeteroConv

from foundationmed.dataset._dataset import FeatureSpec
from foundationmed.graph._schema import EDGE_DIM
from foundationmed.models._hetero import (
    HETERO_EDGE_TYPES,
    HeteroBackbone,
    bidirectional_edge_attr_dict,
)
from foundationmed.models._registry import register_model


@register_model("hetero_gatv2")
class HeteroGATv2Model(HeteroBackbone):
    """HeteroConv of per-relation edge-aware GATv2, with edge_attr routed in."""

    def __init__(
        self,
        spec: FeatureSpec,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        heads: int = 4,
    ):
        super().__init__(spec, hidden_dim, dropout)
        self.convs = nn.ModuleList(
            HeteroConv(
                {
                    et: GATv2Conv(
                        (hidden_dim, hidden_dim),
                        hidden_dim,
                        heads=heads,
                        concat=False,
                        edge_dim=EDGE_DIM,
                        add_self_loops=False,
                    )
                    for et in HETERO_EDGE_TYPES
                }
            )
            for _ in range(num_layers)
        )

    def _edges(self, data):
        return bidirectional_edge_attr_dict(data)

    def _has_edges(self, edges) -> bool:
        return bool(edges[0])

    def _call_conv(self, conv, x_dict, edges):
        edge_index_dict, edge_attr_dict = edges
        return conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
