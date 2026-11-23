"""HGT model: Heterogeneous Graph Transformer over the native typed graph."""

from __future__ import annotations

from torch import nn
from torch_geometric.nn import HGTConv

from foundationmed.dataset._dataset import FeatureSpec
from foundationmed.models._hetero import HETERO_METADATA, HeteroBackbone
from foundationmed.models._registry import register_model


@register_model("hgt")
class HGTModel(HeteroBackbone):
    """Heterogeneous Graph Transformer: typed attention over the full schema."""

    def __init__(
        self,
        spec: FeatureSpec,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        heads: int = 4,
    ):
        if hidden_dim % heads != 0:
            raise ValueError(
                f"hgt needs hidden_dim divisible by heads: {hidden_dim} % {heads}"
            )
        super().__init__(spec, hidden_dim, dropout)
        self.convs = nn.ModuleList(
            HGTConv(hidden_dim, hidden_dim, HETERO_METADATA, heads)
            for _ in range(num_layers)
        )
