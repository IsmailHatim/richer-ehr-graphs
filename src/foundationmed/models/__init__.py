"""Model architectures for patient-graph prediction."""

from foundationmed.models._encoders import (
    EVENT_TYPES,
    NODE_ORDER,
    HeteroNodeEncoder,
    homogenize,
)
from foundationmed.models._gat import GATCore, GATModel, homogenize_with_edges
from foundationmed.models._gcn import GCNCore, SimpleGCN
from foundationmed.models._hetero import (
    HETERO_EDGE_TYPES,
    HETERO_METADATA,
    HETERO_NODE_TYPES,
    HeteroBackbone,
    bidirectional_edge_attr_dict,
    bidirectional_edge_index_dict,
)
from foundationmed.models._hetero_gatv2 import HeteroGATv2Model
from foundationmed.models._hgt import HGTModel
from foundationmed.models._registry import (
    available_models,
    build_model,
    register_model,
)

__all__ = [
    "EVENT_TYPES",
    "NODE_ORDER",
    "HeteroNodeEncoder",
    "homogenize",
    "GATCore",
    "GATModel",
    "homogenize_with_edges",
    "GCNCore",
    "SimpleGCN",
    "HETERO_EDGE_TYPES",
    "HETERO_METADATA",
    "HETERO_NODE_TYPES",
    "HeteroBackbone",
    "HeteroGATv2Model",
    "HGTModel",
    "bidirectional_edge_attr_dict",
    "bidirectional_edge_index_dict",
    "available_models",
    "build_model",
    "register_model",
]
