"""Configurable graph-construction subsystem: composable transform representations."""

from foundationmed.graph._config import GraphConfig, load_graph_config
from foundationmed.graph._registry import build_pipeline, register_component
from foundationmed.graph._schema import EDGE_DIM

__all__ = [
    "EDGE_DIM",
    "GraphConfig",
    "load_graph_config",
    "build_pipeline",
    "register_component",
]
