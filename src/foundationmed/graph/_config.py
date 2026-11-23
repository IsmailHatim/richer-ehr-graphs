"""Declarative graph-version configuration (a [graph] TOML block)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GraphConfig:
    """A named graph version: which components to apply and their parameters."""

    name: str
    components: list[str] = field(default_factory=list)
    params: dict[str, dict] = field(default_factory=dict)


def load_graph_config(path: str | Path) -> GraphConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    section = raw.get("graph", raw)
    return GraphConfig(
        name=section.get("name", Path(path).stem),
        components=list(section.get("components", [])),
        params=dict(section.get("params", {})),
    )
