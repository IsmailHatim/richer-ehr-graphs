"""Registry mapping backbone names to graph-model factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from torch import nn

from foundationmed.dataset._dataset import FeatureSpec

_MODELS: dict[str, Callable[..., nn.Module]] = {}

# Bound to the model class so the decorator returns the *same* type it receives;
# a plain Callable annotation would erase SimpleGCN/GATModel down to nn.Module.
_ModelT = TypeVar("_ModelT", bound=type[nn.Module])


def register_model(name: str) -> Callable[[_ModelT], _ModelT]:
    """Decorator: register a graph-model class under a config name (type-preserving)."""

    def deco(cls: _ModelT) -> _ModelT:
        _MODELS[name] = cls
        return cls

    return deco


def available_models() -> list[str]:
    """Names accepted by ``backbone`` in the training config."""
    import foundationmed.models  # noqa: F401  ensure built-ins register

    return sorted(_MODELS)


def build_model(
    name: str,
    spec: FeatureSpec,
    *,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
) -> nn.Module:
    """Instantiate the registered model selected by ``name``."""
    import foundationmed.models  # noqa: F401  ensure built-ins register

    if name not in _MODELS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(_MODELS)}")
    return _MODELS[name](
        spec, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout
    )
