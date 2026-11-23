"""Registry mapping component names to BaseTransform factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
import structlog
from torch_geometric.transforms import BaseTransform, Compose

from foundationmed.graph._config import GraphConfig

log = structlog.get_logger()

_COMPONENTS: dict[str, Callable[..., BaseTransform]] = {}

F = TypeVar("F", bound=Callable[..., BaseTransform])


def register_component(name: str) -> Callable[[F], F]:
    """Decorator: register a BaseTransform factory under a config name."""

    def deco(factory: F) -> F:
        _COMPONENTS[name] = factory
        return factory

    return deco


def build_pipeline(cfg: GraphConfig) -> Compose:
    """Turn a GraphConfig into a composed transform. Empty gives identity."""
    import foundationmed.graph.components  # noqa: F401  ensure built-ins register

    steps: list[Callable] = []
    for name in cfg.components:
        if name not in _COMPONENTS:
            raise KeyError(
                f"unknown graph component {name!r}; available: {sorted(_COMPONENTS)}"
            )
        steps.append(_COMPONENTS[name](**cfg.params.get(name, {})))
    return Compose(steps)


@runtime_checkable
class Fittable(Protocol):
    """A graph component that must be fitted on the TRAIN split before use.

    Any component estimating population statistics implements this, so that
    training/_train.py can fit it after the split rather than before -- fitting
    before the split would leak val/test patients into the representation.
    """

    def fit(self, dataset, train_idx: np.ndarray) -> None: ...


def fit_pipeline(pipeline: Compose, dataset, train_idx: np.ndarray) -> None:
    """Fit every Fittable member of `pipeline` on `train_idx`. Others are no-ops."""
    for step in pipeline.transforms:
        if isinstance(step, Fittable):
            log.info("fitting_component", component=type(step).__name__)
            step.fit(dataset, train_idx)
