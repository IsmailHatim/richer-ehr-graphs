"""Precomputed edges."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import structlog
import torch
from torch_geometric.transforms import BaseTransform, Compose

from foundationmed.graph._registry import Fittable
from foundationmed.graph._schema import EDGE_DIM
from foundationmed.graph.components._coupling import COUPLING_EDGE, CouplingEdges

log = structlog.get_logger()

CACHE_VERSION = 1
DEFAULT_CACHE_DIR = Path("data/cache")


def cache_key(
    dataset_name: str,
    task: str,
    min_events: int,
    max_visits: int | None,
    params: dict,
) -> str:
    """Identity of a cache: `task`/`min_events`/`max_visits`."""
    payload = json.dumps(
        {
            "v": CACHE_VERSION,
            "dataset": dataset_name,
            "task": task,
            "min_events": int(min_events),
            "max_visits": None if max_visits is None else int(max_visits),
            "params": {k: params[k] for k in sorted(params)},
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_path(cache_dir: str | Path, key: str) -> Path:
    return Path(cache_dir) / f"coupling-{key}.pt"


def _digest(samples: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(samples).tobytes()).hexdigest()[:16]


def build_cache(dataset, coupling, path: str | Path, key: str) -> Path:
    """Run `coupling` over every sample on raw values and store the edges."""
    if isinstance(coupling, Fittable):
        raise TypeError(
            f"{type(coupling).__name__} is Fittable: it estimates population "
            "statistics and must be fitted on the train split, so caching it "
            "cohort-wide would leak val/test into the representation."
        )
    stats = dataset.lab_stats
    dataset.lab_stats = None  # raw values: cache independent of the split
    try:
        idx_parts: list[torch.Tensor] = []
        attr_parts: list[torch.Tensor] = []
        offsets = [0]
        for i in range(len(dataset)):
            data = coupling(dataset.get(i))
            ei = data[COUPLING_EDGE].edge_index
            idx_parts.append(ei.to(torch.long))
            attr_parts.append(data[COUPLING_EDGE].edge_attr.to(torch.float))
            offsets.append(offsets[-1] + int(ei.size(1)))
            if (i + 1) % 20000 == 0:
                log.info("cache_build_progress", done=i + 1, total=len(dataset))
    finally:
        dataset.lab_stats = stats

    blob = {
        "version": CACHE_VERSION,
        "key": key,
        "samples_digest": _digest(dataset.samples),
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "edge_index": torch.cat(idx_parts, dim=1)
        if idx_parts
        else torch.zeros((2, 0), dtype=torch.long),
        "edge_attr": torch.cat(attr_parts, dim=0)
        if attr_parts
        else torch.zeros((0, EDGE_DIM), dtype=torch.float),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path)
    log.info(
        "coupling_cache_built",
        path=str(path),
        samples=len(dataset),
        edges=int(blob["edge_index"].size(1)),
        mb=round(path.stat().st_size / 1e6, 1),
    )
    return path


class CachedCouplingEdges(BaseTransform):
    """Attach precomputed coupling edges instead of recomputing them."""

    def __init__(
        self, edge_index: torch.Tensor, edge_attr: torch.Tensor, offsets: torch.Tensor
    ):
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.offsets = offsets

    def forward(self, data):
        i = int(data.sample_idx)
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        store = data[COUPLING_EDGE]
        if hi > lo:
            # Clone: a view would keep the whole cache storage alive and drag it
            # through DataLoader worker pickling.
            store.edge_index = self.edge_index[:, lo:hi].clone()
            store.edge_attr = self.edge_attr[lo:hi].clone()
        else:
            store.edge_index = torch.zeros((2, 0), dtype=torch.long)
            store.edge_attr = torch.zeros((0, EDGE_DIM), dtype=torch.float)
        return data


def load_cache(path: str | Path, key: str, samples: np.ndarray) -> CachedCouplingEdges:
    """Load a cache, refusing one that was built for a different sample set."""
    blob = torch.load(path, weights_only=True)
    if blob.get("version") != CACHE_VERSION or blob.get("key") != key:
        raise ValueError(
            f"coupling cache at {path} has key {blob.get('key')!r} "
            f"(version {blob.get('version')}), expected {key!r} v{CACHE_VERSION}"
        )
    if blob.get("samples_digest") != _digest(samples):
        raise ValueError(
            f"coupling cache at {path} was built for a different samples array; "
            "rebuild it with `fmed cache-graph`"
        )
    return CachedCouplingEdges(blob["edge_index"], blob["edge_attr"], blob["offsets"])


def attach_cache(
    pipeline: Compose, path: str | Path, key: str, samples: np.ndarray
) -> Compose:
    """Swap `CouplingEdges` for its cached twin when a matching cache exists."""
    if not Path(path).exists():
        log.info(
            "coupling_cache_miss",
            path=str(path),
            note="computing coupling every epoch; `fmed cache-graph` precomputes it",
        )
        return pipeline
    cached = load_cache(path, key, samples)
    steps = [
        cached if isinstance(step, CouplingEdges) else step
        for step in pipeline.transforms
    ]
    log.info("coupling_cache_loaded", path=str(path))
    return Compose(steps)
