"""Evolution edges: consecutive same-code lab readings, carrying (Δt, Δv)."""

from __future__ import annotations

import torch
from torch_geometric.transforms import BaseTransform

from foundationmed.graph._registry import register_component
from foundationmed.graph._schema import attach_edges, trajectory_attr

LAB = "lab"
TRAJECTORY_EDGE = (LAB, "trajectory", LAB)


@register_component("trajectory")
class TrajectoryEdges(BaseTransform):
    """Link consecutive occurrences of the same lab code in time."""

    def forward(self, data):
        store = data[LAB]
        src: list[int] = []
        dst: list[int] = []
        attr = None
        if store.num_nodes > 1:
            codes = store.x[:, 0]
            values = store.x[:, 1]
            times = store.t
            # Lexsort by (code, time): a stable sort on time, then a stable sort
            # on code, leaves each code's readings contiguous and time-ordered,
            # with equal timestamps still breaking by ascending node index.
            order = torch.argsort(times, stable=True)
            order = order[torch.argsort(codes[order], stable=True)]
            # Consecutive pairs in that order are the trajectory edges, minus
            # the ones straddling a code boundary.
            a, b = order[:-1], order[1:]
            same_code = codes[a] == codes[b]
            a, b = a[same_code], b[same_code]
            if a.numel():
                src, dst = a.tolist(), b.tolist()
                attr = trajectory_attr(times[b] - times[a], values[b] - values[a])
        attach_edges(
            data,
            TRAJECTORY_EDGE,
            src,
            dst,
            attr if attr is not None else torch.empty(0),
        )
        return data
