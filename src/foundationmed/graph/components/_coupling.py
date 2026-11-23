"""local: cross-variable coupling edges from windowed Pearson over lab values.

Deterministic, no learned topology. Slides a fixed window over the patient's lab
timeline; within each window, for every pair of distinct codes with enough aligned
readings, computes Pearson rho and adds an edge iff |rho| >= threshold. Edge feature
= (window span /norm, signed rho).
"""

from __future__ import annotations

import itertools

import torch
from torch import Tensor
from torch_geometric.transforms import BaseTransform

from foundationmed.graph._registry import register_component
from foundationmed.graph._schema import attach_edges, coupling_attr

LAB = "lab"
COUPLING_EDGE = (LAB, "couples", LAB)


def _bucket_means(times: Tensor, values: Tensor, buckets: Tensor) -> Tensor:
    """One mean value per bucket; NaN where a bucket has no reading."""
    out = torch.full((buckets.numel() - 1,), float("nan"))
    for i in range(buckets.numel() - 1):
        m = (times >= buckets[i]) & (times < buckets[i + 1])
        if i == buckets.numel() - 2:  # last bucket is closed on the right
            m = m | (times == buckets[i + 1])
        if bool(m.any()):
            out[i] = values[m].mean()
    return out


def _pearson(a: Tensor, b: Tensor) -> float | None:
    if a.numel() < 2:
        return None
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if float(denom) == 0.0:
        return None
    return float((a * b).sum() / denom)


@register_component("coupling")
class CouplingEdges(BaseTransform):
    def __init__(
        self,
        window_h: float = 24.0,
        threshold: float = 0.5,
        min_overlap: int = 2,
        n_buckets: int | None = None,
    ):
        self.window_h = float(window_h)
        self.threshold = float(threshold)
        self.min_overlap = int(min_overlap)
        self.n_buckets = int(n_buckets) if n_buckets is not None else None

    def forward(self, data):
        store = data[LAB]
        src: list[int] = []
        dst: list[int] = []
        dts: list[float] = []
        rhos: list[float] = []
        n = store.num_nodes
        if n >= 2 * self.min_overlap:
            codes, values, times = store.x[:, 0], store.x[:, 1], store.t
            t0 = float(times.min())
            # Only Windows that hold readings.
            order = torch.argsort(times, stable=True)
            ts = times[order]
            kidx = torch.div(ts - t0, self.window_h, rounding_mode="floor")
            uniq, counts = torch.unique_consecutive(kidx, return_counts=True)
            ends = torch.cumsum(counts, 0).tolist()
            start = 0
            for k, end in zip(uniq.tolist(), ends):
                in_win, _ = torch.sort(order[start:end])  # ascending node index
                start = end
                if in_win.numel() < 2 * self.min_overlap:
                    continue
                lo = t0 + k * self.window_h
                hi = lo + self.window_h
                nb = self.n_buckets or self.min_overlap
                buckets = torch.linspace(lo, hi, nb + 1)
                series: dict[int, tuple[Tensor, int]] = {}
                for code in torch.unique(codes[in_win]):
                    cidx = in_win[codes[in_win] == code]
                    if cidx.numel() < self.min_overlap:
                        continue
                    bvals = _bucket_means(times[cidx], values[cidx], buckets)
                    latest = int(cidx[torch.argmax(times[cidx])])
                    series[int(code)] = (bvals, latest)
                for (_, (va, la)), (_, (vb, lb)) in itertools.combinations(
                    series.items(), 2
                ):
                    mask = ~(torch.isnan(va) | torch.isnan(vb))
                    if int(mask.sum()) < self.min_overlap:
                        continue
                    rho = _pearson(va[mask], vb[mask])
                    if rho is None or abs(rho) < self.threshold:
                        continue
                    src.append(la)
                    dst.append(lb)
                    dts.append(self.window_h)
                    rhos.append(rho)
        attr = coupling_attr(torch.tensor(dts), torch.tensor(rhos)) if src else None
        attach_edges(
            data, COUPLING_EDGE, src, dst, attr if attr is not None else torch.empty(0)
        )
        return data
