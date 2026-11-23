"""prior: coupling edges drawn from a population lab-code prior.

local scores each patient's lab pairs independently, so an edge exists only where the
patient has enough aligned readings -- reaching 25-47% of patients, with the median
patient carrying none, and making edge existence a severity proxy (patients with
more labs drawn are sicker).

prior inverts that: the population prior says which code pairs couple, so any patient
with both codes present gets the candidate edge (~98% coverage), carrying the
population correlation as its edge attribute. Coupling strength therefore stops
being an existence criterion; selecting among the edges is left to the model's
attention gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog
import torch
from torch_geometric.transforms import BaseTransform

from foundationmed.graph._prior import CouplingPrior, _lab_arrays, fit_prior
from foundationmed.graph._registry import register_component
from foundationmed.graph._schema import attach_edges, coupling_attr

log = structlog.get_logger()

LAB = "lab"
COUPLING_EDGE = (LAB, "couples", LAB)

# Packs a canonical (code_a, code_b) pair into one int64 so prior lookup is a
# single searchsorted instead of a per-pair dict hit. Lab vocabularies are a few
# hundred codes wide; the guard in `_pack_pairs` refuses anything that would
# collide rather than silently mis-resolving a pair.
_CODE_STRIDE = 1 << 20


@register_component("coupling_prior")
class PriorCouplingEdges(BaseTransform):
    """Emit one coupling edge per present code pair the fitted prior knows."""

    def __init__(
        self,
        bin_h: float = 6.0,
        min_bins: int = 3,
        min_support: int = 30,
        max_patients: int = 20000,
        seed: int = 0,
        cache_dir: str | None = None,
    ):
        self.bin_h = float(bin_h)
        self.min_bins = int(min_bins)
        self.min_support = int(min_support)
        self.max_patients = int(max_patients)
        self.seed = int(seed)
        self.cache_dir = cache_dir
        self.prior: CouplingPrior | None = None

    def fit(self, dataset, train_idx: np.ndarray) -> None:
        """Fit the population prior on the TRAIN split only (Fittable protocol)."""
        self.prior = fit_prior(
            dataset,
            train_idx,
            bin_h=self.bin_h,
            min_bins=self.min_bins,
            min_support=self.min_support,
            max_patients=self.max_patients,
            seed=self.seed,
            cache_dir=Path(self.cache_dir) if self.cache_dir else None,
        )

    def forward(self, data):
        if self.prior is None:
            raise RuntimeError(
                "PriorCouplingEdges is not fitted; run_training fits the pipeline "
                "on the train split before installing it as the dataset transform"
            )
        src = np.zeros(0, dtype=np.int64)
        dst = np.zeros(0, dtype=np.int64)
        rho_bars = np.zeros(0)

        arrays = _lab_arrays(data)
        if arrays is not None:
            codes, _, times = arrays
            uniq, latest = _latest_node_per_code(codes, times)

            # Every present pair, in the same lexicographic (code_a < code_b)
            # order the combinations() loop emitted.
            ia, ib = np.triu_indices(uniq.size, k=1)
            keys, prior_rho = self._prior_arrays(self.prior)
            if keys.size:
                wanted = _pack_pairs(uniq[ia], uniq[ib])
                pos = np.clip(np.searchsorted(keys, wanted), 0, keys.size - 1)
                known = keys[pos] == wanted
                ia, ib, pos = ia[known], ib[known], pos[known]
            else:
                ia = ib = pos = np.zeros(0, dtype=np.int64)

            rho_bars = prior_rho[pos]
            src, dst = latest[ia], latest[ib]

        attr = (
            coupling_attr(torch.full((src.size,), self.bin_h), torch.tensor(rho_bars))
            if src.size
            else torch.empty(0)
        )
        attach_edges(data, COUPLING_EDGE, src.tolist(), dst.tolist(), attr)
        return data

    def _prior_arrays(self, prior: CouplingPrior) -> tuple[np.ndarray, np.ndarray]:
        """Sorted (packed key, rho_bar) arrays for the fitted prior.

        Built once per prior object and reused for every sample -- the table is
        the same for the whole cohort, and rebuilding it per call would put the
        per-pair dict lookup straight back into the hot path.
        """
        if getattr(self, "_lookup_for", None) is not prior:
            items = prior.items()
            if items:
                codes = np.array([pair for pair, _ in items], dtype=np.int64)
                order = np.argsort(_pack_pairs(codes[:, 0], codes[:, 1]), kind="stable")
                stats = [items[k][1] for k in order]
                lookup = (
                    _pack_pairs(codes[order, 0], codes[order, 1]),
                    np.array([s.rho_bar for s in stats]),
                )
            else:
                lookup = (np.zeros(0, dtype=np.int64), np.zeros(0))
            self._lookup_for = prior
            self._lookup = lookup
        return self._lookup


def _pack_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pack canonical code pairs into one sortable int64 key."""
    hi = max(int(a.max()) if a.size else 0, int(b.max()) if b.size else 0)
    if hi >= _CODE_STRIDE:
        raise ValueError(
            f"lab code {hi} exceeds the pair-packing stride {_CODE_STRIDE}; "
            "raise _CODE_STRIDE in _coupling_prior.py"
        )
    return a.astype(np.int64) * _CODE_STRIDE + b.astype(np.int64)


def _latest_node_per_code(
    codes: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Ascending unique codes and the node index of each one's last reading."""
    order = np.lexsort((np.arange(codes.size), -times, codes))
    ordered = codes[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    return ordered[starts], order[starts]
