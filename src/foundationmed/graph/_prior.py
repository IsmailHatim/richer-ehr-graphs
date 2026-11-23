"""Population lab-code coupling prior, fitted on the training split only."""

from __future__ import annotations

import hashlib
import itertools
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

log = structlog.get_logger()

# Bounded per-pair reservoir of (va, vb) vectors kept for the permutation null.
# rho_bar uses streaming sums over every observation; only the null needs vectors.
_NULL_CAP = 200

# Minimum points a null sample needs. Mirrors the `min_bins=3` overlap gate in
# fit_prior: at 2 points _pearson returns +-1 by construction, which is not a
# correlation but an artefact, and would corrupt the null's moments.
_MIN_NULL_POINTS = 3


def _cache_key(
    train_idx: np.ndarray,
    bin_h: float,
    min_bins: int,
    min_support: int,
    max_patients: int,
    seed: int,
) -> str:
    """Hash over the split AND every parameter that changes the estimate.

    train_idx is part of the key: the split changes per seed, and reusing another
    split's prior would leak val/test patients into training.
    """
    h = hashlib.sha256()
    h.update(np.asarray(train_idx, dtype=np.int64).tobytes())
    h.update(repr((bin_h, min_bins, min_support, max_patients, seed)).encode())
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class PairStat:
    """Population coupling statistics for one lab-code pair."""

    rho_bar: float  # signed mean correlation across patients
    z: float  # Welch: (rho_bar - null_mean) / sqrt(s^2/support + s^2/n_null)
    support: int  # patients contributing an estimate


class CouplingPrior:
    """Frozen (code_i, code_j) -> PairStat table, canonical i < j."""

    def __init__(self, stats: dict[tuple[int, int], PairStat]):
        self._stats = dict(stats)

    def get(self, i: int, j: int) -> PairStat | None:
        return self._stats.get((i, j) if i < j else (j, i))

    def items(self) -> list[tuple[tuple[int, int], PairStat]]:
        """Every (pair, stat), for consumers building a lookup table in bulk."""
        return list(self._stats.items())

    def __len__(self) -> int:
        return len(self._stats)


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    """Pearson correlation, or None if undefined (< 2 points or zero variance)."""
    if a.size < 2:
        return None
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return None
    return float(np.dot(a, b) / denom)


def bin_series(
    codes: np.ndarray,
    values: np.ndarray,
    times: np.ndarray,
    bin_h: float,
    min_bins: int,
) -> dict[int, np.ndarray]:
    """Per-code binned-mean vectors over the whole stay; NaN where a bin is empty.

    Bins start at the patient's first reading. Codes with fewer than `min_bins`
    non-empty bins are dropped -- they cannot support a correlation.
    """
    if times.size == 0:
        return {}
    bin_id = ((times - times.min()) / bin_h).astype(np.int64)
    n_bins = int(bin_id.max()) + 1
    if n_bins < min_bins:
        return {}
    out: dict[int, np.ndarray] = {}
    for code in np.unique(codes):
        m = codes == code
        v = np.full(n_bins, np.nan)
        b, vv = bin_id[m], values[m]
        # mean per bin: sum / count, restricted to bins this code touches
        sums = np.bincount(b, weights=vv, minlength=n_bins)
        cnts = np.bincount(b, minlength=n_bins)
        nz = cnts > 0
        v[nz] = sums[nz] / cnts[nz]
        if int(np.count_nonzero(~np.isnan(v))) >= min_bins:
            out[int(code)] = v
    return out


def _lab_arrays(data) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    # Membership check first: `data["lab"]` *creates* an empty store on a
    # HeteroData that lacks one (mutating the caller's graph and warning), so
    # never index before confirming the store exists.
    if "lab" not in data.node_types:
        return None
    store = data["lab"]
    if not store.num_nodes:
        return None
    x = store.x.numpy()
    return (
        x[:, 0].astype(np.int64),
        x[:, 1].astype(np.float64),
        store.t.numpy().astype(np.float64),
    )


def fit_prior(
    dataset,
    train_idx: np.ndarray,
    *,
    bin_h: float = 6.0,
    min_bins: int = 3,
    min_support: int = 30,
    max_patients: int = 20000,
    seed: int = 0,
    cache_dir: Path | None = None,
) -> CouplingPrior:
    """Fit the population coupling prior over `train_idx` patients ONLY."""
    cache_path: Path | None = None
    if cache_dir is not None:
        key = _cache_key(train_idx, bin_h, min_bins, min_support, max_patients, seed)
        cache_path = Path(cache_dir) / f"coupling_prior_{key}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                prior = pickle.load(f)
            log.info("coupling_prior_cache_hit", path=str(cache_path), pairs=len(prior))
            return prior

    rng = np.random.default_rng(seed)
    idx = np.asarray(train_idx)
    if max_patients and len(idx) > max_patients:
        idx = np.sort(rng.choice(idx, size=max_patients, replace=False))

    rho_sum: dict[tuple[int, int], float] = {}
    count: dict[tuple[int, int], int] = {}
    reservoir: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = {}

    for i in idx:
        arrays = _lab_arrays(dataset[int(i)])
        if arrays is None:
            continue
        series = bin_series(*arrays, bin_h=bin_h, min_bins=min_bins)
        for ca, cb in itertools.combinations(sorted(series), 2):
            va, vb = series[ca], series[cb]
            mask = ~(np.isnan(va) | np.isnan(vb))
            if int(mask.sum()) < min_bins:
                continue
            rho = _pearson(va[mask], vb[mask])
            if rho is None:
                continue
            key = (ca, cb)
            rho_sum[key] = rho_sum.get(key, 0.0) + rho
            count[key] = count.get(key, 0) + 1
            _reservoir_add(
                reservoir.setdefault(key, []),
                (va[mask], vb[mask]),
                count[key],
                _NULL_CAP,
                rng,
            )

    stats: dict[tuple[int, int], PairStat] = {}
    for key, n in count.items():
        if n < min_support:
            continue
        rho_bar = rho_sum[key] / n
        null_mean, null_std, n_null = _null_moments(reservoir[key], rng)
        if n_null < 2 or null_std <= 0.0:
            # The null could not be estimated (too few reservoir entries, or
            # no eligible re-pairing partner survived length-matching). A
            # fabricated near-zero denominator would rank this pair as
            # maximally significant -- exactly backwards -- so drop it
            # instead of emitting a z score.
            continue
        # Welch two-sample SE. `null_mean` is itself estimated from only
        # `n_null` (<= _NULL_CAP) samples, so it carries error null_std/sqrt(
        # n_null). Dividing the difference by rho_bar's SE alone would ignore
        # that and over-disperse z by sqrt(1 + n/n_null) -- an inflation that
        # grows with support, i.e. with how *frequently drawn* a lab is rather
        # than with how coupled it is. z therefore saturates at large support:
        # no pair can claim more significance than the null's own precision
        # supports.
        se = null_std * np.sqrt(1.0 / n + 1.0 / n_null)
        z = (rho_bar - null_mean) / se
        stats[key] = PairStat(rho_bar=float(rho_bar), z=float(z), support=int(n))

    log.info(
        "coupling_prior_fitted",
        patients=len(idx),
        pairs_seen=len(count),
        pairs_kept=len(stats),
        min_support=min_support,
    )
    prior = CouplingPrior(stats)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(prior, f)
        log.info("coupling_prior_cached", path=str(cache_path))
    return prior


def _reservoir_add(
    res: list,
    item,
    seen: int,
    cap: int,
    rng: np.random.Generator,
) -> None:
    """Algorithm R: keep `res` a uniform sample of `cap` items from the stream."""
    if len(res) < cap:
        res.append(item)
        return
    k = int(rng.integers(0, seen))
    if k < cap:
        res[k] = item


def _surrogate_vb(
    vbs: list[np.ndarray],
    sizes: np.ndarray,
    k: int,
    n_k: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """An order-preserving surrogate `vb` for entry `k`: a contiguous window."""
    candidates = [j for j in range(len(vbs)) if j != k and sizes[j] >= n_k]
    if not candidates:
        return None
    j = candidates[int(rng.integers(0, len(candidates)))]
    src = vbs[j]
    start = int(rng.integers(0, src.size - n_k + 1))
    return src[start : start + n_k]


def _null_moments(
    vec_pairs: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator
) -> tuple[float, float, int]:
    """Pairing-surrogate null: re-pair each `va` with another entry's `vb`."""
    if len(vec_pairs) < 2:
        return 0.0, 0.0, 0
    vbs = [vb for _, vb in vec_pairs]
    sizes = np.array([vb.size for vb in vbs])
    null: list[float] = []
    for k, (va, _) in enumerate(vec_pairs):
        n_k = va.size
        if n_k < _MIN_NULL_POINTS:
            continue
        vb = _surrogate_vb(vbs, sizes, k, n_k, rng)
        if vb is None:
            continue
        r = _pearson(va, vb)
        if r is not None:
            null.append(r)
    if not null:
        return 0.0, 0.0, 0
    arr = np.asarray(null)
    return float(arr.mean()), float(arr.std()), int(arr.size)
