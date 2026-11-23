"""Task labelling and leakage-free data splitting."""

from __future__ import annotations

import numpy as np
import torch

from foundationmed.data._data import task_attr


class LabelTransform:
    """PyG transform: attach the target as data.y."""

    def __init__(self, task: str = "mortality") -> None:
        self._field = f"y_{task_attr(task)}"  # y_mortality / y_readmission

    def __call__(self, data):
        data.y = getattr(data["visit"], self._field)[-1:].float()
        return data


def stratified_split(
    labels: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx), stratified by label, reproducible."""
    rng = np.random.default_rng(seed)

    if groups is not None:
        return _grouped_split(labels, groups, val_frac, test_frac, rng)

    train, val, test = [], [], []

    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test.append(idx[:n_test])
        val.append(idx[n_test : n_test + n_val])
        train.append(idx[n_test + n_val :])

    train_idx = np.concatenate(train) if train else np.array([], dtype=np.int64)
    val_idx = np.concatenate(val) if val else np.array([], dtype=np.int64)
    test_idx = np.concatenate(test) if test else np.array([], dtype=np.int64)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def _grouped_split(
    labels: np.ndarray,
    groups: np.ndarray,
    val_frac: float,
    test_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split whole groups, stratified by each group's most severe label."""
    uniq, inverse = np.unique(groups, return_inverse=True)
    group_label = np.zeros(len(uniq), dtype=labels.dtype)
    np.maximum.at(
        group_label, inverse, labels
    )  # A group is positive if any of its graphs is

    assign = np.empty(len(uniq), dtype=np.int8)  # 0 train, 1 val, 2 test
    for cls in np.unique(group_label):
        gidx = np.where(group_label == cls)[0]
        rng.shuffle(gidx)
        n = len(gidx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        assign[gidx[:n_test]] = 2
        assign[gidx[n_test : n_test + n_val]] = 1
        assign[gidx[n_test + n_val :]] = 0

    per_graph = assign[inverse]
    train_idx = np.where(per_graph == 0)[0]
    val_idx = np.where(per_graph == 1)[0]
    test_idx = np.where(per_graph == 2)[0]
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def pos_weight(labels: np.ndarray) -> torch.Tensor:
    """Positive class weight for BCEWithLogitsLoss given class imbalance."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    weight = (n_neg / n_pos) if n_pos > 0 else 1.0
    return torch.tensor(weight, dtype=torch.float)
