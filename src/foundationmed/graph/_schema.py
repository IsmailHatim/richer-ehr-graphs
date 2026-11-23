"""Unified edge-feature schema shared by every graph representation.

One fixed width keeps the GATv2 backbone byte-identical across rungs, so topology
is the only thing that varies between base/evolution/local.

Layout (EDGE_DIM = 6):
    [0:3]  edge-type one-hot  [structural, trajectory, coupling]
    [3]    delta-t  (hours, normalized by DT_SCALE)
    [4]    delta-value  (already z-scored lab units)
    [5]    signed correlation rho
"""

from __future__ import annotations

import torch
from torch import Tensor

EDGE_DIM = 6

_T_STRUCTURAL, _T_TRAJECTORY, _T_COUPLING = 0, 1, 2
_DT, _DV, _RHO = 3, 4, 5

DT_SCALE = 24.0  # hours; normalizes delta-t to ~O(1)


def structural_attr(num_edges: int, device=None) -> Tensor:
    """Constant edge attribute for base (has_/next) edges."""
    attr = torch.zeros((num_edges, EDGE_DIM), dtype=torch.float, device=device)
    attr[:, _T_STRUCTURAL] = 1.0
    return attr


def trajectory_attr(dt: Tensor, dv: Tensor) -> Tensor:
    """Edge attribute for same-code trajectory edges (evolution)."""
    attr = torch.zeros((dt.shape[0], EDGE_DIM), dtype=torch.float, device=dt.device)
    attr[:, _T_TRAJECTORY] = 1.0
    attr[:, _DT] = dt / DT_SCALE
    attr[:, _DV] = dv
    return attr


def coupling_attr(dt: Tensor, rho: Tensor) -> Tensor:
    """Edge attribute for cross-variable coupling edges (local)."""
    attr = torch.zeros((dt.shape[0], EDGE_DIM), dtype=torch.float, device=dt.device)
    attr[:, _T_COUPLING] = 1.0
    attr[:, _DT] = dt / DT_SCALE
    attr[:, _RHO] = rho
    return attr


def reverse_attr(attr: Tensor) -> Tensor:
    """Edge attribute for reversed edges: flip delta-value on trajectory rows only."""
    rev = attr.clone()
    traj = attr[:, _T_TRAJECTORY] == 1.0
    rev[traj, _DV] = -rev[traj, _DV]
    return rev


def attach_edges(data, edge_type, src: list[int], dst: list[int], attr: Tensor) -> None:
    """Attach a (possibly empty) edge set with EDGE_DIM attribute to HeteroData."""
    if src:
        data[edge_type].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data[edge_type].edge_attr = attr
    else:
        data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data[edge_type].edge_attr = torch.zeros((0, EDGE_DIM), dtype=torch.float)
