"""MLflow tracking helpers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from foundationmed.training._config import DataConfig, ExperimentConfig


def git_commit() -> str:
    """Return the current short git commit hash, or 'unknown' outside a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def git_dirty() -> str:
    """Return 'true' when the working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        )
        return "true" if out.strip() else "false"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "false"


def graph_version(graph: str | None) -> str:
    """Derive the graph-version tag: the graph TOML stem, or 'base' if none."""
    return Path(graph).stem if graph else "base"


def build_tags(
    exp: ExperimentConfig,
    data: DataConfig,
    device: torch.device,
    graph: str | None = None,
) -> dict[str, str]:
    """Build the MLflow tag set a run is sliced and grouped by later."""
    tags = {
        "backbone": exp.model.backbone,
        "graph_version": graph_version(graph),
        "dataset": data.name,
        "task": data.task,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "device": device.type,
        "seed": str(exp.training.seed),
    }
    if exp.tracking.config_source:  # omitted rather than empty
        tags["config_source"] = exp.tracking.config_source
    if cluster := os.environ.get("FMED_CLUSTER", ""):  # omitted rather than empty
        tags["cluster"] = cluster
    if exp.tracking.purpose:
        tags["purpose"] = exp.tracking.purpose
    return tags
