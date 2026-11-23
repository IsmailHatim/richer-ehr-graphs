"""Compare graph representations under one fixed protocol."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import structlog

from foundationmed.training._config import DataConfig, ExperimentConfig
from foundationmed.training._train import run_training

log = structlog.get_logger()


def run_comparison(
    exp: ExperimentConfig | list[ExperimentConfig],
    data: DataConfig,
    graph_configs: list[str],
    seeds: int,
    output_path: str,
    config_paths: list[str] | None = None,
    extra_tags: dict[str, str] | None = None,
) -> dict:
    """Train each graph version over seeds and log every run to MLflow."""
    exps: list[ExperimentConfig] = (
        [exp] * len(graph_configs) if isinstance(exp, ExperimentConfig) else exp
    )
    if len(exps) != len(graph_configs):
        raise ValueError(
            f"got {len(exps)} configs for {len(graph_configs)} graphs; "
            "pass one config or one per graph"
        )
    paths: list[str | None]
    if config_paths is None:
        paths = [None] * len(graph_configs)
    elif len(config_paths) == 1:
        paths = [config_paths[0]] * len(graph_configs)
    elif len(config_paths) == len(graph_configs):
        paths = list(config_paths)
    else:
        raise ValueError(
            f"got {len(config_paths)} config paths for {len(graph_configs)} "
            "graphs; pass one config path or one per graph"
        )

    tags = {"sweep": "graph-comparison", **(extra_tags or {})}
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}

    for base_exp, gc, config_path in zip(exps, graph_configs, paths):
        name = Path(gc).stem
        aurocs: list[float] = []
        auprcs: list[float] = []
        for seed in range(seeds):
            log.info("compare_arm", graph=name, seed=seed, config_path=config_path)
            run_exp = dataclasses.replace(
                base_exp,
                training=dataclasses.replace(
                    base_exp.training,
                    seed=seed,
                    output_path=str(out / name / f"seed{seed}"),
                ),
            )
            res = run_training(run_exp, data, graph=gc, extra_tags=tags)
            aurocs.append(res["model"]["auroc"])
            auprcs.append(res["model"]["auprc"])
        summary[name] = {
            "auroc_mean": float(np.nanmean(aurocs)),
            "auroc_std": float(np.nanstd(aurocs)),
            "auprc_mean": float(np.nanmean(auprcs)),
            "auprc_std": float(np.nanstd(auprcs)),
            "seeds": seeds,
        }
        log.info("compare_result", graph=name, **summary[name])

    return summary
