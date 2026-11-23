"""Train and benchmark mortality models on patient graphs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

import mlflow
import numpy as np
import structlog
import torch
from ignite.engine import Engine, Events
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from foundationmed.dataset._dataset import EHRGraphDataset
from foundationmed.graph._cache import (
    DEFAULT_CACHE_DIR,
    attach_cache,
    cache_key,
    cache_path,
)
from foundationmed.graph._config import load_graph_config
from foundationmed.graph._registry import build_pipeline, fit_pipeline
from foundationmed.models import build_model
from foundationmed.training._config import DataConfig, ExperimentConfig
from foundationmed.training._data import (
    LabelTransform,
    pos_weight,
    stratified_split,
)
from foundationmed.training._engines import (
    attach_loss,
    attach_metrics,
    create_evaluator,
    create_trainer,
)
from foundationmed.training._handlers import (
    attach_best_model,
    attach_console_logging,
    attach_divergence_guard,
    attach_early_stopping,
    attach_epoch_callback,
    attach_lr_plateau,
    attach_mlflow,
)
from foundationmed.training._tracking import build_tags

log = structlog.get_logger()

# (epoch, val_metrics) hook invoked after each validation pass
EpochCallback = Callable[[int, dict], None]


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def output_dir(exp: ExperimentConfig, data: DataConfig) -> Path:
    """Per-dataset dir for weights/splits: ``output_path/<dataset>``."""
    return Path(exp.training.output_path) / data.name


def _extra_node_dims(pipeline) -> dict[str, int]:
    """Extra node-feature columns each component appends, summed per node type."""
    dims: dict[str, int] = {}
    for step in getattr(pipeline, "transforms", []):
        node_type = getattr(step, "node_type", None)
        out_dim = getattr(step, "out_dim", None)
        if node_type is not None and out_dim:
            dims[node_type] = dims.get(node_type, 0) + int(out_dim)
    return dims


def _mlflow_params(exp: ExperimentConfig, data: DataConfig) -> dict[str, object]:
    """Flatten the nested config to dotted keys for ``mlflow.log_params``."""
    params: dict[str, object] = {
        "data.processed_path": data.processed_path,
        "data.name": data.name,
        "data.task": data.task,
    }
    for block, sub in asdict(exp).items():  # model, training, tracking
        for key, value in sub.items():
            params[f"{block}.{key}"] = value
    return params


def run_training(
    exp: ExperimentConfig,
    data: DataConfig,
    graph: str | None = None,
    extra_tags: dict[str, str] | None = None,
    epoch_callback: EpochCallback | None = None,
) -> dict:
    """Train the selected model on one dataset and log the run to MLflow."""
    mlflow.set_tracking_uri(exp.tracking.tracking_uri)
    mlflow.set_experiment(exp.tracking.experiment_name)
    with mlflow.start_run() as run:
        training = exp.training
        torch.manual_seed(training.seed)
        device = resolve_device(training.device)
        out = output_dir(exp, data)
        out.mkdir(parents=True, exist_ok=True)

        mlflow.log_params(_mlflow_params(exp, data))
        tags = build_tags(exp, data, device, graph)
        if extra_tags:
            tags.update(extra_tags)
        mlflow.set_tags(tags)

        # Log full config
        log.info(
            "run_config",
            run_id=run.info.run_id,
            graph=graph or "base",
            dataset=data.name,
            task=data.task,
            **{f"model.{k}": v for k, v in asdict(exp.model).items()},
            **{f"training.{k}": v for k, v in asdict(exp.training).items()},
        )

        log.info(
            "loading_graphs",
            path=data.processed_path,
            note="unpickling is silent and takes approximately 1 min for eICU",
        )
        dataset = EHRGraphDataset(
            graphs_path=Path(data.processed_path) / "patient_graphs.pkl",
            vocabs_path=Path(data.processed_path) / "vocabularies.pkl",
            task=data.task,
            min_events=training.min_events,
            max_visits=training.max_visits or None,
        )
        label = LabelTransform(data.task)
        graph_cfg = load_graph_config(graph) if graph else None
        pipeline = build_pipeline(graph_cfg) if graph_cfg else None
        if (
            graph_cfg is not None
            and pipeline is not None
            and "coupling" in graph_cfg.components
        ):
            key = cache_key(
                data.name,
                data.task,
                training.min_events,
                training.max_visits or None,
                graph_cfg.params.get("coupling", {}),
            )
            pipeline = attach_cache(
                pipeline,
                cache_path(DEFAULT_CACHE_DIR, key),
                key,
                dataset.samples,
            )

        all_idx = np.arange(len(dataset))

        # Optionnal Sampling
        rng = np.random.default_rng(training.seed)
        if training.subsample and training.subsample < len(all_idx):
            all_idx = np.sort(
                rng.choice(all_idx, size=training.subsample, replace=False)
            )

        labels = dataset.labels[all_idx]
        groups = dataset.groups[all_idx]  # by subject_id in eICU
        tr_pos, va_pos, te_pos = stratified_split(
            labels,
            training.val_frac,
            training.test_frac,
            training.seed,
            groups=groups if len(np.unique(groups)) < len(groups) else None,
        )
        train_idx, val_idx, test_idx = all_idx[tr_pos], all_idx[va_pos], all_idx[te_pos]

        dataset.fit_lab_normalization(train_idx)

        # Components estimating population statistics (the prior coupling prior) must
        # see the training split ONLY -- hence fit here, after the split, and only
        # then install the transform. Non-Fittable components are unaffected.
        extra_node_dims: dict[str, int] = {}
        if pipeline is not None:
            fit_pipeline(pipeline, dataset, train_idx)
            dataset.transform = Compose([pipeline, label])
            extra_node_dims = _extra_node_dims(pipeline)
        else:
            dataset.transform = label
        log.info(
            "split",
            train=len(train_idx),
            val=len(val_idx),
            test=len(test_idx),
            people=len(np.unique(groups)),
            prevalence=round(float(labels.mean()), 3),
            device=str(device),
        )

        val_metrics, model_metrics = _train_model(
            dataset,
            exp,
            out,
            device,
            train_idx,
            val_idx,
            test_idx,
            labels[tr_pos],
            epoch_callback,
            extra_node_dims,
        )

        prevalence = float(labels.mean())
        results = {
            "backbone": exp.model.backbone,
            "prevalence": prevalence,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "val": val_metrics,
            "model": model_metrics,
        }

        mlflow.log_metrics({"prevalence": prevalence})

        np.savez(out / "split.npz", train=train_idx, val=val_idx, test=test_idx)

        _log_results(results)
        return results


def _train_model(
    dataset,
    exp,
    out,
    device,
    train_idx,
    val_idx,
    test_idx,
    train_labels,
    epoch_callback: EpochCallback | None = None,
    extra_node_dims: dict[str, int] | None = None,
) -> tuple[dict, dict]:
    """Train one model; return (best-val metrics, test metrics)."""
    model_cfg, training = exp.model, exp.training
    workers = training.num_workers
    train_loader = DataLoader(
        dataset.index_select(train_idx.tolist()),
        batch_size=training.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    val_loader = DataLoader(
        dataset.index_select(val_idx.tolist()),
        batch_size=training.batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    test_loader = DataLoader(
        dataset.index_select(test_idx.tolist()),
        batch_size=training.batch_size,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    spec = dataset.feature_spec  # TODO: derive extra_node_dims without
    # replacing the spec
    if extra_node_dims:
        spec = replace(spec, extra_node_dims=extra_node_dims)
    model = build_model(
        model_cfg.backbone,
        spec,
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training.lr, weight_decay=training.weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight(train_labels).to(device)
    )

    trainer = create_trainer(model, optimizer, criterion, device, training.grad_clip)
    val_evaluator = create_evaluator(model, device)
    test_evaluator = create_evaluator(model, device)
    loss_metric = attach_loss(trainer, criterion)
    for engine in (val_evaluator, test_evaluator):
        attach_metrics(engine)

    # The validation pass must run before any handler that reads its metrics.
    @trainer.on(Events.EPOCH_COMPLETED)
    def _validate(engine: Engine) -> None:
        val_evaluator.run(val_loader)

    best = attach_best_model(val_evaluator, model)
    attach_lr_plateau(trainer, val_evaluator, optimizer, training)
    attach_early_stopping(trainer, val_evaluator, training)
    attach_divergence_guard(trainer, val_evaluator, best)
    attach_epoch_callback(trainer, val_evaluator, epoch_callback)
    attach_mlflow(trainer, val_evaluator, test_evaluator, optimizer)
    attach_console_logging(
        trainer, val_evaluator, loss_metric, max(1, len(train_loader) // 5)
    )

    trainer.run(train_loader, max_epochs=training.epochs)

    best.restore(model)
    torch.save(model.state_dict(), out / "model.pt")

    test_evaluator.run(test_loader)
    test_metrics = dict(test_evaluator.state.metrics)
    best_val = best.metrics or dict(val_evaluator.state.metrics)
    return best_val, test_metrics


def _log_results(results: dict) -> None:
    m = results["model"]
    log.info(
        "benchmark_result",
        model=results["backbone"],
        auroc=round(m["auroc"], 3),
        auprc=round(m["auprc"], 3),
        test_prevalence=round(results["prevalence"], 3),
    )
