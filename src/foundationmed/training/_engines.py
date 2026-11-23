"""Ignite engine factories and metric wrappers for the graph benchmark."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from ignite.engine import Engine
from ignite.metrics import EpochMetric, Loss
from torch import nn
from torch.nn.utils import clip_grad_norm_

from foundationmed.training._metrics import auprc, auroc

# The subset of evaluator metrics logged to MLflow every epoch.
METRIC_NAMES = ("auroc", "auprc")


def create_trainer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> Engine:
    """Build the training engine."""

    def train_step(engine: Engine, batch) -> dict[str, torch.Tensor]:
        model.train()
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        if grad_clip:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        return {"y_pred": logits.detach(), "y": batch.y, "loss": loss.detach()}

    return Engine(train_step)


def create_evaluator(model: nn.Module, device: torch.device) -> Engine:
    """Build an inference engine, used for both the validation and test passes."""

    @torch.no_grad()
    def eval_step(engine: Engine, batch) -> dict[str, torch.Tensor]:
        model.eval()
        batch = batch.to(device)
        return {"y_pred": model(batch), "y": batch.y}

    return Engine(eval_step)


class _SklearnMetric(EpochMetric):
    """Accumulate an epoch of (probability, label) pairs, then defer to numpy."""

    def __init__(self, fn: Callable[[np.ndarray, np.ndarray], float]) -> None:
        super().__init__(
            compute_fn=lambda y_pred, y: fn(y.cpu().numpy(), y_pred.cpu().numpy()),
            output_transform=lambda out: (
                torch.sigmoid(out["y_pred"]),
                out["y"],
            ),
            check_compute_fn=False,  # the fns are NaN-safe by design
            device=torch.device("cpu"),
        )


def attach_metrics(engine: Engine) -> None:
    """Attach auroc, auprc, prevalence and n to an evaluator."""
    metrics = {
        "auroc": _SklearnMetric(auroc),
        "auprc": _SklearnMetric(auprc),
        "prevalence": _SklearnMetric(
            lambda y_true, _y_score: (
                float(np.mean(y_true)) if len(y_true) else float("nan")
            ),
        ),
        "n": _SklearnMetric(lambda y_true, _y_score: int(len(y_true))),
    }
    for name, metric in metrics.items():
        metric.attach(engine, name)


def attach_loss(engine: Engine, criterion: nn.Module) -> Loss:
    """Attach the sample-weighted epoch-mean training loss."""
    metric = Loss(criterion, output_transform=lambda out: (out["y_pred"], out["y"]))
    metric.attach(engine, "loss")
    return metric
