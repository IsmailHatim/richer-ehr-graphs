"""Ignite handlers: attach_* per training-loop component."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import structlog
import torch
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, TerminateOnNan, global_step_from_engine
from ignite.handlers.mlflow_logger import (
    MLflowLogger,
    OptimizerParamsHandler,
    OutputHandler,
)
from ignite.handlers.param_scheduler import ReduceLROnPlateauScheduler
from torch import nn

from foundationmed.training._engines import METRIC_NAMES

if TYPE_CHECKING:
    from foundationmed.training._config import TrainingConfig

log = structlog.get_logger()


def _score(engine: Engine) -> float:
    """The stopping/scheduling signal: raw ``auprc``, with NaN mapped to -inf."""
    value = float(engine.state.metrics["auprc"])  # TODO: generalise the metric
    return -math.inf if math.isnan(value) else value


class BestModel:
    """The best-scoring weights seen so far, kept in memory."""

    def __init__(self) -> None:
        self.score: float = -math.inf
        self.metrics: dict | None = None
        self.state_dict: dict | None = None

    def restore(self, model: nn.Module) -> None:
        """Load the best snapshot into ``model``; a no-op if none was taken."""
        if self.state_dict is not None:
            model.load_state_dict(self.state_dict)


def attach_best_model(evaluator: Engine, model: nn.Module) -> BestModel:
    """Snapshot the model at the argmax of **raw** ``auprc``."""
    best = BestModel()

    @evaluator.on(Events.COMPLETED)
    def _snapshot(engine: Engine) -> None:
        score = _score(engine)
        if score > best.score:
            best.score = score
            best.metrics = dict(engine.state.metrics)
            best.state_dict = copy.deepcopy(model.state_dict())

    return best


def attach_console_logging(
    trainer: Engine,
    evaluator: Engine,
    loss_metric,
    log_every: int,
) -> None:
    """Emit the train_step and epoch structlog lines."""

    @trainer.on(Events.ITERATION_COMPLETED(every=log_every))
    def _log_step(engine: Engine) -> None:
        epoch_length = engine.state.epoch_length
        assert epoch_length is not None
        log.info(
            "train_step",
            epoch=engine.state.epoch,
            step=engine.state.iteration % epoch_length or epoch_length,
            total_steps=epoch_length,
            running_loss=round(float(loss_metric.compute()), 4),
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def _log_epoch(engine: Engine) -> None:
        metrics = evaluator.state.metrics
        log.info(
            "epoch",
            epoch=engine.state.epoch,
            loss=round(float(engine.state.metrics["loss"]), 4),
            val_auroc=round(float(metrics["auroc"]), 3),
            val_auprc=round(float(metrics["auprc"]), 3),
        )


def attach_lr_plateau(
    trainer: Engine,
    evaluator: Engine,
    optimizer: torch.optim.Optimizer,
    training: TrainingConfig,
) -> None:
    """Halve the LR when val_auprc plateaus, and say so in the log."""
    scheduler = ReduceLROnPlateauScheduler(
        optimizer,
        metric_name="auprc",
        trainer=trainer,
        save_history=True,
        mode="max",
        factor=training.lr_factor,
        patience=training.lr_patience,
        threshold=training.min_delta,  # min auprc gain that resets lr_patience
    )
    evaluator.add_event_handler(Events.COMPLETED, scheduler)

    last_lr = float(optimizer.param_groups[0]["lr"])

    @evaluator.on(Events.COMPLETED)
    def _log_decay(engine: Engine) -> None:
        nonlocal last_lr
        current = float(optimizer.param_groups[0]["lr"])
        if current != last_lr:
            log.info("lr_decay", epoch=trainer.state.epoch, lr=current)
            last_lr = current


def attach_early_stopping(
    trainer: Engine, evaluator: Engine, training: TrainingConfig
) -> None:
    """Stop when val_auprc stops improving by at least min_delta."""
    handler = EarlyStopping(
        patience=training.patience,
        score_function=_score,
        trainer=trainer,
        threshold=training.min_delta,
        cumulative=training.cumulative_delta,
    )
    evaluator.add_event_handler(Events.COMPLETED, handler)


def attach_divergence_guard(
    trainer: Engine, evaluator: Engine, best: BestModel
) -> None:
    """Kill a collapsed run instead of spending the whole patience budget on it."""
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        TerminateOnNan(output_transform=lambda out: out["loss"]),
    )

    div_patience = 0

    @evaluator.on(Events.COMPLETED)
    def _guard(engine: Engine) -> None:
        nonlocal div_patience
        score = _score(engine)
        if best.score > -math.inf and score < 0.5 * best.score:
            div_patience += 1
            if div_patience >= 2:
                log.warning(
                    "diverged",
                    epoch=trainer.state.epoch,
                    auprc=score,
                    best=best.score,
                )
                trainer.terminate()
        else:
            div_patience = 0


class _UnderscoreOutputHandler(OutputHandler):
    """Log <tag>_<metric> where stock Ignite logs '<tag> <metric>'."""

    def __call__(
        self, engine: Engine, logger: MLflowLogger, event_name: str | Events
    ) -> None:
        """Emit the configured metrics under underscore-joined keys."""
        metrics = {
            f"{self.tag}_{name}": float(engine.state.metrics[name])
            for name in (self.metric_names or [])
            if name in engine.state.metrics
        }
        logger.log_metrics(metrics, step=self.global_step_transform(engine, event_name))


class _LRHandler(OptimizerParamsHandler):
    """Log the learning rate as lr rather than stock's lr group_0."""

    def __call__(
        self, engine: Engine, logger: MLflowLogger, event_name: str | Events
    ) -> None:
        """Emit the first param group's LR at the current epoch."""
        logger.log_metrics(
            {self.param_name: float(self.optimizer.param_groups[0][self.param_name])},
            step=engine.state.get_event_attrib_value(event_name),
        )


def attach_mlflow(
    trainer: Engine,
    val_evaluator: Engine,
    test_evaluator: Engine,
    optimizer: torch.optim.Optimizer,
) -> MLflowLogger:
    """Wire per-epoch metric logging to the ambient MLflow run."""
    logger = MLflowLogger()
    step = global_step_from_engine(trainer)

    logger.attach(
        trainer,
        log_handler=_UnderscoreOutputHandler(
            "train", metric_names=["loss"], global_step_transform=step
        ),
        event_name=Events.EPOCH_COMPLETED,
    )
    logger.attach(
        val_evaluator,
        log_handler=_UnderscoreOutputHandler(
            "val", metric_names=list(METRIC_NAMES), global_step_transform=step
        ),
        event_name=Events.COMPLETED,
    )
    logger.attach(
        test_evaluator,
        log_handler=_UnderscoreOutputHandler(
            "test", metric_names=list(METRIC_NAMES), global_step_transform=step
        ),
        event_name=Events.COMPLETED,
    )
    logger.attach(
        trainer,
        log_handler=_LRHandler(optimizer),
        event_name=Events.EPOCH_COMPLETED,
    )
    return logger


def attach_epoch_callback(
    trainer: Engine,
    evaluator: Engine,
    epoch_callback: Callable[[int, dict], None] | None,
) -> None:
    """Forward each epoch's validation metrics to a caller-supplied hook."""
    if epoch_callback is None:
        return

    @evaluator.on(Events.COMPLETED)
    def _report(engine: Engine) -> None:
        epoch_callback(trainer.state.epoch, dict(engine.state.metrics))
