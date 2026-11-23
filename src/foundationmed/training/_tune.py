"""Ray Tune + Optuna hyperparameter search over the existing training loop."""

from __future__ import annotations

import dataclasses
import os
import statistics
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from foundationmed.training._config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrackingConfig,
    TrainingConfig,
)

# keep relative paths (data/, mlflow.db) valid inside Ray trial workers
os.environ.setdefault("RAY_CHDIR_TO_TRIAL_DIR", "0")

log = structlog.get_logger()

_METRICS = ("val_auprc", "val_auroc")  # metrics reported each epoch


@dataclass
class SearchConfig:
    """Study knobs + flattened search space from a config/search TOML."""

    study_name: str = "hpo"
    num_trials: int = 20
    metric: str = "val_auprc"
    mode: str = "max"
    sampler_seed: int = 0
    grace_period: int = 5  # min epochs before ASHA prunes
    reduction_factor: int = 3  # keep top 1/3 at each rung
    cpus_per_trial: float = 4.0
    gpus_per_trial: float = 0.0
    max_concurrent_trials: int = 0  # 0 = auto
    progress_interval_s: int = 300  # seconds between study_progress lines
    time_budget_s: float | None = None  # wall-clock cap on the whole study
    storage_path: str = "ray_results"
    space: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_search_config(path: str | Path) -> SearchConfig:
    """Load a search TOML: [study] knobs + [space.*] tables."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    space = _flatten_space(raw.get("space", {}))
    return SearchConfig(**raw.get("study", {}), space=space)


def _flatten_space(node: dict, prefix: str = "") -> dict[str, dict[str, Any]]:
    """Nested [space.*] tables turned to a  flat dotted keys; a leaf has a type."""
    out: dict[str, dict[str, Any]] = {}
    for key, value in node.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and isinstance(value.get("type"), str):
            out[dotted] = value
        elif isinstance(value, dict):
            out.update(_flatten_space(value, dotted))
        else:
            raise ValueError(f"search-space entry {dotted!r} is not a table")
    return out


def _fit_workers_to_trial(
    exp: ExperimentConfig, search: SearchConfig
) -> ExperimentConfig:
    """Cap a trial's DataLoader workers at the CPUs Ray allocated it."""
    if "training.num_workers" in search.space:
        raise ValueError(
            "training.num_workers is a runtime knob, not a hyperparameter: "
            "searching it would vary each trial's real CPU footprint while "
            "Ray's cpus_per_trial stays fixed. Set it in the training TOML."
        )
    allowed = max(0, int(search.cpus_per_trial) - 1)
    requested = exp.training.num_workers
    if requested <= allowed:
        return exp
    log.warning(
        "num_workers_capped_for_trial",
        requested=requested,
        used=allowed,
        cpus_per_trial=search.cpus_per_trial,
        note="Ray cannot see DataLoader workers; the config value would "
        "oversubscribe the node. Raise cpus_per_trial to use more.",
    )
    return dataclasses.replace(
        exp, training=dataclasses.replace(exp.training, num_workers=allowed)
    )


def _build_param_space(space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Typed leaves to Ray Tune sampling primitives."""
    from ray import tune

    out: dict[str, Any] = {}
    for name, leaf in space.items():
        kind = leaf["type"]
        if kind == "loguniform":
            out[name] = tune.loguniform(leaf["low"], leaf["high"])
        elif kind == "uniform":
            out[name] = tune.uniform(leaf["low"], leaf["high"])
        elif kind == "choice":
            out[name] = tune.choice(leaf["values"])
        elif kind == "int":
            out[name] = tune.randint(leaf["low"], leaf["high"] + 1)  # inclusive
        else:
            raise ValueError(f"unknown search type {kind!r} for {name!r}")
    return out


def _apply(exp: ExperimentConfig, sampled: dict[str, Any]) -> ExperimentConfig:
    """Sampled dotted keys to new ExperimentConfig; unknown fields fail loudly."""
    model_kw: dict[str, Any] = {}
    training_kw: dict[str, Any] = {}
    for key, value in sampled.items():
        if hasattr(value, "item"):  # numpy scalar to plain python
            value = value.item()
        section, _, name = key.partition(".")
        if section == "model" and name:
            model_kw[name] = value
        elif section == "training" and name:
            training_kw[name] = value
        else:
            raise ValueError(
                f"search key {key!r} must be 'model.<field>' or 'training.<field>'"
            )
    return dataclasses.replace(
        exp,
        model=dataclasses.replace(exp.model, **model_kw),
        training=dataclasses.replace(exp.training, **training_kw),
    )


def _experiment_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
        tracking=TrackingConfig(**raw["tracking"]),
    )


def _make_trainable(
    exp: ExperimentConfig, data: DataConfig, study_name: str, graph: str | None
):
    """Closure over plain dicts so cloudpickle ships no live config objects."""
    exp_dict, data_dict = dataclasses.asdict(exp), dataclasses.asdict(data)

    def trainable(sampled: dict[str, Any]) -> None:
        from ray import tune

        from foundationmed.training._train import run_training

        trial_id = tune.get_context().get_trial_id()
        trial_exp = _apply(_experiment_from_dict(exp_dict), sampled)
        trial_exp = dataclasses.replace(
            trial_exp,
            training=dataclasses.replace(
                trial_exp.training,  # per-trial dir: model.pt/split.npz
                output_path=str(
                    Path(trial_exp.training.output_path) / "hpo" / trial_id
                ),
            ),
            # A tuning run is its own provenance; the exported best_config.toml
            # carries the same value so a later train/compare stays attributable.
            tracking=dataclasses.replace(trial_exp.tracking, config_source=study_name),
        )
        run_training(
            trial_exp,
            DataConfig(**data_dict),
            graph=graph,
            extra_tags={"sweep": "hpo", "study": study_name, "trial_id": trial_id},
            epoch_callback=lambda epoch, m: tune.report(
                {"val_auprc": m["auprc"], "val_auroc": m["auroc"]}
            ),
        )

    return trainable


def _asha_budget(
    num_trials: int, grace_period: int, reduction_factor: int, max_t: int
) -> int:
    """Total epochs an ASHA study runs."""
    if reduction_factor < 2:  # no pruning: every trial runs the full budget
        return num_trials * max_t
    rungs, rung = [], grace_period
    while rung < max_t:
        rungs.append(rung)
        rung *= reduction_factor
    rungs.append(max_t)
    total, alive, previous = 0, num_trials, 0
    for rung in rungs:
        total += alive * (rung - previous)
        previous = rung
        alive = max(1, alive // reduction_factor)
    return total


def _make_progress_callback(search: SearchConfig, num_trials: int, max_epochs: int):
    """One throttled `study_progress` line, plus a line per errored trial."""
    from ray import tune

    budget_s = search.time_budget_s
    budget = (
        None
        if budget_s is not None or num_trials < 0
        else _asha_budget(
            num_trials, search.grace_period, search.reduction_factor, max_epochs
        )
    )
    better = (
        (lambda new, old: new > old)
        if search.mode == "max"
        else (lambda new, old: new < old)
    )

    class _Progress(tune.Callback):
        def __init__(self) -> None:
            self.epochs: dict[str, int] = {}
            self.epoch_times: list[float] = []
            self.best: float | None = None
            self.started = time.monotonic()
            self.logged_at: float | None = None

        def on_trial_result(self, iteration, trials, trial, result, **info) -> None:
            try:
                done = int(result.get("training_iteration", 0))
                self.epochs[trial.trial_id] = done
                elapsed = result.get("time_this_iter_s")
                if elapsed and done > 1:  # iteration 1 carries the dataset load
                    self.epoch_times.append(float(elapsed))
                    del self.epoch_times[:-200]  # keep the estimate recent
                value = result.get(search.metric)
                if value is not None and (
                    self.best is None or better(value, self.best)
                ):
                    self.best = float(value)
            except Exception as exc:  # noqa: BLE001 - logging must not kill a study
                log.warning(
                    "progress_hook_failed", hook="on_trial_result", error=str(exc)
                )

        def on_trial_error(self, iteration, trials, trial, **info) -> None:
            log.error(
                "trial_error",
                trial=trial.trial_id,
                epochs=self.epochs.get(trial.trial_id, 0),
            )

        def on_step_end(self, iteration, trials, **info) -> None:
            try:
                now = time.monotonic()
                if (
                    self.logged_at is not None
                    and now - self.logged_at < search.progress_interval_s
                ):
                    return
                if len(self.epoch_times) < 2:  # no usable rate yet
                    return
                self.logged_at = now
                running = sum(1 for t in trials if t.status == "RUNNING")
                finished = sum(1 for t in trials if t.status in ("TERMINATED", "ERROR"))
                epochs = sum(self.epochs.values())
                per_epoch = statistics.median(self.epoch_times)
                elapsed = now - self.started
                if budget is None:
                    eta = (budget_s - elapsed) / 3600 if budget_s else 0.0
                else:
                    eta = (budget - epochs) * per_epoch / max(running, 1) / 3600
                log.info(
                    "study_progress",
                    elapsed_h=round(elapsed / 3600, 2),
                    trials=f"{finished}/{num_trials}" if num_trials > 0 else finished,
                    running=running,
                    epochs=f"{epochs}/{budget}" if budget is not None else str(epochs),
                    best=round(self.best, 4) if self.best is not None else None,
                    s_epoch=round(per_epoch),
                    eta_h=round(max(eta, 0.0), 1),
                )
            except Exception as exc:  # noqa: BLE001 - logging must not kill a study
                log.warning("progress_hook_failed", hook="on_step_end", error=str(exc))

    return _Progress()


def _tune_config(
    search: SearchConfig,
    total_trials: int,
    max_concurrent: int | None,
    max_t: int = 100,
):
    """Build the TuneConfig. A wall-clock budget may stand in for a trial cap."""
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    if total_trials < 0 and search.time_budget_s is None:
        raise ValueError(
            "num_trials < 0 keeps sampling until something stops the study, so "
            "it needs a wall-clock bound: set time_budget_s in [study]."
        )
    return tune.TuneConfig(
        num_samples=total_trials,
        time_budget_s=search.time_budget_s,
        search_alg=OptunaSearch(
            metric=search.metric, mode=search.mode, seed=search.sampler_seed
        ),
        scheduler=ASHAScheduler(
            metric=search.metric,
            mode=search.mode,
            max_t=max_t,
            grace_period=search.grace_period,
            reduction_factor=search.reduction_factor,
        ),
        max_concurrent_trials=max_concurrent or None,
    )


def run_tuning(
    exp: ExperimentConfig,
    data: DataConfig,
    search: SearchConfig,
    graph: str | None = None,
    num_trials: int | None = None,
    max_concurrent: int | None = None,
):
    """Run the study and return the best Ray Result; exports its config."""
    from ray import tune

    if search.metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}, got {search.metric!r}")

    exp = _fit_workers_to_trial(exp, search)
    total_trials = num_trials or search.num_trials

    tuner = tune.Tuner(
        tune.with_resources(
            _make_trainable(exp, data, search.study_name, graph),
            {"cpu": search.cpus_per_trial, "gpu": search.gpus_per_trial},
        ),
        param_space=_build_param_space(search.space),
        tune_config=_tune_config(
            search,
            total_trials,
            (max_concurrent or search.max_concurrent_trials) or None,
            max_t=exp.training.epochs + 1,
        ),
        run_config=tune.RunConfig(
            name=search.study_name,
            storage_path=Path(search.storage_path).resolve().as_uri(),
            callbacks=[
                _make_progress_callback(search, total_trials, exp.training.epochs)
            ],
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(metric=search.metric, mode=search.mode)
    export = _export_best(best, exp, search)
    log.info(
        "best_trial",
        config=best.config,
        best_metric={search.metric: (best.metrics or {}).get(search.metric)},
        export=str(export),
    )
    return best


def _export_best(best, exp: ExperimentConfig, search: SearchConfig) -> Path:
    """Write a ready-to-train [model]/[training] TOML with the winning values."""
    merged = dataclasses.asdict(_apply(exp, best.config))
    lines = [
        f"# best of study {search.study_name!r}: "
        f"{search.metric} = {(best.metrics or {}).get(search.metric)}"
    ]
    for section in ("model", "training"):
        lines.append(f"[{section}]")
        lines.extend(
            f"{key} = {_toml_value(value)}" for key, value in merged[section].items()
        )
        lines.append("")
    # Stamp provenance so a later train/compare on this config is attributable
    # even if the --tag flag is forgotten.
    lines.append("[tracking]")
    lines.append(f"config_source = {_toml_value(search.study_name)}")
    lines.append("")
    out = Path(search.storage_path) / search.study_name / "best_config.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    return out


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, int | float):
        return repr(value)
    raise TypeError(f"cannot serialize {value!r} to TOML")
