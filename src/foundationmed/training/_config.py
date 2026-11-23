"""Training and dataset configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Architecture - the model under test."""

    backbone: str = "gcn"  # registered model name; see foundationmed.models.build_model
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2


@dataclass
class TrainingConfig:
    """The learning procedure: cohort shaping, optimisation, split, runtime."""

    # Cohort shaping (which patients from the dataset enter the benchmark)
    min_events: int = 5
    max_visits: int = 0
    subsample: int = 0
    # Optimisation
    lr: float = 1e-3
    weight_decay: float = 3e-5
    epochs: int = 100
    batch_size: int = 64
    grad_clip: float = 1.0  # max grad norm; 0 disables
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-3
    cumulative_delta: bool = True
    # LR schedule
    lr_factor: float = 0.5
    lr_patience: int = 5  # < patience
    # Split / reproducibility
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = 0
    # Runtime
    device: str = "auto"
    output_path: str = "data/models/mortality"
    num_workers: int = 0

    def __post_init__(self) -> None:
        if self.lr_patience >= self.patience:
            raise ValueError(f"lr_patience ({self.lr_patience}) must be < patience")
        if self.num_workers < 0:
            raise ValueError(f"num_workers ({self.num_workers}) must be >= 0")


@dataclass
class TrackingConfig:
    """MLflow tracking — the primary experiment store."""

    tracking_uri: str = "sqlite:///mlflow.db"  # backend store (sqlite locally)
    experiment_name: str = "foundationmed-mortality"  # groups runs
    config_source: str = ""  # Provenance of the [model]/[training] values
    purpose: str = ""  # Why this run exists; free text, logged as a param


@dataclass
class ExperimentConfig:
    """A full training run: model + procedure + tracking."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


@dataclass
class DataConfig:
    """Which processed dataset to train on, its tag, and the prediction task."""

    processed_path: str
    name: str = ""  # MLflow dataset tag; defaults to the processed_path dir name
    task: str = "mortality"  # prediction target; see _data.available_tasks()

    def __post_init__(self) -> None:
        if not self.name:
            self.name = Path(self.processed_path).name or "unknown"


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment TOML."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return ExperimentConfig(
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        tracking=TrackingConfig(**raw.get("tracking", {})),
    )


def load_data_config(path: str | Path) -> DataConfig:
    """Load a dataset TOML."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    section = raw.get("data", raw)
    return DataConfig(**section)
