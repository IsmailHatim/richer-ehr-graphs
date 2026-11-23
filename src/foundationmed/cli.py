"""Command-line interface for FoundationMed."""

from __future__ import annotations

from pathlib import Path

import typer

from foundationmed._logging import setup_logging

app = typer.Typer(
    add_completion=False,
    help="Sparse EHR data → patient-level longitudinal graphs → GNN benchmarks.",
)
preprocess_app = typer.Typer(help="Process a raw dataset into patient graphs.")
app.add_typer(preprocess_app, name="preprocess")


def parse_tags(tags: list[str] | None) -> dict[str, str]:
    """Parse repeated --tag KEY=VALUE options into a dict."""
    parsed: dict[str, str] = {}
    for item in tags or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise typer.BadParameter(f"expected KEY=VALUE, got {item!r}")
        parsed[key] = value
    return parsed


@app.callback()
def main() -> None:
    """Configure logging before any command runs."""
    setup_logging()


@preprocess_app.command()
def eicu(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to the eICU dataset TOML"
    ),
) -> None:
    """Process raw eICU CSVs into patient graphs and vocabularies."""
    from foundationmed.preprocessing._process_eicu import run_pipeline

    run_pipeline(config)


@preprocess_app.command()
def mimic4(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to the MIMIC-IV dataset TOML"
    ),
) -> None:
    """Process raw MIMIC-IV CSVs into patient graphs and vocabularies."""
    from foundationmed.preprocessing._process_mimic4 import run_pipeline

    run_pipeline(config)


@preprocess_app.command("blob")
def preprocess_blob(
    processed_path: Path = typer.Argument(
        ..., help="A processed_path holding patient_graphs.pkl"
    ),
) -> None:
    """Build graphs.bin for an already-processed dataset."""
    # TODO: temporary CLI; delete once every processed_path has a blob.
    import pickle

    import structlog

    from foundationmed.data._blob import write_blob

    graphs_file = processed_path / "patient_graphs.pkl"
    if not graphs_file.exists():
        raise typer.BadParameter(f"no patient_graphs.pkl in {processed_path}")

    structlog.get_logger().info(
        "loading_graphs", path=str(graphs_file), note="unpickling is silent, minutes"
    )
    with open(graphs_file, "rb") as f:
        graphs = pickle.load(f)
    write_blob(graphs, processed_path)


@app.command()
def train(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to a (dataset-agnostic) training TOML"
    ),
    data: Path = typer.Option(
        ..., "--data", "-d", help="Path to a dataset TOML (config/data/*.toml)"
    ),
    graph: Path = typer.Option(
        None,
        "--graph",
        "-g",
        help="Graph-version TOML (config/graph/*.toml); omit for the base graph.",
    ),
    tag: list[str] = typer.Option(
        None,
        "--tag",
        "-t",
        help="Extra MLflow tag KEY=VALUE (repeatable), e.g. config_source=best.",
    ),
) -> None:
    """Train and benchmark a model on one processed dataset."""
    from foundationmed.training._config import (
        load_data_config,
        load_experiment_config,
    )
    from foundationmed.training._train import run_training

    run_training(
        load_experiment_config(config),
        load_data_config(data),
        graph=str(graph) if graph else None,
        extra_tags=parse_tags(tag) or None,
    )


@app.command()
def compare(
    config: list[Path] = typer.Option(
        ...,
        "--config",
        "-c",
        help="Base training TOML. Pass one to share across all graphs, or one "
        "per --graphs to give each its own tuned best_config.toml.",
    ),
    data: Path = typer.Option(
        ..., "--data", "-d", help="Path to a dataset TOML (config/data/*.toml)"
    ),
    graphs: list[Path] = typer.Option(
        ..., "--graphs", "-g", help="Graph-version TOMLs (repeat the flag per version)"
    ),
    seeds: int = typer.Option(5, "--seeds", "-s", help="Seeds per graph version"),
    output_path: Path = typer.Option(
        Path("data/models/comparison"),
        "--output",
        "-o",
        help="Root dir for per-run weights/splits (metrics go to MLflow)",
    ),
    tag: list[str] = typer.Option(
        None,
        "--tag",
        "-t",
        help="Extra MLflow tag KEY=VALUE (repeatable), applied to every run.",
    ),
) -> None:
    """Compare graph representations under one fixed protocol."""
    from foundationmed.training._compare import run_comparison
    from foundationmed.training._config import (
        load_data_config,
        load_experiment_config,
    )

    if len(config) not in (1, len(graphs)):
        raise typer.BadParameter(
            f"pass 1 --config (shared) or one per --graphs ({len(graphs)}); "
            f"got {len(config)}"
        )
    exps = [load_experiment_config(c) for c in config]

    run_comparison(
        exps[0] if len(exps) == 1 else exps,
        load_data_config(data),
        [str(g) for g in graphs],
        seeds,
        str(output_path),
        config_paths=[str(c) for c in config],
        extra_tags=parse_tags(tag) or None,
    )


@app.command()
def tune(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to the base training TOML"
    ),
    data: Path = typer.Option(
        ..., "--data", "-d", help="Path to a dataset TOML (config/data/*.toml)"
    ),
    search: Path = typer.Option(
        ..., "--search", "-s", help="Search TOML (config/search/*.toml)"
    ),
    graph: Path = typer.Option(
        None,
        "--graph",
        "-g",
        help="Graph-version TOML applied to every trial; omit for the base graph.",
    ),
    num_trials: int = typer.Option(
        0, "--num-trials", "-n", help="Override [study] num_trials (0 = keep)"
    ),
    max_concurrent: int = typer.Option(
        0, "--max-concurrent", help="Cap concurrent trials (use 1 on MPS Macs)"
    ),
) -> None:
    """Hyperparameter search: Ray Tune + Optuna (TPE) + ASHA over run_training."""
    from foundationmed.training._config import (
        load_data_config,
        load_experiment_config,
    )
    from foundationmed.training._tune import load_search_config, run_tuning

    run_tuning(
        load_experiment_config(config),
        load_data_config(data),
        load_search_config(search),
        graph=str(graph) if graph else None,
        num_trials=num_trials or None,
        max_concurrent=max_concurrent or None,
    )


@app.command("cache-graph")
def cache_graph(
    data: Path = typer.Option(
        ..., "--data", "-d", help="Path to the dataset TOML to precompute for"
    ),
    graph: Path = typer.Option(
        ..., "--graph", "-g", help="Path to the graph-version TOML (must use coupling)"
    ),
    config: Path = typer.Option(
        None,
        "--config",
        "-c",
        help="Training TOML supplying min_events/max_visits (else defaults)",
    ),
) -> None:
    """Precompute this graph version's coupling edges once, for every later run."""
    from foundationmed.dataset._dataset import EHRGraphDataset
    from foundationmed.graph._cache import (
        DEFAULT_CACHE_DIR,
        build_cache,
        cache_key,
        cache_path,
    )
    from foundationmed.graph._config import load_graph_config
    from foundationmed.graph._registry import build_pipeline
    from foundationmed.graph.components._coupling import CouplingEdges
    from foundationmed.training._config import (
        TrainingConfig,
        load_data_config,
        load_experiment_config,
    )

    data_cfg = load_data_config(data)
    graph_cfg = load_graph_config(graph)
    if "coupling" not in graph_cfg.components:
        raise typer.BadParameter(
            f"graph version {graph_cfg.name!r} has no coupling component; "
            "nothing to cache"
        )
    training = load_experiment_config(config).training if config else TrainingConfig()

    dataset = EHRGraphDataset(
        graphs_path=Path(data_cfg.processed_path) / "patient_graphs.pkl",
        vocabs_path=Path(data_cfg.processed_path) / "vocabularies.pkl",
        task=data_cfg.task,
        min_events=training.min_events,
        max_visits=training.max_visits or None,
    )
    params = graph_cfg.params.get("coupling", {})
    key = cache_key(
        data_cfg.name,
        data_cfg.task,
        training.min_events,
        training.max_visits or None,
        params,
    )
    coupling = next(
        step
        for step in build_pipeline(graph_cfg).transforms
        if isinstance(step, CouplingEdges)
    )
    build_cache(dataset, coupling, cache_path(DEFAULT_CACHE_DIR, key), key)


if __name__ == "__main__":
    app()
