# FoundationMed

Sparse, heterogeneous EHR data as patient-level longitudinal graphs, with graph
neural network benchmarks on top. One sample is one *visit* — a visit whose
label is defined, built from that visit and its predecessors — not one patient.

Datasets: eICU 2.0 and MIMIC-IV (raw CSVs, obtained separately under their own
data-use agreements). Task: in-hospital / post-discharge mortality.

## Install

Environments are managed with [Pixi](https://pixi.sh) (conda-based, lockfile in
`pixi.lock`):

```bash
curl -fsSL https://pixi.sh/install.sh | sh    # macOS / Linux; see pixi.sh for Windows
pixi install -e dev                           # solves from pixi.lock
```

## Run

Every command is driven by TOML configs in `config/`. Three orthogonal axes:
`--config` (model + optimisation), `--data` (which processed dataset, which
task), `--graph` (which graph representation; omit for the base graph).

```bash
# 1. raw CSVs → patient graphs (set `path` in the processing TOML first)
pixi run fmed preprocess eicu --config config/processing/eicu.toml

# 2. smoke run: 500 samples, 3 epochs, small model — minutes on a laptop
pixi run -e dev fmed train --config config/training/smoke_gat.toml \
  --data config/data/eicu.toml --graph config/graph/prior.toml

# 3. a full model
pixi run -e dev fmed train --config config/training/mortality_hetero_gatv2.toml \
  --data config/data/eicu.toml --graph config/graph/prior.toml

# 4. one backbone across graph representations, 5 seeds each
pixi run -e dev fmed compare --config config/training/mortality_hetero_gatv2.toml \
  --data config/data/mimic4.toml \
  --graphs config/graph/evolution.toml --graphs config/graph/prior.toml --seeds 5
```

**Backbones** (`backbone` in the training TOML): `gcn`, `gat`, `hetero_gatv2`,
`hgt` — one config each in `config/training/`.

**Graph representations** (`config/graph/`):

| Version | Edges added on top of the visit hierarchy |
|---------|-------------------------------------------|
| `base` | none |
| `evolution` | directed edges between consecutive readings of the same lab code, carrying (Δt, Δv) |
| `local` | evolution + per-patient cross-variable coupling: windowed Pearson over lab values, an edge where \|ρ\| ≥ threshold |
| `prior` | evolution + coupling drawn from a population lab-code prior fitted on the training split only, so any patient carrying both codes gets the edge |

Results land in MLflow (`sqlite:///mlflow.db` by default), one run per
`(graph version, seed)`, tagged with `backbone` / `graph_version` / `dataset` /
`task` / `seed`; `model.pt` + `split.npz` are written under `output_path`.

```bash
pixi run mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://127.0.0.1:5000
```

## Layout

```
src/foundationmed/
  cli.py            # the `fmed` Typer app
  data/             # Event / Visit / PatientGraph + the graphs.bin blob
  preprocessing/    # one pipeline per dataset
  dataset/          # EHRGraphDataset: one HeteroData sample per labelled visit
  graph/            # graph-version registry, components, population prior, cache
  models/           # backbones + the registry `backbone` selects from
  training/         # config, Ignite engines/handlers, train, compare, tune
config/             # processing / data / training / graph / search TOMLs
```

## Development

```bash
pixi run -e dev ruff format src && pixi run -e dev ruff check src
pixi run -e dev ty check src
```
