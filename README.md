# When Do Richer EHR Graphs Help? Representation, Architecture, and Task

Ismail Hatim¹ · Guillaume Lachaud¹ · Yanlei Diao¹ · Benoît Gallix²

¹ École polytechnique, France · ² American Hospital of Paris, France

Code for the ML4H 2026 Findings Track poster. Sparse, heterogeneous EHR data
become patient-level longitudinal graphs, with graph neural network benchmarks
on top. One sample is one *visit* — a visit whose label is defined, built from
that visit and its predecessors — not one patient.

This release covers the outcome benchmark: preprocessing, the four graph
representations, the four backbones, training, multi-seed comparison and
hyperparameter search. The laboratory-prediction tasks and the attention
analyses shown on the poster are not part of this release.

## Data

No patient data is distributed here. Both datasets are available to
credentialed researchers through PhysioNet, under their own data-use
agreements:

- [eICU Collaborative Research Database 2.0](https://physionet.org/content/eicu-crd/2.0/):
  `patient.csv`, `diagnosis.csv`, `treatment.csv`, `medication.csv`, `lab.csv`,
  `apacheApsVar.csv`
- [MIMIC-IV 2.0](https://physionet.org/content/mimiciv/2.0/), `hosp` module:
  `patients.csv`, `admissions.csv`, `diagnoses_icd.csv`, `procedures_icd.csv`,
  `prescriptions.csv`, `labevents.csv`, `d_icd_diagnoses.csv`,
  `d_icd_procedures.csv`, `d_labitems.csv`

The pipelines read uncompressed `.csv` files, so decompress the `.csv.gz`
downloads first.

Task: mortality, defined as death at ICU discharge for eICU and death within 90
days of discharge for MIMIC-IV. Setting `task = "readmission"` in a
`config/data/*.toml` switches the label.

## Install

Environments are managed with [Pixi](https://pixi.sh) (conda-based, lockfile in
`pixi.lock`):

```bash
curl -fsSL https://pixi.sh/install.sh | sh    # macOS / Linux; see pixi.sh for Windows
pixi install -e dev                           # solves from pixi.lock
pixi install -e cuda                          # Linux + NVIDIA GPU; then use `-e cuda` below
```

Supported platforms are `linux-64` and `osx-arm64`. The device is picked
automatically (MPS, then CUDA, then CPU).

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

# 5. hyperparameter search (Ray Tune + Optuna TPE + ASHA); smoke.toml is a 2-trial check
pixi run -e dev fmed tune --config config/training/mortality_gat.toml \
  --data config/data/eicu.toml --search config/search/smoke.toml --max-concurrent 1

# optional: precompute the `local` coupling edges once instead of every epoch
pixi run -e dev fmed cache-graph --data config/data/mimic4.toml \
  --graph config/graph/local.toml --config config/training/mortality_gcn.toml
```

The first run of `train` / `compare` / `tune` on a dataset needs step 1 to have
written `graphs.bin` under its `processed_path`.

**Backbones** (`backbone` in the training TOML): `gcn`, `gat`, `hetero_gatv2`,
`hgt` — one config each in `config/training/`. The full configs set
`num_workers = 32`; lower it to your physical core count, since building
samples on the CPU is what bounds epoch time.

**Graph representations** (`config/graph/`; the poster's G_base, G_evo, G_local
and G_prior):

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

## Citation

```bibtex
@misc{hatim2026richer,
  title  = {When Do Richer {EHR} Graphs Help? Representation, Architecture, and Task},
  author = {Hatim, Ismail and Lachaud, Guillaume and Diao, Yanlei and Gallix, Beno{\^i}t},
  year   = {2026},
  note   = {Machine Learning for Health (ML4H) 2026, Findings Track}
}
```

## License

MIT, see [LICENSE](LICENSE). The licence covers the code only; eICU and MIMIC-IV
remain under their PhysioNet data-use agreements.
