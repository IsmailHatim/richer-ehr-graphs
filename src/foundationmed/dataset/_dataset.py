"""PyG Dataset over processed patient graphs."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog
import torch
from torch_geometric.data import Dataset, HeteroData

from foundationmed.data._blob import BlobGraphs, load_blob
from foundationmed.data._data import PatientGraph, task_attr

log = structlog.get_logger()


@dataclass(frozen=True)
class FeatureSpec:
    """Input dimensions a model needs to build its per-node-type encoders."""

    apache_dim: int
    demo_cont_dim: int
    n_genders: int
    n_ethnicities: int
    vocab_sizes: dict[str, int]
    # Extra feature columns appended to an event node type x
    extra_node_dims: dict[str, int] = field(default_factory=dict)


APACHE_DIM = (
    36  # 21 continuous + one-hot: eyes(4) + motor(6) + verbal(5) - 3 categorical
)

# Continuous demographic features appended to each visit node:
DEMO_CONT_DIM = 2

_UNK = "<unk>"


def _build_categorical_vocab(values) -> dict[str, int]:
    """Map distinct category strings to indices, 0 for unknown."""
    vocab = {_UNK: 0}
    for raw in values:
        key = (raw or "").strip().lower() or _UNK
        if key not in vocab:
            vocab[key] = len(vocab)
    return vocab


def _category_index(vocab: dict[str, int], raw: str | None) -> int:
    return vocab.get((raw or "").strip().lower() or _UNK, 0)


class EHRGraphDataset(Dataset):
    """PyG Dataset returning one HeteroData per *visit*.

    Args:
        graphs_path: Path to patient_graphs.pkl (output of _process.py). Only
                     its directory is read: the graphs themselves come from the
                     `graphs.bin` blob beside it.
        vocabs_path: Path to vocabularies.pkl (output of _process.py).
        task:        Which label to predict; selects which visits are samples.
        min_events:  Minimum number of events across all visits. Patients below
                     this threshold are excluded.
        max_visits:  If set, keep only the most recent N visits per example.
    """

    def __init__(
        self,
        graphs_path: str | Path,
        vocabs_path: str | Path,
        task: str = "mortality",
        min_events: int = 1,
        max_visits: int | None = None,
    ):
        super().__init__()
        self.max_visits = max_visits
        self.task = task
        self.lab_stats: dict[int, tuple[float, float]] | None = (
            None  # (mean, std) train
        )

        with open(vocabs_path, "rb") as f:
            self.vocabs: dict[str, dict[str, int]] = pickle.load(f)

        # Graphs come from the memmapped blob beside the pickle.
        index, blob_file = load_blob(Path(graphs_path).parent)
        keep = np.flatnonzero((index.n_visits > 0) & (index.n_events >= min_events))
        self.graphs = BlobGraphs(blob_file, index.offsets, keep)
        self._index, self._keep = index, keep

        label = getattr(index, task_attr(task))
        self.samples = np.array(
            [
                (i, t)
                for i, g in enumerate(keep)
                for t in range(int(index.n_visits[g]))
                if label[index.visit_ptr[g] + t] != -1  # -1 is None, i.e. censored
            ],
            dtype=np.int64,
        ).reshape(
            -1, 2
        )  # (N, 2) of (graph index, visit carrying this example's label).

        # Categorical demographic vocabularies (0 for unknown).
        self.gender_vocab = _build_categorical_vocab(self._visit_values("gender"))
        self.ethnicity_vocab = _build_categorical_vocab(self._visit_values("ethnicity"))

        log.info(
            "dataset_loaded",
            graphs=len(self.graphs),
            samples=len(self.samples),
            task=task,
            vocab_sizes={k: len(v) for k, v in self.vocabs.items()},
            genders=len(self.gender_vocab),
            ethnicities=len(self.ethnicity_vocab),
        )

    def _visit_values(self, field: str):
        """Yield a raw per-visit value over the filtered graphs, in `get` order."""
        values = getattr(self._index, f"{field}_values")
        codes = getattr(self._index, field)
        for g in self._keep:
            for code in codes[self._index.visit_ptr[g] : self._index.visit_ptr[g + 1]]:
                yield values[code]

    def _sample_rows(self) -> np.ndarray:
        """Flat per-visit row backing each example."""
        return (
            self._index.visit_ptr[self._keep[self.samples[:, 0]]] + self.samples[:, 1]
        )

    @property
    def labels(self) -> np.ndarray:
        """Target per example, read at the visit that example predicts."""
        attr = getattr(self._index, task_attr(self.task))
        return attr[self._sample_rows()].astype(np.int64)

    @property
    def groups(self) -> np.ndarray:
        """Person per example, so grouped splitting keeps them together."""
        return self._index.group_id[self._keep[self.samples[:, 0]]].astype(object)

    def fit_lab_normalization(self, train_idx: np.ndarray) -> None:
        """Fit per-lab-code z-score statistics on the train split only."""
        sums: dict[int, float] = {}
        sq_sums: dict[int, float] = {}
        counts: dict[int, int] = {}
        lab_vocab = self.vocabs["lab"]

        for i in np.unique(self.samples[np.asarray(train_idx), 0]):
            for visit in self.graphs[int(i)].visits:
                for event in visit.events:
                    if event.event_type != "lab" or event.value is None:
                        continue
                    code = lab_vocab.get(event.code, 0)
                    value = float(event.value)
                    sums[code] = sums.get(code, 0.0) + value
                    sq_sums[code] = sq_sums.get(code, 0.0) + value * value
                    counts[code] = counts.get(code, 0) + 1

        stats: dict[int, tuple[float, float]] = {}
        for code, n in counts.items():
            mean = sums[code] / n
            var = max(sq_sums[code] / n - mean * mean, 0.0)
            stats[code] = (mean, max(var**0.5, 1e-8))  # min std fixed to 1e-8
        self.lab_stats = stats

        log.info(
            "lab_normalization_fitted",
            codes=len(stats),
            of_vocab=len(lab_vocab),
            train_samples=len(train_idx),
        )

    def _visits(self, graph: PatientGraph, upto: int):
        """Visits `get` builds features from: through `upto`, then max_visits."""
        visits = graph.visits[: upto + 1]
        if self.max_visits is None:
            return visits
        return visits[-self.max_visits :]

    def _normalize(self, codes: list[int], values: list[float]) -> list[float]:
        """Z-score lab values with the fitted train-split statistics."""
        if self.lab_stats is None:
            return values
        out = []
        for code, value in zip(codes, values):
            stat = self.lab_stats.get(code)
            out.append(0.0 if stat is None else (value - stat[0]) / stat[1])
        return out

    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> HeteroData:
        graph_idx, upto = self.samples[idx]
        graph = self.graphs[graph_idx]
        visits = self._visits(graph, upto)

        data = HeteroData()
        data.patient_id = graph.patient_id
        data.sample_idx = int(idx)
        n_visits = len(visits)

        # Visit nodes
        visit_feats = []
        for visit in visits:
            feat = np.zeros(APACHE_DIM, dtype=np.float32)
            if visit.apache_features is not None:
                raw = np.nan_to_num(
                    np.asarray(visit.apache_features, dtype=np.float32).ravel(), nan=0.0
                )
                n = min(raw.shape[0], APACHE_DIM)
                feat[:n] = raw[:n]
            visit_feats.append(feat)

        data["visit"].x = torch.tensor(np.stack(visit_feats), dtype=torch.float)
        data["visit"].pos = torch.arange(n_visits, dtype=torch.long)

        data["visit"].age = torch.tensor(
            [float(v.age) for v in visits], dtype=torch.float
        )
        data["visit"].duration_h = torch.tensor(
            [v.icu_duration / 60.0 for v in visits], dtype=torch.float
        )
        data["visit"].gender = torch.tensor(
            [_category_index(self.gender_vocab, v.gender) for v in visits],
            dtype=torch.long,
        )
        data["visit"].ethnicity = torch.tensor(
            [_category_index(self.ethnicity_vocab, v.ethnicity) for v in visits],
            dtype=torch.long,
        )
        data["visit"].y_mortality = torch.tensor(
            [v.mortality for v in visits], dtype=torch.bool
        )
        data["visit"].y_readmission = torch.tensor(
            [bool(v.readmission) for v in visits], dtype=torch.bool
        )  # Undefined labels are not read

        # Event nodes + relational edges
        buffers: dict[str, dict] = {
            etype: {
                "codes": [],
                "timestamps": [],
                "values": [],
                "rel_src": [],
                "rel_dst": [],
            }
            for etype in ("diagnosis", "treatment", "medication", "lab")
        }

        for visit_idx, visit in enumerate(visits):
            for event in visit.events:
                buf = buffers[event.event_type]
                event_idx = len(buf["codes"])
                buf["codes"].append(self.vocabs[event.event_type].get(event.code, 0))
                buf["timestamps"].append(
                    (visit.start_offset + event.timestamp) / 60.0
                )  # minutes to hours
                buf["values"].append(event.value if event.value is not None else 0.0)
                buf["rel_src"].append(visit_idx)
                buf["rel_dst"].append(event_idx)

        for etype, buf in buffers.items():
            n = len(buf["codes"])

            if n == 0:
                # still materialize x / t with the correct feature width (0 rows)
                feat_dim = 2 if etype == "lab" else 1
                data[etype].x = torch.zeros((0, feat_dim), dtype=torch.float)
                data[etype].t = torch.zeros(0, dtype=torch.float)
                data["visit", f"has_{etype}", etype].edge_index = torch.zeros(
                    (2, 0), dtype=torch.long
                )
                continue

            codes = torch.tensor(buf["codes"], dtype=torch.long)
            timestamps = torch.tensor(buf["timestamps"], dtype=torch.float)

            if etype == "lab":
                values = torch.tensor(
                    self._normalize(buf["codes"], buf["values"]), dtype=torch.float
                )
                data[etype].x = torch.stack([codes.float(), values], dim=1)
            else:
                data[etype].x = codes.unsqueeze(1).float()

            data[etype].t = timestamps

            data["visit", f"has_{etype}", etype].edge_index = torch.tensor(
                [buf["rel_src"], buf["rel_dst"]], dtype=torch.long
            )

        # Temporal edges
        if n_visits > 1:
            data["visit", "next", "visit"].edge_index = torch.tensor(
                [list(range(n_visits - 1)), list(range(1, n_visits))], dtype=torch.long
            )
        else:
            data["visit", "next", "visit"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long
            )

        return data

    @property
    def vocab_sizes(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.vocabs.items()}

    @property
    def feature_spec(self) -> FeatureSpec:
        """Dimensions needed by the modeling layer to size its encoders."""
        return FeatureSpec(
            apache_dim=APACHE_DIM,
            demo_cont_dim=DEMO_CONT_DIM,
            n_genders=len(self.gender_vocab),
            n_ethnicities=len(self.ethnicity_vocab),
            vocab_sizes=self.vocab_sizes,
        )
