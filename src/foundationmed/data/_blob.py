"""Flat, fork-safe storage for the processed patient graphs."""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from foundationmed.data._data import PatientGraph

log = structlog.get_logger()

BLOB_VERSION = 1
BLOB_FILE = "graphs.bin"
INDEX_FILE = "graph_index.pkl"


@dataclass(frozen=True)
class GraphIndex:
    """Index needed for EHRGraphDataset without unpickling a graph."""

    version: int
    offsets: np.ndarray  # (G+1,) int64  byte spans into the blob
    n_visits: np.ndarray  # (G,)   int32
    n_events: np.ndarray  # (G,)   int64
    patient_id: np.ndarray  # (G,)   object
    group_id: np.ndarray  # (G,)   object
    visit_ptr: np.ndarray  # (G+1,) int64
    mortality: np.ndarray  # (V,)   int8   -1 = None
    readmission: np.ndarray  # (V,)   int8   -1 = None
    gender: np.ndarray  # (V,)   int32  into gender_values
    ethnicity: np.ndarray  # (V,)   int32  into ethnicity_values
    gender_values: list  # raw values, None included
    ethnicity_values: list


def _memmap(blob_file: Path) -> np.ndarray:
    """Map the buffer read-only. np.memmap refuses a zero-length file."""
    if blob_file.stat().st_size == 0:
        return np.empty(0, dtype=np.uint8)
    return np.memmap(blob_file, dtype=np.uint8, mode="r")


class BlobGraphs:
    """The graphs list, backed by a memmapped buffer."""

    def __init__(self, blob_file: Path, offsets: np.ndarray, keep: np.ndarray):
        """Map the buffer and bind its byte offsets and surviving graph ids."""
        self._path = Path(blob_file)
        self._offsets, self._keep = offsets, keep
        self._buf = _memmap(self._path)

    def __getstate__(self) -> dict:
        """Pickle the *path* and not the bytes."""
        return {"path": self._path, "offsets": self._offsets, "keep": self._keep}

    def __setstate__(self, state: dict) -> None:
        """Re-map the buffer from its path in the receiving process."""
        self.__init__(state["path"], state["offsets"], state["keep"])

    def __len__(self) -> int:
        """Number of graphs surviving the inclusion filter."""
        return len(self._keep)

    def __getitem__(self, i: int) -> PatientGraph:
        """Unpickle one graph out of the buffer."""
        g = self._keep[i]
        return pickle.loads(
            self._buf[self._offsets[g] : self._offsets[g + 1]].tobytes()
        )

    def __iter__(self) -> Iterator[PatientGraph]:
        """Yield each surviving graph in order."""
        for i in range(len(self._keep)):
            yield self[i]


def write_blob(graphs: dict[str, PatientGraph], output_path: Path) -> Path:
    """Serialise every graph to `graphs.bin`, streaming, and build its index."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    blob_file, index_file = output_path / BLOB_FILE, output_path / INDEX_FILE

    offsets, n_visits, n_events = [0], [], []
    patient_id, group_id, visit_ptr = [], [], [0]
    mortality, readmission, gender, ethnicity = [], [], [], []
    gender_tab: dict = {}
    ethnicity_tab: dict = {}

    def intern(table: dict, value) -> int:
        if value not in table:
            table[value] = len(table)
        return table[value]

    with open(blob_file, "wb") as f:
        for graph in graphs.values():  # preprocessing order; never re-sorted
            payload = pickle.dumps(graph, protocol=5)
            f.write(payload)
            offsets.append(offsets[-1] + len(payload))
            n_visits.append(graph.n_visits)
            n_events.append(graph.n_events)
            patient_id.append(graph.patient_id)
            group_id.append(graph.group_id)
            for visit in graph.visits:
                mortality.append(
                    -1 if visit.mortality is None else int(visit.mortality)
                )
                readmission.append(
                    -1 if visit.readmission is None else int(visit.readmission)
                )
                gender.append(intern(gender_tab, visit.gender))
                ethnicity.append(intern(ethnicity_tab, visit.ethnicity))
            visit_ptr.append(len(mortality))

    index = GraphIndex(
        version=BLOB_VERSION,
        offsets=np.array(offsets, dtype=np.int64),
        n_visits=np.array(n_visits, dtype=np.int32),
        n_events=np.array(n_events, dtype=np.int64),
        patient_id=np.array(patient_id, dtype=object),
        group_id=np.array(group_id, dtype=object),
        visit_ptr=np.array(visit_ptr, dtype=np.int64),
        mortality=np.array(mortality, dtype=np.int8),
        readmission=np.array(readmission, dtype=np.int8),
        gender=np.array(gender, dtype=np.int32),
        ethnicity=np.array(ethnicity, dtype=np.int32),
        gender_values=list(gender_tab),
        ethnicity_values=list(ethnicity_tab),
    )
    with open(index_file, "wb") as f:
        pickle.dump(index, f, protocol=5)

    log.info(
        "saved_graph_blob",
        path=str(blob_file),
        graphs=len(n_visits),
        visits=len(mortality),
        mb=round(blob_file.stat().st_size / 1e6, 1),
    )
    return blob_file


def load_blob(processed_path: Path) -> tuple[GraphIndex, Path]:
    """Read the index and locate the buffer, or say how to build them."""
    processed_path = Path(processed_path)
    blob_file, index_file = processed_path / BLOB_FILE, processed_path / INDEX_FILE
    if not (blob_file.exists() and index_file.exists()):
        raise FileNotFoundError(
            f"no graph blob in {processed_path}. Datasets processed before the "
            f"blob existed need a one-off migration:\n"
            f"    fmed preprocess blob {processed_path}"
        )
    with open(index_file, "rb") as f:
        index: GraphIndex = pickle.load(f)
    if index.version != BLOB_VERSION:
        raise ValueError(
            f"graph blob in {processed_path} is version {index.version}, "
            f"expected {BLOB_VERSION}; rebuild it with "
            f"`fmed preprocess blob {processed_path}`"
        )
    return index, blob_file
