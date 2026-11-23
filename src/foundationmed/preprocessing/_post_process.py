"""Shared post-processing utilities for the preprocessing pipelines."""

from __future__ import annotations

import pickle
from pathlib import Path

import structlog

from foundationmed.data._blob import write_blob
from foundationmed.data._data import PatientGraph, Visit

log = structlog.get_logger()


def _visit_direct_lookup(graphs: dict[str, PatientGraph]) -> dict[str, Visit]:
    """Return {icu_id/hadm_id: Visit} for event attachment."""
    return {v.icu_id: v for g in graphs.values() for v in g.visits}


def sort_events(graphs: dict[str, PatientGraph]) -> None:
    for graph in graphs.values():
        for visit in graph.visits:
            visit.events.sort(key=lambda e: e.timestamp)


def normalize_visit_offsets(graphs: dict[str, PatientGraph]) -> None:
    """Rebase each graph's `start_offset` on its own first visit."""
    for graph in graphs.values():
        if not graph.visits:
            continue
        base = min(v.start_offset for v in graph.visits)
        for visit in graph.visits:
            visit.start_offset -= base


def build_vocabularies(graphs: dict[str, PatientGraph]) -> dict[str, dict[str, int]]:
    vocabs: dict[str, dict[str, int]] = {
        "diagnosis": {},
        "treatment": {},
        "medication": {},
        "lab": {},
    }
    for graph in graphs.values():
        for visit in graph.visits:
            for event in visit.events:
                vocab = vocabs[event.event_type]
                if event.code not in vocab:
                    vocab[event.code] = len(vocab)
    log.info(
        "vocabularies_built",
        sizes={etype: len(vocab) for etype, vocab in vocabs.items()},
    )
    return vocabs


def log_summary(graphs: dict[str, PatientGraph]) -> None:
    n_patients = len(graphs)
    n_visits = sum(g.n_visits for g in graphs.values())
    n_events = sum(g.n_events for g in graphs.values())
    multi_visit = sum(1 for g in graphs.values() if g.n_visits > 1)
    with_apache = sum(
        1 for g in graphs.values() for v in g.visits if v.apache_features is not None
    )
    mortality_rate = (
        sum(1 for g in graphs.values() for v in g.visits if v.mortality) / n_visits
    )

    log.info(
        "dataset_summary",
        patients=n_patients,
        multi_visit_patients=multi_visit,
        multi_visit_pct=round(100 * multi_visit / n_patients, 1),
        visits=n_visits,
        events=n_events,
        visits_with_apache=with_apache,
        avg_events_per_visit=round(n_events / n_visits, 1),
        mortality_rate_pct=round(100 * mortality_rate, 1),
    )


def save(graphs: dict[str, PatientGraph], vocabs: dict, output_path: Path) -> None:
    graphs_file = output_path / "patient_graphs.pkl"
    vocabs_file = output_path / "vocabularies.pkl"
    with open(graphs_file, "wb") as f:
        pickle.dump(graphs, f)
    with open(vocabs_file, "wb") as f:
        pickle.dump(vocabs, f)
    log.info("saved_patient_graphs", n=len(graphs), path=str(graphs_file))
    log.info("saved_vocabularies", path=str(vocabs_file))
    write_blob(graphs, output_path)
