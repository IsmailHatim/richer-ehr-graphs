"""Core data structures for patient-level longitudinal graphs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Event:
    """A single clinical event attached to a visit node."""

    timestamp: int  # offset in minutes from ICU admission
    event_type: str  # "diagnosis" | "treatment" | "medication" | "lab"
    code: str  # identifier
    value: float | None = None  # numeric value for labs, None otherwise


@dataclass
class Visit:
    """One ICU stay: a visit node in the patient graph."""

    icu_id: str
    admission_id: str
    hospital_id: str
    icu_duration: int  # minutes
    age: int
    gender: str
    ethnicity: str
    mortality: bool
    readmission: bool | None  # Drop no next-visit from MIMIC
    apache_features: np.ndarray | None = None  # (36,) static clinical features
    start_offset: int = 0  # Minutes from the graph's first visit
    events: list[Event] = field(default_factory=list)


@dataclass
class PatientGraph:
    """A patient's full longitudinal record as a graph."""

    patient_id: str
    visits: list[Visit] = field(default_factory=list)
    # TODO: rename; for eICU this is the person behind several graphs
    subject_id: str | None = None

    @property
    def group_id(self) -> str:
        """Person-level key for leakage-free splitting (eICU multi-graph patients)."""
        return getattr(self, "subject_id", None) or self.patient_id

    @property
    def n_visits(self) -> int:
        return len(self.visits)

    @property
    def n_events(self) -> int:
        return sum(len(v.events) for v in self.visits)

    def is_valid(self) -> bool:
        """A graph is valid if it has at least one visit with at least one event."""
        return self.n_visits > 0 and self.n_events > 0


# Prediction tasks
_TASK_ATTR = {
    "mortality": "mortality",
    "readmission": "readmission",
}


def available_tasks() -> tuple[str, ...]:
    """The prediction tasks the training layer knows how to label."""
    return tuple(_TASK_ATTR)


def task_attr(task: str) -> str:
    """Resolve a task name to its `Visit` attribute, or raise on an unknown one."""
    try:
        return _TASK_ATTR[task]
    except KeyError:
        raise ValueError(
            f"unknown task {task!r}; available: {', '.join(sorted(_TASK_ATTR))}"
        ) from None
