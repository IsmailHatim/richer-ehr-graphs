"""eICU preprocessing pipeline: raw CSVs -> processed pickle files."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

import numpy as np
import polars as pl
import structlog
from tqdm import tqdm

from foundationmed.data._data import Event, PatientGraph, Visit
from foundationmed.preprocessing._post_process import (
    _visit_direct_lookup,
    build_vocabularies,
    log_summary,
    normalize_visit_offsets,
    save,
    sort_events,
)

log = structlog.get_logger()

# eICU constants

APACHEAPSVAR_COLS = [
    "intubated",
    "vent",
    "dialysis",
    "eyes",
    "motor",
    "verbal",
    "meds",
    "urine",
    "wbc",
    "temperature",
    "respiratoryrate",
    "sodium",
    "heartrate",
    "meanbp",
    "ph",
    "hematocrit",
    "creatinine",
    "albumin",
    "pao2",
    "pco2",
    "bun",
    "glucose",
    "bilirubin",
    "fio2",
]

APACHEAPSVAR_NCAT = {"eyes": 4, "motor": 6, "verbal": 5}

# 1. Patients


def _build_readmission_labels(df: pl.DataFrame) -> dict[str, bool]:
    readmission: dict[str, bool] = {}
    for _, group in df.group_by("patienthealthsystemstayid"):
        stays = group.with_columns(
            pl.col("hospitaladmitoffset").cast(pl.Int64).neg().alias("_entry")
        ).sort("_entry")
        icu_ids = stays["patientunitstayid"].cast(pl.Utf8).to_list()
        timestamps = stays["_entry"].to_list()
        for i in range(len(icu_ids) - 1):
            readmission[icu_ids[i]] = timestamps[i + 1] - timestamps[i] <= 15 * 24 * 60
        readmission[icu_ids[-1]] = False
    return readmission


def process_patients(df: pl.DataFrame) -> dict[str, PatientGraph]:
    """Build one graph per hospital admission."""
    readmission = _build_readmission_labels(df)
    offset_map = {
        str(row["patientunitstayid"]): -int(row["hospitaladmitoffset"])
        for row in df.iter_rows(named=True)
    }
    graphs: dict[str, PatientGraph] = {}
    excl_duration = excl_age = 0

    for row in df.iter_rows(named=True):
        icu_id = str(row["patientunitstayid"])
        duration = int(row["unitdischargeoffset"])

        if duration > 10 * 24 * 60 or duration < 12 * 60:
            excl_duration += 1
            continue
        try:
            age = int(str(row["age"]))
        except ValueError:
            excl_age += 1
            continue
        if age < 18:
            excl_age += 1
            continue

        pid = str(row["patienthealthsystemstayid"])
        visit = Visit(
            icu_id=icu_id,
            admission_id=str(row["patienthealthsystemstayid"]),
            hospital_id=str(row["hospitalid"]),
            icu_duration=duration,
            age=age,
            gender=str(row["gender"]),
            ethnicity=str(row["ethnicity"]),
            mortality=row["unitdischargestatus"] == "Expired",
            readmission=readmission.get(icu_id, False),
            start_offset=-int(
                row["hospitaladmitoffset"]
            ),  # ICU entry relative to the hospital admission:
        )
        if pid not in graphs:
            graphs[pid] = PatientGraph(patient_id=pid, subject_id=str(row["uniquepid"]))
        graphs[pid].visits.append(visit)

    for graph in graphs.values():
        graph.visits.sort(key=lambda v: offset_map.get(v.icu_id, 0))

    log.info("patients_excluded", duration=excl_duration, age=excl_age)
    log.info(
        "patients_included",
        stays=sum(g.n_visits for g in graphs.values()),
        admissions=len(graphs),
        people=len({g.group_id for g in graphs.values()}),
    )
    return graphs


# 2. Diagnosis


def attach_diagnosis(df: pl.DataFrame, graphs: dict[str, PatientGraph]) -> None:
    visit_lk = _visit_direct_lookup(graphs)
    attached = skipped = 0
    for row in tqdm(
        df.iter_rows(named=True), total=df.height, desc="diagnosis", leave=False
    ):
        icu_id = str(row["patientunitstayid"])
        code = row["diagnosisstring"]
        if not code or icu_id not in visit_lk:
            skipped += 1
            continue
        ts = int(row["diagnosisoffset"])
        visit = visit_lk[icu_id]
        if ts > visit.icu_duration:
            skipped += 1
            continue
        visit.events.append(Event(ts, "diagnosis", code.lower()))
        attached += 1
    log.info("diagnosis_attached", attached=attached, skipped=skipped)


# 3. Treatment


def attach_treatment(df: pl.DataFrame, graphs: dict[str, PatientGraph]) -> None:
    visit_lk = _visit_direct_lookup(graphs)
    attached = skipped = 0
    for row in tqdm(
        df.iter_rows(named=True), total=df.height, desc="treatment", leave=False
    ):
        icu_id = str(row["patientunitstayid"])
        code = row["treatmentstring"]
        if not code or icu_id not in visit_lk:
            skipped += 1
            continue
        ts = int(row["treatmentoffset"])
        visit = visit_lk[icu_id]
        if ts > visit.icu_duration:
            skipped += 1
            continue
        visit.events.append(Event(ts, "treatment", code.lower()))
        attached += 1
    log.info("treatment_attached", attached=attached, skipped=skipped)


# 4. Medication


def attach_medication(df: pl.DataFrame, graphs: dict[str, PatientGraph]) -> None:
    visit_lk = _visit_direct_lookup(graphs)
    attached = skipped = 0
    for row in tqdm(
        df.iter_rows(named=True), total=df.height, desc="medication", leave=False
    ):
        icu_id = str(row["patientunitstayid"])
        code = row["drugname"]
        if not code or icu_id not in visit_lk:
            skipped += 1
            continue
        ts = int(row["drugstartoffset"])
        visit = visit_lk[icu_id]
        if ts > visit.icu_duration:
            skipped += 1
            continue
        visit.events.append(Event(ts, "medication", code.lower()))
        attached += 1
    log.info("medication_attached", attached=attached, skipped=skipped)


# 5. Lab


def prepare_labs(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows without a numeric result. Values stay in their raw units."""
    return df.filter(pl.col("labresult").is_not_null())


def attach_lab(df: pl.DataFrame, graphs: dict[str, PatientGraph]) -> None:
    visit_lk = _visit_direct_lookup(graphs)
    attached = skipped = 0
    for row in tqdm(df.iter_rows(named=True), total=df.height, desc="lab", leave=False):
        icu_id = str(row["patientunitstayid"])
        code = row["labname"]
        if not code or icu_id not in visit_lk:
            skipped += 1
            continue
        ts = int(row["labresultoffset"])
        val = row["labresult"]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            skipped += 1
            continue
        visit = visit_lk[icu_id]
        if ts > visit.icu_duration:
            skipped += 1
            continue
        visit.events.append(Event(ts, "lab", code.lower(), float(val)))
        attached += 1
    log.info("lab_attached", attached=attached, skipped=skipped)


# 6. ApacheApsVar


def attach_apache(df: pl.DataFrame, graphs: dict[str, PatientGraph]) -> None:
    visit_lk = _visit_direct_lookup(graphs)
    attached = skipped = 0
    for row in df.iter_rows(named=True):
        icu_id = str(row["patientunitstayid"])
        if icu_id not in visit_lk:
            skipped += 1
            continue
        features: list[float] = []
        for col in APACHEAPSVAR_COLS:
            raw = row[col]
            val = np.nan if (raw is None or raw == "" or raw == -1) else float(raw)
            if col in APACHEAPSVAR_NCAT:
                ncat = APACHEAPSVAR_NCAT[col]
                if np.isnan(val):
                    features.extend([np.nan] * ncat)
                else:
                    onehot = [0.0] * ncat
                    onehot[int(val) - 1] = 1.0
                    features.extend(onehot)
            else:
                features.append(val)
        visit_lk[icu_id].apache_features = np.array(features, dtype=np.float32)
        attached += 1
    log.info("apache_attached", attached=attached, skipped=skipped)


# Pipeline entry point


def run_pipeline(config_path: Path) -> None:
    """Process raw eICU CSVs into patient graphs and vocabularies."""
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    raw_path = Path(config["eicu"]["path"])
    output_path = Path(config["eicu"].get("processed_path", "data/processed/eicu"))
    output_path.mkdir(parents=True, exist_ok=True)

    log.info("loading_csvs", path=str(raw_path))
    patient_df = pl.read_csv(raw_path / "patient.csv")
    diagnosis_df = pl.read_csv(raw_path / "diagnosis.csv")
    treatment_df = pl.read_csv(raw_path / "treatment.csv")
    medication_df = pl.read_csv(raw_path / "medication.csv")
    lab_df = pl.read_csv(raw_path / "lab.csv")
    apache_df = pl.read_csv(
        raw_path / "apacheApsVar.csv",
        schema_overrides={col: pl.Float64 for col in APACHEAPSVAR_COLS},
    )

    log.info("step", n=1, total=7, name="patients")
    graphs = process_patients(patient_df)

    log.info("step", n=2, total=7, name="diagnosis")
    attach_diagnosis(diagnosis_df, graphs)

    log.info("step", n=3, total=7, name="treatment")
    attach_treatment(treatment_df, graphs)

    log.info("step", n=4, total=7, name="medication")
    attach_medication(medication_df, graphs)

    log.info("step", n=5, total=7, name="lab")
    lab_df = prepare_labs(lab_df)
    attach_lab(lab_df, graphs)

    log.info("step", n=6, total=7, name="apache")
    attach_apache(apache_df, graphs)

    log.info("step", n=7, total=7, name="post_process")
    normalize_visit_offsets(graphs)
    sort_events(graphs)
    vocabs = build_vocabularies(graphs)
    log_summary(graphs)
    save(graphs, vocabs, output_path)
