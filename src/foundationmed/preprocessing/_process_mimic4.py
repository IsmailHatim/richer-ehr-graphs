"""MIMIC-IV preprocessing pipeline: raw CSVs → processed pickle files."""

from __future__ import annotations

import math
import tomllib
from pathlib import Path

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

# MIMIC-IV constants

# Keep the N most frequently measured lab items (99.1% of all lab rows)
N_TOP_LABS = 150

# 1. Patients


def process_patients(
    patients_df: pl.DataFrame,
    admissions_df: pl.DataFrame,  # admittime/dischtime already parsed as Datetime
) -> dict[str, PatientGraph]:
    """Build PatientGraph objects from patients.csv + admissions.csv."""
    patients_df = patients_df.with_columns(
        pl.col("dod").str.to_datetime(format="%Y-%m-%d", strict=False)
    )
    merged = admissions_df.join(patients_df, on="subject_id", how="inner")
    merged = merged.with_columns(
        [
            (
                pl.col("admittime").dt.year()
                - pl.col("anchor_year")
                + pl.col("anchor_age")
            ).alias("admit_age"),
            ((pl.col("dischtime") - pl.col("admittime")).dt.total_seconds() / 60).alias(
                "duration_min"
            ),
        ]
    )
    merged = merged.sort(["subject_id", "admittime"])
    # Minutes from the patient's first admission
    merged = merged.with_columns(
        (
            (
                pl.col("admittime") - pl.col("admittime").min().over("subject_id")
            ).dt.total_seconds()
            / 60
        ).alias("_start_offset")
    )

    # Readmission: next admission within 15 days of discharge
    merged = merged.with_columns(
        pl.col("admittime").shift(-1).over("subject_id").alias("_next_admittime")
    )
    merged = merged.with_columns(
        [
            pl.when(pl.col("_next_admittime").is_not_null())
            .then(
                (pl.col("_next_admittime") - pl.col("dischtime")).dt.total_seconds()
                / 86400
                <= 15
            )
            .otherwise(
                None
            )  # In MIMIC "None" is no info about next admission (approx. 40%)
            .alias("readmission"),
            # 90-day post-discharge mortality
            pl.when(pl.col("dod").is_not_null())
            .then(
                (pl.col("dod") - pl.col("dischtime")).dt.total_seconds() / 86400 <= 90
                # 86400 = 3600x24
            )
            .otherwise(False)
            .alias("mortality_90d"),
        ]
    )

    graphs: dict[str, PatientGraph] = {}
    excl_duration = excl_age = excl_inhospital = 0

    for row in merged.iter_rows(named=True):
        duration = row["duration_min"]
        if duration is None or duration < 12 * 60 or duration > 10 * 24 * 60:
            # exclude if less than 12 hours or more than 10 days
            excl_duration += 1
            continue
        age = row["admit_age"]
        if age is None or age < 18 or age > 89:
            excl_age += 1
            continue
        if row["hospital_expire_flag"] == 1:
            excl_inhospital += 1
            continue

        pid = str(row["subject_id"])
        hadm_id = str(row["hadm_id"])
        visit = Visit(
            icu_id=hadm_id,
            admission_id=pid,
            hospital_id="",
            icu_duration=int(duration),
            age=int(age),
            gender=str(row["gender"]),
            ethnicity=str(row["race"]),
            mortality=bool(row["mortality_90d"]),
            readmission=(
                None if row["readmission"] is None else bool(row["readmission"])
            ),
            start_offset=int(row["_start_offset"]),
        )
        if pid not in graphs:
            graphs[pid] = PatientGraph(patient_id=pid)
        graphs[pid].visits.append(visit)
        # visits already chronological since merged is sorted by admittime

    log.info(
        "patients_excluded",
        duration=excl_duration,
        age=excl_age,
        in_hospital_death=excl_inhospital,
    )
    log.info(
        "patients_included",
        admissions=sum(g.n_visits for g in graphs.values()),
        patients=len(graphs),
    )
    return graphs


# 2. Diagnoses ICD


def attach_diagnoses_icd(
    diagnoses_df: pl.DataFrame,
    icd_dict_df: pl.DataFrame,
    graphs: dict[str, PatientGraph],
) -> None:
    """Discharge diagnoses carry no timestamp — placed at t=0."""
    visit_lk = _visit_direct_lookup(graphs)
    df = diagnoses_df.join(
        icd_dict_df.select(["icd_code", "icd_version", "long_title"]),
        on=["icd_code", "icd_version"],
        how="inner",
    ).filter(pl.col("long_title").is_not_null())
    attached = skipped = 0
    for row in tqdm(
        df.iter_rows(named=True), total=df.height, desc="diagnoses_icd", leave=False
    ):
        hadm_id = str(row["hadm_id"])
        if hadm_id not in visit_lk:
            skipped += 1
            continue
        visit_lk[hadm_id].events.append(
            Event(0, "diagnosis", row["long_title"].lower())
        )
        attached += 1
    log.info("diagnoses_icd_attached", attached=attached, skipped=skipped)


# 3. Procedures ICD


def attach_procedures_icd(
    procedures_df: pl.DataFrame,
    icd_dict_df: pl.DataFrame,
    admissions_df: pl.DataFrame,
    graphs: dict[str, PatientGraph],
) -> None:
    visit_lk = _visit_direct_lookup(graphs)
    df = (
        procedures_df.join(
            icd_dict_df.select(["icd_code", "icd_version", "long_title"]),
            on=["icd_code", "icd_version"],
            how="inner",
        )
        .filter(pl.col("long_title").is_not_null())
        .join(admissions_df.select(["hadm_id", "admittime"]), on="hadm_id", how="inner")
        .with_columns(
            pl.col("chartdate").str.to_datetime(format="%Y-%m-%d", strict=False)
        )
        .filter(pl.col("chartdate").is_not_null())
        .with_columns(
            ((pl.col("chartdate") - pl.col("admittime")).dt.total_seconds() / 60)
            .cast(pl.Int64)
            .alias("timestamp")
        )
    )
    attached = skipped = 0
    for row in tqdm(
        df.iter_rows(named=True), total=df.height, desc="procedures_icd", leave=False
    ):
        hadm_id = str(row["hadm_id"])
        if hadm_id not in visit_lk:
            skipped += 1
            continue
        ts = max(0, row["timestamp"] or 0)
        visit = visit_lk[hadm_id]
        if ts > visit.icu_duration:
            skipped += 1
            continue
        visit.events.append(Event(ts, "treatment", row["long_title"].lower()))
        attached += 1
    log.info("procedures_icd_attached", attached=attached, skipped=skipped)


# 4. Prescriptions


def attach_prescriptions(
    prescriptions_df: pl.DataFrame,
    admissions_df: pl.DataFrame,
    graphs: dict[str, PatientGraph],
) -> None:
    visit_lk = _visit_direct_lookup(graphs)
    df = (
        prescriptions_df.filter(
            pl.col("hadm_id").is_not_null() & pl.col("drug").is_not_null()
        )
        .join(admissions_df.select(["hadm_id", "admittime"]), on="hadm_id", how="inner")
        .with_columns(
            pl.col("starttime").str.to_datetime(
                format="%Y-%m-%d %H:%M:%S", strict=False
            )
        )
        .filter(pl.col("starttime").is_not_null())
        .with_columns(
            ((pl.col("starttime") - pl.col("admittime")).dt.total_seconds() / 60)
            .cast(pl.Int64)
            .alias("timestamp")
        )
    )
    attached = skipped = 0
    for row in tqdm(
        df.iter_rows(named=True), total=df.height, desc="prescriptions", leave=False
    ):
        hadm_id = str(int(row["hadm_id"]))
        if hadm_id not in visit_lk:
            skipped += 1
            continue
        ts = max(0, row["timestamp"] or 0)
        visit = visit_lk[hadm_id]
        if ts > visit.icu_duration:
            skipped += 1
            continue
        visit.events.append(Event(ts, "medication", row["drug"].lower()))
        attached += 1
    log.info("prescriptions_attached", attached=attached, skipped=skipped)


# 5. Lab events


def _lab_codes(selected: pl.DataFrame) -> pl.DataFrame:
    """Name each selected itemid, disambiguating by fluid only where needed."""
    n_fluids = selected.group_by("label").agg(
        pl.col("fluid").n_unique().alias("_n_fluids")
    )
    return (
        selected.join(n_fluids, on="label")
        .with_columns(
            pl.when(pl.col("_n_fluids") > 1)
            .then(pl.col("label") + pl.lit(", ") + pl.col("fluid"))
            .otherwise(pl.col("label"))
            .str.to_lowercase()
            .alias("code")
        )
        .select(["itemid", "code"])
    )


def prepare_labs(raw_path: Path, d_labitems_df: pl.DataFrame) -> pl.DataFrame:
    """Lazy-scan labevents.csv and keep the `N_TOP_LABS` most frequent items."""
    labevents = (
        pl.scan_csv(
            raw_path / "labevents.csv",
            schema_overrides={"hadm_id": pl.Float64, "valuenum": pl.Float32},
            infer_schema_length=10_000,
        )
        .filter(pl.col("hadm_id").is_not_null() & pl.col("valuenum").is_not_null())
        .select(["subject_id", "hadm_id", "storetime", "itemid", "valuenum"])
        .collect()
    )
    log.info("labs_collected", rows=len(labevents))

    freq = labevents.group_by("itemid").len().sort("len", descending=True)
    top = freq.head(N_TOP_LABS)
    kept = int(top["len"].sum())
    log.info(
        "labs_selected",
        item_ids=top.height,
        of_total=freq.height,
        row_coverage_pct=round(100 * kept / max(labevents.height, 1), 2),
    )

    codes = _lab_codes(
        d_labitems_df.filter(pl.col("itemid").is_in(top["itemid"])).select(
            ["itemid", "label", "fluid"]
        )
    )
    out = labevents.join(codes, on="itemid", how="inner")
    log.info("lab_codes_built", codes=out["code"].n_unique())
    return out


def attach_labevents(
    labevents_df: pl.DataFrame,
    admissions_df: pl.DataFrame,
    graphs: dict[str, PatientGraph],
) -> None:
    visit_lk = _visit_direct_lookup(graphs)
    df = (
        labevents_df.with_columns(pl.col("hadm_id").cast(pl.Int64, strict=False))
        .filter(pl.col("hadm_id").is_not_null())
        .join(admissions_df.select(["hadm_id", "admittime"]), on="hadm_id", how="inner")
        .with_columns(
            pl.col("storetime").str.to_datetime(
                format="%Y-%m-%d %H:%M:%S", strict=False
            )
        )
        .filter(pl.col("storetime").is_not_null())
        .with_columns(
            ((pl.col("storetime") - pl.col("admittime")).dt.total_seconds() / 60)
            .cast(pl.Int64)
            .alias("timestamp")
        )
    )
    attached = skipped = 0
    for row in tqdm(
        df.iter_rows(named=True), total=df.height, desc="labevents", leave=False
    ):
        hadm_id = str(row["hadm_id"])
        if hadm_id not in visit_lk:
            skipped += 1
            continue
        ts = row["timestamp"] or 0
        val = row["valuenum"]
        if val is None or (isinstance(val, float) and math.isnan(val)):
            skipped += 1
            continue
        visit = visit_lk[hadm_id]
        if ts > visit.icu_duration:
            skipped += 1
            continue
        visit.events.append(Event(max(0, ts), "lab", row["code"], float(val)))
        attached += 1
    log.info("labevents_attached", attached=attached, skipped=skipped)


# Pipeline entry point


def run_pipeline(config_path: Path) -> None:
    """Process raw MIMIC-IV CSVs into patient graphs and vocabularies."""
    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    raw_path = Path(config["mimic4"]["path"])
    output_path = Path(config["mimic4"].get("processed_path", "data/processed/mimic4"))
    output_path.mkdir(parents=True, exist_ok=True)

    log.info("loading_csvs", path=str(raw_path))
    patients_df = pl.read_csv(raw_path / "patients.csv")
    admissions_df = pl.read_csv(raw_path / "admissions.csv").with_columns(
        [
            pl.col("admittime").str.to_datetime(
                format="%Y-%m-%d %H:%M:%S", strict=False
            ),
            pl.col("dischtime").str.to_datetime(
                format="%Y-%m-%d %H:%M:%S", strict=False
            ),
        ]
    )
    _icd_schema = {"icd_code": pl.Utf8, "icd_version": pl.Utf8}
    diagnoses_df = pl.read_csv(
        raw_path / "diagnoses_icd.csv", schema_overrides=_icd_schema
    )
    icd_diag_df = pl.read_csv(
        raw_path / "d_icd_diagnoses.csv", schema_overrides=_icd_schema
    )
    procedures_df = pl.read_csv(
        raw_path / "procedures_icd.csv", schema_overrides=_icd_schema
    )
    icd_proc_df = pl.read_csv(
        raw_path / "d_icd_procedures.csv", schema_overrides=_icd_schema
    )
    prescriptions_df = pl.read_csv(
        raw_path / "prescriptions.csv",
        columns=["subject_id", "hadm_id", "starttime", "drug"],
    )
    d_labitems_df = pl.read_csv(raw_path / "d_labitems.csv")

    log.info("step", n=1, total=6, name="patients")
    graphs = process_patients(patients_df, admissions_df)

    log.info("step", n=2, total=6, name="diagnoses_icd")
    attach_diagnoses_icd(diagnoses_df, icd_diag_df, graphs)

    log.info("step", n=3, total=6, name="procedures_icd")
    attach_procedures_icd(procedures_df, icd_proc_df, admissions_df, graphs)

    log.info("step", n=4, total=6, name="prescriptions")
    attach_prescriptions(prescriptions_df, admissions_df, graphs)

    log.info("step", n=5, total=6, name="labevents")
    labevents_df = prepare_labs(raw_path, d_labitems_df)
    attach_labevents(labevents_df, admissions_df, graphs)

    log.info("step", n=6, total=6, name="post_process")
    normalize_visit_offsets(graphs)
    sort_events(graphs)
    vocabs = build_vocabularies(graphs)
    log_summary(graphs)
    save(graphs, vocabs, output_path)
