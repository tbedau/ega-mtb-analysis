"""Workbook parsing and validation (synthetic data)."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from polars.exceptions import InvalidOperationError

from ega_mtb.data import (
    ALTERATIONS_FILE,
    CLINICAL_COLUMNS,
    CLINICAL_FILE,
    build_cohort,
    load_cohort,
)


def _with(df: pl.DataFrame, column: str, row: int, value: str | None) -> pl.DataFrame:
    """Copy of ``df`` with one cell replaced."""
    values = df[column].to_list()
    values[row] = value
    return df.with_columns(pl.Series(column, values, dtype=pl.String))


def test_patients_are_tidied(cohort):
    p = cohort.patients
    assert p["patient_id"].to_list() == ["1", "2", "3", "4", "5", "6", "7", "8"]
    assert p["wes"].to_list() == [False] * 6 + [True] * 2
    assert p["sex"].to_list()[:2] == ["Male", "Female"]
    assert p["location"].to_list()[:3] == ["Esophagogastric junction", "Stomach", "Esophagus"]
    assert p["her2"].to_list() == ["0", "1+", "2+", "3+", "Unknown", "Unknown", "0", "1+"]
    assert p["pdl1"].to_list() == ["0", "1–4", "5–9", "≥10", "Unknown", "0", "≥10", "Unknown"]
    assert p["event"].to_list() == [1, 1, 0, 1, 1, 0, 1, 0]
    assert p["os_days"].to_list() == [100, 200, 300, 150, 50, 400, 250, 0]
    assert p["days_to_mtb"].to_list() == [40, 50, None, 45, None, 60, 70, None]
    # "NA" (no MTB decision on record) counts as no recommendation.
    assert p["mtb_recommendation"].to_list() == [True, False, False, True] + [False] * 4


def test_markers_parse_decimal_commas_and_evaluability(cohort):
    p = cohort.patients
    assert p["tmb"].to_list() == [5.5, 10.2, None, 3.1, 7.0, 1.6, 12.0, None]
    assert p["msi_score"].to_list() == [2.04, None, 0.0, 4.8, None, 1.5, 3.2, 0.8]
    assert p["fusion"].to_list() == [None, "TMPRSS2(Exon 7)::RIPK4(Exon8)"] + [None] * 6


def test_alteration_events(cohort):
    events = {tuple(row) for row in cohort.alterations.iter_rows()}
    assert events == {
        ("1", "TP53", "Missense"),
        ("1", "KRAS", "Missense"),
        ("1", "ERBB2", "CNV Gain"),
        ("2", "TP53", "Nonsense"),
        ("2", "ARID1A", "Frameshift"),
        ("4", "TP53", "Splice-Site"),
        ("4", "CDKN2A", "Small deletion"),
        ("4", "KRAS", "CNV Gain"),  # found at low tumor cell content (status 3)
        ("4", "CDKN2A", "CNV Loss"),
        ("4", "CDKN2B", "CNV Loss"),
        ("5", "APC", "Frameshift"),
        ("6", "TP53", "Missense"),  # listed twice, counted once
        ("6", "MYC", "CNV Gain"),
        ("6", "CCNE1", "CNV Gain"),
        ("8", "PIK3CA", "Missense"),
    }
    assert cohort.alterations.height == len(events)


def test_only_analysis_columns_are_kept(cohort):
    assert set(cohort.patients.columns) == {
        "patient_id",
        "wes",
        "sex",
        "age",
        "location",
        "m_status",
        "prior_resection",
        "her2",
        "msi_status",
        "pdl1",
        "claudin",
        "therapy_line",
        "days_to_report",
        "mtb_presented",
        "days_to_mtb",
        "mtb_recommendation",
        "therapy_started",
        "event",
        "os_days",
        "tmb",
        "msi_score",
        "fusion",
    }


def test_load_cohort_reads_only_allow_listed_columns(tmp_path: Path, clinical_raw, alterations_raw):
    identifiers = {"Name": ["Doe"] * 8, "Geburtsdatum": ["01.01.1950"] * 8, "PID": ["123"] * 8}
    clinical_raw.with_columns(**{k: pl.Series(v) for k, v in identifiers.items()}).write_excel(
        tmp_path / CLINICAL_FILE
    )
    alterations_raw.write_excel(tmp_path / ALTERATIONS_FILE)
    loaded = load_cohort(tmp_path)
    assert loaded.patients.equals(build_cohort(clinical_raw, alterations_raw).patients)
    assert not set(identifiers) & set(CLINICAL_COLUMNS)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("Geschlecht", "U"),
        ("HER2", "4"),
        ("Verstorben?", "unklar"),
        ("Alter bei Eingang", "sixty"),
        ("d Eingang-Versterbedatum/Zensur", "NA"),
    ],
)
def test_unexpected_clinical_values_raise(clinical_raw, alterations_raw, column, value):
    with pytest.raises(InvalidOperationError):
        build_cohort(_with(clinical_raw, column, 0, value), alterations_raw)


def test_missing_clinical_value_raises(clinical_raw, alterations_raw):
    with pytest.raises(ValueError, match="missing values"):
        build_cohort(_with(clinical_raw, "Lokalisation", 0, None), alterations_raw)


def test_recommendation_requires_mtb_presentation(clinical_raw, alterations_raw):
    with pytest.raises(ValueError, match="not presented"):
        build_cohort(_with(clinical_raw, "Empfehlung aus Basis TSO500", 2, "ja"), alterations_raw)


def test_days_to_mtb_required_for_presented_patients(clinical_raw, alterations_raw):
    with pytest.raises(ValueError, match="days to MTB"):
        build_cohort(_with(clinical_raw, "molTb-Eingang", 0, "NA"), alterations_raw)


def test_workbooks_must_list_the_same_patients(clinical_raw, alterations_raw):
    with pytest.raises(ValueError, match="different patients"):
        build_cohort(clinical_raw, _with(alterations_raw, "ID_neu", 7, "9"))


def test_evaluable_measurement_must_parse(clinical_raw, alterations_raw):
    with pytest.raises(InvalidOperationError):
        build_cohort(clinical_raw, _with(alterations_raw, "TMB", 0, "n.n."))


def test_variant_needs_gene_and_type(clinical_raw, alterations_raw):
    with pytest.raises(ValueError, match="gene and a mutation type"):
        build_cohort(clinical_raw, _with(alterations_raw, "Mutation Type 1", 0, None))


def test_unknown_variant_type_raises(clinical_raw, alterations_raw):
    with pytest.raises(InvalidOperationError):
        build_cohort(clinical_raw, _with(alterations_raw, "Mutation Type 1", 0, "Inversion"))


def test_cnv_genes_require_a_found_status(clinical_raw, alterations_raw):
    with pytest.raises(ValueError, match="CNV Gain genes"):
        build_cohort(clinical_raw, _with(alterations_raw, "CNV Gain", 0, "2"))
