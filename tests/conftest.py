"""Shared fixtures: a small synthetic cohort in the workbook layout, and the real data.

The synthetic workbooks hold every column the analysis reads, as text (the way
``load_cohort`` reads them). Their values exercise each code path: the WES
marker, mixed-case codes, decimal commas, "NA" and not-evaluable entries, a
duplicated variant, padded gene names, CNVs found at low tumor cell content and
trailing legend rows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl
import pytest

from ega_mtb.data import (
    ALTERATION_COLUMNS,
    ALTERATIONS_FILE,
    CLINICAL_COLUMNS,
    CLINICAL_FILE,
    DATA_DIR,
    Cohort,
    build_cohort,
    load_cohort,
)

CLINICAL = {
    "ID neu": ["1", "2", "3", "4", "5", "6", "7*", "8*"],
    "Geschlecht": ["M", "W", "M", "M", "W", "M", "M", "W"],
    "Alter bei Eingang": ["45", "70", "58", "66", "39", "81", "52", "62"],
    "Lokalisation": ["AEG", "Magen", "Ösophagus", "AEG", "Magen", "AEG", "Magen", "AEG"],
    "cM (ED)": ["0", "1", "x", "0", "1", "0", "1", "0"],
    "OP?": ["ja", "nein", "unklar", "ja", "nein", "ja", "nein", "ja"],
    "HER2": ["0", "1", "2", "3", "x", "X", "0", "1"],
    "MSI-h": ["MSS", "MSS", "MSS", "MSS", "MSS", "MSS", "MSS", "x"],
    "PD-L1": ["0", "1-4", "5-9", "10+", "x", "0", "10+", "X"],
    "Claudin 18.2": ["0", "1", "x", "0", "1", "x", "x", "0"],
    "Therapielinie zum Zeitpunkt der Diagnose": ["0", "1", "2", "3", "4", "5", "1", "2"],
    "Eingang-Ausgang": ["20", "25", "30", "27", "22", "35", "40", "28"],
    "molTB durchgeführt?": ["ja", "ja", "nein", "ja", "nein", "ja", "ja", "nein"],
    "molTb-Eingang": ["40", "50", "NA", "45", "NA", "60", "70", "NA"],
    "Empfehlung aus Basis TSO500": ["ja", "keine", "NA", "ja", "NA", "keine", "NA", "NA"],
    "Therapieeinleitung aufgrund molTB oder TSO 500 erfolgt?": [
        "nein",
        "nein",
        "ja",
        "nein",
        "nein",
        "nein",
        "nein",
        "nein",
    ],
    "Verstorben?": ["ja", "ja", "nein", "ja", "ja", "nein", "ja", "nein"],
    "d Eingang-Versterbedatum/Zensur": ["100", "200", "300", "150", "50", "400", "250", "0"],
}

# Patient rows 1–8, then two legend rows without a patient number.
ALTERATIONS = {
    "ID_neu": ["1", "2", "3", "4", "5", "6", "7", "8", None, None],
    "TMB verwertbar?": ["ja", "ja", "nein", "ja", "ja", "ja", "ja", "nein", None, None],
    "TMB": ["5,5", "10,2", "n.n. <40% TZG", "3.1", "7", "1,6", "12.0", None, None, None],
    "MSI Score verwertbar": ["ja", "nein", "ja", "ja", "nein", "ja", "ja", "ja", None, None],
    "MSI-Score": ["2,04", "n.n.<40% TZG", "0", "4.8", None, "1,5", "3,2", "0,8", None, None],
    "Fusion": [
        "keine",
        "TMPRSS2(Exon 7)::RIPK4(Exon8)",
        "n.n.",
        "keine",
        "keine Analyse",
        "keine",
        "keine",
        "keine",
        None,
        None,
    ],
    "Alteration 1": ["TP53", "TP53", None, "TP53", "APC", "TP53", None, "PIK3CA", None, None],
    "Mutation Type 1": [
        "Missense-Mutation",
        "Nonsense-Mutation",
        None,
        "Splice-Site-Mutation",
        "Frameshift-Mutation",
        "Missense-Mutation",
        None,
        "missense-mutation",
        None,
        None,
    ],
    "Alteration 2": [" KRAS", "ARID1A ", None, "CDKN2A", None, "TP53", None, None, None, None],
    "Mutation Type 2": [
        "Missense-Mutation",
        "Frameshift-Mutation",
        None,
        "Deletion",
        None,
        "Missense-Mutation",
        None,
        None,
        None,
        None,
    ],
    "CNV Gain": ["1", "0", "2", "3", "0", "1", "0", "0", "0=nein", "1=ja"],
    "CNV Gain 1": ["ERBB2", None, None, "KRAS", None, "MYC", None, None, None, None],
    "CNV Gain 2": [None, None, None, None, None, "CCNE1", None, None, None, None],
    "CNV Loss": ["0", "0", "2", "1", "0", "0", "0", "0", "0=nein", "1=ja"],
    "CNV Loss 1": [None, None, None, "CDKN2A", None, None, None, None, None, None],
    "CNV Loss 2": [None, None, None, "CDKN2B", None, None, None, None, None, None],
}


def _frame(values: Mapping[str, Sequence[str | None]], columns: tuple[str, ...]) -> pl.DataFrame:
    """All ``columns`` as text; those not given in ``values`` are empty."""
    height = len(next(iter(values.values())))
    return pl.DataFrame(
        {c: list(values.get(c, [None] * height)) for c in columns},
        schema=dict.fromkeys(columns, pl.String),
    )


@pytest.fixture
def clinical_raw() -> pl.DataFrame:
    return _frame(CLINICAL, CLINICAL_COLUMNS)


@pytest.fixture
def alterations_raw() -> pl.DataFrame:
    return _frame(ALTERATIONS, ALTERATION_COLUMNS)


@pytest.fixture
def cohort(clinical_raw: pl.DataFrame, alterations_raw: pl.DataFrame) -> Cohort:
    return build_cohort(clinical_raw, alterations_raw)


@pytest.fixture(scope="session")
def study() -> Cohort:
    """The real study cohort; tests using it are skipped without the data."""
    if not (DATA_DIR / CLINICAL_FILE).exists() or not (DATA_DIR / ALTERATIONS_FILE).exists():
        pytest.skip(f"study data not found in {DATA_DIR}")
    return load_cohort()
