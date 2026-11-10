"""Loading and validation of the two pseudonymised study workbooks.

The analysis reads two Excel workbooks that are not distributed with this
repository (see README):

* ``clinical.xlsx`` — one row per patient: demographics, standard biomarkers,
  molecular tumor board (MTB) pathway and survival.
* ``alterations.xlsx`` — one row per patient: functionally relevant small
  variants, copy-number variants (CNVs), TMB, MSI score and gene fusions.

Only the columns listed in this module are read (an explicit allow-list), so
quasi-identifiers in the source files never enter the analysis. Every coded
value is mapped strictly: an unexpected code raises an error instead of being
silently miscounted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
#: Directory holding the two workbooks; override with ``EGA_MTB_DATA_DIR``.
DATA_DIR = Path(os.environ.get("EGA_MTB_DATA_DIR", PROJECT_ROOT / "data"))
CLINICAL_FILE = "clinical.xlsx"
ALTERATIONS_FILE = "alterations.xlsx"

#: Average month length; survival times are recorded in days, reported in months.
DAYS_PER_MONTH = 365.25 / 12

# --------------------------------------------------------------------------- #
# Clinical workbook
# --------------------------------------------------------------------------- #

#: Clinical workbook columns read by the analysis.
CLINICAL_COLUMNS = (
    "ID neu",  # pseudonymised patient number; a trailing "*" marks WES patients
    "Geschlecht",  # sex
    "Alter bei Eingang",  # age at molecular testing (sample receipt)
    "Lokalisation",  # tumor localization
    "cM (ED)",  # M-status at primary diagnosis
    "OP?",  # prior surgical resection with curative intent
    "HER2",  # HER2 IHC score
    "MSI-h",  # MMR/MSI status from the electronic health record
    "PD-L1",  # PD-L1 combined positivity score
    "Claudin 18.2",
    "Therapielinie zum Zeitpunkt der Diagnose",  # line of therapy at testing
    "Eingang-Ausgang",  # days from sample receipt to molecular report
    "molTB durchgeführt?",  # presented at the MTB
    "molTb-Eingang",  # days from sample receipt to MTB presentation
    "Empfehlung aus Basis TSO500",  # MTB recommendation based on TSO500/WES
    "Therapieeinleitung aufgrund molTB oder TSO 500 erfolgt?",  # targeted therapy started
    "Verstorben?",  # deceased
    "d Eingang-Versterbedatum/Zensur",  # days from sample receipt to death/censoring
)

# Codes used in the clinical workbook (matched case-insensitively) -> labels.
SEX = {"m": "Male", "w": "Female"}
LOCATION = {"aeg": "Esophagogastric junction", "magen": "Stomach", "ösophagus": "Esophagus"}
M_STATUS = {"0": "M0", "1": "M1", "x": "Unknown"}
PRIOR_RESECTION = {"ja": "Yes", "nein": "No", "unklar": "Unknown"}
HER2 = {"0": "0", "1": "1+", "2": "2+", "3": "3+", "x": "Unknown"}
MSI_STATUS = {"mss": "pMMR/MSS", "x": "Unknown"}
PDL1 = {"0": "0", "1-4": "1–4", "5-9": "5–9", "10+": "≥10", "x": "Unknown"}
CLAUDIN = {"0": "Negative", "1": "Positive", "x": "Unknown"}
YES_NO = {"ja": True, "nein": False}
# "keine" = presented without a recommendation; "NA" = no MTB decision on record.
RECOMMENDATION = {"ja": True, "keine": False, "na": False}

# --------------------------------------------------------------------------- #
# Alterations workbook
# --------------------------------------------------------------------------- #

SMALL_VARIANT_SLOTS = 7  # "Alteration i" / "Mutation Type i" column pairs
CNV_GAIN_SLOTS = 6  # "CNV Gain 1" … "CNV Gain 6"
CNV_LOSS_SLOTS = 3  # "CNV Loss 1" … "CNV Loss 3"

#: Alterations workbook columns read by the analysis.
ALTERATION_COLUMNS = (
    "ID_neu",
    "TMB verwertbar?",  # TMB evaluable (ja/nein)
    "TMB",  # mut/Mb, decimal comma
    "MSI Score verwertbar",  # MSI score evaluable (ja/nein)
    "MSI-Score",  # % unstable microsatellite sites, decimal comma
    "Fusion",  # "GENE1(exon)::GENE2(exon)", or a note that none was found
    *(f"Alteration {i}" for i in range(1, SMALL_VARIANT_SLOTS + 1)),
    *(f"Mutation Type {i}" for i in range(1, SMALL_VARIANT_SLOTS + 1)),
    "CNV Gain",  # status code, see CNV_FOUND
    *(f"CNV Gain {i}" for i in range(1, CNV_GAIN_SLOTS + 1)),
    "CNV Loss",  # status code, see CNV_FOUND
    *(f"CNV Loss {i}" for i in range(1, CNV_LOSS_SLOTS + 1)),
)

#: Small-variant types (workbook code -> label used in Figure 2).
SMALL_VARIANT_TYPES = {
    "missense-mutation": "Missense",
    "nonsense-mutation": "Nonsense",
    "frameshift-mutation": "Frameshift",
    "deletion": "Small deletion",
    "splice-site-mutation": "Splice-Site",
}
CNV_GAIN = "CNV Gain"
CNV_LOSS = "CNV Loss"
#: All alteration types, in legend order.
ALTERATION_TYPES = (*SMALL_VARIANT_TYPES.values(), CNV_GAIN, CNV_LOSS)

#: CNV status codes: 0 = none found; 1 = found; 2 = none found, but tumor cell
#: content <40% (not evaluable); 3 = found despite tumor cell content <40%.
CNV_FOUND = {"0": False, "1": True, "2": False, "3": True}


@dataclass(frozen=True)
class Cohort:
    """The study cohort.

    ``patients`` has one row per patient (clinical data plus TMB, MSI score and
    fusion status); ``alterations`` has one row per (patient, gene, alteration
    type) and lists functionally relevant alterations only.
    """

    patients: pl.DataFrame
    alterations: pl.DataFrame


def load_cohort(data_dir: Path = DATA_DIR) -> Cohort:
    """Read and validate both workbooks from ``data_dir``."""
    clinical = pl.read_excel(
        data_dir / CLINICAL_FILE, columns=list(CLINICAL_COLUMNS), infer_schema_length=0
    )
    alterations = pl.read_excel(
        data_dir / ALTERATIONS_FILE, columns=list(ALTERATION_COLUMNS), infer_schema_length=0
    )
    return build_cohort(clinical, alterations)


def build_cohort(clinical: pl.DataFrame, alterations: pl.DataFrame) -> Cohort:
    """Tidy and cross-validate the raw workbook frames (all columns as text)."""
    # Trailing legend rows in the alterations workbook carry no patient number.
    alterations = alterations.filter(pl.col("ID_neu").is_not_null())
    patients = tidy_clinical(clinical)
    markers = tidy_markers(alterations)
    if set(patients["patient_id"]) != set(markers["patient_id"]):
        raise ValueError("the two workbooks list different patients")
    return Cohort(
        patients=patients.join(markers, on="patient_id", how="left", validate="1:1"),
        alterations=alteration_events(alterations),
    )


def tidy_clinical(raw: pl.DataFrame) -> pl.DataFrame:
    """Clinical workbook -> one validated row per patient."""
    patients = raw.select(
        patient_id=_text("ID neu").str.strip_chars_end("*"),
        wes=_text("ID neu").str.ends_with("*"),
        sex=_coded("Geschlecht", SEX),
        age=_number("Alter bei Eingang"),
        location=_coded("Lokalisation", LOCATION),
        m_status=_coded("cM (ED)", M_STATUS),
        prior_resection=_coded("OP?", PRIOR_RESECTION),
        her2=_coded("HER2", HER2),
        msi_status=_coded("MSI-h", MSI_STATUS),
        pdl1=_coded("PD-L1", PDL1),
        claudin=_coded("Claudin 18.2", CLAUDIN),
        therapy_line=_number("Therapielinie zum Zeitpunkt der Diagnose"),
        days_to_report=_number("Eingang-Ausgang"),
        mtb_presented=_coded("molTB durchgeführt?", YES_NO, pl.Boolean),
        days_to_mtb=_number("molTb-Eingang", not_applicable=True),
        mtb_recommendation=_coded("Empfehlung aus Basis TSO500", RECOMMENDATION, pl.Boolean),
        therapy_started=_coded(
            "Therapieeinleitung aufgrund molTB oder TSO 500 erfolgt?", YES_NO, pl.Boolean
        ),
        event=_coded("Verstorben?", YES_NO, pl.Boolean).cast(pl.Int64),
        os_days=_number("d Eingang-Versterbedatum/Zensur"),
    )
    _require_complete(patients.drop("days_to_mtb"))
    if patients["patient_id"].is_duplicated().any():
        raise ValueError("duplicate patient numbers in the clinical workbook")
    if (patients["mtb_recommendation"] & ~patients["mtb_presented"]).any():
        raise ValueError("MTB recommendation recorded for a patient not presented at the MTB")
    if (patients["mtb_presented"] != patients["days_to_mtb"].is_not_null()).any():
        raise ValueError("days to MTB presentation must be given exactly for presented patients")
    return patients


def tidy_markers(raw: pl.DataFrame) -> pl.DataFrame:
    """Alterations workbook -> one row per patient with TMB, MSI score and fusion."""
    markers = raw.select(
        patient_id=_text("ID_neu"),
        tmb_evaluable=_coded("TMB verwertbar?", YES_NO, pl.Boolean),
        msi_evaluable=_coded("MSI Score verwertbar", YES_NO, pl.Boolean),
        tmb=_measurement("TMB", "TMB verwertbar?"),
        msi_score=_measurement("MSI-Score", "MSI Score verwertbar"),
        fusion=pl.when(_text("Fusion").str.contains("::", literal=True)).then(_text("Fusion")),
    )
    _require_complete(markers.select("patient_id", "tmb_evaluable", "msi_evaluable"))
    if markers["patient_id"].is_duplicated().any():
        raise ValueError("duplicate patient numbers in the alterations workbook")
    return markers.drop("tmb_evaluable", "msi_evaluable")


def alteration_events(raw: pl.DataFrame) -> pl.DataFrame:
    """Alterations workbook -> one row per (patient, gene, alteration type)."""
    small = pl.concat(
        raw.select(
            patient_id=_text("ID_neu"),
            gene=_text(f"Alteration {i}"),
            code=_text(f"Mutation Type {i}"),
        )
        for i in range(1, SMALL_VARIANT_SLOTS + 1)
    )
    if (small["gene"].is_null() != small["code"].is_null()).any():
        raise ValueError("every small variant needs both a gene and a mutation type")
    small = small.drop_nulls("gene").select(
        "patient_id",
        "gene",
        alteration=pl.col("code").str.to_lowercase().replace_strict(SMALL_VARIANT_TYPES),
    )
    events = pl.concat(
        [
            small,
            _cnv_events(raw, CNV_GAIN, CNV_GAIN_SLOTS),
            _cnv_events(raw, CNV_LOSS, CNV_LOSS_SLOTS),
        ]
    )
    return events.unique().sort("patient_id", "gene", "alteration")


def _cnv_events(raw: pl.DataFrame, kind: str, slots: int) -> pl.DataFrame:
    """Genes listed under a CNV gain or loss; the status code must say 'found'."""
    listed = pl.concat(
        raw.select(
            patient_id=_text("ID_neu"),
            found=_coded(kind, CNV_FOUND, pl.Boolean),
            gene=_text(f"{kind} {i}"),
        )
        for i in range(1, slots + 1)
    )
    _require_complete(listed.select("found"))
    listed = listed.drop_nulls("gene")
    if not listed["found"].all():
        raise ValueError(f"{kind} genes listed for a patient whose status is not 'found'")
    return listed.select("patient_id", "gene", alteration=pl.lit(kind))


# --------------------------------------------------------------------------- #
# Parsing helpers (all workbook columns are read as text)
# --------------------------------------------------------------------------- #


def _text(column: str) -> pl.Expr:
    """Stripped text; empty cells become null."""
    value = pl.col(column).str.strip_chars()
    return pl.when(value != "").then(value)


def _coded(column: str, codes: dict, dtype: pl.DataType | type[pl.DataType] = pl.String) -> pl.Expr:
    """Map a coded column (case-insensitive); an unknown code raises."""
    return _text(column).str.to_lowercase().replace_strict(codes, return_dtype=dtype)


def _number(column: str, *, not_applicable: bool = False) -> pl.Expr:
    """Parse an integer column; a non-numeric value raises.

    With ``not_applicable``, the code ``NA`` is accepted and read as missing.
    """
    value = _text(column)
    if not_applicable:
        value = pl.when(value.str.to_lowercase() != "na").then(value)
    return value.cast(pl.Int64)


def _measurement(column: str, evaluable_flag: str) -> pl.Expr:
    """Parse a decimal-comma value where the evaluability flag says 'ja', else null.

    A value flagged as evaluable that cannot be parsed raises.
    """
    evaluable = _coded(evaluable_flag, YES_NO, pl.Boolean)
    value = _text(column).str.replace(",", ".", literal=True)
    return pl.when(evaluable).then(value).cast(pl.Float64)


def _require_complete(df: pl.DataFrame) -> None:
    missing = [name for name, n in zip(df.columns, df.null_count().row(0), strict=True) if n]
    if missing:
        raise ValueError(f"missing values in required columns: {missing}")
