"""Table 1 — patient characteristics."""

from __future__ import annotations

import polars as pl

from ega_mtb.stats import median_iqr

#: Categorical sections of Table 1: column -> (heading, [(value, row label)]), in
#: table order. The listed values of a section must account for every patient.
CATEGORICAL_SECTIONS = {
    "sex": (
        "Sex",
        [("Male", "Male"), ("Female", "Female")],
    ),
    "location": (
        "Tumor localization",
        [
            ("Esophagogastric junction", "Esophagogastric junction (EGJ)"),
            ("Stomach", "Stomach"),
            ("Esophagus", "Esophagus"),
        ],
    ),
    "m_status": (
        "M-status at primary diagnosis",
        [("M0", "M0"), ("M1", "M1"), ("Unknown", "Unknown")],
    ),
    "prior_resection": (
        "Prior surgical resection with curative intent",
        [("Yes", "Yes"), ("No", "No"), ("Unknown", "Unknown")],
    ),
    "therapy_line": (
        "Line of therapy at molecular testing",
        [
            (0, "Curative setting"),
            (1, "1st line"),
            (2, "2nd line"),
            (3, "3rd line"),
            (4, "4th line"),
            (5, "5th line"),
        ],
    ),
    "her2": (
        "HER2 IHC score",
        [(v, v) for v in ("0", "1+", "2+", "3+", "Unknown")],
    ),
    "pdl1": (
        "PD-L1 CPS",
        [(v, v) for v in ("0", "1–4", "5–9", "≥10", "Unknown")],
    ),
    "claudin": (
        "Claudin 18.2",
        [(v, v) for v in ("Positive", "Negative", "Unknown")],
    ),
    "msi_status": (
        "Microsatellite status",
        [("pMMR/MSS", "pMMR/MSS"), ("MSI-high", "MSI-high"), ("Unknown", "Unknown")],
    ),
}


def table1(patients: pl.DataFrame) -> pl.DataFrame:
    """Patient characteristics as rows of (Characteristic, Level, Value)."""
    total = patients.height
    rows: list[tuple[str, str, str]] = []

    def section(column: str) -> None:
        heading, levels = CATEGORICAL_SECTIONS[column]
        counts = patients[column].value_counts()
        count = dict(zip(counts[column].to_list(), counts["count"].to_list(), strict=True))
        unlisted = set(count) - {value for value, _ in levels}
        if unlisted:
            raise ValueError(f"{column}: values not shown in Table 1: {sorted(unlisted)}")
        for value, label in levels:
            rows.append((heading, label, percent(count.get(value, 0), total)))

    section("sex")
    median, q1, q3 = median_iqr(patients["age"].to_numpy())
    age = "Age at molecular testing, years"
    rows += [
        (
            age,
            "Median (IQR; range)",
            f"{median:.0f} ({q1:.0f}–{q3:.0f}; {patients['age'].min()}–{patients['age'].max()})",
        ),
        (age, "<65 years", percent(int((patients["age"] < 65).sum()), total)),
        (age, "≥65 years", percent(int((patients["age"] >= 65).sum()), total)),
    ]
    for column in CATEGORICAL_SECTIONS:
        if column != "sex":
            section(column)
    return pl.DataFrame(rows, schema=["Characteristic", "Level", "Value"], orient="row")


def percent(n: int, total: int) -> str:
    """For example ``55 (71.4%)``."""
    return f"{n} ({100 * n / total:.1f}%)"


def to_markdown(df: pl.DataFrame) -> str:
    """Render a frame as a GitHub-flavoured Markdown table."""

    def cell(value: object) -> str:
        return "" if value is None else str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(df.columns) + " |",
        "|" + "---|" * df.width,
        *("| " + " | ".join(cell(v) for v in row) + " |" for row in df.iter_rows()),
    ]
    return "\n".join(lines)
