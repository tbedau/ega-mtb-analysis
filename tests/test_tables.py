"""Table 1 (synthetic data)."""

from __future__ import annotations

import polars as pl
import pytest

from ega_mtb.tables import CATEGORICAL_SECTIONS, percent, table1, to_markdown


def test_table1_rows(cohort):
    table = table1(cohort.patients)
    rows = {(c, level): value for c, level, value in table.iter_rows()}
    assert rows[("Sex", "Male")] == "5 (62.5%)"
    assert rows[("Age at molecular testing, years", "Median (IQR; range)")] == "60 (50–67; 39–81)"
    assert rows[("Age at molecular testing, years", "≥65 years")] == "3 (37.5%)"
    assert rows[("Prior surgical resection with curative intent", "Unknown")] == "1 (12.5%)"
    assert rows[("Line of therapy at molecular testing", "Curative setting")] == "1 (12.5%)"
    assert rows[("HER2 IHC score", "Unknown")] == "2 (25.0%)"
    assert rows[("Microsatellite status", "MSI-high")] == "0 (0.0%)"


def test_every_section_accounts_for_all_patients(cohort):
    table = table1(cohort.patients)
    headings = [heading for heading, _ in CATEGORICAL_SECTIONS.values()]
    for heading in headings:
        counts = table.filter(pl.col("Characteristic") == heading)["Value"]
        assert sum(int(v.split()[0]) for v in counts) == cohort.patients.height, heading


def test_unlisted_value_raises(cohort):
    patients = cohort.patients.with_columns(location=pl.lit("Duodenum"))
    with pytest.raises(ValueError, match="location"):
        table1(patients)


def test_percent():
    assert percent(55, 77) == "55 (71.4%)"


def test_to_markdown_escapes_pipes():
    frame = pl.DataFrame({"a": ["x|y", None]})
    assert to_markdown(frame) == "| a |\n|---|\n| x\\|y |\n|  |"
