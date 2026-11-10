"""The numbers reported in the manuscript, recomputed from the study data.

Skipped when the (non-public) study data are not available. Survival times are
in days (1 month = 30.4375 days); hazard ratios and P values agree with R's
survival package to the stated precision.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ega_mtb import figures, stats
from ega_mtb.tables import table1


def _median(group: stats.Group) -> tuple:
    m = stats.km_median(group.days, group.events)
    return group.n, m.median, m.ci_low, m.ci_high


def _hr(hr: stats.HazardRatio) -> tuple[float, float, float, float]:
    return round(hr.hr, 4), round(hr.ci_low, 4), round(hr.ci_high, 4), round(hr.p, 4)


def test_cohort(study):
    p = study.patients
    assert p.height == 77
    assert int(p["wes"].sum()) == 10
    assert figures.figure1(study) is not None  # screening counts add up to 77


def test_table1(study):
    rows = [(level, value) for _, level, value in table1(study.patients).iter_rows()]
    assert rows == [
        ("Male", "55 (71.4%)"),
        ("Female", "22 (28.6%)"),
        ("Median (IQR; range)", "60 (50–66; 32–88)"),
        ("<65 years", "53 (68.8%)"),
        ("≥65 years", "24 (31.2%)"),
        ("Esophagogastric junction (EGJ)", "41 (53.2%)"),
        ("Stomach", "31 (40.3%)"),
        ("Esophagus", "5 (6.5%)"),
        ("M0", "38 (49.4%)"),
        ("M1", "36 (46.8%)"),
        ("Unknown", "3 (3.9%)"),
        ("Yes", "40 (51.9%)"),
        ("No", "35 (45.5%)"),
        ("Unknown", "2 (2.6%)"),
        ("Curative setting", "2 (2.6%)"),
        ("1st line", "30 (39.0%)"),
        ("2nd line", "28 (36.4%)"),
        ("3rd line", "11 (14.3%)"),
        ("4th line", "5 (6.5%)"),
        ("5th line", "1 (1.3%)"),
        ("0", "41 (53.2%)"),
        ("1+", "15 (19.5%)"),
        ("2+", "7 (9.1%)"),
        ("3+", "7 (9.1%)"),
        ("Unknown", "7 (9.1%)"),
        ("0", "20 (26.0%)"),
        ("1–4", "16 (20.8%)"),
        ("5–9", "17 (22.1%)"),
        ("≥10", "14 (18.2%)"),
        ("Unknown", "10 (13.0%)"),
        ("Positive", "14 (18.2%)"),
        ("Negative", "20 (26.0%)"),
        ("Unknown", "43 (55.8%)"),
        ("pMMR/MSS", "76 (98.7%)"),
        ("MSI-high", "0 (0.0%)"),
        ("Unknown", "1 (1.3%)"),
    ]


def test_molecular_landscape(study):
    alterations = study.alterations
    per_patient = (
        study.patients.select("patient_id")
        .join(alterations.group_by("patient_id").len(), on="patient_id", how="left")
        .fill_null(0)["len"]
    )
    assert int((per_patient > 0).sum()) == 73
    assert stats.median_iqr(per_patient.to_numpy()) == (2, 2, 4)

    def altered(gene: str, *types: str) -> int:
        rows = alterations.filter(pl.col("gene") == gene)
        if types:
            rows = rows.filter(pl.col("alteration").is_in(types))
        return rows["patient_id"].n_unique()

    small = ("Missense", "Nonsense", "Frameshift", "Small deletion", "Splice-Site")
    assert altered("TP53") == 50
    assert altered("ARID1A") == altered("ARID1A", *small) == 10
    assert (altered("KRAS"), altered("KRAS", *small), altered("KRAS", "CNV Gain")) == (10, 4, 7)
    assert altered("APC") == altered("APC", *small) == 7
    assert (altered("CDKN2A"), altered("CDKN2A", *small), altered("CDKN2A", "CNV Loss")) == (
        7,
        4,
        3,
    )
    assert altered("CDKN2B", "CNV Loss") == 3
    assert altered("PIK3CA") == altered("PIK3CA", *small) == 5
    for gene, n in {"ERBB2": 7, "EGFR": 5, "MYC": 5, "CCNE1": 4, "CDK6": 3}.items():
        assert altered(gene) == altered(gene, "CNV Gain") == n, gene
    assert altered("CDH1") == 4
    assert study.patients["fusion"].drop_nulls().len() == 3


def test_tmb_and_msi(study):
    p = study.patients
    tmb = p["tmb"].drop_nulls().to_numpy()
    assert len(tmb) == 74
    assert stats.median_iqr(tmb) == pytest.approx((5.5, 3.275, 7.1))
    assert int(np.sum(tmb >= 10)) == 5
    msi = p["msi_score"].drop_nulls().to_numpy()
    assert (len(msi), msi.min(), msi.max()) == (57, 0, 4.8)
    assert stats.cohort_median(p, "msi_score") == 2.04
    assert int((p["msi_status"] == "Unknown").sum()) == 1
    assert int((p["msi_status"] == "MSI-high").sum()) == 0


def test_clinical_impact(study):
    p = study.patients
    presented = p.filter(pl.col("mtb_presented"))
    assert stats.median_iqr(p["days_to_report"].to_numpy()) == (27, 22, 33)
    assert stats.median_iqr(presented["days_to_mtb"].to_numpy()) == (47.5, 33.75, 67.5)
    assert presented.height == 52
    assert int(p["mtb_recommendation"].sum()) == 13
    started = p.filter(pl.col("therapy_started"))
    assert started.height == 1
    assert not started["mtb_presented"].any()
    nodes = figures.pathway_counts(p).nodes
    assert nodes[1:5] == [
        "MTB presented (52)",
        "Not presented (25)",
        "Recommendation based on<br>TSO500 or WES (13)",
        "No recommendation (39)",
    ]


def test_presented_vs_not_presented(study):
    p = study.patients
    presented = p.filter(pl.col("mtb_presented"))
    not_presented = p.filter(~pl.col("mtb_presented"))
    assert (presented["age"].median(), not_presented["age"].median()) == (60, 59)
    later_line = [round(100 * (g["therapy_line"] >= 2).mean()) for g in (presented, not_presented)]
    stomach = [round(100 * (g["location"] == "Stomach").mean()) for g in (presented, not_presented)]
    assert (later_line, stomach) == ([58, 60], [44, 32])


def test_overall_survival_and_follow_up(study):
    p = study.patients
    days, events = p["os_days"].to_numpy(), p["event"].to_numpy()
    assert stats.km_median(days, events) == stats.Median(77, 52, 239, 168, 321)
    assert stats.follow_up_median(days, events) == stats.Median(77, 25, 944, 279, 1022)
    assert stats.median_iqr(days[events == 0]) == (186, 99, 329)


def test_tmb_survival(study):
    p = study.patients
    low, high = stats.by_median(p, "tmb", "TMB")
    assert stats.cohort_median(p, "tmb") == 5.5
    assert (_median(low), _median(high)) == ((33, 142, 120, 237), (41, 338, 207, 639))
    assert stats.logrank_p([low, high]) == pytest.approx(0.0138, abs=1e-4)
    assert _hr(stats.group_hazard_ratio(low, high)) == (0.4965, 0.2817, 0.8752, 0.0155)

    evaluable = p.filter(pl.col("tmb").is_not_null())
    continuous = stats.cox_hazard_ratio(
        evaluable["os_days"].to_numpy(), evaluable["event"].to_numpy(), evaluable["tmb"].to_numpy()
    )
    assert _hr(continuous) == (0.8862, 0.7942, 0.9888, 0.0307)

    below, above = stats.by_cutoff(p, "tmb", 10, ("TMB <10", "TMB ≥10"))
    assert (_median(below), _median(above)) == ((69, 239, 164, 321), (5, 252, 24, None))
    assert stats.logrank_p([below, above]) == pytest.approx(0.7417, abs=1e-4)
    assert _hr(stats.group_hazard_ratio(below, above)) == (0.823, 0.2543, 2.663, 0.7451)


def test_therapy_line_survival(study):
    early, late = stats.by_therapy_line(study.patients)
    assert (_median(early), _median(late)) == ((32, 256, 168, 742), (45, 221, 140, 321))
    assert stats.logrank_p([early, late]) == pytest.approx(0.0328, abs=1e-4)
    assert _hr(stats.group_hazard_ratio(early, late)) == (1.8679, 1.0438, 3.3424, 0.0353)


def test_biomarker_survival(study):
    p = study.patients
    panels = {
        "her2": (stats.by_her2(p), [(14, 338, 143, 541), (56, 247, 152, 313)], 0.8781),
        "pdl1": (
            stats.by_pdl1(p),
            [(14, 282, 143, 464), (17, 239, 86, 742), (16, 669, 131, None), (20, 207, 120, 313)],
            0.8048,
        ),
        "claudin": (stats.by_claudin(p), [(14, 247, 140, None), (20, 282, 88, 338)], 0.7577),
        "msi": (
            stats.by_median(p, "msi_score", "MSI score"),
            [(28, 247, 143, 639), (29, 252, 135, 707)],
            0.7988,
        ),
    }
    for name, (groups, medians, p_value) in panels.items():
        assert [_median(g) for g in groups] == medians, name
        assert stats.logrank_p(groups) == pytest.approx(p_value, abs=1e-4), name
