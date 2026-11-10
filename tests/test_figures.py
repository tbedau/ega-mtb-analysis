"""Figure construction (synthetic data; nothing is rendered to disk)."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from ega_mtb import figures


def _annotations(fig: go.Figure) -> list[str]:
    return [a.text for a in fig.layout.annotations]


def test_flowchart_counts():
    fig = figures.patient_flowchart(77)
    text = " ".join(_annotations(fig))
    for n in (80, 78, 77, 2, 1):
        assert f"(<i>n</i> = {n})" in text


def test_flowchart_must_match_cohort(cohort):
    with pytest.raises(ValueError, match="cohort size"):
        figures.figure1(cohort)


def test_oncoprint_data(cohort):
    data = figures.oncoprint_data(cohort)
    assert data.genes[:2] == ["TP53", "KRAS"]
    assert data.genes[2:] == sorted(data.genes[2:])  # ties ordered by gene name
    # Memo sort: TP53 + KRAS first (patient 4 also has CDKN2A), then TP53 alone, …
    assert data.patients == ["4", "1", "2", "6", "5", "8", "7", "3"]
    assert data.labels == ["4", "1", "2", "6", "5", "8*", "7*", "3"]
    assert data.cells[("CDKN2A", "4")] == ["Small deletion", "CNV Loss"]
    assert data.gene_pct["TP53"] == 50
    assert data.tmb == [3.1, 5.5, 10.2, 1.6, 7.0, None, 12.0, None]


def test_oncoprint_figure(cohort):
    fig = figures.figure2(cohort)
    footnote = _annotations(fig)[-1]
    assert footnote.startswith("* whole-exome sequencing (WES)")
    assert figures.FLAG_FOOTNOTE in footnote  # CDKN2A/B are among the genes shown
    legend = [t.name for t in fig.data if isinstance(t, go.Scatter)]
    assert legend == list(figures.ALTERATION_TYPES)


def test_pathway_counts(cohort):
    pathway = figures.pathway_counts(cohort.patients)
    assert pathway.nodes[1] == "MTB presented (5)"
    assert pathway.nodes[3] == "Recommendation based on<br>TSO500 or WES (2)"
    assert pathway.nodes[6] == "Therapy initiated (1)"
    flow_in = {}
    flow_out = {}
    for source, target, n in pathway.links:
        flow_out[source] = flow_out.get(source, 0) + n
        flow_in[target] = flow_in.get(target, 0) + n
    for node in (1, 2, 3, 4, 5):  # every intermediate node conserves patients
        assert flow_in[node] == flow_out[node]
    assert flow_out[0] == flow_in[6] + flow_in[7] == 8
    assert (5, 6, 1) in pathway.links  # therapy started without MTB presentation


@pytest.mark.parametrize(
    ("build", "letters"),
    [
        (figures.figure4, ["(A)", "(B)"]),
        (figures.figure_s1, []),
        (figures.figure_s2, ["(A)", "(B)", "(C)", "(D)"]),
    ],
)
def test_km_panel_letters(cohort, build, letters):
    texts = _annotations(build(cohort))
    assert [t for t in texts if t.startswith("<b>(")] == [f"<b>{x}</b>" for x in letters]


def test_km_curve_reaches_last_follow_up(cohort):
    fig = figures.figure4(cohort)
    overall_curve = fig.data[0]
    assert overall_curve.x[-1] * figures.DAYS_PER_MONTH == pytest.approx(400)


def test_km_number_at_risk(cohort):
    fig = figures.figure_s1(cohort)
    risk_rows = [t for t in fig.data if t.mode == "text"]
    # Line 0–1: patients 1, 2 and 7 (100, 200 and 250 days) at risk at 0, 5 and 10 months.
    assert list(risk_rows[0].text[:3]) == ["3", "2", "0"]
