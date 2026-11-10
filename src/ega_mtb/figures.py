"""Figures 1–4 and Supplementary Figures 1–2.

Each ``figure*`` function takes the :class:`~ega_mtb.data.Cohort` and returns a
plotly figure; ``ega_mtb.export`` writes them to disk.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from ega_mtb import stats
from ega_mtb.data import ALTERATION_TYPES, CNV_GAIN, CNV_LOSS, DAYS_PER_MONTH, Cohort

# --------------------------------------------------------------------------- #
# Style
# --------------------------------------------------------------------------- #

FONT_FAMILY = "Arial, Helvetica, sans-serif"
INK = "#1A1A1A"  # primary text
INK_MUTED = "#5B6670"  # secondary text
GRID_COLOR = "#E7EAEC"
NEUTRAL = "#606A73"  # TMB bars

# Okabe–Ito colour-blind-safe palette.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}
NAVY = "#1A3E6F"
RED = "#C0392B"
#: Survival-curve colours: navy/red for two groups, plus green/amber for four.
GROUP_COLORS = (NAVY, RED, OKABE_ITO["green"], OKABE_ITO["orange"])

TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(family=FONT_FAMILY, size=12, color=INK),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
)


# --------------------------------------------------------------------------- #
# Figure 1 — patient selection flowchart
# --------------------------------------------------------------------------- #

#: Screening counts (Methods, Section 2.1). The excluded reports are not part of
#: the analysis data set, so these are fixed inputs; the number of included
#: patients is checked against the cohort.
SCREENING = {"identified": 80, "duplicates": 2, "no_follow_up": 1}

SEARCH_TERMS = (
    '"Adenokarzinom des Ösophagus", "Adenokarzinom des Magens", "Magenkarzinom",<br>'
    '"EAC", "AEG-Tumor", "AEG I/II/III", "Ösophaguskarzinom"'
)

_FLOW_MAIN_FILL = "#CBD9E9"
_FLOW_FINAL_FILL = "#9DBAD8"
_FLOW_EXCLUSION_FILL = "#EEF0F2"


def figure1(cohort: Cohort) -> go.Figure:
    """Patient selection flowchart."""
    return patient_flowchart(cohort.patients.height)


def patient_flowchart(n_included: int) -> go.Figure:
    """Vertical stage flow (identification → eligibility → included) with exclusions."""
    identified = SCREENING["identified"]
    unique = identified - SCREENING["duplicates"]
    if unique - SCREENING["no_follow_up"] != n_included:
        raise ValueError("screening counts do not add up to the cohort size")

    fig = go.Figure()
    # Invisible anchor trace so that the (hidden) axes and their ranges exist.
    fig.add_trace(
        go.Scatter(x=[0, 100], y=[0, 100], mode="markers", marker_opacity=0, hoverinfo="skip")
    )
    for y0, y1, stage in (
        (78, 96, "Identification"),
        (54, 66, "Eligibility"),
        (24, 40, "Included"),
    ):
        _flow_box(
            fig,
            (3, y0, 9, y1),
            f"<b>{stage}</b>",
            fill=NAVY,
            size=13,
            color="white",
            centered=True,
            angle=-90,
        )
    _flow_box(
        fig,
        (11, 78, 52, 96),
        f"<b>Sequencing reports identified (<i>n</i> = {identified})</b><br>"
        "Institutional database searched for TSO500<br>"
        "and WES sequencing reports issued<br>1 Oct 2022 – 30 Mar 2026",
        fill=_FLOW_MAIN_FILL,
        size=14,
    )
    _flow_box(
        fig,
        (11, 54, 52, 66),
        f"<b>Unique patients assessed for eligibility<br>(<i>n</i> = {unique})</b>",
        fill=_FLOW_MAIN_FILL,
        size=14,
    )
    _flow_box(
        fig,
        (11, 24, 52, 40),
        f"<b>Patients included in the final cohort<br>(<i>n</i> = {n_included})</b><br>"
        "gastroesophageal adenocarcinoma",
        fill=_FLOW_FINAL_FILL,
        size=14,
    )
    _flow_box(
        fig,
        (60, 65, 99, 77),
        f"<b>Duplicate reports removed (<i>n</i> = {SCREENING['duplicates']})</b><br>"
        "first sequencing result per patient retained",
        fill=_FLOW_EXCLUSION_FILL,
        size=12.5,
        color=INK_MUTED,
    )
    _flow_box(
        fig,
        (60, 41, 99, 53),
        f"<b>Patients excluded (<i>n</i> = {SCREENING['no_follow_up']})</b><br>"
        "no follow-up data available",
        fill=_FLOW_EXCLUSION_FILL,
        size=12.5,
        color=INK_MUTED,
    )
    centre = 31.5
    for x0, y0, x1, y1 in (
        (centre, 78, centre, 66),  # identified -> unique
        (centre, 54, centre, 40),  # unique -> included
        (centre, 71, 60, 71),  # -> duplicates
        (centre, 47, 60, 47),  # -> no follow-up
    ):
        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.1,
            arrowwidth=1.5,
            arrowcolor=INK,
        )
    fig.add_annotation(
        x=3,
        y=20,
        xref="x",
        yref="y",
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        text=f"Diagnostic search terms: {SEARCH_TERMS}.",
        font=dict(size=11, color=INK_MUTED),
    )
    fig.update_xaxes(range=[0, 100], visible=False, fixedrange=True)
    fig.update_yaxes(range=[14, 100], visible=False, fixedrange=True)
    fig.update_layout(
        template=TEMPLATE,
        width=880,
        height=536,
        showlegend=False,
        margin=dict(l=10, r=10, t=16, b=10),
    )
    return fig


def _flow_box(
    fig: go.Figure,
    box: tuple[float, float, float, float],
    text: str,
    *,
    fill: str,
    size: float,
    color: str = INK,
    centered: bool = False,
    angle: float = 0,
) -> None:
    """Borderless filled rectangle (x0, y0, x1, y1) with a text block."""
    x0, y0, x1, y1 = box
    fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, fillcolor=fill, line_width=0)
    fig.add_annotation(
        x=(x0 + x1) / 2 if centered else x0 + 1.6,
        y=(y0 + y1) / 2,
        xref="x",
        yref="y",
        xanchor="center" if centered else "left",
        yanchor="middle",
        align="center" if centered else "left",
        text=text,
        showarrow=False,
        textangle=angle,
        font=dict(size=size, color=color),
    )


# --------------------------------------------------------------------------- #
# Figure 2 — molecular alteration landscape (oncoprint)
# --------------------------------------------------------------------------- #

ALTERATION_COLORS = {
    "Missense": OKABE_ITO["green"],
    "Nonsense": OKABE_ITO["black"],
    "Frameshift": OKABE_ITO["orange"],
    "Small deletion": OKABE_ITO["purple"],
    "Splice-Site": OKABE_ITO["sky"],
    CNV_GAIN: OKABE_ITO["vermillion"],
    CNV_LOSS: OKABE_ITO["blue"],
}
EMPTY_CELL_COLOR = "#EEF1F3"

#: Genes marked with "#" and explained in a footnote.
FLAGGED_GENES = ("CDKN2A", "CDKN2B")
FLAG_FOOTNOTE = (
    "# The low frequency of CDKN2A/B copy-number losses reflects a reporting artefact: prior "
    "to 2025, copy<br>number losses were not consistently included in molecular pathology "
    "reports within this cohort,<br>resulting in systematic underascertainment."
)


@dataclass(frozen=True)
class Oncoprint:
    """Oncoprint content: genes (rows), patients (columns) and altered cells."""

    genes: list[str]  # most frequently altered first
    patients: list[str]  # column order
    labels: list[str]  # column labels; "*" marks whole-exome sequencing
    cells: dict[tuple[str, str], list[str]]  # (gene, patient) -> alteration types
    tmb: list[float | None]  # per column
    gene_pct: dict[str, float]  # % of all patients with the gene altered


def figure2(cohort: Cohort) -> go.Figure:
    """Oncoprint of the 30 most frequently altered genes with a TMB track."""
    return make_oncoprint(oncoprint_data(cohort, n_genes=30))


def oncoprint_data(cohort: Cohort, n_genes: int = 30) -> Oncoprint:
    """Rank genes by the number of altered patients and memo-sort the patients."""
    patients = cohort.patients
    per_gene = (
        cohort.alterations.group_by("gene")
        .agg(n=pl.col("patient_id").n_unique())
        .sort(["n", "gene"], descending=[True, False])
    )
    genes = per_gene.head(n_genes)["gene"].to_list()

    cells: dict[tuple[str, str], list[str]] = {}
    top = cohort.alterations.filter(pl.col("gene").is_in(genes))
    for patient, gene, alteration in top.select("patient_id", "gene", "alteration").iter_rows():
        cells.setdefault((gene, patient), []).append(alteration)
    for types in cells.values():
        types.sort(key=ALTERATION_TYPES.index)

    # Memo sort: alterations in the most frequent genes cluster to the left; ties
    # are broken by the number of altered genes, then by patient number.
    altered = {p: {g for g, q in cells if q == p} for p in patients["patient_id"]}

    def memo_key(patient: str) -> tuple:
        presence = tuple(gene in altered[patient] for gene in genes)
        return presence, len(altered[patient]), patient

    order = sorted(altered, key=memo_key, reverse=True)
    by_id = {row["patient_id"]: row for row in patients.iter_rows(named=True)}
    return Oncoprint(
        genes=genes,
        patients=order,
        labels=[p + ("*" if by_id[p]["wes"] else "") for p in order],
        cells=cells,
        tmb=[by_id[p]["tmb"] for p in order],
        gene_pct={g: 100 * sum(1 for p in order if g in altered[p]) / len(order) for g in genes},
    )


def make_oncoprint(data: Oncoprint) -> go.Figure:
    """Alteration grid (split cells for two types) under a TMB bar track."""
    n, g = len(data.patients), len(data.genes)
    row_of = {gene: g - 1 - i for i, gene in enumerate(data.genes)}  # top gene at the top
    col_of = {p: i for i, p in enumerate(data.patients)}

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.14, 0.86], vertical_spacing=0.02
    )
    fig.add_trace(
        go.Bar(
            x=list(range(n)),
            y=data.tmb,
            marker_color=NEUTRAL,
            marker_line_width=0,
            showlegend=False,
            customdata=data.labels,
            hovertemplate="%{customdata}<br>TMB: %{y} mut/Mb<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(  # grey background grid
        go.Heatmap(
            x=list(range(n)),
            y=list(range(g)),
            z=[[0] * n for _ in range(g)],
            colorscale=[[0, EMPTY_CELL_COLOR], [1, EMPTY_CELL_COLOR]],
            showscale=False,
            xgap=1,
            ygap=1,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    half = 0.46
    for (gene, patient), types in data.cells.items():
        x, y = col_of[patient], row_of[gene]
        for band, alteration in enumerate(types):
            fig.add_shape(
                type="rect",
                x0=x - half,
                x1=x + half,
                y0=y - half + 2 * half * band / len(types),
                y1=y - half + 2 * half * (band + 1) / len(types),
                fillcolor=ALTERATION_COLORS[alteration],
                line_width=0,
                row=2,
                col=1,
            )
    present = {t for types in data.cells.values() for t in types}
    for alteration in ALTERATION_TYPES:  # legend entries
        if alteration in present:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=11, color=ALTERATION_COLORS[alteration], symbol="square"),
                    name=alteration,
                ),
                row=2,
                col=1,
            )

    fig.update_yaxes(
        title_text="TMB<br>(mut/Mb)",
        title_font=dict(size=13, color="black"),
        tickfont=dict(size=12, color="black"),
        gridcolor=GRID_COLOR,
        showline=False,
        zeroline=False,
        ticks="outside",
        tickcolor="black",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        tickvals=[row_of[gene] for gene in data.genes],
        ticktext=[f"{gene} #" if gene in FLAGGED_GENES else gene for gene in data.genes],
        tickfont=dict(size=13, color="black"),
        tickcolor="black",
        range=[-0.5, g - 0.5],
        autorange=False,
        showgrid=False,
        zeroline=False,
        row=2,
        col=1,
    )
    x_max = n - 0.5 + max(3.0, n * 0.05)  # room for the per-gene percentages
    fig.update_xaxes(
        tickvals=list(range(n)),
        ticktext=data.labels,
        tickangle=-90,
        tickfont=dict(size=8.5, color="black"),
        tickcolor="black",
        showgrid=False,
        zeroline=False,
        range=[-0.5, x_max],
        autorange=False,
        row=2,
        col=1,
    )
    fig.update_xaxes(
        range=[-0.5, x_max], autorange=False, showgrid=False, zeroline=False, row=1, col=1
    )
    for gene in data.genes:
        fig.add_annotation(
            x=n - 0.3,
            y=row_of[gene],
            row=2,
            col=1,
            xanchor="left",
            yanchor="middle",
            showarrow=False,
            text=f"{data.gene_pct[gene]:.0f}%",
            font=dict(size=11, color=INK_MUTED),
        )
    fig.update_layout(
        template=TEMPLATE,
        height=700,
        width=max(958, n * 11 + 108),
        bargap=0.12,
        legend=dict(
            title=dict(text="Alteration type", font=dict(size=13, color=INK)),
            font=dict(size=12.5, color=INK),
            x=1.02,
            y=1,
            itemsizing="constant",
            tracegroupgap=2,
        ),
        margin=dict(l=90, r=138, t=30, b=122),
    )
    footnotes = []
    if any(label.endswith("*") for label in data.labels):
        footnotes.append("* whole-exome sequencing (WES)")
    if any(gene in FLAGGED_GENES for gene in data.genes):
        footnotes.append(FLAG_FOOTNOTE)
    if footnotes:
        fig.add_annotation(
            x=0,
            y=-0.085,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            text="<br>".join(footnotes),
            font=dict(size=11.5, color=INK_MUTED),
        )
    return fig


# --------------------------------------------------------------------------- #
# Figure 3 — clinical impact pathway (Sankey)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Pathway:
    """Sankey nodes and (source, target, patients) links."""

    nodes: list[str]
    links: list[tuple[int, int, int]]


def figure3(cohort: Cohort) -> go.Figure:
    """Clinical impact pathway from molecular testing to therapy initiation."""
    return make_pathway(pathway_counts(cohort.patients))


def pathway_counts(patients: pl.DataFrame) -> Pathway:
    """Testing → MTB presentation → recommendation → therapy initiation.

    Patients not presented at the MTB bypass the recommendation stage.
    """
    presented = patients["mtb_presented"]
    recommended = presented & patients["mtb_recommendation"]
    not_recommended = presented & ~patients["mtb_recommendation"]
    therapy = patients["therapy_started"]

    def count(mask: pl.Series) -> int:
        return int(mask.sum())

    n_therapy = count(therapy)
    nodes = [
        f"Molecular testing (n={patients.height})",  # 0
        f"MTB presented ({count(presented)})",  # 1
        f"Not presented ({count(~presented)})",  # 2
        f"Recommendation based on<br>TSO500 or WES ({count(recommended)})",  # 3
        f"No recommendation ({count(not_recommended)})",  # 4
        f"Board skipped ({count(~presented)})",  # 5
        f"Therapy initiated ({n_therapy})",  # 6
        f"No therapy ({patients.height - n_therapy})",  # 7
    ]
    links = [
        (0, 1, count(presented)),
        (0, 2, count(~presented)),
        (1, 3, count(recommended)),
        (1, 4, count(not_recommended)),
        (2, 5, count(~presented)),
    ]
    for node, mask in ((3, recommended), (4, not_recommended), (5, ~presented)):
        links += [(node, 6, count(mask & therapy)), (node, 7, count(mask & ~therapy))]
    return Pathway(nodes=nodes, links=[link for link in links if link[2] > 0])


def make_pathway(pathway: Pathway) -> go.Figure:
    grey, light_grey = "#A9B2B9", "#CBD2D7"
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=pathway.nodes,
                x=[0.001, 0.33, 0.33, 0.66, 0.66, 0.66, 0.999, 0.999],
                y=[0.5, 0.25, 0.8, 0.12, 0.42, 0.8, 0.2, 0.7],
                color=[
                    "#2E4A6B",  # molecular testing
                    NAVY,  # MTB presented
                    grey,  # not presented
                    OKABE_ITO["green"],  # recommendation
                    grey,  # no recommendation
                    light_grey,  # board skipped
                    RED,  # therapy initiated
                    light_grey,  # no therapy
                ],
                pad=24,
                thickness=18,
                line=dict(color="white", width=1),
            ),
            link=dict(
                source=[s for s, _, _ in pathway.links],
                target=[t for _, t, _ in pathway.links],
                value=[v for _, _, v in pathway.links],
                color=["rgba(26,62,111,0.16)"] * len(pathway.links),
            ),
        )
    )
    fig.update_layout(
        template=TEMPLATE,
        font=dict(size=14),
        height=430,
        width=860,
        margin=dict(l=20, r=20, t=54, b=20),
    )
    stages = (
        "Molecular testing",
        "MTB presentation",
        "Recommendation<br>(TSO500 or WES)",
        "Therapy initiation",
    )
    for x, stage in zip((0.04, 0.36, 0.66, 0.96), stages, strict=True):
        fig.add_annotation(
            x=x,
            y=1.08,
            xref="paper",
            yref="paper",
            text=f"<b>{stage}</b>",
            showarrow=False,
            font=dict(size=14, color=INK_MUTED),
        )
    return fig


# --------------------------------------------------------------------------- #
# Figure 4 and Supplementary Figures 1–2 — Kaplan–Meier curves
# --------------------------------------------------------------------------- #


def figure4(cohort: Cohort) -> go.Figure:
    """OS of (A) the whole cohort and (B) by TMB split at the cohort median."""
    patients = cohort.patients
    return make_km_figure([stats.overall(patients), stats.by_median(patients, "tmb", "TMB")])


def figure_s1(cohort: Cohort) -> go.Figure:
    """OS by line of therapy at molecular testing (0–1 vs ≥2)."""
    return make_km_figure([stats.by_therapy_line(cohort.patients)], ncols=1)


def figure_s2(cohort: Cohort) -> go.Figure:
    """OS by (A) HER2, (B) PD-L1, (C) Claudin 18.2 and (D) MSI score (median split)."""
    patients = cohort.patients
    return make_km_figure(
        [
            stats.by_her2(patients),
            stats.by_pdl1(patients),
            stats.by_claudin(patients),
            stats.by_median(patients, "msi_score", "MSI score"),
        ]
    )


# Panel geometry in layout pixels, shared by all Kaplan–Meier figures.
_KM_MARGIN = dict(l=122, r=24, t=38, b=44)  # left margin fits the risk-row labels
_KM_PANEL_W = 340  # curve area width per column
_KM_COL_GAP = 108  # clears the next column's risk-row labels
_KM_CURVE_H = 300  # curve area height
_KM_INNER_GAP = 92  # x-axis ticks and title between curves and risk table
_KM_RISK_ROW = 19  # height of one number-at-risk row
_KM_BAND_GAP = 72  # vertical gap between panel rows


def make_km_figure(panels: Sequence[Sequence[stats.Group]], ncols: int = 2) -> go.Figure:
    """Grid of Kaplan–Meier panels, each with a number-at-risk table.

    Each panel shows the median OS (95% CI) per group and, for two or more
    groups, the log-rank P value. Colour identifies the groups. Panels are
    labelled (A), (B), … when there is more than one.
    """
    n = len(panels)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    x_max = max(float(np.max(g.days)) for groups in panels for g in groups) / DAYS_PER_MONTH
    ticks = _time_ticks(x_max)

    # Risk tables are n_groups rows tall; panels of a grid row share the band height.
    risk_h = [len(groups) * _KM_RISK_ROW for groups in panels]
    band_top, band_h = [], []
    top = 0.0
    for r in range(nrows):
        band_top.append(top)
        band_h.append(_KM_CURVE_H + _KM_INNER_GAP + max(risk_h[r * ncols : (r + 1) * ncols]))
        top += band_h[-1] + _KM_BAND_GAP
    plot_h = sum(band_h) + (nrows - 1) * _KM_BAND_GAP
    plot_w = ncols * _KM_PANEL_W + (ncols - 1) * _KM_COL_GAP

    fig = make_subplots(
        rows=2 * nrows,
        cols=ncols,
        vertical_spacing=0.02,
        horizontal_spacing=_KM_COL_GAP / plot_w if ncols > 1 else 0.0,
    )
    for i, groups in enumerate(panels):
        r, c = divmod(i, ncols)
        curve_bottom = band_top[r] + _KM_CURVE_H
        risk_top = curve_bottom + _KM_INNER_GAP
        fig.update_yaxes(
            domain=[1 - curve_bottom / plot_h, 1 - band_top[r] / plot_h], row=2 * r + 1, col=c + 1
        )
        fig.update_yaxes(
            domain=[1 - (risk_top + risk_h[i]) / plot_h, 1 - risk_top / plot_h],
            row=2 * r + 2,
            col=c + 1,
        )
        _draw_km_panel(
            fig,
            groups,
            letter=chr(ord("A") + i) if n > 1 else "",
            row=2 * r + 1,
            col=c + 1,
            ncols=ncols,
            ticks=ticks,
            y_title=c == 0,
        )
    fig.update_layout(
        template=TEMPLATE,
        height=int(plot_h + _KM_MARGIN["t"] + _KM_MARGIN["b"]),
        width=int(plot_w + _KM_MARGIN["l"] + _KM_MARGIN["r"]),
        showlegend=False,
        margin=_KM_MARGIN,
    )
    return fig


def _draw_km_panel(
    fig: go.Figure,
    groups: Sequence[stats.Group],
    *,
    letter: str,
    row: int,
    col: int,
    ncols: int,
    ticks: list[int],
    y_title: bool,
) -> None:
    """Curves in subplot ``row``; the number-at-risk table in ``row + 1``."""
    risk_row = row + 1
    # Subplot axes are numbered row-major; axis 1 has no suffix ("x", "y").
    curve_axis = (row - 1) * ncols + col
    risk_axis = (risk_row - 1) * ncols + col
    cx = "" if curve_axis == 1 else str(curve_axis)
    rx = "" if risk_axis == 1 else str(risk_axis)
    single = len(groups) == 1
    stat_size = 12 if len(groups) <= 2 else 11

    for i, group in enumerate(groups):
        color = GROUP_COLORS[i]
        km = stats.kaplan_meier(group.days, group.events)
        curve = km.survival_function_.iloc[:, 0]  # runs to the last follow-up time
        fig.add_trace(
            go.Scatter(
                x=curve.index.to_numpy() / DAYS_PER_MONTH,
                y=curve.to_numpy(),
                mode="lines",
                line=dict(color=color, width=2, shape="hv"),
                hovertemplate="t=%{x:.1f}<br>S(t)=%{y:.3f}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        censored = group.days[group.events == 0]
        fig.add_trace(
            go.Scatter(
                x=censored / DAYS_PER_MONTH,
                y=km.survival_function_at_times(censored).to_numpy(),
                mode="markers",
                marker=dict(symbol="line-ns-open", size=8, color=color, line_width=1.4),
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # Number at risk under each time tick, with the group name to the left.
        y = len(groups) - 1 - i
        fig.add_trace(
            go.Scatter(
                x=ticks,
                y=[y] * len(ticks),
                mode="text",
                text=[str(int(np.sum(group.days / DAYS_PER_MONTH >= t))) for t in ticks],
                textfont=dict(size=12, color=color),
                cliponaxis=False,
                hoverinfo="skip",
            ),
            row=risk_row,
            col=col,
        )
        fig.add_annotation(
            x=0,
            y=y,
            xanchor="right",
            xshift=-14,
            text=group.label,
            showarrow=False,
            font=dict(size=12, color=color),
            row=risk_row,
            col=col,
        )

        median = stats.km_median(group.days, group.events)
        if single and median.median is not None:
            m = median.median / DAYS_PER_MONTH
            for x0, x1, y0, y1 in ((0, m, 0.5, 0.5), (m, m, 0, 0.5)):
                fig.add_shape(
                    type="line",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    line=dict(color=INK_MUTED, width=1, dash="dot"),
                    row=row,
                    col=col,
                )
        summary = (
            f"median {stats.months(median.median)} mo (95% CI "
            f"{stats.months(median.ci_low)}–{stats.months(median.ci_high)}), <i>n</i>={group.n}"
        )
        fig.add_annotation(
            x=0.97,
            y=0.99,
            yshift=-i * (stat_size + 6),
            xref=f"x{cx} domain",
            yref=f"y{cx} domain",
            xanchor="right",
            yanchor="top",
            text=summary if single else f"<b>{group.label}</b>  {summary}",
            showarrow=False,
            font=dict(size=stat_size, color=color),
        )

    if not single:
        p = stats.logrank_p(groups)
        fig.add_annotation(
            x=0.03,
            y=0.04,
            xref=f"x{cx} domain",
            yref=f"y{cx} domain",
            xanchor="left",
            yanchor="bottom",
            text="Log-rank " + ("<i>P</i> < 0.001" if p < 0.001 else f"<i>P</i> = {p:.3f}"),
            showarrow=False,
            font=dict(size=13, color=INK),
        )
    if letter:
        fig.add_annotation(
            x=0,
            y=1.04,
            xref=f"x{cx} domain",
            yref=f"y{cx} domain",
            xanchor="left",
            yanchor="bottom",
            text=f"<b>({letter})</b>",
            showarrow=False,
            font=dict(size=18, color=INK),
        )
    fig.add_annotation(
        x=0,
        y=1,
        yshift=6,
        xref=f"x{rx} domain",
        yref=f"y{rx} domain",
        xanchor="left",
        yanchor="bottom",
        text="Number at risk",
        showarrow=False,
        font=dict(size=12.5, color=INK),
    )

    axis = dict(
        showline=True,
        linecolor="black",
        linewidth=1,
        ticks="outside",
        tickcolor="black",
        ticklen=4,
        tickfont=dict(size=14, color="black"),
        title_font=dict(size=15, color="black"),
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(range=[0, 1.02], dtick=0.25, row=row, col=col, **axis)
    if y_title:
        fig.update_yaxes(title_text="Survival probability", row=row, col=col)
    fig.update_xaxes(
        range=[0, ticks[-1]],
        tickvals=ticks,
        title_text="Time since molecular diagnosis (months)",
        title_standoff=12,
        row=row,
        col=col,
        **axis,
    )
    hidden = dict(showline=False, showticklabels=False, showgrid=False, zeroline=False, ticks="")
    fig.update_xaxes(range=[0, ticks[-1]], row=risk_row, col=col, **hidden)
    fig.update_yaxes(range=[-0.5, len(groups) - 0.5], row=risk_row, col=col, **hidden)


def _time_ticks(x_max: float) -> list[int]:
    """Evenly spaced month ticks from 0 to just beyond ``x_max``."""
    step = 5 if x_max <= 30 else 6 if x_max <= 54 else 12
    return list(range(0, math.ceil(x_max / step) * step + 1, step))
