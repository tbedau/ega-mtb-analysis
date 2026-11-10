import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl

    from ega_mtb import figures, stats, tables
    from ega_mtb.data import DAYS_PER_MONTH, load_cohort

    return DAYS_PER_MONTH, figures, load_cohort, mo, np, pl, stats, tables


@app.cell
def _(mo):
    mo.md(r"""
    # From sequencing to treatment — analysis

    Reproduces Table 1, Figures 1–4, Supplementary Figures 1–2 and the statistics
    reported in the Results (Sections 3.1–3.4) of *From sequencing to treatment: a
    real-world analysis of molecular tumor board recommendations and their clinical
    implementation in esophagogastric adenocarcinoma*. The patient data are not
    included in this repository (see README).
    """)
    return


@app.cell
def _(load_cohort, pl, tables):
    cohort = load_cohort()
    patients = cohort.patients
    n = patients.height

    def show(rows):
        """Two-column Markdown table of (statistic, value) pairs."""
        frame = pl.DataFrame(rows, schema=["Statistic", "Value"], orient="row")
        return tables.to_markdown(frame)

    return cohort, n, patients, show


@app.cell
def _(mo, n, patients, show):
    mo.vstack(
        [
            mo.md("## 3.1 Patient characteristics"),
            mo.md(
                show(
                    [
                        ("Patients", n),
                        ("TSO500", int((~patients["wes"]).sum())),
                        ("Whole-exome sequencing", int(patients["wes"].sum())),
                    ]
                )
            ),
        ]
    )
    return


@app.cell
def _(mo, patients, tables):
    mo.vstack(
        [
            mo.md("### Table 1. Patient characteristics"),
            mo.md(tables.to_markdown(tables.table1(patients))),
        ]
    )
    return


@app.cell
def _(cohort, figures, mo):
    mo.vstack([mo.md("### Figure 1. Patient selection flowchart"), figures.figure1(cohort)])
    return


@app.cell
def _(cohort, n, np, patients, pl, show, stats, tables):
    _alterations = cohort.alterations
    _per_patient = (
        patients.select("patient_id")
        .join(_alterations.group_by("patient_id").len(), on="patient_id", how="left")
        .fill_null(0)["len"]
    )
    _altered = int((_per_patient > 0).sum())
    _count_median, _count_q1, _count_q3 = stats.median_iqr(_per_patient.to_numpy())
    _tmb = patients["tmb"].drop_nulls().to_numpy()
    _tmb_median, _tmb_q1, _tmb_q3 = stats.median_iqr(_tmb)
    _msi = patients["msi_score"].drop_nulls().to_numpy()
    _fusions = patients["fusion"].drop_nulls().to_list()
    _msi_status_known = patients.filter(pl.col("msi_status") != "Unknown")
    molecular_summary = show(
        [
            (
                "Patients with ≥1 functionally relevant alteration",
                tables.percent(_altered, n),
            ),
            (
                "Alterations per patient, median (IQR)",
                f"{_count_median:.0f} ({_count_q1:.0f}–{_count_q3:.0f})",
            ),
            ("Gene fusions", f"{tables.percent(len(_fusions), n)}: {'; '.join(_fusions)}"),
            ("TMB evaluable", tables.percent(len(_tmb), n)),
            (
                "TMB, median (IQR), mut/Mb",
                f"{_tmb_median:.1f} ({_tmb_q1:.1f}–{_tmb_q3:.1f})",
            ),
            ("TMB ≥10 mut/Mb", tables.percent(int(np.sum(_tmb >= 10)), n)),
            (
                "MSI-high by health record (evaluable patients)",
                f"{int((_msi_status_known['msi_status'] == 'MSI-high').sum())}"
                f"/{_msi_status_known.height}",
            ),
            ("NGS MSI score evaluable", tables.percent(len(_msi), n)),
            ("NGS MSI score, range (%)", f"{_msi.min():g}–{_msi.max():g}"),
            ("NGS MSI-high (score ≥20%)", f"{int(np.sum(_msi >= 20))}/{len(_msi)}"),
        ]
    )

    # Patients with an alteration in each gene: any, small variant, CNV gain, CNV loss.
    _by_type = (
        _alterations.with_columns(
            kind=pl.col("alteration").replace_strict(
                {"CNV Gain": "CNV gain", "CNV Loss": "CNV loss"}, default="Small variant"
            )
        )
        .group_by("gene", "kind")
        .agg(pl.col("patient_id").n_unique())
        .pivot(on="kind", index="gene", values="patient_id")
    )
    _any = _alterations.group_by("gene").agg(Altered=pl.col("patient_id").n_unique())
    gene_table = (
        _any.join(_by_type, on="gene")
        .fill_null(0)
        .sort(["Altered", "gene"], descending=[True, False])
        .head(30)
        .select(
            pl.col("gene").alias("Gene"),
            *(
                pl.format("{} ({}%)", pl.col(c), (100 * pl.col(c) / n).round(1)).alias(c)
                for c in ("Altered", "Small variant", "CNV gain", "CNV loss")
            ),
        )
    )
    return gene_table, molecular_summary


@app.cell
def _(gene_table, mo, molecular_summary, tables):
    mo.vstack(
        [
            mo.md("## 3.2 Molecular alteration landscape"),
            mo.md(molecular_summary),
            mo.md("**Patients with an alteration, top 30 genes** (% of all patients)"),
            mo.md(tables.to_markdown(gene_table)),
        ]
    )
    return


@app.cell
def _(cohort, figures, mo):
    mo.vstack([mo.md("### Figure 2. Molecular alteration landscape"), figures.figure2(cohort)])
    return


@app.cell
def _(n, patients, pl, show, stats, tables):
    def _days(values):
        median, q1, q3 = stats.median_iqr(values)
        return f"{median:.0f} ({q1:.0f}–{q3:.0f})"

    _presented = patients.filter(pl.col("mtb_presented"))
    _not_presented = patients.filter(~pl.col("mtb_presented"))
    _recommended = int(patients["mtb_recommendation"].sum())

    def _compare(label, expr):
        """Characteristic in MTB-presented vs not presented patients."""
        return label, " vs ".join(f"{expr(group):.0f}" for group in (_presented, _not_presented))

    pathway_summary = show(
        [
            ("Days from sample receipt to report, median (IQR)", _days(patients["days_to_report"])),
            (
                "Days from sample receipt to MTB presentation, median (IQR)",
                _days(_presented["days_to_mtb"]),
            ),
            ("Presented at the MTB", tables.percent(_presented.height, n)),
            (
                "MTB recommendation (% of presented)",
                tables.percent(_recommended, _presented.height),
            ),
            ("MTB recommendation (% of all patients)", tables.percent(_recommended, n)),
            (
                "Targeted therapy initiated",
                tables.percent(int(patients["therapy_started"].sum()), n),
            ),
            (
                "… of which after an MTB recommendation",
                int((patients["therapy_started"] & patients["mtb_recommendation"]).sum()),
            ),
            _compare("Presented vs not: median age, years", lambda g: g["age"].median()),
            _compare(
                "Presented vs not: second line or later, %",
                lambda g: 100 * (g["therapy_line"] >= 2).mean(),
            ),
            _compare(
                "Presented vs not: stomach, %",
                lambda g: 100 * (g["location"] == "Stomach").mean(),
            ),
        ]
    )
    return (pathway_summary,)


@app.cell
def _(cohort, figures, mo, pathway_summary):
    mo.vstack(
        [
            mo.md("## 3.3 Clinical impact of molecular diagnostics"),
            mo.md(pathway_summary),
            mo.md("### Figure 3. Clinical impact pathway"),
            figures.figure3(cohort),
        ]
    )
    return


@app.cell
def _(DAYS_PER_MONTH, n, patients, pl, show, stats, tables):
    def _logrank(groups):
        return f"log-rank P = {stats.logrank_p(groups):.3f}"

    def _medians(groups):
        return "; ".join(
            f"{g.label} (n={g.n}): {stats.km_median(g.days, g.events).describe()}" for g in groups
        )

    _days = patients["os_days"].to_numpy()
    _events = patients["event"].to_numpy()
    _alive = patients.filter(pl.col("event") == 0)["os_days"].to_numpy()
    _alive_median, _alive_q1, _alive_q3 = (x / DAYS_PER_MONTH for x in stats.median_iqr(_alive))
    _tmb = patients.filter(pl.col("tmb").is_not_null())
    _tmb_median = stats.by_median(patients, "tmb", "TMB")
    _tmb_10 = stats.by_cutoff(patients, "tmb", 10, ("TMB <10", "TMB ≥10"))
    _line = stats.by_therapy_line(patients)

    survival_summary = show(
        [
            ("OS events", f"{int(_events.sum())}/{n}"),
            ("Median OS", stats.km_median(_days, _events).describe()),
            (
                "Median follow-up (reverse Kaplan–Meier)",
                stats.follow_up_median(_days, _events).describe(),
            ),
            ("Alive at last contact", tables.percent(len(_alive), n)),
            (
                "Follow-up of patients alive, median (IQR)",
                f"{_alive_median:.1f} months ({_alive_q1:.1f}–{_alive_q3:.1f})",
            ),
            (
                f"TMB median split at {stats.cohort_median(patients, 'tmb'):g} mut/Mb",
                _medians(_tmb_median),
            ),
            ("TMB median split: test", _logrank(_tmb_median)),
            ("TMB median split: high vs low", stats.group_hazard_ratio(*_tmb_median).describe()),
            (
                "TMB continuous (Cox, per mut/Mb)",
                stats.cox_hazard_ratio(
                    _tmb["os_days"].to_numpy(), _tmb["event"].to_numpy(), _tmb["tmb"].to_numpy()
                ).describe(),
            ),
            ("TMB ≥10 sensitivity analysis", _medians(_tmb_10)),
            ("TMB ≥10: test", _logrank(_tmb_10)),
            ("TMB ≥10 vs <10", stats.group_hazard_ratio(*_tmb_10).describe()),
            ("Line of therapy", _medians(_line)),
            ("Line of therapy: test", _logrank(_line)),
            ("Line ≥2 vs 0–1", stats.group_hazard_ratio(*_line).describe()),
        ]
    )
    return (survival_summary,)


@app.cell
def _(mo, survival_summary):
    mo.vstack([mo.md("## 3.4 Exploratory survival analyses"), mo.md(survival_summary)])
    return


@app.cell
def _(cohort, figures, mo):
    mo.vstack(
        [
            mo.md(
                "### Figure 4. Overall survival: (A) entire cohort, (B) by TMB "
                "(cohort median; 3 patients without evaluable TMB excluded)"
            ),
            figures.figure4(cohort),
        ]
    )
    return


@app.cell
def _(cohort, figures, mo):
    mo.vstack(
        [
            mo.md("### Supplementary Figure 1. Overall survival by line of therapy"),
            figures.figure_s1(cohort),
        ]
    )
    return


@app.cell
def _(cohort, figures, mo):
    mo.vstack(
        [
            mo.md(
                "### Supplementary Figure 2. Overall survival by (A) HER2, (B) PD-L1, "
                "(C) Claudin 18.2 and (D) MSI score (cohort median); unknown values excluded"
            ),
            figures.figure_s2(cohort),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
