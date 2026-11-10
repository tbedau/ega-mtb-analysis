# From sequencing to treatment — analysis code

Statistical analyses and figures for the study *From sequencing to treatment: a
real-world analysis of molecular tumor board recommendations and their clinical
implementation in esophagogastric adenocarcinoma* (manuscript in revision; the
full reference will be added on publication).

The study is a retrospective analysis of 77 patients with esophagogastric
adenocarcinoma (EGA) whose tumors were profiled by TruSight Oncology 500
(TSO500) or whole-exome sequencing (WES) at the University Hospital Cologne.
The molecular findings come from routine diagnostic reports; no sequencing data
are re-analysed here.

## What the code reproduces

| Manuscript item | Code |
|---|---|
| Table 1 — patient characteristics | `ega_mtb.tables.table1` |
| Figure 1 — patient selection flowchart | `ega_mtb.figures.figure1` |
| Figure 2 — molecular alteration landscape (oncoprint) | `ega_mtb.figures.figure2` |
| Figure 3 — clinical impact pathway | `ega_mtb.figures.figure3` |
| Figure 4 — overall survival: (A) cohort, (B) by TMB | `ega_mtb.figures.figure4` |
| Supplementary Figure 1 — overall survival by line of therapy | `ega_mtb.figures.figure_s1` |
| Supplementary Figure 2 — overall survival by HER2, PD-L1, Claudin 18.2, MSI score | `ega_mtb.figures.figure_s2` |
| Statistics reported in the Results (Sections 3.1–3.4) | `analysis.py` |

Table 2 and Supplementary Tables 1–2 were compiled from the clinical records and
assay documentation and are not computed here. The screening counts in Figure 1
(80 reports, 2 duplicates, 1 patient without follow-up) refer to cases outside
the analysis data set and are fixed inputs; the code checks that they add up to
the cohort size.

## Statistical methods

Overall survival (OS) runs from molecular testing (sample receipt) to death from
any cause; patients alive at last contact are censored. Survival curves are
Kaplan–Meier estimates. The median is the first time at which the estimate falls
to 0.5 or below, and its 95% confidence interval is read off the log–log
(exponential Greenwood) pointwise band (Brookmeyer–Crowley). Groups are compared
with the two-sided log-rank test. Hazard ratios come from univariable Cox
proportional-hazards models (Efron's method for ties) with Wald confidence
intervals and P values. Median follow-up is estimated by the reverse
Kaplan–Meier method. TMB and the MSI score are dichotomised at their cohort
medians; TMB is additionally analysed as a continuous variable and at 10 mut/Mb.
Patients with an unknown or non-evaluable value are excluded from the
corresponding comparison. All subgroup analyses are exploratory.

Survival statistics use [lifelines](https://lifelines.readthedocs.io); the test
suite checks them against R's `survival` package (`survfit` with
`conf.type = "log-log"`, `survdiff`, `coxph`).

## Data availability

The patient-level data are not included in this repository because of data
protection regulations; they are available from the corresponding author on
reasonable request. The analysis expects two Excel workbooks in `data/` (or in
the directory given by the environment variable `EGA_MTB_DATA_DIR`), with one row
per patient. Only the columns listed below are read; codes are matched without
regard to case, and any value not listed stops the analysis with an error.

`clinical.xlsx`

| Column | Content | Values |
|---|---|---|
| `ID neu` | pseudonymised patient number | trailing `*` marks WES |
| `Geschlecht` | sex | `M`, `W` (female) |
| `Alter bei Eingang` | age at molecular testing, years | integer |
| `Lokalisation` | tumor localization | `Ösophagus`, `AEG` (esophagogastric junction), `Magen` (stomach) |
| `cM (ED)` | M-status at primary diagnosis | `0`, `1`, `x` (unknown) |
| `OP?` | prior surgical resection with curative intent | `ja`, `nein`, `unklar` (unknown) |
| `HER2` | HER2 IHC score | `0`–`3`, `x` |
| `MSI-h` | MMR/MSI status from the health record | `MSS`, `x` |
| `PD-L1` | PD-L1 combined positivity score | `0`, `1-4`, `5-9`, `10+`, `x` |
| `Claudin 18.2` | Claudin 18.2 status | `0` (negative), `1` (positive), `x` |
| `Therapielinie zum Zeitpunkt der Diagnose` | line of therapy at molecular testing | `0` (curative setting) to `5` |
| `Eingang-Ausgang` | days from sample receipt to molecular report | integer |
| `molTB durchgeführt?` | presented at the molecular tumor board (MTB) | `ja`, `nein` |
| `molTb-Eingang` | days from sample receipt to MTB presentation | integer; `NA` if not presented |
| `Empfehlung aus Basis TSO500` | MTB therapy recommendation based on TSO500 or WES | `ja`, `keine` (none), `NA` |
| `Therapieeinleitung aufgrund molTB oder TSO 500 erfolgt?` | targeted therapy initiated | `ja`, `nein` |
| `Verstorben?` | deceased | `ja`, `nein` |
| `d Eingang-Versterbedatum/Zensur` | days from sample receipt to death or last contact | integer |

`alterations.xlsx` (rows without a patient number, such as a legend, are ignored)

| Column | Content | Values |
|---|---|---|
| `ID_neu` | patient number | as in `clinical.xlsx`, without `*` |
| `Alteration 1`–`7`, `Mutation Type 1`–`7` | gene and type of each functionally relevant small variant | `Missense-Mutation`, `Nonsense-Mutation`, `Frameshift-Mutation`, `Deletion` (small in-frame deletion), `Splice-Site-Mutation` |
| `CNV Gain`, `CNV Gain 1`–`6` | copy-number gain status, and the genes gained | `0` none, `1` found, `2` none found but tumor cell content <40%, `3` found at tumor cell content <40% |
| `CNV Loss`, `CNV Loss 1`–`3` | copy-number loss status, and the genes lost | as for `CNV Gain` |
| `TMB verwertbar?`, `TMB` | TMB evaluable; TMB in mut/Mb | `ja`, `nein`; decimal comma or point |
| `MSI Score verwertbar`, `MSI-Score` | MSI score evaluable; % unstable microsatellite sites | `ja`, `nein`; decimal comma or point |
| `Fusion` | gene fusion | `GENE1(exon)::GENE2(exon)`; any other entry means none |

## Usage

Requires [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and the
locked dependencies. Writing figure files needs a Chrome installation for
kaleido; if none is found, run `uv run plotly_get_chrome` once.

```bash
uv sync                                              # install
uv run marimo edit analysis.py                       # interactive notebook
uv run marimo export html analysis.py -o output/analysis.html   # static report
uv run export-figures                                # figures -> output/ (PDF, 600-dpi PNG)
uv run pytest                                        # tests
```

Tests that need the study data are skipped when it is absent; the others run on
a small synthetic cohort and on R reference values.

## Repository layout

```
analysis.py           marimo notebook: Table 1, all figures, Results statistics
src/ega_mtb/
    data.py           loading and validation of the two workbooks
    stats.py          Kaplan–Meier, log-rank, Cox, follow-up, patient groups
    tables.py         Table 1
    figures.py        Figures 1–4, Supplementary Figures 1–2
    export.py         writes all figures to output/
tests/                pytest suite
```

## License

MIT — see [LICENSE](LICENSE).
