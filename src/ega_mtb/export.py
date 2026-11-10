"""Write all figures to ``output/``: vector PDF and 600-dpi PNG.

Run with ``uv run export-figures``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import plotly.graph_objects as go

from ega_mtb import figures
from ega_mtb.data import PROJECT_ROOT, Cohort, load_cohort

OUTPUT_DIR = PROJECT_ROOT / "output"

#: Output name -> (figure builder, printed width in mm).
FIGURES: dict[str, tuple[Callable[[Cohort], go.Figure], float]] = {
    "figure1_patient_selection": (figures.figure1, 174),
    "figure2_molecular_landscape": (figures.figure2, 174),
    "figure3_clinical_impact": (figures.figure3, 174),
    "figure4_overall_survival": (figures.figure4, 174),
    "figureS1_survival_by_therapy_line": (figures.figure_s1, 84),
    "figureS2_survival_by_biomarker": (figures.figure_s2, 174),
}

PNG_DPI = 600
_PT_PER_PX = 0.75  # kaleido maps one layout pixel to 0.75 pt in vector output
_PT_PER_MM = 72 / 25.4


def save_figure(fig: go.Figure, stem: Path, width_mm: float) -> list[Path]:
    """Write ``stem``.pdf and ``stem``.png, scaled to ``width_mm`` printed width."""
    layout_width = float(fig.layout.width)
    pdf, png = stem.with_suffix(".pdf"), stem.with_suffix(".png")
    fig.write_image(pdf, scale=width_mm * _PT_PER_MM / (_PT_PER_PX * layout_width))
    fig.write_image(png, scale=width_mm / 25.4 * PNG_DPI / layout_width)
    return [pdf, png]


def main() -> None:
    cohort = load_cohort()
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name, (build, width_mm) in FIGURES.items():
        for path in save_figure(build(cohort), OUTPUT_DIR / name, width_mm):
            print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
