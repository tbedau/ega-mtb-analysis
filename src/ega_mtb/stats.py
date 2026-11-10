"""Survival analysis and descriptive statistics.

Kaplan–Meier estimates, log-rank tests and Cox proportional-hazards models are
computed with lifelines. The median survival time is the first time at which
S(t) ≤ 0.5; its 95% confidence limits are read off the log–log (exponential
Greenwood) pointwise confidence band (Brookmeyer–Crowley). Cox models handle tied
event times with Efron's method. The test suite checks these against R's
survival package.

Overall survival (OS) runs from molecular testing (sample receipt) to death from
any cause; patients alive at last contact are censored. Times are in days.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from ega_mtb.data import DAYS_PER_MONTH


@dataclass(frozen=True)
class Group:
    """Survival data of one patient group: days and events (1 = death)."""

    label: str
    days: np.ndarray
    events: np.ndarray

    @property
    def n(self) -> int:
        return len(self.days)


@dataclass(frozen=True)
class Median:
    """Kaplan–Meier median with 95% CI, in days (``None`` = not reached)."""

    n: int
    events: int
    median: float | None
    ci_low: float | None
    ci_high: float | None

    def describe(self) -> str:
        """For example ``7.9 months (95% CI 5.5–10.5)``."""
        return f"{months(self.median)} months (95% CI {months(self.ci_low)}–{months(self.ci_high)})"


@dataclass(frozen=True)
class HazardRatio:
    """Cox hazard ratio with 95% CI and two-sided Wald P value."""

    hr: float
    ci_low: float
    ci_high: float
    p: float

    def describe(self) -> str:
        """For example ``HR 0.89 (95% CI 0.79–0.99), P = 0.031``."""
        return f"HR {self.hr:.2f} (95% CI {self.ci_low:.2f}–{self.ci_high:.2f}), P = {self.p:.3f}"


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #


def kaplan_meier(days: np.ndarray, events: np.ndarray) -> KaplanMeierFitter:
    """Fitted Kaplan–Meier estimator (log–log 95% confidence band)."""
    return KaplanMeierFitter().fit(np.asarray(days, float), np.asarray(events, int))


def km_median(days: np.ndarray, events: np.ndarray) -> Median:
    """Kaplan–Meier median survival with its Brookmeyer–Crowley 95% CI."""
    km = kaplan_meier(days, events)
    survival = km.survival_function_[km.label]
    # The log–log band is undefined where S(t) = 0. lifelines reports 0 there,
    # which would turn an unreached confidence limit into a finite one; R
    # reports it as missing, and so do we.
    defined = survival > 0
    band = km.confidence_interval_
    return Median(
        n=len(days),
        events=int(np.sum(events)),
        median=_first_at_or_below_half(survival),
        ci_low=_first_at_or_below_half(band[f"{km.label}_lower_0.95"].where(defined)),
        ci_high=_first_at_or_below_half(band[f"{km.label}_upper_0.95"].where(defined)),
    )


def follow_up_median(days: np.ndarray, events: np.ndarray) -> Median:
    """Median follow-up by the reverse Kaplan–Meier method.

    Censoring is treated as the event and deaths as censored observations, so
    ``Median.events`` counts the patients alive at last contact.
    """
    return km_median(days, 1 - np.asarray(events, int))


def logrank_p(groups: Sequence[Group]) -> float:
    """Two-sided log-rank P value across two or more groups."""
    days = np.concatenate([g.days for g in groups])
    events = np.concatenate([g.events for g in groups])
    labels = np.repeat(np.arange(len(groups)), [g.n for g in groups])
    return float(multivariate_logrank_test(days, labels, events).p_value)


def cox_hazard_ratio(days: np.ndarray, events: np.ndarray, covariate: np.ndarray) -> HazardRatio:
    """Univariable Cox model; hazard ratio per unit increase of ``covariate``."""
    frame = pd.DataFrame(
        {
            "days": np.asarray(days, float),
            "event": np.asarray(events, int),
            "x": np.asarray(covariate, float),
        }
    )
    row = CoxPHFitter().fit(frame, duration_col="days", event_col="event").summary.loc["x"]
    return HazardRatio(
        hr=float(row["exp(coef)"]),
        ci_low=float(row["exp(coef) lower 95%"]),
        ci_high=float(row["exp(coef) upper 95%"]),
        p=float(row["p"]),
    )


def group_hazard_ratio(reference: Group, comparison: Group) -> HazardRatio:
    """Cox hazard ratio of ``comparison`` relative to ``reference``."""
    return cox_hazard_ratio(
        np.concatenate([reference.days, comparison.days]),
        np.concatenate([reference.events, comparison.events]),
        np.repeat([0, 1], [reference.n, comparison.n]),
    )


def median_iqr(values: np.ndarray) -> tuple[float, float, float]:
    """Median, first and third quartile (linear interpolation, as R's default)."""
    q1, median, q3 = np.percentile(np.asarray(values, float), [25, 50, 75])
    return float(median), float(q1), float(q3)


def months(days: float | None) -> str:
    """Days formatted as months with one decimal; ``NR`` = not reached."""
    return "NR" if days is None else f"{days / DAYS_PER_MONTH:.1f}"


def _first_at_or_below_half(curve: pd.Series) -> float | None:
    """First time at which ``curve`` is ≤ 0.5; ``None`` if never (not reached)."""
    times = curve.index[curve.to_numpy() <= 0.5]
    return float(times[0]) if len(times) else None


# --------------------------------------------------------------------------- #
# Patient groups (Figure 4, Supplementary Figures 1–2, Section 3.4)
# --------------------------------------------------------------------------- #


def survival_group(patients: pl.DataFrame, label: str, where: pl.Expr | None = None) -> Group:
    """Patients matching ``where`` (all patients if ``None``) as a survival group."""
    selected = patients if where is None else patients.filter(where)
    return Group(label, selected["os_days"].to_numpy(), selected["event"].to_numpy())


def overall(patients: pl.DataFrame) -> list[Group]:
    return [survival_group(patients, "All patients")]


def by_therapy_line(patients: pl.DataFrame) -> list[Group]:
    """Line of therapy at molecular testing: 0–1 vs ≥2 (0 = curative setting)."""
    return [
        survival_group(patients, "Line 0–1", pl.col("therapy_line") <= 1),
        survival_group(patients, "Line ≥2", pl.col("therapy_line") >= 2),
    ]


def by_cutoff(
    patients: pl.DataFrame, column: str, cutoff: float, labels: tuple[str, str]
) -> list[Group]:
    """``column`` below vs at or above ``cutoff``; patients without a value are excluded."""
    low, high = labels
    return [
        survival_group(patients, low, pl.col(column) < cutoff),
        survival_group(patients, high, pl.col(column) >= cutoff),
    ]


def cohort_median(patients: pl.DataFrame, column: str) -> float:
    """Median of ``column`` over the patients with an evaluable value."""
    return float(np.median(patients[column].drop_nulls().to_numpy()))


def by_median(patients: pl.DataFrame, column: str, name: str) -> list[Group]:
    """Dichotomised at the cohort median: ``{name} low`` vs ``{name} high``."""
    cutoff = cohort_median(patients, column)
    return by_cutoff(patients, column, cutoff, (f"{name} low", f"{name} high"))


def by_her2(patients: pl.DataFrame) -> list[Group]:
    """HER2 IHC 2+/3+ vs 0/1+; unknown status excluded."""
    return [
        survival_group(patients, "HER2 2+/3+", pl.col("her2").is_in(["2+", "3+"])),
        survival_group(patients, "HER2 0/1+", pl.col("her2").is_in(["0", "1+"])),
    ]


def by_pdl1(patients: pl.DataFrame) -> list[Group]:
    """PD-L1 CPS ≥10 vs 5–9 vs 1–4 vs 0; unknown status excluded."""
    return [
        survival_group(patients, f"PD-L1 {level}", pl.col("pdl1") == level)
        for level in ("≥10", "5–9", "1–4", "0")
    ]


def by_claudin(patients: pl.DataFrame) -> list[Group]:
    """Claudin 18.2 positive vs negative; unknown status excluded."""
    return [
        survival_group(patients, "Claudin 18.2 pos", pl.col("claudin") == "Positive"),
        survival_group(patients, "Claudin 18.2 neg", pl.col("claudin") == "Negative"),
    ]
