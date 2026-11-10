"""Survival statistics against reference values from R.

Reference values were computed with R 4.4.3 / survival 3.8.3 on the classic
``aml`` data set (acute myelogenous leukemia, maintained vs non-maintained
chemotherapy): ``survfit(..., conf.type = "log-log")``, ``survdiff`` and
``coxph`` (Efron ties).
"""

from __future__ import annotations

import numpy as np
import pytest

from ega_mtb import stats

# survival::aml — time (weeks), status (1 = relapse); the first 11 patients
# received maintenance chemotherapy, the other 12 did not.
TIME = np.array(
    [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161]  # maintained
    + [5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45]  # non-maintained
)
STATUS = np.array([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0] + [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1])
NONMAINTAINED = np.array([0] * 11 + [1] * 12)
# An arbitrary continuous covariate, and three arbitrary groups.
Z = np.array(
    [3.1, 0.5, 2.2, 4.8, 1.0, 2.9, 0.7, 3.6, 1.8, 2.4, 0.2, 4.1]
    + [1.5, 3.3, 2.0, 0.9, 4.4, 2.7, 1.2, 3.8, 0.4, 2.5, 1.7]
)
THREE_GROUPS = np.arange(len(TIME)) % 3


def _groups(labels: np.ndarray) -> list[stats.Group]:
    return [stats.Group(str(k), TIME[labels == k], STATUS[labels == k]) for k in np.unique(labels)]


def test_km_median_with_log_log_ci():
    assert stats.km_median(TIME, STATUS) == stats.Median(
        n=23, events=18, median=27, ci_low=13, ci_high=34
    )


def test_km_median_ci_bound_not_reached():
    maintained = stats.km_median(TIME[:11], STATUS[:11])
    assert (maintained.median, maintained.ci_low, maintained.ci_high) == (31, 13, None)
    nonmaintained = stats.km_median(TIME[11:], STATUS[11:])
    assert (nonmaintained.median, nonmaintained.ci_low, nonmaintained.ci_high) == (23, 5, 33)


def test_km_median_not_reached():
    median = stats.km_median(np.array([5, 10, 15]), np.array([0, 0, 0]))
    assert (median.median, median.ci_low, median.ci_high) == (None, None, None)
    assert median.describe() == "NR months (95% CI NR–NR)"


def test_reverse_km_follow_up():
    follow_up = stats.follow_up_median(TIME, STATUS)
    assert (follow_up.events, follow_up.median, follow_up.ci_low, follow_up.ci_high) == (
        5,
        161,
        28,
        None,
    )


def test_logrank_two_groups():
    assert stats.logrank_p(_groups(NONMAINTAINED)) == pytest.approx(0.0653393220, abs=1e-9)


def test_logrank_three_groups():
    assert stats.logrank_p(_groups(THREE_GROUPS)) == pytest.approx(0.6516597086, abs=1e-9)


# lifelines and R stop their Newton–Raphson iterations at slightly different
# tolerances, so Cox estimates agree to about six significant digits.
COX_TOLERANCE = 1e-5


def test_cox_binary_covariate():
    hr = stats.cox_hazard_ratio(TIME, STATUS, NONMAINTAINED)
    expected = (2.4981053262, 0.9159072574, 6.8134957661, 0.0737148606)
    assert (hr.hr, hr.ci_low, hr.ci_high, hr.p) == pytest.approx(expected, rel=COX_TOLERANCE)


def test_cox_continuous_covariate():
    hr = stats.cox_hazard_ratio(TIME, STATUS, Z)
    expected = (1.2174001298, 0.8290790378, 1.7876016743, 0.3155476019)
    assert (hr.hr, hr.ci_low, hr.ci_high, hr.p) == pytest.approx(expected, rel=COX_TOLERANCE)


def test_group_hazard_ratio_matches_binary_cox():
    maintained, nonmaintained = _groups(NONMAINTAINED)
    assert stats.group_hazard_ratio(maintained, nonmaintained) == stats.cox_hazard_ratio(
        TIME, STATUS, NONMAINTAINED
    )


def test_median_iqr_matches_r_type_7():
    assert stats.median_iqr(TIME) == (23, 12.5, 33.5)


def test_formatting():
    median = stats.Median(n=77, events=52, median=239, ci_low=168, ci_high=321)
    assert median.describe() == "7.9 months (95% CI 5.5–10.5)"
    hr = stats.HazardRatio(hr=0.88620, ci_low=0.79425, ci_high=0.98880, p=0.03066)
    assert hr.describe() == "HR 0.89 (95% CI 0.79–0.99), P = 0.031"


def test_groups_exclude_missing_values(cohort):
    p = cohort.patients
    low, high = stats.by_cutoff(p, "tmb", 10, ("TMB <10", "TMB ≥10"))
    assert (low.n, high.n) == (4, 2)  # two patients without an evaluable TMB
    assert stats.cohort_median(p, "tmb") == 6.25
    assert [g.label for g in stats.by_median(p, "tmb", "TMB")] == ["TMB low", "TMB high"]
    assert [g.n for g in stats.by_her2(p)] == [2, 4]
    assert [g.n for g in stats.by_pdl1(p)] == [2, 1, 1, 2]
    assert [g.n for g in stats.by_claudin(p)] == [2, 3]
    assert [g.n for g in stats.by_therapy_line(p)] == [3, 5]
    assert stats.overall(p)[0].n == 8
