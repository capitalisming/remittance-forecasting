import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import realtime  # noqa: E402

ISSUE_DATES = list(realtime.issue_dates())
index, wide, regime, first_vintage = realtime._load()


@pytest.mark.parametrize("v", ISSUE_DATES, ids=lambda d: d.strftime("%Y-%m"))
def test_no_month_after_publication_cutoff(v):
    known = realtime.info_set(v)
    if len(known):
        assert known.index.max() <= v.to_period("M") - 2


@pytest.mark.parametrize("v", ISSUE_DATES, ids=lambda d: d.strftime("%Y-%m"))
def test_unfilled_values_come_from_vintages_published_by_v(v):
    known = realtime.info_set(v)
    direct = known[~known.filled]
    assert (direct.source_vintage <= v).all()


@pytest.mark.parametrize("v", ISSUE_DATES, ids=lambda d: d.strftime("%Y-%m"))
def test_filled_values_respect_lag_and_regime(v):
    known = realtime.info_set(v)
    filled = known[known.filled]
    base = realtime.base_vintage(v)
    base_regime = regime[base] if base is not None else "mts"
    assert (filled.index <= v.to_period("M") - 2).all()
    assert all(regime[s] == base_regime for s in filled.source_vintage)


def test_known_months_never_disappear():
    previous = set()
    for v in ISSUE_DATES:
        current = set(realtime.info_set(v).index)
        assert previous <= current, f"months lost at {v.date()}: {sorted(previous - current)[:3]}"
        previous = current


@pytest.mark.parametrize("v", ISSUE_DATES[::7], ids=lambda d: d.strftime("%Y-%m"))
def test_covariates_respect_publication_lag(v):
    for name, series in realtime.covariates_at(v).items():
        assert series.index.max() <= v.to_period("M") - realtime.COVARIATE_LAG[name]


def test_values_match_source_vintage():
    for v in ISSUE_DATES[::5]:
        known = realtime.info_set(v)
        for m, row in known.sample(min(5, len(known)), random_state=0).iterrows():
            assert wide.at[m, row.source_vintage] == pytest.approx(row.value)


def test_no_regime_leakage_before_redefinition():
    """Direct reviewer challenge: does info_set(v) for v before the 2024-02-14 redefinition
    ever return a value sourced from a cross_border-regime vintage (i.e. does the forecaster
    at v see a retroactively-redefined number they could not have known)?"""
    first_cross_border = index.loc[index.regime == "cross_border", "vintage_date"].min()
    checked = 0
    for v in ISSUE_DATES:
        if v >= first_cross_border:
            continue
        known = realtime.info_set(v)
        if known.empty:
            continue
        regimes_seen = {regime[s] for s in known.source_vintage}
        assert regimes_seen <= {"mts"}, f"{v.date()}: saw regimes {regimes_seen}, expected only mts"
        checked += 1
    assert checked > 0
    print(f"\nchecked {checked} pre-redefinition issue dates, all mts-only")
