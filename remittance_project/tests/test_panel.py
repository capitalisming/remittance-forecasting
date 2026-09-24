import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import realtime  # noqa: E402
import panel  # noqa: E402

ALL_DATES = list(realtime.issue_dates("2005-03", "2026-06"))
USABLE_DATES = []
for _v in ALL_DATES:
    try:
        panel.build_row(_v)
        USABLE_DATES.append(_v)
    except AssertionError:
        pass  # not enough lag history yet (only true near the very start of the series)


def test_most_dates_have_enough_history():
    assert len(USABLE_DATES) >= len(ALL_DATES) - 15


@pytest.mark.parametrize("v", USABLE_DATES, ids=lambda d: d.strftime("%Y-%m"))
def test_target_month_excluded_from_training_rows(v):
    X, y, q, target_month = panel.build_row(v)
    assert target_month not in X.index
    assert target_month not in y.index


@pytest.mark.parametrize("v", USABLE_DATES, ids=lambda d: d.strftime("%Y-%m"))
def test_no_training_row_is_after_issue_date(v):
    X, y, q, target_month = panel.build_row(v)
    if len(X):
        assert X.index.max() <= v.to_period("M")
        assert y.index.max() <= v.to_period("M")


@pytest.mark.parametrize("v", USABLE_DATES, ids=lambda d: d.strftime("%Y-%m"))
def test_shapes_and_finiteness(v):
    X, y, q, target_month = panel.build_row(v)
    assert len(X) == len(y)
    assert list(X.columns) == list(q.columns)
    assert np.isfinite(X.to_numpy()).all()
    assert np.isfinite(y.to_numpy()).all()
    assert np.isfinite(q.to_numpy()).all()


def test_target_is_h1_ahead():
    for v in USABLE_DATES[::11]:
        _, _, _, target_month = panel.build_row(v)
        assert target_month == v.to_period("M") + 1


def test_month_dummies_are_one_hot_with_december_dropped():
    X, y, q, target_month = panel.build_row(USABLE_DATES[50])
    month_cols = [c for c in X.columns if c.startswith("month_")]
    assert len(month_cols) == 11
    assert (X[month_cols].sum(axis=1) <= 1).all()


def test_stale_issue_dates_still_produce_a_query_row():
    import pandas as pd
    for v in (pd.Timestamp("2023-12-31"), pd.Timestamp("2024-01-31")):
        X, y, q, target_month = panel.build_row(v)
        assert len(q) == 1
        assert y.index.max() == pd.Period("2023-09", "M")
