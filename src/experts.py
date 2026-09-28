"""Point-forecast experts (h=1) and a generic real-time backtest loop.

Every expert takes the same real-time-safe (X_train, y_train, x_query) triple from panel.build_row
and returns a single point forecast in level units (mln USD). Ridge is fit on log-level targets;
the exp() back-transform is a stated modelling choice (median under symmetric log-space errors),
matching the original proposal.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import panel
import realtime

# Tuned by tune_ridge.py: grid search over alpha in {0.01,...,100}, dev-period MAE only
# (results/ridge_alpha_tuning.json). E1 and E2 are tuned separately since expanding vs. rolling
# training-set size changes the bias-variance trade-off the penalty has to correct for.
EXPANDING_RIDGE_ALPHA = 0.1
ROLLING_RIDGE_ALPHA = 1.0


def expanding_ridge(X_train, y_train, x_query, alpha=EXPANDING_RIDGE_ALPHA, **_ignored):
    if len(X_train) < 13:
        return np.exp(y_train.iloc[-1]) if len(y_train) else np.nan
    model = Ridge(alpha=alpha).fit(X_train, y_train)
    return float(np.exp(model.predict(x_query)[0]))


def rolling_ridge(X_train, y_train, x_query, window, alpha=ROLLING_RIDGE_ALPHA, **_ignored):
    Xw, yw = X_train.iloc[-window:], y_train.iloc[-window:]
    return expanding_ridge(Xw, yw, x_query, alpha=alpha)


def seasonal_naive(X_train, y_train, x_query, target_month, **_ignored):
    prior_year = target_month - 12
    if prior_year in y_train.index:
        return float(np.exp(y_train.loc[prior_year]))
    return float(np.exp(y_train.iloc[-1])) if len(y_train) else np.nan


def backtest(issue_dates, forecast_fn, include_covariates=False, **kwargs):
    """forecast_fn(X_train, y_train, x_query, target_month, **kwargs) -> point forecast (level).
    Returns a DataFrame indexed by issue_date with columns target_month, forecast, actual."""
    truth = panel.first_release_series()
    rows = []
    for v in issue_dates:
        try:
            X, y, q, target_month = panel.build_row(v, include_covariates=include_covariates)
        except AssertionError:
            continue  # not enough lag history yet
        if len(y) < 13:
            continue  # not enough training history for a meaningful fit
        pred = forecast_fn(X, y, q, target_month=target_month, **kwargs)
        actual = truth.get(target_month, np.nan)
        rows.append(dict(issue_date=v, target_month=target_month, forecast=pred, actual=actual))
    out = pd.DataFrame(rows).set_index("issue_date")
    out["abs_err"] = (out.forecast - out.actual).abs()
    return out


DEV_DATES = realtime.issue_dates("2006-02", "2021-12")
EVAL_DATES = realtime.issue_dates("2022-01", "2026-06")
