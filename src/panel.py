"""Build the real-time feature/target panel every expert reads from.

Ground truth = first release (the value in the first vintage that ever reported that month),
per the regime-consistency argument: evaluating against the final, retroactively-redefined
value would require a real-time forecaster to predict a future change of methodology.

Covariates (Stage 2, 2026-09-25): FX/Brent/IMOEX are wired in as optional features
(`include_covariates=True`) via `realtime.covariates_at(v)`, which already handles each
covariate's own publication lag leakage-safely (see realtime.py and
test_covariates_respect_publication_lag). The earlier "2026-09-24" scope note above this said
per-covariate lag handling didn't exist; it did (covariates_at was already built and tested),
it just wasn't wired into the feature matrix. The fix: shift each covariate's log by
(1 + its own lag) before differencing, so the query row (target_month = v+1) always uses the
freshest value actually known at v -- e.g. Brent (lag 1) contributes its month-(v-1) log-change,
not month-v, which would not yet be published. Default is unchanged (no covariates) so the
original E1/E2/fixed-share pipeline and its results are untouched; covariates are an additive
comparison, not a replacement.
"""
import numpy as np
import pandas as pd

import realtime

TRANSITION_ZONE = pd.period_range("2023-10", "2024-01", freq="M")  # excluded from the main window

# Effective feature lag = 1 (the usual "most recent complete month") + the covariate's own
# publication lag, so the query row for target_month=v+1 never reaches for an unpublished value.
COVARIATE_FEATURE_LAG = {name: 1 + lag for name, lag in realtime.COVARIATE_LAG.items()}

# Fixed a priori (lags 1-3 for short-run momentum, lag 12 for year-over-year seasonality),
# not grid-searched, unlike alpha/w/eta elsewhere in the pipeline. A post hoc dev-only
# sensitivity check across 7 alternatives (src/tune_lags.py, results/lag_selection.json) finds
# a marginally better candidate (1,3,6,12); the gap is small enough (6.3%/2.7% dev MAE for
# E1/E2) relative to the paper's effect sizes that we keep this set frozen rather than
# re-tuning everything downstream against it.
LAGS = (1, 2, 3, 12)


def first_release_series():
    return realtime.release_table()["first_release"].rename("y")


def make_features(history: pd.Series) -> pd.DataFrame:
    """history: index = Period[M], values = mln USD level. Target column is log(history);
    rows with any missing lag are NOT dropped here (the query row needs to survive), the caller
    decides which rows are usable for training vs. which one is the forecast query."""
    z = np.log(history)
    df = pd.DataFrame(index=history.index)
    for lag in LAGS:
        df[f"log_y_lag{lag}"] = z.shift(lag)
    for m in range(1, 12):
        df[f"month_{m}"] = (df.index.month == m).astype(float)
    df["target"] = z
    return df


def make_covariate_features(v, idx) -> pd.DataFrame:
    """Leakage-safe covariate features aligned to `idx` (the same ref_month index as the
    remittance panel). For each covariate, a one-month log-change, shifted by its own
    COVARIATE_FEATURE_LAG: the value used at row-month m is always known at v for any m <=
    target_month, because covariates_at(v) already truncates each series to what publication
    lag allows, and the shift/diff below never looks past that truncation."""
    cov = realtime.covariates_at(v)
    cols = {}
    for name, series in cov.items():
        # Reindex onto the full panel index (extending past the covariate's own truncation
        # cutoff, e.g. up to target_month) *before* shifting, so shift() has a row to place the
        # value in for every month in idx -- shift() only moves values within an existing index,
        # it does not extend it, so shifting the truncated series first would silently drop the
        # very rows (including the query row) this feature is meant to fill in.
        z = np.log(series).reindex(idx)
        shifted = z.shift(COVARIATE_FEATURE_LAG[name])
        cols[f"cov_{name}_chg"] = shifted.diff()
    return pd.DataFrame(cols, index=idx)


def build_row(v, include_covariates=False):
    """One issue date v -> (X_train, y_train, x_query, target_month).

    X_train/y_train: every past month whose lag features are fully known at v.
    x_query: feature row for target_month = month(v)+1, the h=1 forecast we must produce.
    """
    hist = realtime.info_set(v)["value"]
    target_month = v.to_period("M") + 1
    idx = hist.index.union([target_month])
    extended = hist.reindex(idx)
    feats = make_features(extended)

    if include_covariates:
        feats = feats.join(make_covariate_features(v, idx))

    query = feats.loc[[target_month]].drop(columns=["target"])
    assert not query.isna().any().any(), f"query row for {target_month} has missing lags at v={v.date()}"

    train = feats.drop(index=target_month).dropna()
    return train.drop(columns=["target"]), train["target"], query, target_month


def is_in_transition_zone(target_month) -> bool:
    return target_month in TRANSITION_ZONE
