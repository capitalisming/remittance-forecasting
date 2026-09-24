"""Build the real-time feature/target panel every expert reads from.

Ground truth = first release (the value in the first vintage that ever reported that month),
per the regime-consistency argument: evaluating against the final, retroactively-redefined
value would require a real-time forecaster to predict a future change of methodology.

Scope decision (2026-09-24): no covariates in the MIDI pilot. FX/Brent/IMOEX were tried, but a
lagged covariate (Brent, published ~1 month after its reference month) cannot be computed
without leakage for the row being forecast (t = v+1): "log-change at lag 1" needs the covariate
value at t-1 = month(v), which is not yet published when the covariate's own lag is >=1. Fixing
this properly needs per-covariate row-specific lag handling; deferred to the covariate stage
(Stage "multimodal signals" in the original 12-week proposal), out of scope for the pilot.
Pure autoregressive + seasonal ridge below.
"""
import numpy as np
import pandas as pd

import realtime

TRANSITION_ZONE = pd.period_range("2023-10", "2024-01", freq="M")  # excluded from the main window
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


def build_row(v):
    """One issue date v -> (X_train, y_train, x_query, target_month).

    X_train/y_train: every past month whose lag features are fully known at v.
    x_query: feature row for target_month = month(v)+1, the h=1 forecast we must produce.
    """
    hist = realtime.info_set(v)["value"]
    target_month = v.to_period("M") + 1
    idx = hist.index.union([target_month])
    extended = hist.reindex(idx)
    feats = make_features(extended)

    query = feats.loc[[target_month]].drop(columns=["target"])
    assert not query.isna().any().any(), f"query row for {target_month} has missing lags at v={v.date()}"

    train = feats.drop(index=target_month).dropna()
    return train.drop(columns=["target"]), train["target"], query, target_month


def is_in_transition_zone(target_month) -> bool:
    return target_month in TRANSITION_ZONE
