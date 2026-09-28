"""Stage 3: feature/target panel for the Armenia forecasting comparison, mirroring panel.py's
design (real-time-safe lag features, h=1) but with LAGS=(1,2,3) only -- no lag-12 seasonal term,
because the archive is missing all of 2020 and most of 2021 (see armenia_corrupted_months.txt),
so lag-12 features for the 2022 shock-window query rows (which need the corresponding 2021
months) would be NaN for exactly the rows this comparison cares about. Revisit once/if the
retry downloads fill that gap.
"""
import numpy as np
import pandas as pd

import armenia_realtime as art

LAGS = (1, 2, 3)


def make_features(history: pd.Series) -> pd.DataFrame:
    z = np.log(history)
    df = pd.DataFrame(index=history.index)
    for lag in LAGS:
        df[f"log_y_lag{lag}"] = z.shift(lag)
    for m in range(1, 12):
        df[f"month_{m}"] = (df.index.month == m).astype(float)
    df["target"] = z
    return df


def build_row(v):
    hist = art.info_set(v)
    target_month = v.to_period("M") + 1
    idx = hist.index.union([target_month])
    extended = hist.reindex(idx)
    feats = make_features(extended)

    query = feats.loc[[target_month]].drop(columns=["target"])
    if query.isna().any().any():
        raise AssertionError(f"query row for {target_month} has missing lags at v={v.date()}")

    train = feats.drop(index=target_month).dropna()
    return train.drop(columns=["target"]), train["target"], query, target_month


def first_release_series():
    return art._series().rename("y")
