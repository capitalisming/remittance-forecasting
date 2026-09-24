"""Real-time information sets for the NBKR remittance series.

Rule for what a forecaster knows at issue date v (a month end):
  1. Base vintage B(v): the latest archived NBKR file whose upload date is <= v.
     Every month it contains is known, with the value it shows.
  2. Archive-gap fill: a month m <= month(v) - 2 missing from B(v) is also known, with the value
     from the earliest archived vintage that contains m, provided that vintage uses the same
     series definition as B(v). Justification: observed publication lags are 35-54 days, so month
     m is always published before the end of month m+2, and within a definition regime values are
     almost never revised afterwards (see revision check). Vintages the Wayback archive missed
     therefore do not make the forecaster look staler than they were.
  3. Before the first archived vintage (2015-12-16) the base regime is money-transfer systems and
     all months are known through rule 2 (publication-lag reconstruction for early history).
Months after month(v) - 2 are never known.
"""
import os
from functools import lru_cache

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
KNOWN_LAG_MONTHS = 2


@lru_cache(maxsize=None)
def _load():
    index = pd.read_csv(os.path.join(PROC, "vintage_index.csv"), parse_dates=["vintage_date"])
    index = index.sort_values("vintage_date").reset_index(drop=True)
    long = pd.read_csv(os.path.join(PROC, "vintages_long.csv"), parse_dates=["vintage_date"])
    long["ref_month"] = pd.PeriodIndex(long["ref_month"], freq="M")
    wide = long.pivot(index="ref_month", columns="vintage_date", values="inflow").sort_index()
    regime = dict(zip(index.vintage_date, index.regime))
    # earliest archived vintage containing each month
    first_vintage = wide.notna().idxmax(axis=1)
    return index, wide, regime, first_vintage


def base_vintage(v):
    """Latest archived vintage uploaded on or before issue date v (None before the archive starts)."""
    index, _, _, _ = _load()
    avail = index[index.vintage_date <= pd.Timestamp(v)]
    return None if avail.empty else avail.vintage_date.iloc[-1]


def info_set(v):
    """Remittance inflow (mln USD) known at issue date v.

    Returns a DataFrame indexed by reference month (Period) with columns
    value, source_vintage, filled.
    """
    _, wide, regime, first_vintage = _load()
    v = pd.Timestamp(v)
    cutoff = v.to_period("M") - KNOWN_LAG_MONTHS
    base = base_vintage(v)
    base_regime = regime[base] if base is not None else "mts"

    rows = {}
    if base is not None:
        col = wide[base].dropna()
        for m, val in col.items():
            rows[m] = (val, base, False)
    for m in wide.index[wide.index <= cutoff]:
        if m in rows:
            continue
        src = first_vintage[m]
        if regime[src] == base_regime:
            rows[m] = (wide.at[m, src], src, True)
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["value", "source_vintage", "filled"])
    out.index.name = "ref_month"
    return out.sort_index()


def issue_dates(start="2005-03", end="2026-08"):
    return pd.period_range(start, end, freq="M").to_timestamp(how="end").normalize()


@lru_cache(maxsize=None)
def release_table():
    """For every month: the first issue date at which it is known and its value at that moment
    (the real-time 'first release'), plus its value in the latest vintage."""
    _, wide, _, _ = _load()
    first_known, first_value = {}, {}
    for v in issue_dates():
        known = info_set(v)
        for m, row in known.iterrows():
            if m not in first_known:
                first_known[m] = v
                first_value[m] = row["value"]
    latest = wide[wide.columns.max()]
    out = pd.DataFrame({"first_known_issue_date": pd.Series(first_known),
                        "first_release": pd.Series(first_value),
                        "latest": latest}).sort_index()
    out.index.name = "ref_month"
    return out


# ---- covariates -------------------------------------------------------------------------------
# Availability at issue date v (month end), in months back from month(v):
#   rub_per_usd_eom  official CBR rate on the last day of the month -> known at v (lag 0)
#   imoex_close      MOEX index close of the month                  -> known at v (lag 0)
#   brent_avg        EIA monthly average spot price, published after the month -> lag 1
COVARIATE_LAG = {"rub_per_usd_eom": 0, "imoex_close": 0, "brent_avg": 1}


@lru_cache(maxsize=None)
def _covariates():
    path = os.path.join(PROC, "covariates_monthly.csv")
    cov = pd.read_csv(path, index_col=0)
    cov.index = pd.PeriodIndex(cov.index, freq="M")
    return cov


def covariates_at(v):
    """Covariate values known at issue date v, each at its own publication lag."""
    cov = _covariates()
    m = pd.Timestamp(v).to_period("M")
    out = {}
    for name, lag in COVARIATE_LAG.items():
        s = cov[name].dropna()
        s = s[s.index <= m - lag]
        out[name] = s
    return out
