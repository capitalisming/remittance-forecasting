"""Stage 3: real-time information set for the Armenia (CBA) money-transfer series.

Simpler than realtime.py's NBKR logic because the series is not revised after publication
(spot-checked: the March-2021 value is byte-identical whether read from the March-2021 issue or
from the March-2022 issue's year-ago column -- see commit message for
parse_cba_armenia.py). So "known at issue date v" just needs a fixed publication lag, not a
vintage/revision model.

Lag assumption: the bulletin for month m already reports month m's own transfers (e.g. the
"March 2022" bulletin's 2022 table has a March column), so the effective lag is however long
after month-end the bulletin itself takes to compile and publish. We do not have an official
release calendar for this specific table (unlike the BPM6 quarterly remittances series, whose
CBA metadata page states a documented 3-month lag) and could not reach the CBA site to check
(network-restricted sandbox); ARMENIA_LAG_MONTHS=1 is a conservative placeholder -- month m is
treated as known starting at issue date = end of month m+1 -- consistent with typical monthly
external-sector reporting lags and explicitly flagged as an assumption, not a measured fact, in
the paper.
"""
import os
from functools import lru_cache

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")
ARMENIA_LAG_MONTHS = 1


@lru_cache(maxsize=None)
def _series():
    df = pd.read_csv(os.path.join(PROC, "armenia_total_transfers_monthly.csv"))
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    return df.set_index("month")["total_inflow_usd_thousand"].sort_index()


def info_set(v):
    """Armenia transfer values (thousand USD) known at issue date v: every parsed month up to
    and including v_period - ARMENIA_LAG_MONTHS. Months the archive is missing (corrupted or
    never-downloaded bulletins, see armenia_corrupted_months.txt) are simply absent -- there is
    no reconstruction fallback the way NBKR's archive-gap-fill rule provides one."""
    s = _series()
    cutoff = pd.Timestamp(v).to_period("M") - ARMENIA_LAG_MONTHS
    return s[s.index <= cutoff]


def issue_dates(start, end):
    return pd.period_range(start, end, freq="M").to_timestamp(how="end").normalize()
