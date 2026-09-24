"""Reviewer's catch #1: fixed-share vs naive/Chronos on the shock window was stated as "устойчиво"
without bootstrap, while fixed-share vs equal-weight got the full treatment. Close that double
standard by running the same moving-block bootstrap on all three shock-window comparisons.

Also addresses catch #3: report the number of valid block start positions at n=8 for each block
length, not just note that block=12 degenerates.
"""
import json

import numpy as np
import pandas as pd

import bootstrap as bs
import fixed_share as fs
import realtime

DEV_END, EVAL_END = "2021-12", "2026-06"


def main():
    cfg = json.load(open("../results/fixed_share_config.json"))
    eta, alpha, clip_c = cfg["eta"], cfg["alpha"], cfg["clip_c"]
    fs_start = pd.Timestamp(cfg["fs_start"])
    all_dates = list(realtime.issue_dates(fs_start.strftime("%Y-%m"), EVAL_END))
    eval_dates = [v for v in all_dates if v.strftime("%Y-%m") > DEV_END]

    archive = fs.precompute_forecasts(all_dates)
    truth = fs.panel.first_release_series()
    fixed = fs.simulate(archive, all_dates, eta=eta, alpha=alpha, clip_c=clip_c).loc[eval_dates]
    fixed = fixed[fixed.actual.notna()]

    shock_mask = fixed.target_month.between("2022-02", "2022-09")
    shock = fixed[shock_mask]
    print(f"shock window: n={len(shock)} target months {shock.target_month.tolist()}\n")

    for other in ("naive", "chronos"):
        other_err = np.array([abs(archive[m][other] - truth[m]) for m in shock.target_month])
        diff = shock.abs_err.to_numpy() - other_err
        print(f"--- fixed-share vs {other}, shock window ---")
        bs.summarize(diff, f"fixed-share - {other}")
        print()

    n = len(shock)
    print("valid block-start positions at n=%d (n - block_length + 1, or 1 if block_length>=n):" % n)
    for bl in (3, 6, 12):
        starts = max(1, n - min(bl, n) + 1)
        print(f"  block={bl:>2}mo -> effective block={min(bl,n)}, valid starts={starts}")


if __name__ == "__main__":
    main()
