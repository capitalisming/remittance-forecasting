"""Bootstrap CIs for the two headline comparisons: fixed-share vs equal-weight, and Experiment D's
sensitivity to extra feedback delay k. Reuses the tuned config and re-derives the same archive
(deterministic given the same issue dates)."""
import json

import numpy as np
import pandas as pd

import bootstrap as bs
import fixed_share as fs
import realtime

DEV_END, EVAL_END = "2021-12", "2026-06"
TZ = pd.period_range("2023-10", "2024-01", freq="M")


def main():
    cfg = json.load(open("../results/fixed_share_config.json"))
    eta, alpha, clip_c = cfg["eta"], cfg["alpha"], cfg["clip_c"]
    fs_start = pd.Timestamp(cfg["fs_start"])
    all_dates = list(realtime.issue_dates(fs_start.strftime("%Y-%m"), EVAL_END))
    eval_dates = [v for v in all_dates if v.strftime("%Y-%m") > DEV_END]

    archive = fs.precompute_forecasts(all_dates)

    fixed = fs.simulate(archive, all_dates, eta=eta, alpha=alpha, clip_c=clip_c).loc[eval_dates]
    equal = fs.equal_weight_baseline(archive, all_dates).loc[eval_dates]
    assert (fixed.target_month.values == equal.target_month.values).all()

    has_truth = fixed.actual.notna() & equal.actual.notna()
    n_dropped = (~has_truth).sum()
    if n_dropped:
        print(f"dropping {n_dropped} origin(s) with no known outcome yet "
              f"(target month(s) {fixed.loc[~has_truth, 'target_month'].tolist()}, "
              f"publication hasn't reached them as of the data snapshot)\n")
    fixed, equal = fixed[has_truth], equal[has_truth]

    main_mask = ~fixed.target_month.isin(TZ)
    shock_mask = fixed.target_month.between("2022-02", "2022-09")

    print("=== fixed-share vs equal-weight: |err_fixed_share| - |err_equal_weight| ===")
    print("(positive = fixed-share worse; 0 in the CI = not distinguishable)\n")
    diff_main = (fixed.abs_err - equal.abs_err)[main_mask.values].to_numpy()
    bs.summarize(diff_main, "main window (n=%d)" % main_mask.sum())
    print()
    diff_shock = (fixed.abs_err - equal.abs_err)[shock_mask.values].to_numpy()
    bs.summarize(diff_shock, "shock window (n=%d)" % shock_mask.sum())

    print("\n=== Experiment D: |err(k)| - |err(k=0)|, main window ===")
    print("(positive = extra delay makes the loss worse)\n")
    sims = {k: fs.simulate(archive, all_dates, eta=eta, alpha=alpha, clip_c=clip_c, delay_k=k).loc[fixed.index]
            for k in (0, 1, 2, 3)}
    for k in (1, 2, 3):
        d = (sims[k].abs_err - sims[0].abs_err)[main_mask.values].to_numpy()
        bs.summarize(d, f"k={k} vs k=0 (n={main_mask.sum()})")
        print()


if __name__ == "__main__":
    main()
