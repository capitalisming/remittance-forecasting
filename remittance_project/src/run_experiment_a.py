"""Experiment A (operational): precompute all four experts' forecasts once, tune eta/alpha on the
development period only, then report the fixed-share combination on the evaluation window."""
import itertools
import json
import time

import numpy as np
import pandas as pd

import fixed_share as fs
import panel
import realtime

DEV_END, EVAL_END = "2021-12", "2026-06"


def main():
    t0 = time.time()
    fs_start = fs.first_valid_issue_date(realtime.issue_dates("2006-01", DEV_END))
    all_dates = list(realtime.issue_dates(fs_start.strftime("%Y-%m"), EVAL_END))
    dev_dates = [v for v in all_dates if v.strftime("%Y-%m") <= DEV_END]
    eval_dates = [v for v in all_dates if v.strftime("%Y-%m") > DEV_END]
    print(f"fixed-share starts {fs_start.date()} | dev origins {len(dev_dates)} | eval origins {len(eval_dates)}")

    archive = fs.precompute_forecasts(all_dates)
    print(f"forecasts precomputed for {len(archive)} target months in {time.time()-t0:.1f}s")

    # clip constant c: 95th percentile of pooled |error| across the four experts, dev period only
    truth = panel.first_release_series()
    dev_targets = {v.to_period("M") + 1 for v in dev_dates}
    pooled_errs = [abs(p - truth[m]) for m in dev_targets if m in truth.index and m in archive
                  for p in archive[m].values() if p is not None and not pd.isna(p)]
    clip_c = float(np.percentile(pooled_errs, 95))
    print(f"clip constant c (95th pct of dev |error|, pooled over experts): {clip_c:.2f}")

    grid = list(itertools.product([0.5, 1, 2, 4], [0.0, 0.01, 0.05]))
    dev_scores = []
    for eta, alpha in grid:
        sim = fs.simulate(archive, dev_dates, eta=eta, alpha=alpha, clip_c=clip_c)
        dev_scores.append((eta, alpha, sim.abs_err.mean()))
    dev_scores.sort(key=lambda r: r[2])
    print("\ntop 5 (eta, alpha) by dev MAE:")
    for eta, alpha, mae in dev_scores[:5]:
        print(f"  eta={eta:<4} alpha={alpha:<5} dev MAE={mae:.2f}")
    best_eta, best_alpha, _ = dev_scores[0]

    full_sim = fs.simulate(archive, all_dates, eta=best_eta, alpha=best_alpha, clip_c=clip_c)
    eval_sim = full_sim.loc[full_sim.index.isin(eval_dates)]
    eq = fs.equal_weight_baseline(archive, all_dates)
    eq_eval = eq.loc[eq.index.isin(eval_dates)]

    tz = pd.period_range("2023-10", "2024-01", freq="M")
    main_mask = ~eval_sim.target_month.isin(tz)
    shock_mask = eval_sim.target_month.between("2022-02", "2022-09")

    print(f"\n=== Experiment A, chosen eta={best_eta}, alpha={best_alpha}, c={clip_c:.2f} ===")
    print(f"fixed-share MAE  | full eval: {eval_sim.abs_err.mean():.2f}"
          f" | main window: {eval_sim[main_mask].abs_err.mean():.2f}"
          f" | shock window: {eval_sim[shock_mask].abs_err.mean():.2f}")
    print(f"equal-weight MAE | full eval: {eq_eval.abs_err.mean():.2f}"
          f" | main window: {eq_eval[main_mask.values].abs_err.mean():.2f}"
          f" | shock window: {eq_eval[shock_mask.values].abs_err.mean():.2f}")

    print("\nreference (individual experts, same eval window, from expert_comparison_eval.csv):")
    ref = pd.read_csv("../results/expert_comparison_eval.csv", index_col=0)
    print("  full eval MAE:", ref.mean().round(2).to_dict())

    print("\nweight trajectory around the shock (2022-02..2022-09 targets):")
    cols = ["target_month"] + [f"w_{k}" for k in fs.EXPERTS]
    print(eval_sim[eval_sim.target_month.between("2022-02", "2022-09")][cols].round(3).to_string(index=False))

    eval_sim.to_csv("../results/fixed_share_eval.csv")
    json.dump(dict(eta=best_eta, alpha=best_alpha, clip_c=clip_c, fs_start=str(fs_start.date())),
             open("../results/fixed_share_config.json", "w"), indent=2)
    print(f"\ntotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
