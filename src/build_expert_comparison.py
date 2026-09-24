"""Regenerate results/expert_comparison_eval.csv: per-issue-date abs_err for all four experts on
the evaluation window (2022-01..2026-06). Rerun whenever an expert's hyperparameters change --
last regenerated after tune_ridge.py fixed the previously-untuned ridge alpha (see git history).
"""
import pandas as pd

import chronos_expert as ce
import experts

EVAL_DATES = experts.EVAL_DATES


def main():
    naive = experts.backtest(EVAL_DATES, experts.seasonal_naive).abs_err.rename("naive")
    exp_ridge = experts.backtest(EVAL_DATES, experts.expanding_ridge).abs_err.rename("expanding_ridge")
    roll_ridge = experts.backtest(EVAL_DATES, experts.rolling_ridge, window=48).abs_err.rename("rolling_ridge_w48")
    chronos = ce.backtest_chronos(EVAL_DATES).abs_err.rename("chronos_zeroshot")

    out = pd.concat([naive, exp_ridge, roll_ridge, chronos], axis=1)
    out.to_csv("../results/expert_comparison_eval.csv")
    print(out.shape)
    print(out.mean().round(2))


if __name__ == "__main__":
    main()
