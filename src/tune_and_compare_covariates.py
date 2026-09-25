"""Stage 2: does adding FX/Brent/IMOEX (leakage-safe, per-covariate lag) help forecast the 2022
shock, i.e. did early leading indicators exist? Same dev-tune-then-freeze protocol as
tune_ridge.py, but for the covariate-augmented expanding ridge (E1), then compares dev, main,
and shock-window MAE against the no-covariate baseline (frozen alpha=0.1 from
results/ridge_alpha_tuning.json).
"""
import json

import experts
import panel  # noqa: F401

ALPHA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
BASELINE_ALPHA = experts.EXPANDING_RIDGE_ALPHA

MAIN_WINDOW_EXCLUDE = panel.TRANSITION_ZONE
SHOCK_WINDOW = experts.realtime.issue_dates("2022-01", "2022-08")  # target months 2022-02..2022-09


def window_mae(bt, window_dates=None, exclude_target_months=None):
    df = bt
    if window_dates is not None:
        df = df.loc[df.index.isin(window_dates)]
    if exclude_target_months is not None:
        df = df.loc[~df.target_month.isin(exclude_target_months)]
    return df.abs_err.mean(), len(df)


def main():
    dev_dates = experts.DEV_DATES
    print(f"dev origins: {len(dev_dates)}\n")

    print("=== E1 + covariates, dev MAE by alpha ===")
    scores = []
    for a in ALPHA_GRID:
        mae = experts.backtest(dev_dates, experts.expanding_ridge, alpha=a,
                                include_covariates=True).abs_err.mean()
        scores.append((a, mae))
        print(f"  alpha={a:<7} dev MAE={mae:.3f}")
    best_alpha, best_dev_mae = min(scores, key=lambda t: t[1])
    print(f"\nbest alpha (dev): {best_alpha}, dev MAE={best_dev_mae:.3f}")

    eval_dates = experts.EVAL_DATES

    bt_cov = experts.backtest(eval_dates, experts.expanding_ridge, alpha=best_alpha,
                               include_covariates=True)
    bt_base = experts.backtest(eval_dates, experts.expanding_ridge, alpha=BASELINE_ALPHA,
                                include_covariates=False)

    main_mae_cov, n_main = window_mae(bt_cov, exclude_target_months=MAIN_WINDOW_EXCLUDE)
    main_mae_base, _ = window_mae(bt_base, exclude_target_months=MAIN_WINDOW_EXCLUDE)

    shock_mae_cov, n_shock = window_mae(bt_cov, window_dates=SHOCK_WINDOW)
    shock_mae_base, _ = window_mae(bt_base, window_dates=SHOCK_WINDOW)

    print(f"\n=== Main window (n={n_main}) ===")
    print(f"  E1 no covariates (alpha={BASELINE_ALPHA}): MAE={main_mae_base:.3f}")
    print(f"  E1 + covariates  (alpha={best_alpha}):      MAE={main_mae_cov:.3f}")

    print(f"\n=== Shock window 2022 (n={n_shock}) ===")
    print(f"  E1 no covariates (alpha={BASELINE_ALPHA}): MAE={shock_mae_base:.3f}")
    print(f"  E1 + covariates  (alpha={best_alpha}):      MAE={shock_mae_cov:.3f}")

    out = {
        "grid": ALPHA_GRID,
        "full_scores": [[a, round(m, 3)] for a, m in scores],
        "best_alpha": best_alpha,
        "dev_mae": round(best_dev_mae, 3),
        "main_window": {"n": n_main, "no_covariates": round(main_mae_base, 3),
                         "with_covariates": round(main_mae_cov, 3)},
        "shock_window": {"n": n_shock, "no_covariates": round(shock_mae_base, 3),
                          "with_covariates": round(shock_mae_cov, 3)},
    }
    json.dump(out, open("../results/covariate_comparison.json", "w"), indent=2)
    print("\nwritten to ../results/covariate_comparison.json")


if __name__ == "__main__":
    main()
