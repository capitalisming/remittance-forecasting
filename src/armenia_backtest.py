"""Stage 3: does the Kyrgyzstan pattern (fitted models beat naive ordinarily, lose to it during
the 2022 shock) replicate for Armenia? Same dev-tune-then-freeze protocol, h=1, real-time-safe
features (armenia_panel.py). Naive here is "repeat last known month" (last-value naive), not
seasonal-naive, because lag-12 isn't available (see armenia_panel.py) to build a proper
year-ago comparison either.
"""
import json

import numpy as np
from sklearn.linear_model import Ridge

import armenia_panel as apanel
import armenia_realtime as art

ALPHA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

DEV_DATES = art.issue_dates("2015-06", "2019-12")
SHOCK_DATES = art.issue_dates("2022-01", "2022-08")  # target months 2022-02..2022-09


def expanding_ridge(X_train, y_train, x_query, alpha):
    if len(X_train) < 13:
        return np.exp(y_train.iloc[-1]) if len(y_train) else np.nan
    model = Ridge(alpha=alpha).fit(X_train, y_train)
    return float(np.exp(model.predict(x_query)[0]))


def last_value_naive(y_train):
    return float(np.exp(y_train.iloc[-1])) if len(y_train) else np.nan


def backtest(issue_dates, alpha=None):
    truth = apanel.first_release_series()
    rows = []
    for v in issue_dates:
        try:
            X, y, q, target_month = apanel.build_row(v)
        except AssertionError:
            continue
        if len(y) < 13:
            continue
        ridge_pred = expanding_ridge(X, y, q, alpha) if alpha is not None else np.nan
        naive_pred = last_value_naive(y)
        actual = truth.get(target_month, np.nan)
        rows.append(dict(issue_date=v, target_month=target_month,
                          ridge=ridge_pred, naive=naive_pred, actual=actual))
    import pandas as pd
    out = pd.DataFrame(rows).set_index("issue_date")
    out["ridge_abs_err"] = (out.ridge - out.actual).abs()
    out["naive_abs_err"] = (out.naive - out.actual).abs()
    return out


def main():
    print(f"dev origins: {len(DEV_DATES)}")
    scores = []
    for a in ALPHA_GRID:
        bt = backtest(DEV_DATES, alpha=a)
        mae = bt.ridge_abs_err.mean()
        scores.append((a, mae))
        print(f"  alpha={a:<7} dev MAE={mae:.3f}  (n={len(bt)})")
    best_alpha, best_mae = min(scores, key=lambda t: t[1])
    print(f"\nbest alpha (dev): {best_alpha}, dev MAE={best_mae:.3f}")

    shock_bt = backtest(SHOCK_DATES, alpha=best_alpha)
    print(f"\n=== Shock window 2022 (n={len(shock_bt)}) ===")
    print(shock_bt[["target_month", "ridge", "naive", "actual", "ridge_abs_err", "naive_abs_err"]])
    print(f"\nRidge (lags 1-3) MAE: {shock_bt.ridge_abs_err.mean():.3f}")
    print(f"Last-value naive MAE:  {shock_bt.naive_abs_err.mean():.3f}")

    out = {
        "grid": ALPHA_GRID,
        "full_scores": [[a, round(m, 3)] for a, m in scores],
        "best_alpha": best_alpha,
        "dev_mae": round(best_mae, 3),
        "shock_window": {
            "n": len(shock_bt),
            "ridge_mae": round(shock_bt.ridge_abs_err.mean(), 3),
            "naive_mae": round(shock_bt.naive_abs_err.mean(), 3),
        },
    }
    json.dump(out, open("../results/armenia_comparison.json", "w"), indent=2)
    print("\nwritten to ../results/armenia_comparison.json")


if __name__ == "__main__":
    main()
