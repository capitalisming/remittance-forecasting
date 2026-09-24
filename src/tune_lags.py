"""Justify the lag set (1, 2, 3, 12) used by E1/E2, closing the P1 gap flagged in peer review
(lag choice was asserted, not justified). Same protocol as tune_ridge.py: dev-period-only grid
search, ridge alpha held at its already-tuned value, then freeze. Candidate sets are chosen a
priori from domain reasoning (short-run momentum via lags 1-6, year-over-year seasonality via
lag 12, combinations of both) rather than searched exhaustively, to keep this a single
pre-registered comparison rather than a second hyperparameter search stacked on the first.
"""
import json

import experts
import panel

CANDIDATE_LAG_SETS = [
    (1,),
    (12,),
    (1, 12),
    (1, 2, 12),
    (1, 2, 3, 12),  # current choice
    (1, 3, 6, 12),
    (1, 2, 3, 6, 12),
    (1, 2, 3, 4, 5, 6, 12),
]


def main():
    dev_dates = experts.DEV_DATES
    print(f"dev origins: {len(dev_dates)}\n")

    results = {"expanding_ridge": [], "rolling_ridge_w48": []}
    orig_lags = panel.LAGS
    try:
        for lags in CANDIDATE_LAG_SETS:
            panel.LAGS = lags
            mae_e1 = experts.backtest(dev_dates, experts.expanding_ridge,
                                       alpha=experts.EXPANDING_RIDGE_ALPHA).abs_err.mean()
            mae_e2 = experts.backtest(dev_dates, experts.rolling_ridge, window=48,
                                       alpha=experts.ROLLING_RIDGE_ALPHA).abs_err.mean()
            results["expanding_ridge"].append((lags, mae_e1))
            results["rolling_ridge_w48"].append((lags, mae_e2))
            print(f"  lags={lags!s:<24} E1 dev MAE={mae_e1:.3f}   E2 dev MAE={mae_e2:.3f}")
    finally:
        panel.LAGS = orig_lags

    best = {name: min(scores, key=lambda t: t[1]) for name, scores in results.items()}
    print("\nbest lag set per expert (dev MAE):")
    for name, (lags, mae) in best.items():
        print(f"  {name}: lags={lags}, dev MAE={mae:.3f}")

    out = {
        "candidates": [list(lags) for lags in CANDIDATE_LAG_SETS],
        "current_choice": list(panel.LAGS),
        "expanding_ridge": {"lags": list(best["expanding_ridge"][0]), "dev_mae": round(best["expanding_ridge"][1], 3)},
        "rolling_ridge_w48": {"lags": list(best["rolling_ridge_w48"][0]), "dev_mae": round(best["rolling_ridge_w48"][1], 3)},
        "full_scores": {name: [[list(lags), round(m, 3)] for lags, m in scores] for name, scores in results.items()},
    }
    json.dump(out, open("../results/lag_selection.json", "w"), indent=2)
    print("\nwritten to ../results/lag_selection.json")


if __name__ == "__main__":
    main()
