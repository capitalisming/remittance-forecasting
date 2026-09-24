"""Tune ridge alpha for E1 (expanding) and E2 (rolling, w=48) on the development period only,
the same grid-search-on-dev-then-freeze protocol already used for eta/alpha (fixed-share) and for
the rolling window w. Closes the P0 gap flagged in peer review: RIDGE_ALPHA was a hardcoded
placeholder (alpha=1.0) that a nonexistent tune_ridge.py was supposed to have produced.
"""
import json

import experts
import panel  # noqa: F401  (imported for side effect parity with other scripts; experts imports it)

ALPHA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def main():
    dev_dates = experts.DEV_DATES
    print(f"dev origins: {len(dev_dates)}\n")

    results = {"expanding_ridge": [], "rolling_ridge_w48": []}

    print("=== expanding_ridge, dev MAE by alpha ===")
    for a in ALPHA_GRID:
        mae = experts.backtest(dev_dates, experts.expanding_ridge, alpha=a).abs_err.mean()
        results["expanding_ridge"].append((a, mae))
        print(f"  alpha={a:<7} dev MAE={mae:.3f}")

    print("\n=== rolling_ridge (window=48), dev MAE by alpha ===")
    for a in ALPHA_GRID:
        mae = experts.backtest(dev_dates, experts.rolling_ridge, window=48, alpha=a).abs_err.mean()
        results["rolling_ridge_w48"].append((a, mae))
        print(f"  alpha={a:<7} dev MAE={mae:.3f}")

    best = {name: min(scores, key=lambda t: t[1]) for name, scores in results.items()}
    print("\nbest alpha per expert (dev MAE):")
    for name, (a, mae) in best.items():
        print(f"  {name}: alpha={a}, dev MAE={mae:.3f}")

    out = {
        "expanding_ridge": {"alpha": best["expanding_ridge"][0], "dev_mae": round(best["expanding_ridge"][1], 3)},
        "rolling_ridge_w48": {"alpha": best["rolling_ridge_w48"][0], "dev_mae": round(best["rolling_ridge_w48"][1], 3)},
        "grid": ALPHA_GRID,
        "full_scores": {name: [[a, round(m, 3)] for a, m in scores] for name, scores in results.items()},
    }
    json.dump(out, open("../results/ridge_alpha_tuning.json", "w"), indent=2)
    print("\nwritten to ../results/ridge_alpha_tuning.json")


if __name__ == "__main__":
    main()
