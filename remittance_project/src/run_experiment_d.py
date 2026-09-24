"""Experiment D: extra delay k in {0,1,2,3} months on top of the natural publication lag,
applied only to the eligibility of scoring/weight-update events (forecasts and training data
are untouched). Uses the same archive and the eta/alpha tuned on the development period."""
import json

import pandas as pd

import fixed_share as fs
import panel
import realtime

DEV_END, EVAL_END = "2021-12", "2026-06"


def main():
    cfg = json.load(open("../results/fixed_share_config.json"))
    eta, alpha, clip_c = cfg["eta"], cfg["alpha"], cfg["clip_c"]
    fs_start = pd.Timestamp(cfg["fs_start"])
    all_dates = list(realtime.issue_dates(fs_start.strftime("%Y-%m"), EVAL_END))
    eval_dates = [v for v in all_dates if v.strftime("%Y-%m") > DEV_END]

    archive = fs.precompute_forecasts(all_dates)
    tz = pd.period_range("2023-10", "2024-01", freq="M")

    print(f"Experiment D | eta={eta} alpha={alpha} c={clip_c:.2f} (tuned on dev, reused as-is)\n")
    results = {}
    for k in (0, 1, 2, 3):
        sim = fs.simulate(archive, all_dates, eta=eta, alpha=alpha, clip_c=clip_c, delay_k=k)
        ev = sim.loc[sim.index.isin(eval_dates)]
        main_mask = ~ev.target_month.isin(tz)
        results[k] = dict(full=ev.abs_err.mean(), main=ev[main_mask].abs_err.mean())
        print(f"k={k}: full-eval MAE {results[k]['full']:.2f} | main-window MAE {results[k]['main']:.2f}")

    g0 = results[0]["main"]
    print("\nG_{k=0} - G_k (main window; positive = extra delay makes the loss WORSE, i.e. hurts adaptation):")
    for k in (1, 2, 3):
        print(f"  k={k}: {results[k]['main'] - g0:+.2f}")

    json.dump(results, open("../results/experiment_d.json", "w"), indent=2)


if __name__ == "__main__":
    main()
