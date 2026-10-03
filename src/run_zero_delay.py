"""Zero-delay counterfactual for the fixed-share combiner.

Question: is the combiner's weakness in the shock caused by delayed feedback or by a slow learning
rate? We re-run the weight recursion with (a) the real publication lag, (b) feedback available
immediately (month m's outcome is usable at the end of month m, i.e. at issue date v with
m <= period(v)), each crossed with eta in {0.5, 1, 2, 4}. Forecasts of the four experts are
unchanged; only the timing/size of the weight updates changes.

Chronos forecasts for the full 2007-2026 history are not stored anywhere (the model weights are
not available in every environment), but for the evaluation origins they are recoverable exactly
from the logged fixed-share weights: combined = sum_k w_k f_k, so f_chronos = (combined -
sum_{k != chronos} w_k f_k) / w_chronos (checked against expert_comparison_eval.csv to 1e-11).
The recursion is therefore restarted at the first origin whose operational information set
already covers every month before the evaluation period (2022-03-31, last known month 2022-01),
from the logged operational weights. The (real lag, eta=0.5) arm reproduces the published
fixed-share forecasts exactly; this is asserted below.
"""
import json
import os

import numpy as np
import pandas as pd

import bootstrap as bs
import experts
import panel
import realtime

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ARCHIVE_CSV = os.path.join(RESULTS, "forecast_archive_eval.csv")
NAMES = ["naive", "expanding_ridge", "rolling_ridge_w48", "chronos"]
START = pd.Timestamp("2022-03-31")
ETAS = (0.5, 1.0, 2.0, 4.0)
TZ = panel.TRANSITION_ZONE


def build_archive():
    fs = pd.read_csv(os.path.join(RESULTS, "fixed_share_eval.csv"), index_col=0, parse_dates=True)
    dates = experts.EVAL_DATES
    f = pd.DataFrame({
        "naive": experts.backtest(dates, experts.seasonal_naive).forecast,
        "expanding_ridge": experts.backtest(dates, experts.expanding_ridge).forecast,
        "rolling_ridge_w48": experts.backtest(dates, experts.rolling_ridge, window=48).forecast,
    })
    d = fs.join(f)
    others = sum(d[f"w_{k}"] * d[k] for k in NAMES[:3])
    d["chronos"] = (d.combined_forecast - others) / d.w_chronos
    ev = pd.read_csv(os.path.join(RESULTS, "expert_comparison_eval.csv"), index_col=0, parse_dates=True)
    chk = ((d.chronos - d.actual).abs() - ev.chronos_zeroshot.reindex(d.index)).abs().max()
    assert chk < 1e-8, chk
    out = d[["target_month", "actual"] + NAMES].copy()
    out.to_csv(ARCHIVE_CSV)
    return out


def load_archive():
    if not os.path.exists(ARCHIVE_CSV):
        return build_archive()
    return pd.read_csv(ARCHIVE_CSV, index_col=0, parse_dates=True)


def simulate(arch, w0, eta, alpha, clip_c, zero_delay):
    """Weight recursion from START. w0: weights logged at START (after the operational events
    through month 2022-01 were applied); arch rows are indexed by issue date."""
    w = np.array([w0[f"w_{k}"] for k in NAMES])
    known, rows = set(), []
    targets = {pd.Period(t, "M"): r for t, r in zip(arch.target_month, arch.itertuples())}
    for v in arch.index[arch.index >= START]:
        if zero_delay:
            eligible = [m for m in targets if m <= v.to_period("M")]
        else:
            info = realtime.info_set(v).index
            eligible = [m for m in targets if m in info]
        for m in sorted(m for m in eligible if m not in known):
            known.add(m)
            r = targets[m]
            if pd.isna(r.actual):
                continue
            loss = np.minimum(np.abs(np.array([getattr(r, k) for k in NAMES]) - r.actual) / clip_c, 1.0)
            q = w * np.exp(-eta * loss)
            q /= q.sum()
            w = (1 - alpha) * q + alpha / len(NAMES)
        row = arch.loc[v]
        combined = float(np.dot(w, row[NAMES].to_numpy(float)))
        rows.append(dict(issue_date=v, target_month=pd.Period(row.target_month, "M"),
                         combined=combined, actual=row.actual, **{f"w_{k}": x for k, x in zip(NAMES, w)}))
    out = pd.DataFrame(rows).set_index("issue_date")
    out["abs_err"] = (out.combined - out.actual).abs()
    return out


def windows(df):
    tm = df.target_month
    shock = tm.astype(str).between("2022-02", "2022-09")
    tz = tm.isin(TZ)
    return {"shock": shock, "ordinary": ~shock & ~tz, "transition": tz}


def main():
    cfg = json.load(open(os.path.join(RESULTS, "fixed_share_config.json")))
    alpha, clip_c = cfg["alpha"], cfg["clip_c"]
    arch = load_archive()
    w0 = arch.loc[START]
    w0 = pd.read_csv(os.path.join(RESULTS, "fixed_share_eval.csv"), index_col=0, parse_dates=True).loc[START]

    ref = simulate(arch, w0, 0.5, alpha, clip_c, zero_delay=False)
    pub = pd.read_csv(os.path.join(RESULTS, "fixed_share_eval.csv"), index_col=0, parse_dates=True).loc[ref.index]
    gap = (ref.combined - pub.combined_forecast).abs().max()
    assert gap < 1e-6, f"real-lag eta=0.5 arm does not reproduce published fixed-share forecasts: {gap}"
    print(f"validation: real-lag eta=0.5 arm reproduces published forecasts (max gap {gap:.2e})\n")

    known = arch.dropna(subset=["actual"])
    sims = {}
    for zero in (False, True):
        for eta in ETAS:
            s = simulate(arch, w0, eta, alpha, clip_c, zero)
            sims[(zero, eta)] = s.loc[s.actual.notna()]

    base = sims[(False, 0.5)]
    win = windows(base)
    res = {"start": str(START.date()), "alpha": alpha, "clip_c": clip_c, "n": {k: int(m.sum()) for k, m in win.items()},
           "mae": {}, "ci_zero_vs_real_ordinary_block3": {}}
    ew = known.loc[base.index, NAMES].mean(axis=1)
    ref_rows = {"equal_weight": (ew - base.actual).abs(), "naive": (known.loc[base.index, "naive"] - base.actual).abs()}
    for name, e in ref_rows.items():
        res["mae"][name] = {w: round(float(e[m.values].mean()), 2) for w, m in win.items()}
    for (zero, eta), s in sims.items():
        key = f"{'zero' if zero else 'real'}_eta{eta}"
        res["mae"][key] = {w: round(float(s.abs_err[m.values].mean()), 2) for w, m in win.items()}
        res["mae"][key]["weight_naive_mean_shock"] = round(float(s.w_naive[win['shock'].values].mean()), 3)
    for eta in ETAS:
        d = (sims[(True, eta)].abs_err - sims[(False, eta)].abs_err)[win["ordinary"].values].to_numpy()
        boot = bs.moving_block_bootstrap_mean(d, 3)
        res["ci_zero_vs_real_ordinary_block3"][str(eta)] = [round(float(d.mean()), 2),
                                                           *[round(float(x), 2) for x in np.percentile(boot, [2.5, 97.5])]]
    json.dump(res, open(os.path.join(RESULTS, "zero_delay_counterfactual.json"), "w"), indent=2)

    print("n per window:", res["n"], "\n")
    print(pd.DataFrame(res["mae"]).T.to_string())
    print("\nzero-delay minus real-lag MAE on the ordinary window: [point, 95% CI lo, hi] (block=3)")
    for eta, v in res["ci_zero_vs_real_ordinary_block3"].items():
        print(f"  eta={eta}: {v}")


if __name__ == "__main__":
    main()
