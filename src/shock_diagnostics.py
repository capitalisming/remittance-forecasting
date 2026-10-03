"""Numbers quoted in the paper's abstract and Section 6.1 beyond Table 1: relative MAE and MAPE
by window, year-on-year growth in the shock window, signed errors of naive/E1/E2, and the
year-on-year volatility of the main window. Reads results/expert_comparison_eval.csv (abs errors)
and recomputes naive/E1/E2 forecasts for signed errors (Chronos needs the HF model, so its
signed errors are not reported)."""
import pandas as pd

import experts
import panel

df = pd.read_csv("../results/expert_comparison_eval.csv", index_col=0, parse_dates=True)
df["target"] = pd.PeriodIndex(df.index, freq="M") + 1
df = df.dropna(subset=["naive"])
truth = panel.first_release_series()
df["actual"] = [truth.get(t) for t in df.target]
tz = panel.TRANSITION_ZONE
windows = {"main": ~df.target.isin(tz), "shock": df.target.astype(str).between("2022-02", "2022-09"),
           "transition": df.target.isin(tz)}
cols = ["naive", "expanding_ridge", "rolling_ridge_w48", "chronos_zeroshot"]
for name, m in windows.items():
    d = df[m]
    mae = d[cols].mean()
    mape = d[cols].div(d.actual, axis=0).mean() * 100
    print(f"\n{name} (n={len(d)})")
    print(pd.DataFrame({"MAE": mae, "rel_to_naive": mae / mae["naive"], "MAPE%": mape}).round(2))

shock = df[windows["shock"]].copy()
shock["actual_ly"] = [truth.get(t - 12) for t in shock.target]
shock["yoy%"] = (shock.actual / shock.actual_ly - 1) * 100
print("\nshock: mean YoY %.1f%%, cumulative YoY %.1f%%" % (
    shock["yoy%"].mean(), (shock.actual.sum() / shock.actual_ly.sum() - 1) * 100))
main = df[windows["main"]]
yoy_main = (main.actual / pd.Series([truth.get(t - 12) for t in main.target], index=main.index) - 1) * 100
print("main-window YoY std: %.1f%%" % yoy_main.std())

dates = experts.EVAL_DATES
fc = {"naive": experts.backtest(dates, experts.seasonal_naive),
      "E1": experts.backtest(dates, experts.expanding_ridge),
      "E2": experts.backtest(dates, experts.rolling_ridge, window=48)}
sh = {k: v[v.target_month.astype(str).between("2022-02", "2022-09")] for k, v in fc.items()}
print("\nshock mean signed error (forecast - actual):",
      {k: round((v.forecast - v.actual).mean(), 1) for k, v in sh.items()})
print("signed errors by shock month:")
print(pd.DataFrame({k: (v.forecast - v.actual).round(1).values for k, v in sh.items()},
                   index=sh["naive"].target_month))

# --- additions for the revised draft -------------------------------------------------------
print("\n=== Ordinary window (main minus shock months), n = %d ===" % (windows["main"] & ~windows["shock"]).sum())
ordm = windows["main"] & ~windows["shock"]
d = df[ordm]
mae = d[cols].mean()
mape = d[cols].div(d.actual, axis=0).mean() * 100
print(pd.DataFrame({"MAE": mae, "rel_to_naive": mae / mae["naive"], "MAPE%": mape}).round(2))

import json
c = json.load(open("../results/fixed_share_config.json"))["clip_c"]
print("\n=== share of feedback events with |error| > c (loss clipped at 1), c = %.2f ===" % c)
rows = {}
for name, m in {"all": pd.Series(True, index=df.index), **windows}.items():
    rows[name] = (df.loc[m, cols] > c).mean().mul(100).round(1)
print(pd.DataFrame(rows).T.to_string())
print("pooled over four experts:", {n: round(float((df.loc[m, cols] > c).to_numpy().mean() * 100), 1)
      for n, m in {"all": pd.Series(True, index=df.index), **windows}.items()})

rel = pd.Timestamp("2024-11-26")
post = df.index > rel
print("\n=== Chronos-Bolt released 2024-11-26: origins before (n=%d) vs after (n=%d) ===" % ((~post).sum(), post.sum()))
for lab, m in (("before", ~post), ("after", post)):
    s = df.loc[m, cols].mean()
    print(lab, "MAE", s.round(1).to_dict(), " chronos/naive = %.2f, E1/naive = %.2f" % (
        s["chronos_zeroshot"] / s["naive"], s["expanding_ridge"] / s["naive"]))
