"""Recompute Table 1 (expert-race, main/shock/TZ/full windows) from the regenerated
expert_comparison_eval.csv after the ridge-alpha tuning fix."""
import pandas as pd

import panel

df = pd.read_csv("../results/expert_comparison_eval.csv", index_col=0, parse_dates=True)
df["target_month"] = pd.PeriodIndex(df.index, freq="M") + 1
df = df.dropna(subset=["naive"])  # target month not yet released (2026-07)

tz = panel.TRANSITION_ZONE
main_mask = ~df.target_month.isin(tz)
shock_mask = df.target_month.astype(str).between("2022-02", "2022-09")
tz_mask = df.target_month.isin(tz)

experts_cols = ["naive", "expanding_ridge", "rolling_ridge_w48", "chronos_zeroshot"]
windows = {"main": main_mask, "shock": shock_mask, "TZ": tz_mask, "full": pd.Series(True, index=df.index)}

print(f"{'expert':<20}" + "".join(f"{w:>12}" for w in windows))
for e in experts_cols:
    row = f"{e:<20}"
    for w, mask in windows.items():
        row += f"{df.loc[mask, e].mean():>12.2f}"
    print(row)
print("\nn per window:", {w: int(m.sum()) for w, m in windows.items()})
