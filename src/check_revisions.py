"""Count how many consecutive same-regime vintage pairs revise a shared month, and at what
materiality threshold, closing the P0 gap flagged in peer review: Section 3 says "9 of 97" but
doesn't state the threshold, so a naive recount (any nonzero float difference) gives a different
number. Also reports the true within-regime max revision size so the "<4% in all but one case"
claim can be checked.
"""
import pandas as pd

idx = pd.read_csv("data/processed/vintage_index.csv", parse_dates=["vintage_date"])
idx = idx.sort_values("vintage_date").reset_index(drop=True)
long = pd.read_csv("data/processed/vintages_long.csv", parse_dates=["vintage_date"])
wide = long.pivot(index="ref_month", columns="vintage_date", values="inflow").sort_index()

pairs = []
for regime in ("mts", "cross_border"):
    dates = idx[idx.regime == regime].sort_values("vintage_date").vintage_date.tolist()
    pairs.extend(zip(dates[:-1], dates[1:]))
print(f"consecutive same-regime pairs (mts + cross_border): {len(pairs)}")

thresholds = [0.0, 1e-6, 0.001, 0.01, 0.02, 0.05, 0.1]
counts = {t: 0 for t in thresholds}
max_diffs = []
pct_diffs = []
for d0, d1 in pairs:
    if d0 not in wide.columns or d1 not in wide.columns:
        continue
    shared = wide[[d0, d1]].dropna()
    if shared.empty:
        continue
    diff = (shared[d1] - shared[d0]).abs()
    max_d = diff.max()
    max_diffs.append((d0.date(), d1.date(), max_d))
    for t in thresholds:
        if (diff > t).any():
            counts[t] += 1
    nonzero = diff[diff > 0]
    if len(nonzero):
        pct = (nonzero / shared.loc[nonzero.index, d0].abs()) * 100
        pct_diffs.extend(pct.tolist())

print("\npairs with >=1 shared-month revision exceeding threshold:")
for t in thresholds:
    print(f"  threshold >{t}: {counts[t]} of {len(pairs)} pairs")

print("\nall pairs sorted by max within-pair revision (mln USD), largest first:")
for d0, d1, m in sorted(max_diffs, key=lambda r: -r[2])[:15]:
    print(f"  {d0} -> {d1}: max diff = {m:.4f}")

print(f"\nmax revision as % of prior value, among nonzero diffs: max={max(pct_diffs):.2f}%, "
      f"n_above_4pct={sum(1 for p in pct_diffs if p >= 4)}")
