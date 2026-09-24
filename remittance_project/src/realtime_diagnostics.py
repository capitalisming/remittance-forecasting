"""Coverage of the real-time information sets over the evaluation window and first-release table."""
import os

import pandas as pd

import realtime

OUT_DIR = os.path.join(realtime.ROOT, "results")
EVAL_START, EVAL_END = "2022-01", "2026-06"   # h=1 issue dates; targets 2022-02 .. 2026-07


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for v in realtime.issue_dates(EVAL_START, EVAL_END):
        known = realtime.info_set(v)
        base = realtime.base_vintage(v)
        expected = v.to_period("M") - 2
        rows.append(dict(issue_date=v.date(), base_vintage=base.date(),
                         base_regime=realtime._load()[2][base],
                         last_known=str(known.index.max()), expected_last=str(expected),
                         stale_months=(expected - known.index.max()).n,
                         n_filled=int(known.filled.sum())))
    cov = pd.DataFrame(rows)
    cov.to_csv(os.path.join(OUT_DIR, "realtime_coverage.csv"), index=False)
    print(f"evaluation issue dates: {len(cov)}")
    print("staleness after gap fill:", cov.stale_months.value_counts().sort_index().to_dict())
    print("issue dates that needed an archive-gap fill:", (cov.n_filled > 0).sum())
    print(cov[cov.stale_months > 0].to_string(index=False))

    rel = realtime.release_table()
    rel.to_csv(os.path.join(realtime.PROC, "release_table.csv"))
    sub = rel.loc[pd.Period("2021-12", "M"):pd.Period("2024-02", "M")].copy()
    sub["revision_pct"] = (sub.latest / sub.first_release - 1) * 100
    print("\nfirst release vs latest value (mln USD):")
    print(sub.round(1).to_string())


if __name__ == "__main__":
    main()
