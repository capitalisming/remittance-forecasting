"""Generate the paper figure (paper/figures/fig_series.pdf)
from already-computed data/results -- no new analysis, just visualizing existing numbers
(release_table.csv, expert_comparison_eval.csv) so Table 1 and the data description have a
visual counterpart.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

import panel

FIG_DIR = "../paper/figures"

COLOR_1 = "#2a78d6"  # blue
COLOR_2 = "#eb6834"  # orange
COLOR_3 = "#1baf7a"  # aqua
COLOR_4 = "#eda100"  # yellow
SHOCK_COLOR = "#e34948"
TZ_COLOR = "#9085e9"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
    "axes.axisbelow": True,
})


def fig_series():
    import matplotlib.dates as mdates
    rt = pd.read_csv("../data/processed/release_table.csv")
    rt["ref_month"] = pd.PeriodIndex(rt["ref_month"], freq="M")
    rt = rt.dropna(subset=["latest"])
    rt = rt[rt["ref_month"] >= pd.Period("2018-01", "M")]
    t = rt["ref_month"].dt.to_timestamp()

    fig, ax1 = plt.subplots(figsize=(5.1, 1.2))
    ss = pd.Period("2022-02", "M").to_timestamp(); se = (pd.Period("2022-09", "M") + 1).to_timestamp()
    ts = panel.TRANSITION_ZONE[0].to_timestamp(); te = (panel.TRANSITION_ZONE[-1] + 1).to_timestamp()
    ax1.axvspan(ss, se, color=SHOCK_COLOR, alpha=0.18, lw=0, zorder=0)
    ax1.axvspan(ts, te, color=TZ_COLOR, alpha=0.22, lw=0, zorder=0)
    ax1.plot(t, rt["latest"], color=COLOR_1, lw=1.2, zorder=3)
    fr = rt[rt["ref_month"].between(pd.Period("2022-12", "M"), pd.Period("2024-05", "M"))]
    ax1.plot(fr["ref_month"].dt.to_timestamp(), fr["first_release"], color=COLOR_2, lw=1.3, ls="--", zorder=4)
    ax1.set_ylim(0, 400); ax1.set_yticks([0, 100, 200, 300])
    ax1.text(pd.Timestamp("2018-03-01"), 335, "Latest (revised)", ha="left", va="bottom", fontsize=7, color=COLOR_1)
    ax1.text(pd.Timestamp("2023-02-01"), 132, "First release", ha="center", va="top", fontsize=7, color=COLOR_2)
    ax1.text(ss + (se - ss) / 2, 12, "shock", ha="center", va="bottom", fontsize=6.5, color=SHOCK_COLOR)
    ax1.text(te + pd.Timedelta(days=60), 12, "redefinition", ha="left", va="bottom", fontsize=6.5, color="#5d4fb0")
    ax1.set_ylabel("Million USD")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_series.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    fig_series()
    print("written", FIG_DIR)

