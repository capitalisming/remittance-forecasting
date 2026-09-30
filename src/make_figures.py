"""Generate the two paper figures (paper/figures/fig1_series.pdf, fig2_mae_by_window.pdf)
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


def fig1_series():
    df = pd.read_csv("../data/processed/release_table.csv")
    df["ref_month"] = pd.PeriodIndex(df["ref_month"], freq="M")
    df = df.dropna(subset=["latest"])
    df = df[df["ref_month"] >= pd.Period("2018-01", "M")]
    t = df["ref_month"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(5.1, 1.3))

    shock_start = pd.Period("2022-02", "M").to_timestamp()
    shock_end = (pd.Period("2022-09", "M") + 1).to_timestamp()
    tz_start = panel.TRANSITION_ZONE[0].to_timestamp()
    tz_end = (panel.TRANSITION_ZONE[-1] + 1).to_timestamp()
    ax.axvspan(shock_start, shock_end, color=SHOCK_COLOR, alpha=0.18, lw=0, zorder=0)
    ax.axvspan(tz_start, tz_end, color=TZ_COLOR, alpha=0.22, lw=0, zorder=0)

    ax.plot(t, df["latest"], color=COLOR_1, lw=1.3, ls="-", label="Latest (revised) value", zorder=3)
    ax.plot(t, df["first_release"], color=COLOR_2, lw=1.1, ls="--", label="First-release value", zorder=2)

    ymax = max(df["latest"].max(), df["first_release"].max()) * 1.38
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 100, 200, 300])
    ax.annotate("2022 shock", xy=(shock_start + (shock_end - shock_start) / 2, ymax * 0.90),
                ha="center", va="top", fontsize=7, color=SHOCK_COLOR)
    ax.annotate("2023--24 redefinition", xy=(tz_end + pd.Timedelta(days=200), ymax * 0.90),
                ha="left", va="top", fontsize=7, color="#5d4fb0")

    ax.set_ylabel("Remittances (million USD)")
    ax.set_xlabel("")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=2.2,
              bbox_to_anchor=(0.0, 1.30))
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig1_series.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2_mae_by_window():
    df = pd.read_csv("../results/expert_comparison_eval.csv", index_col=0, parse_dates=True)
    df["target_month"] = pd.PeriodIndex(df.index, freq="M") + 1
    df = df.dropna(subset=["naive"])

    tz = panel.TRANSITION_ZONE
    main_mask = ~df.target_month.isin(tz)
    shock_mask = df.target_month.astype(str).between("2022-02", "2022-09")
    tz_mask = df.target_month.isin(tz)
    windows = [("Main\n(n=49)", main_mask), ("Shock\n(n=8)", shock_mask), ("Transition\n(n=4)", tz_mask)]

    experts_cols = ["naive", "expanding_ridge", "rolling_ridge_w48", "chronos_zeroshot"]
    labels = ["Seasonal naive", "Expanding ridge (E1)", "Rolling ridge (E2)", "Chronos (E3)"]
    colors = [COLOR_3, COLOR_1, COLOR_4, COLOR_2]
    hatches = [None, "//", None, ".."]

    n_experts = len(experts_cols)
    n_windows = len(windows)
    x = range(n_windows)
    width = 0.19

    fig, ax = plt.subplots(figsize=(5.1, 1.3))
    for i, (col, label, color, hatch) in enumerate(zip(experts_cols, labels, colors, hatches)):
        vals = [df.loc[mask, col].mean() for _, mask in windows]
        offsets = [xi + (i - (n_experts - 1) / 2) * width for xi in x]
        ax.bar(offsets, vals, width=width * 0.92, color=color, label=label,
               hatch=hatch, edgecolor="white", linewidth=0.4)

    ax.set_xticks(list(x))
    ax.set_xticklabels([w[0] for w in windows])
    ax.set_ylabel("MAE (million USD)")
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=7,
              bbox_to_anchor=(0.0, 1.28), handlelength=1.4, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig2_mae_by_window.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_combined():
    import matplotlib.dates as mdates
    import numpy as np
    rt = pd.read_csv("../data/processed/release_table.csv")
    rt["ref_month"] = pd.PeriodIndex(rt["ref_month"], freq="M")
    rt = rt.dropna(subset=["latest"])
    rt = rt[rt["ref_month"] >= pd.Period("2018-01", "M")]
    t = rt["ref_month"].dt.to_timestamp()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.1, 1.55), gridspec_kw={"width_ratios": [1.35, 1]})

    ss = pd.Period("2022-02", "M").to_timestamp(); se = (pd.Period("2022-09", "M") + 1).to_timestamp()
    ts = panel.TRANSITION_ZONE[0].to_timestamp(); te = (panel.TRANSITION_ZONE[-1] + 1).to_timestamp()
    ax1.axvspan(ss, se, color=SHOCK_COLOR, alpha=0.18, lw=0, zorder=0)
    ax1.axvspan(ts, te, color=TZ_COLOR, alpha=0.22, lw=0, zorder=0)
    ax1.plot(t, rt["latest"], color=COLOR_1, lw=1.2, label="Latest", zorder=3)
    ax1.plot(t, rt["first_release"], color=COLOR_2, lw=1.0, ls="--", label="First release", zorder=2)
    ymax = 480
    ax1.set_ylim(0, ymax); ax1.set_yticks([0, 100, 200, 300])
    ax1.text(ss + (se - ss) / 2, 12, "shock", ha="center", va="bottom", fontsize=6.5, color=SHOCK_COLOR)
    ax1.text(te + pd.Timedelta(days=60), 12, "redefinition", ha="left", va="bottom", fontsize=6.5, color="#5d4fb0")
    ax1.set_ylabel("Million USD")
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.8, fontsize=7, bbox_to_anchor=(0, 1.02))
    ax1.set_title("(a) Remittance inflows", fontsize=8, loc="left", pad=3)

    df = pd.read_csv("../results/expert_comparison_eval.csv", index_col=0, parse_dates=True)
    df["target_month"] = pd.PeriodIndex(df.index, freq="M") + 1
    df = df.dropna(subset=["naive"])
    tz = panel.TRANSITION_ZONE
    main_m = ~df.target_month.isin(tz)
    shock_m = df.target_month.astype(str).between("2022-02", "2022-09")
    tz_m = df.target_month.isin(tz)
    wins = [("Main", main_m), ("Shock", shock_m), ("Transition", tz_m)]
    cols = ["naive", "expanding_ridge", "rolling_ridge_w48", "chronos_zeroshot"]
    labs = ["Naive", "E1", "E2", "E3"]
    colors = [COLOR_3, COLOR_1, COLOR_4, COLOR_2]
    hatches = [None, "//", None, ".."]
    w = 0.2
    for i, (c, l, col, h) in enumerate(zip(cols, labs, colors, hatches)):
        vals = [df.loc[m, c].mean() for _, m in wins]
        ax2.bar([x + (i - 1.5) * w for x in range(3)], vals, width=w * 0.92, color=col, label=l,
                hatch=h, edgecolor="white", linewidth=0.4)
    ax2.set_xticks(range(3)); ax2.set_xticklabels([n for n, _ in wins], fontsize=7)
    ax2.set_ylabel("MAE (million USD)")
    ax2.set_ylim(0, 150)
    ax2.legend(loc="upper left", frameon=False, ncol=4, fontsize=6.5, handlelength=1.0,
               columnspacing=0.7, handletextpad=0.4, bbox_to_anchor=(-0.02, 1.02))
    ax2.set_title("(b) MAE by window", fontsize=8, loc="left", pad=3)
    fig.tight_layout(w_pad=0.8)
    fig.savefig(f"{FIG_DIR}/fig_combined.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    fig1_series()
    fig2_mae_by_window()
    fig_combined()
    print("written", FIG_DIR)

