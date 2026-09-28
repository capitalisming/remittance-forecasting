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


if __name__ == "__main__":
    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    fig1_series()
    fig2_mae_by_window()
    print("written", FIG_DIR)
