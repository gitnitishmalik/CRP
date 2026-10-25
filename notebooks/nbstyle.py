"""Matplotlib style and chart helpers for the notebooks, matching the dashboard's palette and rules."""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
STATUS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#ec835a", "Very High": "#d03b3b"}
DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#1c5cab", "#86b6ef", "#f0efec", "#f1a09f", "#b8302f"])

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 110, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlecolor": INK, "legend.frameon": False,
})


def rate_barh(ax, labels, values, average=None, title=None, fmt="{:.1%}", color=BLUE):
    """One series of horizontal bars with the value at each tip and an optional average line."""
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.5, color=color)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        ax.text(v, yi, " " + fmt.format(v), va="center", color=INK_2, fontsize=9)
    if average is not None:
        ax.axvline(average, color=INK_2, lw=1)
        ax.annotate(f"average {fmt.format(average)}", (average, -0.6), ha="center", color=INK_2, fontsize=9,
                    annotation_clip=False)
    if "%" in fmt:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, max(values) * 1.22)
    if title:
        ax.set_title(title, pad=14)


def paired_barh(ax, labels, first, second, names, title=None):
    """Two measures in the same unit side by side (blue, orange) with a legend."""
    y = np.arange(len(labels))
    ax.barh(y - 0.17, first, height=0.3, color=BLUE, label=names[0])
    ax.barh(y + 0.17, second, height=0.3, color=ORANGE, label=names[1])
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(loc="lower right", fontsize=9)
    if title:
        ax.set_title(title, pad=14)


def heatmap(ax, rate, count, average, title=None):
    """Default rate per cell, diverging around the average; * marks cells under 100 accounts."""
    z = rate.to_numpy(dtype=float)
    spread = np.nanmax(np.abs(z - average))
    image = ax.imshow(z, cmap=DIVERGING, norm=TwoSlopeNorm(vcenter=average, vmin=average - spread,
                                                           vmax=average + spread), aspect="auto")
    ax.set_xticks(range(z.shape[1]), [str(c) for c in rate.columns])
    ax.set_yticks(range(z.shape[0]), [str(r) for r in rate.index])
    ax.xaxis.tick_top()
    ax.grid(False)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if np.isnan(z[i, j]):
                continue
            n = count.iloc[i, j]
            dark = abs(z[i, j] - average) > 0.6 * spread
            ax.text(j, i, f"{z[i, j]:.1%}{'*' if n < 100 else ''}\nn={n:,}", ha="center", va="center", fontsize=8.5,
                    color="white" if dark else INK)
    bar = plt.colorbar(image, ax=ax, fraction=0.04, pad=0.02, format=PercentFormatter(1.0, decimals=0))
    bar.set_ticks([t for t in np.arange(0, 1.01, 0.1) if average - spread <= t <= average + spread])
    bar.outline.set_visible(False)
    if title:
        ax.set_title(title, pad=28)
