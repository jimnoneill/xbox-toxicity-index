"""Shared style and helpers for the manuscript figures (read-only on all inputs; writes only to output/v7/figs)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.expanduser("~/x_box/pipeline")
V7 = ROOT + "/output/v7"
DATA = ROOT + "/data"
OUT = V7 + "/figs"
os.makedirs(OUT, exist_ok=True)

R_COL = "#B22222"   # Republicans
D_COL = "#1F77B4"   # Democrats
# Okabe-Ito (colorblind safe) for non-party encodings
BLUE, ORANGE, GREEN, YELLOW, SKY, PINK, BLACK, GREY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9", "#CC79A7", "#222222", "#8a8a8a"
W1, W2 = 3.5, 7.2


def style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
        "lines.linewidth": 1.2, "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 100,
    })


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def plab(ax, s, dx=-0.0, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def kde(x, grid, bw=None):
    x = np.asarray(x, float)
    bw = bw or 1.06 * x.std(ddof=1) * len(x) ** (-0.2)
    z = (grid[:, None] - x[None, :]) / bw
    return np.exp(-0.5 * z ** 2).sum(1) / (len(x) * bw * np.sqrt(2 * np.pi))


def mean_ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    return m, m - 1.96 * se, m + 1.96 * se
