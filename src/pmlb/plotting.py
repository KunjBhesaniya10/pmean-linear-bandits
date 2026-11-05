"""Plotting helpers for regret curves and demographic welfare."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Canonical algorithm styles and colors matching presentation figures
ALGO_STYLE = {
    "welfarist_ucb": ("Welfarist-UCB", "#2a78d6"),
    "explore_then_ucb": ("Explore-then-UCB", "#eb6834"),
    "ncb": ("NCB", "#eb6834"),
    "explore_then_linucb_naive": ("Adaptation 1 (Uniform)", "#2a78d6"),
    "explore_then_linucb_design_only": ("Adaptation 2 (D-Optimal)", "#d9534f"),
    "explore_then_linucb": ("Explore-then-LinUCB (Coverage)", "#6250d6"),
    "explore_then_linucb_global": ("Explore-then-LinUCB (Global)", "#eda100"),
    "explore_then_linucb_coverage_only": ("Coverage Only", "#6250d6"),
    "explore_then_greedy": ("Explore-then-Greedy", "#e87ba4"),
    "lin_nash": ("LinNash (Sawarni et al.)", "#2ca02c"),
    "lin_nash_tuned": ("LinNash, tuned c=1", "#008300"),
    "linucb": ("LinUCB", "#1baf7a"),
    "lin_greedy": ("Greedy (ridge)", "#555555"),
    "fair_lin_ucb": ("FairLinUCB (Sarkar et al.)", "#9933cc"),
    "fair_lin_bandit": ("FairLinBandit", "#9933cc"),
}
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def log_grid(T, start=100, num=400):
    """Generate log-spaced 1-based round indices in [start, T]."""
    return np.unique(np.geomspace(start, T, num).astype(np.int64))


def _style_ax(ax):
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def plot_curves(curves, T, title, ylabel, path, start=100):
    """Plot cumulative regret curves over rounds t."""
    ts = log_grid(T, start)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for label, color, curve in curves:
        ax.plot(ts, curve[ts - 1], label=label, color=color, lw=2.2)
    ax.set_xscale("log")
    ax.set_xlabel(f"Round $t$ (Horizon $T = {T:.0e}$)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    _style_ax(ax)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.9, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_grouped_bars(groups, series, title, ylabel, path, ylim=None, figsize=(9, 4.4), rotate=0):
    """Plot grouped bar charts across categories."""
    fig, ax = plt.subplots(figsize=figsize)
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (label, color, vals) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=label, color=color, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=rotate, ha="right" if rotate else "center", fontsize=9.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    _style_ax(ax)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.9, fontsize=9.0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_lines(xs, series, title, xlabel, ylabel, path, logy=False, xticks=None):
    """Plot metric lines vs parameter (e.g. regret vs p)."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for label, color, ys in series:
        ax.plot(xs, ys, label=label, color=color, lw=2.2, marker="o", ms=5)
    if logy:
        ax.set_yscale("log")
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    _style_ax(ax)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.9, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_tradeoff(points, title, xlabel, ylabel, path):
    """Plot scatter tradeoff between aggregate utility and worst-group utility."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for label, color, marker, x, y, note in points:
        ax.scatter([x], [y], color=color, marker=marker, s=60, label=label)
        if note:
            ax.annotate(note, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=8.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    _style_ax(ax)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.9, fontsize=9.0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
