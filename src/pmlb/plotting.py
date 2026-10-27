"""Plot helpers for regret curves."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Fixed colour per algorithm so every figure reads the same way.
ALGO_STYLE = {
    "welfarist_ucb": ("Welfarist-UCB", "#2a78d6"),
    "explore_then_ucb": ("Explore-then-UCB", "#eb6834"),
    "ncb": ("NCB", "#eb6834"),
    "ucb": ("UCB (no exploration phase)", "#1baf7a"),
    # linear contextual bandits
    "explore_then_linucb": ("Explore-then-LinUCB (ours)", "#2a78d6"),
    "explore_then_linucb_global": ("Explore-then-LinUCB, one stop for all groups", "#eda100"),
    "explore_then_linucb_naive": ("Explore-then-LinUCB, naive rule", "#eb6834"),
    "explore_then_linucb_design_only": ("design only (naive rule)", "#008300"),
    "explore_then_linucb_coverage_only": ("coverage rule only (uniform)", "#6250d6"),
    "explore_then_greedy": ("Explore-then-Greedy (ours)", "#e87ba4"),
    "explore_then_greedy_naive": ("Explore-then-Greedy, naive rule", "#e34948"),
    "linucb": ("LinUCB", "#1baf7a"),
    "lin_greedy": ("Greedy (ridge)", "#555555"),
}
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def log_grid(T, start=100, num=400):
    """Log-spaced 1-based round indices in [start, T]."""
    return np.unique(np.geomspace(start, T, num).astype(np.int64))


def plot_curves(curves, T, title, ylabel, path, start=100):
    """curves: list of (label, color, regret_curve) with regret_curve indexed by t-1."""
    ts = log_grid(T, start)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for label, color, curve in curves:
        ax.plot(ts, curve[ts - 1], label=label, color=color, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("round t (horizon T = %.0e)" % T)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(alpha=0.3, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _style(ax):
    ax.grid(alpha=0.3, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_grouped_bars(groups, series, title, ylabel, path, ylim=None, figsize=(9, 4.4), rotate=0):
    """groups: x tick labels; series: list of (label, color, values aligned with groups)."""
    fig, ax = plt.subplots(figsize=figsize)
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (label, color, vals) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=label, color=color)
    ax.set_xticks(x, groups, rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if ylim:
        ax.set_ylim(*ylim)
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lines(xs, series, title, xlabel, ylabel, path, logy=False, xticks=None):
    """series: list of (label, color, ys aligned with xs)."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for label, color, ys in series:
        ax.plot(xs, ys, label=label, color=color, lw=2, marker="o", ms=4)
    if logy:
        ax.set_yscale("log")
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tradeoff(points, title, xlabel, ylabel, path):
    """points: list of (label, color, marker, x, y, annotation or None)."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    seen = set()
    for label, color, marker, x, y, note in points:
        ax.scatter(x, y, color=color, marker=marker, s=60, label=None if label in seen else label, zorder=3)
        seen.add(label)
        if note:
            ax.annotate(note, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7, color="#333333")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
