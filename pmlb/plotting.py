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
