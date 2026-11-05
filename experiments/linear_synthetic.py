"""Synthetic linear bandit experiments matching the project presentation.

Usage:
    python -m experiments.linear_synthetic       # run both Experiment 1 & 2
    python -m experiments.linear_synthetic 1     # run Experiment 1 (Benign arms)
    python -m experiments.linear_synthetic 2     # run Experiment 2 (Rare niche arms)
    python -m experiments.linear_synthetic --force

- Experiment 1 (Slide 16): Benign feature arms (K=50, d=6), rapid Phase I exit.
- Experiment 2 (Slide 17): Rare niche features (K=100, d=6), Adaptation 2 achieves 1.8x lower regret.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import dirs, write_curves, write_table
from pmlb.envs import BERNOULLI
from pmlb.linear.algorithms import (
    explore_then_linucb_design_only,
    explore_then_linucb_naive,
)
from pmlb.linear.envs import fixed_arm_env, random_instance
from pmlb.plotting import log_grid
from pmlb.welfare import contextual_p_mean_regret_curve

NAME = "linear_synthetic"
COLOR_V1 = "#2a78d6"  # Royal blue (Adaptation 1: Uniform)
COLOR_V2 = "#d9534f"  # Crimson / Coral (Adaptation 2: D-Optimal)


def _style_ax(ax):
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def expt1(force=False, n_seeds=15):
    """Experiment 1: Benign Feature Arms (Presentation Slide 16)."""
    res_dir, plot_dir = dirs(NAME)
    K, d, T = 50, 6, 50_000
    print(f"[Expt 1] Running Benign Feature Arms (K={K}, d={d}, T={T}, {n_seeds} seeds)...")

    X, theta = random_instance(K=K, d=d, seed=42)
    env = fixed_arm_env(X, theta, sigma2=0.25, reward_type=BERNOULLI)
    args = env.kernel_args(T)
    rstar = env.optimal_rewards(T)

    algos = [
        ("Adaptation 1 (Uniform)", explore_then_linucb_naive, COLOR_V1, "-"),
        ("Adaptation 2 (D-Optimal)", explore_then_linucb_design_only, COLOR_V2, "--"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ts = log_grid(T, start=100)
    named_curves = []
    rows = []

    for label, fn, color, ls in algos:
        er = np.zeros(T)
        taus = []
        for s in range(n_seeds):
            out = np.zeros(T)
            tau = fn(*args, 0, 10.0, s, out)
            er += out
            taus.append(tau)
        er /= n_seeds
        reg = contextual_p_mean_regret_curve(er, rstar, 0)
        avg_tau = int(np.mean(taus))
        legend_label = f"{label} (stop $\\tau \\approx {avg_tau:,}$)"
        ax.plot(ts, reg[ts - 1], label=legend_label, color=color, linestyle=ls, linewidth=2.5)
        named_curves.append((label, reg))
        rows.append([label, f"{avg_tau:,}", f"{reg[9_999]:.4f}", f"{reg[-1]:.4f}"])

    _style_ax(ax)
    ax.set_xscale("log")
    ax.set_xlabel("Round $t$ (log scale)")
    ax.set_ylabel("Nash Regret ($p = 0$)")
    ax.set_title(r"Experiment 1: Benign Feature Arms ($\mu \in [0.5, 0.95]$)", loc="left", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.9)
    fig.tight_layout()

    plot_path = os.path.join(plot_dir, "expt1_variant1_wins.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    write_table(os.path.join(res_dir, "expt1_summary.md"),
                ["Algorithm", "Exploration Rounds (tau)", "Nash Regret (t=1e4)", "Final Regret (T)"], rows)
    write_curves(os.path.join(res_dir, "expt1_curves.csv"), T, named_curves)
    print(f"--> Saved Expt 1 plot: {plot_path}")


def expt2(force=False, n_seeds=15):
    """Experiment 2: Rare Niche Features (Presentation Slide 17)."""
    res_dir, plot_dir = dirs(NAME)
    K, d, T = 100, 6, 30_000
    print(f"[Expt 2] Running Rare Niche Features (K={K}, d={d}, T={T}, {n_seeds} seeds)...")

    theta = np.array([0.3, 0.3, 0.3, 0.1, 0.8, 0.0])
    theta = theta / np.linalg.norm(theta)
    X = np.zeros((K, d))
    rng = np.random.default_rng(42)

    for i in range(K - 2):
        v = rng.uniform(0.1, 0.6, 4)
        v = v / np.linalg.norm(v) * 0.6
        X[i, :4] = v

    X[98] = np.array([0.0, 0.0, 0.0, 0.0, 0.75 / theta[4], 0.2])
    X[98] = X[98] / np.linalg.norm(X[98]) * 0.95
    v = X[98] - (X[98] @ theta) * theta
    v = v / np.linalg.norm(v)
    X[98] = 0.75 * theta + np.sqrt(max(0, 0.95**2 - 0.75**2)) * v
    X[99] = theta * 0.95

    env = fixed_arm_env(X, theta, sigma2=0.25, reward_type=BERNOULLI)
    args = env.kernel_args(T)
    rstar = env.optimal_rewards(T)

    algos = [
        ("Adaptation 1 (Uniform)", explore_then_linucb_naive, COLOR_V1, "-"),
        ("Adaptation 2 (D-Optimal)", explore_then_linucb_design_only, COLOR_V2, "--"),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ts = log_grid(T, start=100)
    named_curves = []
    rows = []

    for label, fn, color, ls in algos:
        er = np.zeros(T)
        taus = []
        for s in range(n_seeds):
            out = np.zeros(T)
            tau = fn(*args, 0, 10.0, s, out)
            er += out
            taus.append(tau)
        er /= n_seeds
        reg = contextual_p_mean_regret_curve(er, rstar, 0)
        avg_tau = int(np.mean(taus))
        legend_label = f"{label} (stop $\\tau \\approx {avg_tau:,}$)"
        ax.plot(ts, reg[ts - 1], label=legend_label, color=color, linestyle=ls, linewidth=2.5)
        named_curves.append((label, reg))
        rows.append([label, f"{avg_tau:,}", f"{reg[9_999]:.4f}", f"{reg[-1]:.4f}"])

    _style_ax(ax)
    ax.set_xscale("log")
    ax.set_xlabel("Round $t$ (log scale)")
    ax.set_ylabel("Nash Regret ($p = 0$)")
    ax.set_title(r"Experiment 2: Rare Niche ($\mu_{\mathrm{common}} \approx 0.33$) — Adaptation 2 Wins ($1.8\times$)",
                 loc="left", fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.9)
    fig.tight_layout()

    plot_path = os.path.join(plot_dir, "expt2_variant2_wins.png")
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    write_table(os.path.join(res_dir, "expt2_summary.md"),
                ["Algorithm", "Exploration Rounds (tau)", "Nash Regret (t=1e4)", "Final Regret (T)"], rows)
    write_curves(os.path.join(res_dir, "expt2_curves.csv"), T, named_curves)
    print(f"--> Saved Expt 2 plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Run synthetic linear bandit experiments.")
    parser.add_argument("expt", nargs="?", choices=["1", "2"], help="Run specific experiment (1 or 2)")
    parser.add_argument("--force", action="store_true", help="Force recomputation")
    args = parser.parse_args()

    if args.expt == "1":
        expt1(force=args.force)
    elif args.expt == "2":
        expt2(force=args.force)
    else:
        expt1(force=args.force)
        expt2(force=args.force)


if __name__ == "__main__":
    main()
