"""MovieLens-1M dynamic streaming contextual bandit benchmark.

Evaluates Adaptation 1 (Uniform Exploration) vs Adaptation 2 (50/50 D-Optimal Exploration)
on recommendation regret dynamics and demographic subgroup equity.

Usage:
    python scripts/download_movielens.py      # download dataset if not already present
    python -m experiments.movielens [--force]
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import cached_linear, dirs, write_csv, write_curves, write_table
from pmlb.data.movielens import make_env
from pmlb.plotting import ALGO_STYLE, log_grid
from pmlb.welfare import contextual_p_mean_regret_curve, group_utilities, p_mean

NAME = "movielens"
T = 50_000
N_TRIALS = 20
KAPPA = 1.0  # kappa=1.0 for contextual high-dimensional recommendation (d=180)
PS = [0, -2]  # evaluations; p=1 shares the p=0 exploration run
REGRET_PS = [1, 0, -2]
ALGOS = ["explore_then_linucb_naive", "explore_then_linucb_design_only"]
ATTRS = ["gender", "age", "occupation"]

COLOR_V1 = "#2a78d6"  # Royal blue (Adaptation 1: Uniform)
COLOR_V2 = "#d9534f"  # Crimson / Coral (Adaptation 2: D-Optimal)


def _style_ax(ax):
    ax.grid(True, linestyle="--", alpha=0.4, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def generate_presentation_plots(plot_dir, t_sub, v1_curve, v2_curve, tau_v1, tau_v2,
                                v1_regrets, v2_regrets, v1_demog, v2_demog):
    """Generate the two presentation figures (Slide 18 & 19)."""
    # 1. Regret Dynamics & Scaling across p (Slide 18)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax1.plot(t_sub, v1_curve, label=f"Adaptation 1 (Uniform, stop $\\tau \\approx {int(tau_v1):,}$)",
             color=COLOR_V1, linewidth=2.5)
    ax1.plot(t_sub, v2_curve, label=f"Adaptation 2 (D-Optimal, stop $\\tau \\approx {int(tau_v2):,}$)",
             color=COLOR_V2, linewidth=2.5, linestyle="--")
    ax1.set_xscale("log")
    ax1.set_xlabel("Round $t$ (log scale)")
    ax1.set_ylabel("Nash Regret ($p = 0$)")
    ax1.set_title("Nash Regret Trajectory ($p = 0$)", loc="left", fontweight="bold", fontsize=12)
    _style_ax(ax1)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.9, fontsize=9.5)

    objectives = ["Utilitarian ($p=1$)", "Nash ($p=0$)", "Inequity-Averse ($p=-2$)"]
    x = np.arange(len(objectives))
    width = 0.32
    r1 = ax2.bar(x - width/2, v1_regrets, width, label="Adaptation 1 (Uniform)", color=COLOR_V1, alpha=0.9)
    r2 = ax2.bar(x + width/2, v2_regrets, width, label="Adaptation 2 (D-Optimal)", color=COLOR_V2, alpha=0.9)
    _style_ax(ax2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(objectives, fontsize=10)
    ax2.set_ylabel("Cumulative Regret at $T=50{,}000$")
    ax2.set_title("Regret by Welfare Objective", loc="left", fontweight="bold", fontsize=12)
    ax2.legend(frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.9, fontsize=9.5)

    for rect in list(r1) + list(r2):
        h = rect.get_height()
        ax2.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9.5)
    ax2.set_ylim(0, max(max(v1_regrets), max(v2_regrets)) * 1.25)
    fig.tight_layout()

    out1 = os.path.join(plot_dir, "movielens_clean.png")
    fig.savefig(out1, dpi=200)
    plt.close(fig)

    # 2. Demographic Equity & Group Welfare (Slide 19)
    groups = [
        "Farmer (0.3% pop)",
        "Unemployed",
        "Tradesman",
        "Elderly (Age 56+)",
        "Female Users",
        "Male Users",
        "Group Welfare $M_{-2}$"
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    y = np.arange(len(groups))
    height = 0.35

    rects1 = ax.barh(y + height/2, v1_demog, height, label="Adaptation 1 (Uniform + Best-Arm)",
                     color=COLOR_V1, alpha=0.9)
    rects2 = ax.barh(y - height/2, v2_demog, height, label="Adaptation 2 (D-Optimal + Best-Arm)",
                     color=COLOR_V2, alpha=0.9)
    _style_ax(ax)
    ax.set_yticks(y)
    ax.set_yticklabels(groups, fontsize=10)
    ax.set_xlabel("Normalized Group Utility (Share of Optimum, 1.0 = Perfect)")
    ax.set_title("MovieLens-1M Demographic Equity: Utility by Subgroup", loc="left", fontweight="bold", fontsize=12)
    ax.set_xlim(0.75, 1.0)
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.9, fontsize=9.5)

    for rect in rects1:
        w = rect.get_width()
        ax.annotate(f"{w:.3f}", xy=(w, rect.get_y() + rect.get_height()/2),
                     xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                     fontsize=9, fontweight="bold", color=COLOR_V1)
    for rect in rects2:
        w = rect.get_width()
        ax.annotate(f"{w:.3f}", xy=(w, rect.get_y() + rect.get_height()/2),
                     xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                     fontsize=9, fontweight="bold", color=COLOR_V2)
    fig.tight_layout()

    out2 = os.path.join(plot_dir, "movielens_demographics.png")
    fig.savefig(out2, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force recomputation")
    args = parser.parse_args()

    res_dir, plot_dir = dirs(NAME)
    env = make_env(T)
    rstar = env.optimal_rewards(T)
    rg = env.round_groups(T)
    attr = np.array(env.group_attr)
    names = np.array(env.group_names)
    share = np.bincount(rg.ravel(), minlength=env.G) / T

    rows, util_rows, curves = [], [], []
    algo_taus, algo_curves, algo_regrets, algo_utils = {}, {}, {}, {}

    for a in ALGOS:
        algo_regrets[a] = {}
        for p in PS:
            er, taus, _ = cached_linear(NAME, a, env, T, p, N_TRIALS, KAPPA, args.force)
            u = group_utilities(er, rstar, rg, env.G)
            agg = er.sum() / rstar.sum()
            regrets = {q: contextual_p_mean_regret_curve(er, rstar, q)[-1] for q in REGRET_PS}
            if p == 0:
                algo_taus[a] = taus.mean()
                algo_curves[a] = contextual_p_mean_regret_curve(er, rstar, 0)
                algo_utils[a] = u
                algo_regrets[a][1] = regrets[1]
                algo_regrets[a][0] = regrets[0]
            if p == -2:
                algo_regrets[a][-2] = regrets[-2]

            worst = [f"{u[attr == x].min():.3f} ({names[attr == x][np.argmin(u[attr == x])]})" for x in ATTRS]
            gw = p_mean(u, -2)
            label = ALGO_STYLE[a][0]
            rows.append([label, f"{p}", f"{KAPPA:g}", f"{taus.mean():.0f}", f"{agg:.4f}"]
                        + [f"{regrets[q]:.4f}" for q in REGRET_PS] + worst + [f"{gw:.4f}"])
            util_rows.append([a, p, KAPPA] + [f"{v:.5f}" for v in u])

    header = (["Algorithm", "p", "kappa", "Exploration Rounds", "Aggregate Utility"]
              + [f"p-mean Regret (p={q})" for q in REGRET_PS]
              + [f"Worst {x.capitalize()}" for x in ATTRS] + ["Group Welfare M_-2"])
    write_table(os.path.join(res_dir, "summary.md"), header, rows)
    write_csv(os.path.join(res_dir, "group_utilities.csv"), ["algorithm", "p", "kappa"] + list(names), util_rows)
    write_csv(os.path.join(res_dir, "groups.csv"), ["group", "attribute", "share_of_rounds"],
              [[n, x, f"{s:.4f}"] for n, x, s in zip(names, attr, share)])

    # Extract curves and demographic values for presentation plots
    ts = log_grid(T, start=100)
    v1_curve = algo_curves["explore_then_linucb_naive"][ts - 1]
    v2_curve = algo_curves["explore_then_linucb_design_only"][ts - 1]
    tau_v1 = algo_taus["explore_then_linucb_naive"]
    tau_v2 = algo_taus["explore_then_linucb_design_only"]

    v1_regs = [algo_regrets["explore_then_linucb_naive"][q] for q in [1, 0, -2]]
    v2_regs = [algo_regrets["explore_then_linucb_design_only"][q] for q in [1, 0, -2]]

    u1 = algo_utils["explore_then_linucb_naive"]
    u2 = algo_utils["explore_then_linucb_design_only"]

    # Demographic indices:
    # 0: Male, 1: Female, 2-8: Age bands (8 is 56+), 9-29: Occupations (17 is farmer, 28 is tradesman, 29 is unemployed)
    # Get exact group indices by name:
    name_list = list(names)
    i_farmer = name_list.index("farmer")
    i_unemp = name_list.index("unemployed")
    i_trades = name_list.index("tradesman/craftsman")
    i_elderly = name_list.index("age 56+")
    i_female = name_list.index("female")
    i_male = name_list.index("male")

    v1_demog = [
        u1[i_farmer], u1[i_unemp], u1[i_trades], u1[i_elderly], u1[i_female], u1[i_male], p_mean(u1, -2)
    ]
    v2_demog = [
        u2[i_farmer], u2[i_unemp], u2[i_trades], u2[i_elderly], u2[i_female], u2[i_male], p_mean(u2, -2)
    ]

    generate_presentation_plots(plot_dir, ts, v1_curve, v2_curve, tau_v1, tau_v2,
                                v1_regs, v2_regs, v1_demog, v2_demog)
    print(f"--> Saved MovieLens clean plots in: {plot_dir}")


if __name__ == "__main__":
    main()
