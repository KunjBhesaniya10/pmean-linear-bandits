"""Phase 3, MovieLens-1M: utility vs fairness across gender, age and occupation groups.

    python scripts/download_movielens.py      # once
    python -m experiments.phase3_movielens [--force]

Semi-synthetic linear contextual bandit built in `pmlb.data.movielens`
(d = 180, K = 20 candidate movies per round, Bernoulli "like" rewards).
Explore-then algorithms are run with the paper's kappa = 10 and with kappa = 1.
"""
import argparse
import os

import numpy as np

from experiments.common import cached_linear, dirs, write_csv, write_curves, write_table
from pmlb.data.movielens import make_env
from pmlb.plotting import ALGO_STYLE, plot_curves, plot_grouped_bars, plot_tradeoff
from pmlb.welfare import contextual_p_mean_regret_curve, group_utilities, p_mean

PHASE = "phase3_movielens"
T = 50_000
N_TRIALS = 20
KAPPAS = [10.0, 1.0]
PS = [0, -2]  # runs; p = 1 and -1 share the p = 0 run (p_a = 1)
REGRET_PS = [1, 0, -2]
ALGOS = ["linucb", "lin_greedy", "explore_then_linucb_naive", "explore_then_linucb",
         "explore_then_linucb_global", "explore_then_greedy"]
ATTRS = ["gender", "age", "occupation"]


def configs():
    for a in ALGOS:
        if a in ("linucb", "lin_greedy"):
            yield a, 0, None
        else:
            for kappa in KAPPAS:
                for p in PS:
                    yield a, p, kappa


def label(a, p, kappa):
    name = ALGO_STYLE[a][0]
    return name if kappa is None else f"{name} [p={p}, κ={kappa:g}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    res_dir, plot_dir = dirs(PHASE)
    env = make_env(T)
    rstar = env.optimal_rewards(T)
    rg = env.round_groups(T)
    attr = np.array(env.group_attr)
    names = np.array(env.group_names)
    share = np.bincount(rg.ravel(), minlength=env.G) / T

    rows, util_rows, points, curves, bars = [], [], [], [], {}
    for a, p, kappa in configs():
        er, taus, _ = cached_linear(PHASE, a, env, T, p, N_TRIALS, kappa or 10.0, args.force)
        u = group_utilities(er, rstar, rg, env.G)
        agg = er.sum() / rstar.sum()
        regrets = [contextual_p_mean_regret_curve(er, rstar, q)[-1] for q in REGRET_PS]
        worst = [f"{u[attr == x].min():.3f} ({names[attr == x][np.argmin(u[attr == x])]})" for x in ATTRS]
        gw = p_mean(u, -2)
        rows.append([ALGO_STYLE[a][0], "-" if kappa is None else f"{p}", "-" if kappa is None else f"{kappa:g}",
                     f"{taus.mean():.0f}", f"{agg:.4f}"] + [f"{r:.4f}" for r in regrets] + worst + [f"{gw:.4f}"])
        util_rows.append([a, p, kappa or ""] + [f"{v:.5f}" for v in u])
        lab = label(a, p, kappa)
        points.append((ALGO_STYLE[a][0], ALGO_STYLE[a][1], "o" if kappa in (None, 1.0) else "s",
                       agg, u.min(), None if kappa is None else f"p={p}, κ={kappa:g}"))
        if p == 0 and kappa in (None, 1.0):
            curves.append((ALGO_STYLE[a][0], ALGO_STYLE[a][1],
                           contextual_p_mean_regret_curve(er, rstar, 0)))
            bars[a] = u

    header = (["algorithm", "p", "κ", "exploration rounds", "aggregate utility"]
              + [f"p-mean regret, p={q}" for q in REGRET_PS]
              + [f"worst {x} group" for x in ATTRS] + ["group welfare M_-2 over all 30 groups"])
    write_table(os.path.join(res_dir, "summary.md"), header, rows)
    write_csv(os.path.join(res_dir, "group_utilities.csv"), ["algorithm", "p", "kappa"] + list(names), util_rows)
    write_csv(os.path.join(res_dir, "groups.csv"), ["group", "attribute", "share_of_rounds"],
              [[n, x, f"{s:.4f}"] for n, x, s in zip(names, attr, share)])
    write_curves(os.path.join(res_dir, "nash_regret_curves.csv"), T, [(c[0], c[2]) for c in curves])
    plot_curves(curves, T, "MovieLens-1M, Nash regret (κ = 1 for explore-then)", "Nash regret",
                os.path.join(plot_dir, "nash_regret.png"))

    shown = ["linucb", "lin_greedy", "explore_then_linucb_naive", "explore_then_linucb"]
    for x in ATTRS:
        idx = np.where(attr == x)[0]
        ticks = [f"{names[i]} ({100 * share[i]:.1f}%)" for i in idx]
        lo = min(bars[a][idx].min() for a in shown)
        plot_grouped_bars(ticks, [(ALGO_STYLE[a][0], ALGO_STYLE[a][1], bars[a][idx]) for a in shown],
                          f"MovieLens-1M, utility by {x} group (share of rounds in brackets; p = 0, κ = 1)",
                          "utility (share of optimum)", os.path.join(plot_dir, f"groups_{x}.png"),
                          ylim=(max(0.0, lo - 0.05), 1.0), figsize=(12 if x == "occupation" else 8, 4.8),
                          rotate=45 if x == "occupation" else 0)
    plot_tradeoff(points, "MovieLens-1M: aggregate vs worst-group utility (circle κ = 1, square κ = 10)",
                  "aggregate utility (share of optimum)", "worst-group utility (share of optimum)",
                  os.path.join(plot_dir, "tradeoff.png"))


if __name__ == "__main__":
    main()
