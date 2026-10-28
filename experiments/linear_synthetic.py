"""Synthetic data: p-mean regret and group fairness of Explore-then-LinUCB vs LinUCB.

    python -m experiments.linear_synthetic [a] [b] [c] [--force]

(a) benign fixed-arm linear bandit (K = 50, d = 6, i.i.d. positive features);
(b) the rare-direction instance of `linear_stopping_rule`, final p-mean regret
    across p (reuses its runs from results/cache/);
(c) two-group contextual bandit where the minority owns a feature block that
    is non-zero only in its own rounds, for minority shares 30%, 10% and 3%.
"""
import argparse
import os

import numpy as np

from experiments.common import cached_linear, dirs, write_csv, write_curves, write_table
from experiments.linear_stopping_rule import NAME as STOPPING_TAG, make_env as rare_env
from pmlb.linear.envs import BERNOULLI, fixed_arm_env, random_instance, two_group_instance
from pmlb.plotting import ALGO_STYLE, plot_curves, plot_lines, plot_tradeoff
from pmlb.welfare import contextual_p_mean_regret_curve, group_utilities

NAME = "linear_synthetic"
KAPPA = 10.0
PS = [1, 0, -1, -2, -5]
MAIN = ["lin_nash", "lin_nash_tuned", "explore_then_linucb_naive", "explore_then_linucb", "explore_then_greedy"]
CONTEXTUAL = ["explore_then_linucb_naive", "explore_then_linucb", "explore_then_linucb_global",
              "explore_then_greedy"]
SHARES = [0.3, 0.1, 0.03]
MARKERS = {1.0: "o", -2.0: "^"}


def regret_table(tag, env, T, n_trials, algos, force):
    rstar = env.optimal_rewards(T)
    rows, final = [], {a: [] for a in algos}
    curves = {p: [] for p in PS}
    for a in algos:
        for p in PS:
            er, taus, _ = cached_linear(tag, a, env, T, p, n_trials, KAPPA, force)
            reg = contextual_p_mean_regret_curve(er, rstar, p)
            final[a].append(reg[-1])
            curves[p].append((ALGO_STYLE[a][0], ALGO_STYLE[a][1], reg))
            rows.append([ALGO_STYLE[a][0], f"{p}", f"{taus.mean():.0f}", f"{reg[9_999]:.4f}", f"{reg[-1]:.4f}"])
    return rows, final, curves


def expt_a(force):
    res_dir, plot_dir = dirs(NAME)
    T, n = 200_000, 50
    X, theta = random_instance(K=50, d=6)
    env = fixed_arm_env(X, theta, sigma2=0.25, reward_type=BERNOULLI)
    rows, final, curves = regret_table(f"{NAME}_a", env, T, n, MAIN, force)
    write_table(os.path.join(res_dir, "a_benign_summary.md"),
                ["algorithm", "p", "exploration rounds", "p-mean regret at t=1e4", "p-mean regret at T"], rows)
    for p in (0, -2):
        name = "nash" if p == 0 else f"p{p}"
        plot_curves(curves[p], T, f"(a) benign instance, p-mean regret, p = {p}", "p-mean regret",
                    os.path.join(plot_dir, f"a_benign_{name}_regret.png"))
        write_curves(os.path.join(res_dir, f"a_benign_{name}_curves.csv"), T, [(c[0], c[2]) for c in curves[p]])
    m = env.all_means()[0]
    return final, f"(a) K = 50, d = 6, sigma^2 = 0.25, mu* = {m.max():.3f}, min mean = {m.min():.3f}, T = {T}, {n} runs"


def expt_b(force):
    from experiments.linear_stopping_rule import N_TRIALS, T

    env = rare_env()
    _, final, _ = regret_table(STOPPING_TAG, env, T, N_TRIALS, MAIN, force)
    return final


def expt_c(force):
    res_dir, plot_dir = dirs(NAME)
    T, n = 100_000, 30
    rows, csv_rows, points = [], [], []
    min_u = {a: [] for a in CONTEXTUAL}
    agg_u = {a: [] for a in CONTEXTUAL}
    for share in SHARES:
        env = two_group_instance(T, minority_share=share)
        rstar = env.optimal_rewards(T)
        rg = env.round_groups(T)
        for a in CONTEXTUAL:
            for p in (0, -2):
                er, taus, _ = cached_linear(f"{NAME}_c_share{share}", a, env, T, p, n, KAPPA, force)
                u = group_utilities(er, rstar, rg, env.G)
                agg = er.sum() / rstar.sum()
                r0 = contextual_p_mean_regret_curve(er, rstar, 0)[-1]
                r2 = contextual_p_mean_regret_curve(er, rstar, -2)[-1]
                rows.append([f"{100 * share:g}%", ALGO_STYLE[a][0], f"{p}", f"{taus.mean():.0f}",
                             f"{agg:.3f}", f"{u[0]:.3f}", f"{u[1]:.3f}", f"{r0:.4f}", f"{r2:.4f}"])
                csv_rows.append([share, a, p, taus.mean(), agg, u[0], u[1], r0, r2])
                if p == 0:
                    min_u[a].append(u[1])
                    agg_u[a].append(agg)
                points.append((ALGO_STYLE[a][0], ALGO_STYLE[a][1], MARKERS[1.0 if p == 0 else -2.0],
                               agg, u[1], f"{100 * share:g}%"))
    header = ["minority share", "algorithm", "p", "exploration rounds", "aggregate utility",
              "majority utility", "minority utility", "Nash regret at T", "p=-2 regret at T"]
    write_table(os.path.join(res_dir, "c_two_groups_summary.md"), header, rows)
    write_csv(os.path.join(res_dir, "c_two_groups.csv"),
              ["share", "algorithm", "p", "exploration_rounds", "aggregate_utility", "majority_utility",
               "minority_utility", "nash_regret", "p_minus2_regret"], csv_rows)
    xs = [100 * s for s in SHARES]
    plot_lines(xs, [(ALGO_STYLE[a][0], ALGO_STYLE[a][1], min_u[a]) for a in CONTEXTUAL],
               "(c) Minority utility vs minority share (p = 0 runs)", "minority share of rounds (%)",
               "minority utility (share of optimum)", os.path.join(plot_dir, "c_minority_utility.png"), xticks=xs)
    plot_tradeoff(points, "(c) Aggregate utility vs minority utility (circle p = 0, triangle p = -2)",
                  "aggregate utility (share of optimum)", "minority utility (share of optimum)",
                  os.path.join(plot_dir, "c_tradeoff.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", default=["a", "b", "c"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    res_dir, plot_dir = dirs(NAME)
    if "a" in args.experiments or "b" in args.experiments:
        final_a, desc = expt_a(args.force)
        final_b = expt_b(args.force)
        for name, final in (("a_benign", final_a), ("b_rare_direction", final_b)):
            plot_lines(PS, [(ALGO_STYLE[a][0], ALGO_STYLE[a][1], final[a]) for a in MAIN],
                       f"({name[0]}) p-mean regret at T vs p", "p", "p-mean regret at T",
                       os.path.join(plot_dir, f"{name}_regret_vs_p.png"), logy=True, xticks=PS)
            write_csv(os.path.join(res_dir, f"{name}_regret_vs_p.csv"), ["algorithm"] + [f"p={p}" for p in PS],
                      [[a] + [f"{v:.6g}" for v in final[a]] for a in MAIN])
        with open(os.path.join(res_dir, "instances.md"), "w") as f:
            f.write(desc + "\n\n(b) see results/linear_stopping_rule/instance.md\n\n"
                    "(c) d = 8 (z = [1, 1{minority}], 4 item features), K = 10 of 200 items per round, "
                    "Bernoulli rewards, T = 100000, 30 runs\n")
    if "c" in args.experiments:
        expt_c(args.force)


if __name__ == "__main__":
    main()
