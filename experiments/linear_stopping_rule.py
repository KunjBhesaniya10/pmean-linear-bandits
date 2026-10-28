"""Does the adapted stopping rule learn rare feature directions before exploiting?

    python -m experiments.linear_stopping_rule [--force]

Fixed-arm linear bandit (d = 6, K = 400) where only 4 arms (1%) use the two
rare feature directions, and the optimum (0.95) is one of them; the best
common arm earns 0.85. Ablation of Explore-then-LinUCB's two changes
(D-optimal sampling, plausible-set stopping rule) against the direct transfer
of Welfarist-UCB, with a LinUCB and a greedy Phase II, for p in {1, 0, -1, -2, -5}.
"""
import argparse
import os

import numpy as np

from experiments.common import cached_linear, dirs, write_curves, write_table
from pmlb.linear.envs import BERNOULLI, fixed_arm_env, rare_direction_instance
from pmlb.plotting import ALGO_STYLE, plot_curves, plot_grouped_bars, plot_lines
from pmlb.welfare import contextual_p_mean_regret_curve

NAME = "linear_stopping_rule"
T = 200_000
N_TRIALS = 50
KAPPA = 10.0
PS = [1, 0, -1, -2, -5]
ALGOS = ["explore_then_linucb_naive", "explore_then_linucb_design_only", "explore_then_linucb_coverage_only",
         "explore_then_linucb", "explore_then_greedy_naive", "explore_then_greedy", "lin_nash", "lin_nash_tuned"]


def make_env():
    X, theta = rare_direction_instance(n_common=396, common_high=0.85)
    return fixed_arm_env(X, theta, sigma2=0.25, reward_type=BERNOULLI)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    res_dir, plot_dir = dirs(NAME)
    env = make_env()
    means = env.all_means()[0]
    mu_star, mu_2 = np.sort(means)[-1], np.sort(means)[-2]
    rstar = env.optimal_rewards(T)

    rows, curves0, tau_by_p, found0 = [], [], {a: [] for a in ALGOS}, []
    for algo in ALGOS:
        for p in PS:
            er, taus, finals = cached_linear(NAME, algo, env, T, p, N_TRIALS, KAPPA, args.force)
            regret = contextual_p_mean_regret_curve(er, rstar, p)
            found = np.mean(finals > (mu_star + mu_2) / 2)
            rows.append([ALGO_STYLE[algo][0], f"{p}", f"{taus.mean():.0f} ± {taus.std():.0f}",
                         f"{100 * found:.0f}%", f"{regret[9_999]:.4f}", f"{regret[-1]:.4f}"])
            tau_by_p[algo].append(taus.mean())
            if p == 0:
                curves0.append((ALGO_STYLE[algo][0], ALGO_STYLE[algo][1], regret))
                found0.append(100 * found)

    write_table(os.path.join(res_dir, "summary.md"),
                ["algorithm", "p", "exploration rounds (mean ± sd)", "runs ending on the optimal arm",
                 "p-mean regret at t=1e4", "p-mean regret at T"], rows)
    write_curves(os.path.join(res_dir, "nash_regret_curves.csv"), T, [(c[0], c[2]) for c in curves0])
    plot_curves(curves0, T, "Nash regret, rare-direction instance (1% of arms span 2 of 6 dims)",
                "Nash regret", os.path.join(plot_dir, "nash_regret.png"))
    explore_algos = [a for a in ALGOS if not a.startswith("lin_nash")]
    plot_lines(PS, [(ALGO_STYLE[a][0], ALGO_STYLE[a][1], tau_by_p[a]) for a in explore_algos],
               "Exploration length vs p (horizon 2e5)", "p", "exploration rounds",
               os.path.join(plot_dir, "exploration_vs_p.png"), logy=True, xticks=PS)
    plot_grouped_bars([ALGO_STYLE[a][0] for a in ALGOS],
                      [("p = 0", "#2a78d6", found0)], "Runs that end on the optimal (rare-direction) arm",
                      "% of 50 runs", os.path.join(plot_dir, "found_optimum.png"), ylim=(0, 105),
                      figsize=(10, 5.2), rotate=30)
    with open(os.path.join(res_dir, "instance.md"), "w") as f:
        f.write(f"d = {env.d}, K = {env.K}, sigma^2 = {env.sigma2}, T = {T}, {N_TRIALS} runs, kappa = {KAPPA}\n\n")
        f.write(f"mu* = {mu_star:.3f} (rare-direction arm), best common arm = {mu_2:.3f}, "
                f"min = {means.min():.3f}\n")


if __name__ == "__main__":
    main()
