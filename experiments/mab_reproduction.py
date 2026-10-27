"""Reproduce Section 5 of Sarkar, Pandey & Ray Chowdhury (2025), experiments (a)-(f).

    python -m experiments.mab_reproduction            # all experiments
    python -m experiments.mab_reproduction a c        # a subset

Instances are drawn exactly as in the authors' main.py (np.random.seed(42),
then the four instances in order), so the arm means match theirs. Per-run
randomness uses our own seeds. Averaged expected-reward curves are cached in
results/cache/mab_reproduction/ and reused on later runs unless --force is given.
Figures go to plots/mab_reproduction/, tables and curves to results/mab_reproduction/.
"""
import argparse
import csv
import os
import zlib

import numpy as np

from pmlb.envs import bernoulli_instance, gaussian_instance
from pmlb.plotting import ALGO_STYLE, SERIES_COLORS, log_grid, plot_curves
from pmlb.runner import run_mab
from pmlb.welfare import p_mean_regret_curve

N_TRIALS = 50
K = 50
RESULTS = os.path.join("results", "mab_reproduction")
PLOTS = os.path.join("plots", "mab_reproduction")
CACHE = os.path.join("results", "cache", "mab_reproduction")


def make_instances():
    np.random.seed(42)
    return {
        "a": bernoulli_instance(K),
        "b": gaussian_instance(K),
        "cde": gaussian_instance(K),
        "f": gaussian_instance(K),
    }


def experiment_specs(inst):
    """name -> (instance, T, title, [(algorithm, p), ...])"""
    specs = {
        "a": (inst["a"], 10_000_000, "(a) Nash regret, Bernoulli arms",
              [("ncb", 0), ("welfarist_ucb", 0), ("ucb", 0)]),
        "b": (inst["b"], 1_000_000, "(b) Nash regret, Gaussian arms (σ = 20)",
              [("ncb", 0), ("welfarist_ucb", 0), ("ucb", 0)]),
        "f": (inst["f"], 1_000_000, "(f) Welfarist-UCB across p, Gaussian arms",
              [("welfarist_ucb", p) for p in (1, 0, -1, -2, -5)]),
    }
    for name, p in (("c", 0.5), ("d", -0.5), ("e", -1.5)):
        specs[name] = (inst["cde"], 1_000_000, f"({name}) p-mean regret, p = {p}, Gaussian arms",
                       [("explore_then_ucb", p), ("welfarist_ucb", p), ("ucb", p)])
    return dict(sorted(specs.items()))


def cached_run(expt, algorithm, instance, T, p, force):
    path = os.path.join(CACHE, f"{expt}_{algorithm}_p{p}.npz")
    if os.path.exists(path) and not force:
        d = np.load(path)
        return d["expected_rewards"], d["taus"]
    seed = zlib.crc32(f"{expt}/{algorithm}/{float(p)}".encode())
    res = run_mab(algorithm, instance, T, p, N_TRIALS, seed=seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, expected_rewards=res.expected_rewards, taus=res.taus)
    return res.expected_rewards, res.taus


def run_experiment(expt, spec, force=False):
    instance, T, title, runs = spec
    curves, rows = [], []
    for i, (algorithm, p) in enumerate(runs):
        print(f"[{expt}] {algorithm} p={p} T={T:.0e} ({N_TRIALS} runs)", flush=True)
        er, taus = cached_run(expt, algorithm, instance, T, p, force)
        regret = p_mean_regret_curve(er, instance.mu_star, p)
        label, color = ALGO_STYLE[algorithm]
        if expt == "f":
            label, color = f"p = {p}", SERIES_COLORS[i]
        curves.append((label, color, regret))
        rows.append(dict(experiment=expt, algorithm=algorithm, p=p, T=T,
                         final_regret=regret[-1], regret_at_1e4=regret[9999],
                         mean_tau=taus.mean(), std_tau=taus.std()))
    ylabel = "Nash regret" if all(p == 0 for _, p in runs) else "p-mean regret"
    os.makedirs(PLOTS, exist_ok=True)
    plot_curves(curves, T, title, ylabel, os.path.join(PLOTS, f"expt_{expt}.png"))

    # Downsampled curves (log grid) so the numbers are in the repo without the caches.
    os.makedirs(os.path.join(RESULTS, "curves"), exist_ok=True)
    ts = log_grid(T, start=1, num=500)
    with open(os.path.join(RESULTS, "curves", f"expt_{expt}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + [f"{a}_p{p}" for a, p in runs])
        for t in ts:
            w.writerow([t] + [f"{c[2][t - 1]:.6g}" for c in curves])
    return rows


def write_summary(rows, specs):
    path = os.path.join(RESULTS, "summary.md")
    with open(path, "w") as f:
        f.write("| expt | algorithm | p | T | regret at t=1e4 | regret at T | exploration rounds (mean ± sd) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['experiment']} | {r['algorithm']} | {r['p']} | {r['T']:.0e} | "
                    f"{r['regret_at_1e4']:.4g} | {r['final_regret']:.4g} | "
                    f"{r['mean_tau']:.0f} ± {r['std_tau']:.0f} |\n")
        f.write("\nInstances (mu*):\n\n")
        for name, (inst, *_rest) in specs.items():
            f.write(f"- ({name}) k={inst.k}, mu*={inst.mu_star:.4g}, min mean={inst.means.min():.4g}, "
                    f"sigma^2={inst.sigma2}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="*", default=list("abcdef"))
    ap.add_argument("--force", action="store_true", help="ignore cached runs")
    args = ap.parse_args()
    specs = experiment_specs(make_instances())
    rows = []
    for name, spec in specs.items():
        rows += run_experiment(name, spec, args.force) if name in args.experiments else []
    if set(args.experiments) == set(specs):
        write_summary(rows, specs)


if __name__ == "__main__":
    main()
