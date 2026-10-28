"""Helpers shared by the linear-bandit experiment scripts: cached Monte Carlo runs and output paths."""
import csv
import os
import zlib

import numpy as np

from pmlb.plotting import log_grid
from pmlb.runner import run_linear

RESULTS = "results"
PLOTS = "plots"
NO_EXPLORATION = ("linucb", "lin_greedy", "lin_nash", "lin_nash_tuned")  # runs do not depend on p or kappa


def dirs(name):
    res, plots = os.path.join(RESULTS, name), os.path.join(PLOTS, name)
    os.makedirs(res, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return res, plots


def p_run(algorithm, p):
    """The p an algorithm's run depends on: p_a = 1 for p >= -1, so one run serves p in {1, 0, -1}."""
    if algorithm in NO_EXPLORATION:
        return 1.0
    return 1.0 if p >= -1 else float(p)


def cached_linear(tag, algorithm, env, T, p, n_trials, kappa=10.0, force=False):
    """run_linear, cached in results/cache/<tag>/ keyed by algorithm, p_a and kappa."""
    pa = p_run(algorithm, p)
    k = 0.0 if algorithm in NO_EXPLORATION else float(kappa)
    path = os.path.join(RESULTS, "cache", tag, f"{algorithm}_p{pa}_k{k}_T{T}_n{n_trials}.npz")
    if os.path.exists(path) and not force:
        d = np.load(path)
        return d["expected_rewards"], d["taus"], d["final_rewards"]
    print(f"  [{tag}] {algorithm} p_a={pa} kappa={k} T={T:.0e} ({n_trials} runs)", flush=True)
    seed = zlib.crc32(f"{tag}/{algorithm}/{pa}/{k}".encode())
    res = run_linear(algorithm, env, T, pa, n_trials, seed=seed, kappa=kappa)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, expected_rewards=res.expected_rewards, taus=res.taus, final_rewards=res.final_rewards)
    return res.expected_rewards, res.taus, res.final_rewards


def write_curves(path, T, named_curves):
    """Regret curves downsampled on a log grid of rounds."""
    ts = log_grid(T, start=1, num=300)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + [n for n, _ in named_curves])
        for t in ts:
            w.writerow([t] + [f"{c[t - 1]:.6g}" for _, c in named_curves])


def write_table(path, header, rows):
    """Markdown table; rows are lists already formatted as strings."""
    with open(path, "w") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "---|" * len(header) + "\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
