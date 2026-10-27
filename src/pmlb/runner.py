"""Monte Carlo runner shared by all bandit settings.

A *kernel* is any function `kernel(*args, seed, out) -> tau` that plays one
run, writes per-round expected rewards into `out` and returns the length of
its exploration phase. `run_trials` averages `out` over independent runs in
parallel worker processes, which is exactly the E[mu_{I_t}] needed by
`pmlb.welfare.p_mean_regret_curve`.
"""
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib import import_module

import numpy as np


@dataclass
class TrialResult:
    expected_rewards: np.ndarray  # mean over runs of the expected reward in round t
    taus: np.ndarray  # exploration length of each run
    final_rewards: np.ndarray = None  # mean expected reward of each run's last 1% of rounds


def _resolve(kernel_path):
    module, name = kernel_path.rsplit(":", 1)
    return getattr(import_module(module), name)


def _run_chunk(kernel_path, args, T, seeds):
    kernel = _resolve(kernel_path)
    acc = np.zeros(T)
    out = np.empty(T)
    taus, finals = [], []
    tail = max(1, T // 100)
    for s in seeds:
        taus.append(kernel(*args, int(s), out))
        finals.append(out[-tail:].mean())
        acc += out
    return acc, taus, finals


def run_trials(kernel_path, args, T, n_trials, seed=0, n_jobs=None):
    """Run `n_trials` independent runs of the kernel named by `kernel_path`
    ("package.module:function") and average their expected rewards.

    `args` are the kernel's leading arguments (everything before seed, out).
    """
    seeds = np.random.SeedSequence(seed).generate_state(n_trials) % (2**31 - 1)
    n_jobs = min(n_jobs or os.cpu_count() or 1, n_trials)
    chunks = [seeds[i::n_jobs] for i in range(n_jobs)]
    acc = np.zeros(T)
    taus, finals = [], []
    if n_jobs == 1:
        results = [_run_chunk(kernel_path, args, T, chunks[0])]
    else:
        with ProcessPoolExecutor(n_jobs) as pool:
            results = list(pool.map(_run_chunk, [kernel_path] * n_jobs, [args] * n_jobs,
                                    [T] * n_jobs, chunks))
    for a, t, f in results:
        acc += a
        taus.extend(t)
        finals.extend(f)
    return TrialResult(acc / n_trials, np.asarray(taus), np.asarray(finals))


def run_mab(algorithm, instance, T, p, n_trials, seed=0, n_jobs=None):
    """Convenience wrapper for the MAB kernels in `pmlb.mab.algorithms`."""
    args = (np.ascontiguousarray(instance.means, dtype=np.float64), int(T),
            float(instance.sigma2), int(instance.reward_type), float(p))
    return run_trials(f"pmlb.mab.algorithms:{algorithm}", args, T, n_trials, seed, n_jobs)


def run_linear(algorithm, env, T, p, n_trials, seed=0, n_jobs=None, kappa=10.0):
    """Convenience wrapper for the linear kernels in `pmlb.linear.algorithms`."""
    args = env.kernel_args(T) + (float(p), float(kappa))
    return run_trials(f"pmlb.linear.algorithms:{algorithm}", args, T, n_trials, seed, n_jobs)
