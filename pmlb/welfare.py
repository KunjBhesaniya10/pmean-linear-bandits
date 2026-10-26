"""p-mean welfare and p-mean regret.

Everything here works on a sequence of *expected* per-round rewards
(mu_{I_t} for MAB, <x_t, theta*> for linear bandits), so it is shared by
every bandit setting in the project.
"""
import numpy as np


def p_mean(values, p, axis=-1):
    """Generalized (power) mean M_p of positive values.

    p = 1 is the arithmetic mean, p = 0 the geometric (Nash) mean,
    p = -inf the minimum (egalitarian).
    """
    v = np.asarray(values, dtype=np.float64)
    if p == 0:
        return np.exp(np.mean(np.log(v), axis=axis))
    if np.isneginf(p):
        return np.min(v, axis=axis)
    if np.isposinf(p):
        return np.max(v, axis=axis)
    return np.mean(v ** p, axis=axis) ** (1.0 / p)


def running_p_mean(expected_rewards, p):
    """M_p of the first t entries, for every prefix t = 1..T."""
    r = np.asarray(expected_rewards, dtype=np.float64)
    t = np.arange(1, r.size + 1, dtype=np.float64)
    if p == 0:
        return np.exp(np.cumsum(np.log(np.maximum(r, 1e-300))) / t)
    if np.isneginf(p):
        return np.minimum.accumulate(r)
    return (np.cumsum(r ** p) / t) ** (1.0 / p)


def p_mean_regret_curve(expected_rewards, mu_star, p):
    """R^p_t = mu* - M_p(E[mu_{I_1}], ..., E[mu_{I_t}]) for every t.

    `expected_rewards[t]` should already be averaged over independent runs,
    i.e. an estimate of E[mu_{I_t}], as in the paper's definition.
    """
    return mu_star - running_p_mean(expected_rewards, p)
