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


def contextual_p_mean_regret_curve(expected_rewards, optimal_rewards, p):
    """M_p(r*_1..r*_t) - M_p(E[r_1], ..., E[r_t]) for every t.

    With a fixed arm set r*_t = mu* and this is `p_mean_regret_curve`. With
    contexts the benchmark is the p-mean welfare of the policy that always
    shows each round's best candidate.
    """
    return running_p_mean(optimal_rewards, p) - running_p_mean(expected_rewards, p)


def group_utilities(expected_rewards, optimal_rewards, round_groups, G):
    """Normalised utility of each group: sum of E[reward] over the rounds a member
    of the group arrived, divided by the same sum for the per-round optimum.

    `round_groups` is (T, m): a user can belong to one group per attribute.
    Groups that never arrive get NaN.
    """
    er = np.asarray(expected_rewards, dtype=np.float64)
    rs = np.asarray(optimal_rewards, dtype=np.float64)
    got, best = np.zeros(G), np.zeros(G)
    for col in np.asarray(round_groups).T:
        got += np.bincount(col, weights=er, minlength=G)
        best += np.bincount(col, weights=rs, minlength=G)
    with np.errstate(invalid="ignore", divide="ignore"):
        return got / best
