"""Numba kernels for the MAB algorithms compared in Sarkar, Pandey & Ray Chowdhury (2025).

Every kernel has the same signature

    kernel(means, T, sigma2, reward_type, p, seed, out) -> tau

It plays T rounds, writes the expected reward of the arm pulled in round t
(means[I_t]) into out[t], and returns tau, the length of the initial
exploration phase (0 if the algorithm has none). Constants follow the
authors' reference code (github.com/NP-Hardest/UCBisAllYouNeed).
"""
import numpy as np
from numba import njit

from ..envs import sample_reward


@njit(cache=True)
def _pull(a, means, sigma, reward_type, n, sums, mu):
    r = sample_reward(means[a], sigma, reward_type)
    n[a] += 1
    sums[a] += r
    mu[a] = sums[a] / n[a]


@njit(cache=True)
def _ucb_arm(mu, n, scale):
    """argmax_i mu_i + scale / sqrt(n_i); unpulled arms first."""
    best, best_val = 0, -np.inf
    for i in range(mu.size):
        val = 1e18 if n[i] == 0 else mu[i] + scale / np.sqrt(n[i])
        if val > best_val:
            best, best_val = i, val
    return best


@njit(cache=True)
def welfarist_continue(mu, n, sigma2, logT, c, p_a):
    """Phase I continues until some arm i has a positive LCB *and*
    n_i mu_i >= 200 c^2 p_a^2 sigma^2 log T / LCB_i + c sqrt(2 n_i sigma^2 log T)."""
    for i in range(mu.size):
        if n[i] == 0:
            continue
        bonus = c * np.sqrt(2.0 * sigma2 * logT / n[i])
        lcb = mu[i] - bonus
        if lcb <= 0:
            continue
        rhs = 200.0 * c * c * p_a * p_a * sigma2 * logT / lcb + c * np.sqrt(2.0 * n[i] * sigma2 * logT)
        if n[i] * mu[i] >= rhs:
            return False
    return True


@njit(cache=True)
def welfarist_ucb(means, T, sigma2, reward_type, p, seed, out):
    """Welfarist-UCB: round-robin (random permutation per block of k) exploration
    with a data-dependent stopping rule, then UCB."""
    np.random.seed(seed)
    k = means.size
    sigma = np.sqrt(sigma2)
    n = np.zeros(k, np.int64)
    sums = np.zeros(k)
    mu = np.zeros(k)
    c = 2.0
    logT = np.log(T)
    p_a = 1.0 if p >= -1 else p
    perm = np.random.permutation(k)

    t = 0
    while t < T and welfarist_continue(mu, n, sigma2, logT, c, p_a):
        if t % k == 0:
            perm = np.random.permutation(k)
        a = perm[t % k]
        _pull(a, means, sigma, reward_type, n, sums, mu)
        out[t] = means[a]
        t += 1
    tau = t

    scale = c * np.sqrt(2.0 * sigma2 * logT)
    while t < T:
        a = _ucb_arm(mu, n, scale)
        _pull(a, means, sigma, reward_type, n, sums, mu)
        out[t] = means[a]
        t += 1
    return tau


@njit(cache=True)
def explore_then_ucb_length(T, k, p):
    """Fixed exploration length of Krishna et al. (2025), as in the reference code."""
    if p == 0:
        return 16.0 * np.sqrt(T * k * np.log(T) / np.log(k))
    if p > 0:
        return 16.0 * np.sqrt(T * float(k) ** p * np.log(T) / np.log(k))
    return 16.0 * np.sqrt(T * np.log(T) / float(k) ** (-p))


@njit(cache=True)
def explore_then_ucb(means, T, sigma2, reward_type, p, seed, out):
    """Explore-then-UCB (Krishna, John, Barik & Tan, AAAI 2025): uniform random
    exploration for a fixed, p-dependent number of rounds, then UCB with the
    bonus 4 sqrt(log T / n_i) (no sigma scaling, as in the reference code)."""
    np.random.seed(seed)
    k = means.size
    sigma = np.sqrt(sigma2)
    n = np.zeros(k, np.int64)
    sums = np.zeros(k)
    mu = np.zeros(k)
    tilde_T = min(explore_then_ucb_length(T, k, p), T)

    t = 0
    while t < tilde_T:
        a = np.random.randint(k)
        _pull(a, means, sigma, reward_type, n, sums, mu)
        out[t] = means[a]
        t += 1
    tau = t

    scale = 4.0 * np.sqrt(np.log(T))
    while t < T:
        a = _ucb_arm(mu, n, scale)
        _pull(a, means, sigma, reward_type, n, sums, mu)
        out[t] = means[a]
        t += 1
    return tau


@njit(cache=True)
def ncb(means, T, sigma2, reward_type, p, seed, out):
    """Nash Confidence Bound (Barman, Khan, Maiti & Sawarni, AAAI 2023).
    Uniform random exploration until max_i n_i mu_i > 420 c^2 log T, then
    argmax mu_i + 2c sqrt(2 mu_i log T / n_i). Targets Nash regret (p = 0)."""
    np.random.seed(seed)
    k = means.size
    sigma = np.sqrt(sigma2)
    n = np.zeros(k, np.int64)
    sums = np.zeros(k)
    mu = np.zeros(k)
    c = 3.1
    logT = np.log(T)
    threshold = 420.0 * c * c * logT

    t = 0
    while t < T and np.max(n * mu) <= threshold:
        a = np.random.randint(k)
        _pull(a, means, sigma, reward_type, n, sums, mu)
        out[t] = means[a]
        t += 1
    tau = t

    while t < T:
        best, best_val = 0, -np.inf
        for i in range(k):
            if n[i] == 0:
                val = 1e18
            else:
                val = mu[i] + 2.0 * c * np.sqrt(2.0 * max(mu[i], 0.0) * logT / n[i])
            if val > best_val:
                best, best_val = i, val
        _pull(best, means, sigma, reward_type, n, sums, mu)
        out[t] = means[best]
        t += 1
    return tau


ALGORITHMS = {
    "welfarist_ucb": welfarist_ucb,
    "explore_then_ucb": explore_then_ucb,
    "ncb": ncb,
}
