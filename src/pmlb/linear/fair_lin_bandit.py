"""FairLinBandit (Sarkar, Pandey & Ray Chowdhury, "Improved Algorithms for Nash Welfare in Linear Bandits", 2026).

Implements FairLinUCB (arXiv:2601.22969v2 / github.com/NP-Hardest/FairLinBandit):
- Phase I: Exploration alternating between D-optimal design round-robin and
  sampling from the John ellipsoid / MVEE center distribution with probability 0.5 each.
- Adaptive stopping condition (Equation 1 of Sarkar et al. 2026):
      t <= max( 48 * sigma^2 * d^2 * log(T) / (max_x <x, th>)^2,
                900 * p_a^2 * sigma^2 * d^2 * log(T) / (max_x <x, th> - sqrt(48*sigma^2*d^2*log(T)/t))^2 )
  evaluated at doubling epochs starting at ceil(72 * log(T)).
- Phase II: LinUCB initialized with the design matrix and estimates from Phase I.
"""
import numpy as np
from numba import njit

from ..envs import sample_reward
from .algorithms import (
    LAMBDA,
    _argmax_rand,
    _beta,
    _build,
    _dot_rows,
    _update,
    _widths,
)
from .lin_nash import d_optimal_design, mvee_weights


@njit(cache=True)
def _fair_lin_ucb_loop(X, means, design_arms, cdf, theta, T, sigma2, reward_type, p_a, seed, out):
    np.random.seed(seed)
    K, d = X.shape
    sigma = np.sqrt(sigma2)
    logT = np.log(float(max(T, 2)))
    beta = _beta(theta, sigma2, T)

    Vinv = np.eye(d) / LAMBDA
    b = np.zeros(d)
    th = np.zeros(d)
    tmp = np.zeros(d)
    est = np.zeros(K)
    w = np.zeros(K)
    idx = np.zeros(K)

    n_design = design_arms.size
    rr = 0
    t0 = int(np.ceil(72.0 * logT))
    if t0 < 1:
        t0 = 1
    next_check = t0

    C1 = 48.0 * sigma2 * float(d * d) * logT
    C2 = 900.0 * float(p_a * p_a) * sigma2 * float(d * d) * logT

    tau = T
    t = 0

    # Phase I: Exploration
    while t < T:
        if t == next_check:
            _dot_rows(X, th, est)
            mu_max = -1e300
            for a in range(K):
                if est[a] > mu_max:
                    mu_max = est[a]
            rad = np.sqrt(C1 / float(t))
            if mu_max > rad:
                term1 = C1 / (mu_max * mu_max)
                diff = mu_max - rad
                term2 = C2 / (diff * diff)
                thresh = term1 if term1 > term2 else term2
                if float(t) > thresh:
                    tau = t
                    break
            next_check = 2 * next_check

        if np.random.rand() < 0.5:
            # Sample from MVEE center distribution
            r_unif = np.random.rand()
            a = cdf.size - 1
            for i in range(cdf.size):
                if r_unif < cdf[i]:
                    a = i
                    break
        else:
            # D-optimal design round-robin
            a = design_arms[rr % n_design]
            rr += 1

        r = sample_reward(means[a], sigma, reward_type)
        _update(Vinv, b, th, X[a], r, tmp)
        out[t] = means[a]
        t += 1

    # Phase II: LinUCB
    while t < T:
        _dot_rows(X, th, est)
        _widths(X, Vinv, w)
        for i in range(K):
            idx[i] = est[i] + beta * w[i]
        a = _argmax_rand(idx)
        r = sample_reward(means[a], sigma, reward_type)
        _update(Vinv, b, th, X[a], r, tmp)
        out[t] = means[a]
        t += 1

    return tau


def fair_lin_ucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out):
    """FairLinUCB (Sarkar et al. 2026): D-optimal + MVEE exploration with adaptive doubling stopping rule, then LinUCB."""
    if users.size != 1 or Z.shape[0] != 1:
        raise ValueError("FairLinUCB is defined for a fixed arm set")
    u = users[0]
    items = cands[0]
    K = items.size
    d = Z.shape[1] * Y.shape[1]
    X = np.empty((K, d), dtype=np.float64)
    _build(Z, Y, u, items, X)
    means = X @ theta

    lam = d_optimal_design(X)
    u_dist = mvee_weights(X)
    design_arms = np.flatnonzero(lam > 0).astype(np.int64)
    if design_arms.size == 0:
        design_arms = np.arange(K, dtype=np.int64)

    u_sum = np.sum(u_dist)
    if u_sum > 0:
        u_dist = u_dist / u_sum
    else:
        u_dist = np.full(K, 1.0 / K)
    cdf = np.cumsum(u_dist)
    cdf[-1] = 1.0

    p_a = 1.0 if p >= -1.0 else abs(p)

    return _fair_lin_ucb_loop(
        X, means, design_arms, cdf, theta, int(T), float(sigma2), int(reward_type), float(p_a), int(seed), out
    )


fair_lin_bandit = fair_lin_ucb
