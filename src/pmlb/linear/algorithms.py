"""Numba kernels for linear contextual bandits under p-mean social welfare.

Kernel signature:
    kernel(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out) -> tau

Plays T rounds, writes expected reward into out[t], and returns exploration rounds tau.
"""
import numpy as np
from numba import njit

from ..envs import sample_reward
from .lin_nash import lin_nash, lin_nash_tuned

LAMBDA = 1.0  # Ridge regularization parameter
KAPPA = 10.0  # Default stopping constant kappa in LCB >= kappa * |p_a| * width
S_BOUND = 1.0  # Standard a-priori parameter norm bound: ||theta*||_2 <= S


@njit(cache=True)
def _build(Z, Y, u, items, X):
    """Construct Kronecker feature matrix X[a] = kron(Z[u], Y[items[a]])."""
    dz, dy = Z.shape[1], Y.shape[1]
    for a in range(items.size):
        y = Y[items[a]]
        for i in range(dz):
            zi = Z[u, i]
            for j in range(dy):
                X[a, i * dy + j] = zi * y[j]


@njit(cache=True)
def _dot_rows(X, v, res):
    """Compute res = X @ v."""
    for a in range(X.shape[0]):
        s = 0.0
        for j in range(X.shape[1]):
            s += X[a, j] * v[j]
        res[a] = s


@njit(cache=True)
def _widths(X, Vinv, w):
    """Compute directional standard errors w[a] = sqrt(x_a^T V^-1 x_a)."""
    d = X.shape[1]
    nz = np.empty(d, np.int64)
    for a in range(X.shape[0]):
        m = 0
        for i in range(d):
            if X[a, i] != 0.0:
                nz[m] = i
                m += 1
        s = 0.0
        for ii in range(m):
            i = nz[ii]
            t = 0.0
            for jj in range(m):
                t += Vinv[i, nz[jj]] * X[a, nz[jj]]
            s += X[a, i] * t
        w[a] = np.sqrt(max(s, 0.0))


@njit(cache=True)
def _update(Vinv, b, th, x, r, tmp):
    """O(d^2) Sherman-Morrison rank-1 update of V^-1, b += r*x, and th = V^-1 b."""
    d = x.size
    for i in range(d):
        tmp[i] = 0.0
    for j in range(d):
        if x[j] != 0.0:
            for i in range(d):
                tmp[i] += Vinv[i, j] * x[j]
    denom = 1.0
    for i in range(d):
        denom += x[i] * tmp[i]
    for i in range(d):
        for j in range(d):
            Vinv[i, j] -= tmp[i] * tmp[j] / denom
    for i in range(d):
        b[i] += r * x[i]
    for i in range(d):
        t = 0.0
        for j in range(d):
            t += Vinv[i, j] * b[j]
        th[i] = t


@njit(cache=True)
def _argmax_rand(v):
    """Uniform random tie-breaking argmax."""
    best, idx, cnt = -np.inf, 0, 0
    for i in range(v.size):
        if v[i] > best + 1e-12:
            best, idx, cnt = v[i], i, 1
        elif v[i] >= best - 1e-12:
            cnt += 1
            if np.random.rand() * cnt < 1.0:
                idx = i
    return idx


@njit(cache=True)
def _beta(theta, sigma2, T):
    """Abbasi-Yadkori confidence ellipsoid radius: sigma*sqrt(2*log(T)) + sqrt(lambda)*S."""
    return np.sqrt(sigma2) * np.sqrt(2.0 * np.log(T)) + np.sqrt(LAMBDA) * S_BOUND


@njit(cache=True)
def _covered(X, Vinv, th, beta, thresh, est, w, coverage):
    """Evaluate stopping rule on candidate set."""
    _dot_rows(X, th, est)
    _widths(X, Vinv, w)
    if coverage:
        best_lcb = -np.inf
        for a in range(est.size):
            best_lcb = max(best_lcb, est[a] - beta * w[a])
        if best_lcb <= 0:
            return False
        for a in range(est.size):
            if est[a] + beta * w[a] >= best_lcb and best_lcb < thresh * beta * w[a]:
                return False
        return True
    # Best-arm stopping rule (Adaptation 1 & 2)
    for a in range(est.size):
        lcb = est[a] - beta * w[a]
        if lcb > 0 and lcb >= thresh * beta * w[a]:
            return True
    return False


@njit(cache=True)
def _linucb(Z, Y, theta, users, cands, T, sigma2, reward_type, seed, out, scale):
    """LinUCB exploitation loop (scale=1.0 for standard LinUCB, 0.0 for greedy)."""
    np.random.seed(seed)
    K, d = cands.shape[1], Z.shape[1] * Y.shape[1]
    R = users.size
    sigma = np.sqrt(sigma2)
    beta = scale * _beta(theta, sigma2, T)
    Vinv = np.eye(d) / LAMBDA
    b, th, tmp = np.zeros(d), np.zeros(d), np.zeros(d)
    X = np.zeros((K, d))
    mean, est, w, idx = np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K)

    for t in range(T):
        row = t % R
        _build(Z, Y, users[row], cands[row], X)
        _dot_rows(X, theta, mean)
        _dot_rows(X, th, est)
        if beta > 0:
            _widths(X, Vinv, w)
            for a in range(K):
                idx[a] = est[a] + beta * w[a]
        else:
            idx[:] = est
        a = _argmax_rand(idx)
        r = sample_reward(mean[a], sigma, reward_type)
        _update(Vinv, b, th, X[a], r, tmp)
        out[t] = mean[a]
    return 0


@njit(cache=True)
def linucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out):
    """Standard LinUCB baseline."""
    return _linucb(Z, Y, theta, users, cands, T, sigma2, reward_type, seed, out, 1.0)


@njit(cache=True)
def lin_greedy(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out):
    """Greedy ridge regression baseline."""
    return _linucb(Z, Y, theta, users, cands, T, sigma2, reward_type, seed, out, 0.0)


@njit(cache=True)
def _explore_then_linucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out,
                         design, coverage, per_group=True):
    """Two-phase contextual linear bandit engine with adaptive stopping."""
    np.random.seed(seed)
    K, d = cands.shape[1], Z.shape[1] * Y.shape[1]
    R = users.size
    sigma = np.sqrt(sigma2)
    beta = _beta(theta, sigma2, T)
    p_a = 1.0 if p >= -1 else p
    thresh = kappa * abs(p_a)
    Vinv = np.eye(d) / LAMBDA
    b, th, tmp = np.zeros(d), np.zeros(d), np.zeros(d)
    X = np.zeros((K, d))
    mean, est, w, idx = np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K)
    certified = np.zeros(G, np.bool_)
    n_explore = 0

    for t in range(T):
        row = t % R
        u = users[row]
        _build(Z, Y, u, cands[row], X)
        _dot_rows(X, theta, mean)
        exploring = False
        if coverage and not per_group:
            exploring = not certified.all()
        else:
            for g in ugroups[u]:
                if not certified[g]:
                    exploring = True
        if exploring:
            n_explore += 1
            if design and np.random.rand() < 0.5:
                # Active D-optimal design selection: maximize prediction uncertainty ||x_a||_{V^-1}
                _widths(X, Vinv, w)
                a = _argmax_rand(w)
            else:
                # Uniform random candidate selection
                a = np.random.randint(K)
        else:
            # Phase II LinUCB exploitation
            _dot_rows(X, th, est)
            _widths(X, Vinv, w)
            for i in range(K):
                idx[i] = est[i] + beta * w[i]
            a = _argmax_rand(idx)
        r = sample_reward(mean[a], sigma, reward_type)
        _update(Vinv, b, th, X[a], r, tmp)
        out[t] = mean[a]

        if exploring and _covered(X, Vinv, th, beta, thresh, est, w, coverage):
            if coverage:
                for g in ugroups[u]:
                    certified[g] = True
            else:
                certified[:] = True
    return n_explore


@njit(cache=True)
def explore_then_linucb_naive(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out):
    """Contextual Adaptation 1: Baseline Uniform Transfer (Presentation Slide 14).

    Phase I: Uniform random candidate exploration.
    Stopping: Best-arm data-adaptive certificate (LCB* >= kappa * |p_a| * beta * width).
    Phase II: LinUCB initialized with Phase I covariance.
    """
    return _explore_then_linucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed,
                                out, design=False, coverage=False)


@njit(cache=True)
def explore_then_linucb_design_only(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out):
    """Contextual Adaptation 2: 50/50 D-Optimal Exploration (Presentation Slide 15).

    Phase I: 50% Uniform random + 50% Greedy D-optimal design (argmax ||x||_{V^-1}).
    Stopping: Best-arm data-adaptive certificate (LCB* >= kappa * |p_a| * beta * width).
    Phase II: LinUCB initialized with Phase I covariance.
    """
    return _explore_then_linucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed,
                                out, design=True, coverage=False)


@njit(cache=True)
def explore_then_linucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out):
    """Explore-then-LinUCB with plausible-set coverage and per-group stopping."""
    return _explore_then_linucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed,
                                out, design=True, coverage=True, per_group=True)


@njit(cache=True)
def explore_then_linucb_global(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out):
    """Explore-then-LinUCB with plausible-set coverage and global stopping."""
    return _explore_then_linucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed,
                                out, design=True, coverage=True, per_group=False)


@njit(cache=True)
def explore_then_linucb_coverage_only(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out):
    """Ablation: Uniform exploration with plausible-set coverage stopping."""
    return _explore_then_linucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed,
                                out, design=False, coverage=True)


from .fair_lin_bandit import fair_lin_bandit, fair_lin_ucb  # noqa: E402

ALGORITHMS = {
    "explore_then_linucb_naive": explore_then_linucb_naive,
    "explore_then_linucb_design_only": explore_then_linucb_design_only,
    "explore_then_linucb": explore_then_linucb,
    "explore_then_linucb_global": explore_then_linucb_global,
    "explore_then_linucb_coverage_only": explore_then_linucb_coverage_only,
    "linucb": linucb,
    "lin_greedy": lin_greedy,
    "lin_nash": lin_nash,
    "lin_nash_tuned": lin_nash_tuned,
    "fair_lin_ucb": fair_lin_ucb,
    "fair_lin_bandit": fair_lin_bandit,
}
