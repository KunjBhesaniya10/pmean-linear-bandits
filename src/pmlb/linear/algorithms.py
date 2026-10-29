"""Numba kernels for linear contextual bandits under p-mean welfare.

Every kernel has the signature

    kernel(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out) -> tau

(the environment encoding is described in `pmlb.linear.envs`). It plays T
rounds, writes the expected reward <x_t, theta*> of the item it shows in
round t into out[t], and returns tau, the length of its exploration phase
(the number of exploration rounds, 0 if it has none), the same contract
as the MAB kernels.

All kernels share ridge regression with lambda = 1 and the confidence radius

    beta = sigma * sqrt(2 log T) + sqrt(lambda) * ||theta*||,

i.e. the self-normalized radius of Abbasi-Yadkori et al. without its
d log(1 + t L^2 / (lambda d)) term. With that term every algorithm, plain
LinUCB included, keeps exploring for the whole horizon at our T. Ties in
every argmax are broken uniformly at random, so no algorithm gets a
deterministic early sequence from the order of the candidates.

Explore-then-LinUCB carries the two parts of Welfarist-UCB over to features.

* Phase I picks, with probability 1/2, a uniformly random candidate (this
  keeps E[reward of round t] at least half the candidates' average, which
  is what the uniform phase buys the p-mean in the MAB case) and otherwise
  the candidate with the largest ||x||_{V^-1}, a greedy D-optimal design
  step that goes after the least-explored, typically rare, feature directions.
* Stopping rule. Welfarist-UCB's rule (some arm i has LCB_i > 0 and
  n_i mu_i >= 200 c^2 p_a^2 sigma^2 log T / LCB_i + c sqrt(2 n_i sigma^2 log T))
  simplifies to LCB_i >= 10 |p_a| w_i, where w_i is arm i's confidence width.
  We keep the factor as a parameter kappa (10 reproduces the paper's rule).
  Round-robin exploration gives every arm the same width, so checking the
  best arm is enough there. With features the widths differ by direction,
  and an item along a rarely explored direction can look plausible long
  after the common directions are pinned down. Our rule therefore asks the
  best lower bound to dominate the width of every item that UCB could still
  pick (UCB(x_a) >= max LCB):

      LCB* = max_a LCB(x_a) > 0  and  LCB* >= kappa |p_a| beta ||x_a||_{V^-1}
      for every a with UCB(x_a) >= LCB*,

  Items already clearly worse than LCB* need no further exploration. With
  contexts the rule is tested on the candidate set of the current round and
  gives each protected group its own stopping time: a group leaves Phase I
  the first time a context of one of its members passes, and a round is an
  exploration round while any group of its user has not. Frequent groups
  start exploiting early; a rare group keeps exploring until its own
  (minority) feature directions are certified. Phase II can then only choose
  items whose reward is within a constant factor of the best one, in every
  group. With a fixed arm set there is one group and one stopping time, as
  in Welfarist-UCB.

`explore_then_linucb_naive` ("Welfarist-UCB unchanged") is the direct transfer for comparison: uniform
exploration and the paper's best-arm rule on the current round's candidates.
`explore_then_greedy*` replace Phase II by greedy exploitation, the way a
deployed recommender usually runs once it stops exploring; there nothing
after tau can repair a direction the stopping rule left unlearned. The
`_design_only` / `_coverage_only` variants are ablations of the two changes.
"""
import numpy as np
from numba import njit

from ..envs import sample_reward
from .lin_nash import lin_nash, lin_nash_tuned

LAMBDA = 1.0
KAPPA = 10.0  # default kappa in LCB >= kappa |p_a| width: Welfarist-UCB's constant


@njit(cache=True)
def _build(Z, Y, u, items, X):
    dz, dy = Z.shape[1], Y.shape[1]
    for a in range(items.size):
        y = Y[items[a]]
        for i in range(dz):
            zi = Z[u, i]
            for j in range(dy):
                X[a, i * dy + j] = zi * y[j]


@njit(cache=True)
def _dot_rows(X, v, res):
    for a in range(X.shape[0]):
        s = 0.0
        for j in range(X.shape[1]):
            s += X[a, j] * v[j]
        res[a] = s


@njit(cache=True)
def _widths(X, Vinv, w):
    """w_a = ||x_a||_{V^-1}, using only the non-zero entries of x_a
    (one-hot user features make most of kron(z, y) zero)."""
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
    """Sherman-Morrison update of V^-1, b += r x, th = V^-1 b."""
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
    """argmax with uniform random tie-breaking."""
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
    return np.sqrt(sigma2) * np.sqrt(2.0 * np.log(T)) + np.sqrt(LAMBDA) * np.sqrt(np.sum(theta * theta))


@njit(cache=True)
def _covered(X, Vinv, th, beta, thresh, est, w, coverage):
    """Stopping test on one candidate set (see module docstring)."""
    _dot_rows(X, th, est)
    _widths(X, Vinv, w)
    if coverage:
        best_lcb = -np.inf
        for a in range(est.size):
            best_lcb = max(best_lcb, est[a] - beta * w[a])
        if best_lcb <= 0:
            return False
        for a in range(est.size):  # every still-plausible candidate must be certified
            if est[a] + beta * w[a] >= best_lcb and best_lcb < thresh * beta * w[a]:
                return False
        return True
    for a in range(est.size):
        lcb = est[a] - beta * w[a]
        if lcb > 0 and lcb >= thresh * beta * w[a]:
            return True
    return False


@njit(cache=True)
def _linucb(Z, Y, theta, users, cands, T, sigma2, reward_type, seed, out, scale):
    """LinUCB with radius scale * beta (scale 0 is greedy ridge regression)."""
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
    """LinUCB (optimism from the first round, no exploration phase)."""
    return _linucb(Z, Y, theta, users, cands, T, sigma2, reward_type, seed, out, 1.0)


@njit(cache=True)
def lin_greedy(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out):
    """Greedy ridge regression: always show the item with the largest estimate."""
    return _linucb(Z, Y, theta, users, cands, T, sigma2, reward_type, seed, out, 0.0)


@njit(cache=True)
def _explore_then_linucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out,
                         design, coverage, greedy, per_group=True):
    """Shared body. design: mixed uniform / max-width exploration (else uniform).
    coverage: plausible-set rule with a stopping time per group (else the
    paper's best-arm rule and one global stopping time). greedy: Phase II
    exploits the ridge estimate (else LinUCB). per_group=False keeps every
    round in Phase I until all groups are certified. Returns the number of
    exploration rounds."""
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
    certified = np.zeros(G, np.bool_)  # group has passed the stopping rule
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
                _widths(X, Vinv, w)
                a = _argmax_rand(w)
            else:
                a = np.random.randint(K)
        elif greedy:
            _dot_rows(X, th, est)
            a = _argmax_rand(est)
        else:
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
def explore_then_linucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out):
    """Ours: mixed uniform / D-optimal exploration, plausible-set rule with a stopping time per group."""
    return _explore_then_linucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed,
                                out, True, True, False)


@njit(cache=True)
def explore_then_linucb_naive(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out):
    """Direct transfer of Welfarist-UCB: uniform exploration, best-arm stopping rule."""
    return _explore_then_linucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed,
                                out, False, False, False)


@njit(cache=True)
def explore_then_linucb_design_only(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed,
                                    out):
    """Ablation: mixed design exploration with the best-arm stopping rule."""
    return _explore_then_linucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed,
                                out, True, False, False)


@njit(cache=True)
def explore_then_linucb_coverage_only(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed,
                                      out):
    """Ablation: uniform exploration with the coverage stopping rule."""
    return _explore_then_linucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed,
                                out, False, True, False)


@njit(cache=True)
def explore_then_linucb_global(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out):
    """Ours with one global stopping time: everyone explores until every group is certified."""
    return _explore_then_linucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed,
                                out, True, True, False, False)


@njit(cache=True)
def explore_then_greedy(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out):
    """Ours with a greedy Phase II: the stopping rule alone must make exploitation safe."""
    return _explore_then_linucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed,
                                out, True, True, True)


@njit(cache=True)
def explore_then_greedy_naive(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out):
    """Welfarist-UCB unchanged, with a greedy Phase II."""
    return _explore_then_linucb(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed,
                                out, False, False, True)


ALGORITHMS = {
    "lin_nash": lin_nash,
    "lin_nash_tuned": lin_nash_tuned,
    "linucb": linucb,
    "lin_greedy": lin_greedy,
    "explore_then_linucb": explore_then_linucb,
    "explore_then_linucb_naive": explore_then_linucb_naive,
    "explore_then_linucb_global": explore_then_linucb_global,
    "explore_then_greedy": explore_then_greedy,
    "explore_then_greedy_naive": explore_then_greedy_naive,
    "explore_then_linucb_design_only": explore_then_linucb_design_only,
    "explore_then_linucb_coverage_only": explore_then_linucb_coverage_only,
}
