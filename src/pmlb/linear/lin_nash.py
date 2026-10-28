"""LinNash (Sawarni, Pal & Barman, "Nash Regret Guarantees for Linear Bandits", NeurIPS 2023).

Algorithm 2 of the paper for a finite, fixed arm set, used as the prior
Nash-regret baseline for linear bandits. It has the same kernel signature as
the other linear kernels (it ignores p and kappa and needs a fixed arm set
with rewards in [0, 1], which are 1-sub-Poisson by the paper's Lemma 1).

Part I (T~ = 3 sqrt(T d nu log(T|X|)) rounds, Algorithm 1): each round, with
probability 1/2 sample an arm from a distribution U whose mean is the centre
c of an ellipsoid associated with conv(X); otherwise take the next arm of a
D-optimal design, round robin, until each design arm z has been played
ceil(lambda_z T~ / 3) times. Then an OLS estimate and the elimination

    keep x if UNCB(x) >= max_z LNCB(z),  LNCB/UNCB(x, theta, t) = <x,theta> -/+ 6 sqrt(<x,theta> nu d log(T|X|) / t)

with t = T~/3. Part II: phases of length T' = 2 T~/3, doubling each phase.
In each phase, solve the D-optimal design on the surviving arms, play every
support arm a for ceil(lambda_a T') rounds, refit OLS on that phase's data
only and eliminate with t = T'.

One substitution: the paper takes c to be the centre of the John ellipsoid
(largest inscribed) of conv(X). We use the centre of the minimum-volume
enclosing (Loewner) ellipsoid E, computed with Khachiyan's algorithm, whose
dual weights directly give a distribution U over arms with mean c. Since
c + (E - c)/d is contained in conv(X) and x* lies in E, the point
c - (x* - c)/d lies in conv(X), which is all the proof of the paper's Lemma 5
uses, so E_{x~U} <x, theta*> >= <x*, theta*> / (d + 1) still holds.

Second deviation: Line 17 of Algorithm 2 compares against max_z LNCB(z) over
all of X. In Part II the phase's estimate only identifies theta* on the span
of the surviving arms, so the estimates of already-eliminated arms outside
that span are arbitrary and can eliminate the best arm; we take the max over
the surviving arms instead (identical in Part I, where every arm survives).
"""
import numpy as np

from ..envs import BERNOULLI


def d_optimal_design(X, tol=1e-3, max_iter=2000):
    """Frank-Wolfe (Fedorov-Wynn) for max log det sum_x lambda_x x x^T
    (Lattimore & Szepesvari 2020, Ch. 21). Works in the span of X.
    Returns weights trimmed to at most r(r+1)/2 arms, r = rank(X)."""
    K = X.shape[0]
    r = np.linalg.matrix_rank(X)
    lam = np.full(K, 1.0 / K)
    for _ in range(max_iter):
        Vinv = np.linalg.pinv(X.T @ (lam[:, None] * X))
        g = np.einsum("ij,jk,ik->i", X, Vinv, X)
        j = int(np.argmax(g))
        if g[j] <= r * (1 + tol):
            break
        step = (g[j] / r - 1) / (g[j] - 1)
        lam *= 1 - step
        lam[j] += step
    keep = np.argsort(lam)[::-1][: r * (r + 1) // 2]
    keep = keep[lam[keep] > 1e-4]
    out = np.zeros(K)
    out[keep] = lam[keep] / lam[keep].sum()
    return out


def mvee_weights(X, tol=1e-4, max_iter=5000):
    """Khachiyan's algorithm for the minimum-volume enclosing ellipsoid of the
    rows of X. Returns weights u (a distribution over arms); its centre is X^T u."""
    K = X.shape[0]
    Q = np.vstack([X.T, np.ones(K)])  # lifted points
    n = np.linalg.matrix_rank(Q) - 1  # dimension of the affine hull
    u = np.full(K, 1.0 / K)
    for _ in range(max_iter):
        M = np.einsum("ik,ij,jk->k", Q, np.linalg.pinv((Q * u) @ Q.T), Q)
        j = int(np.argmax(M))
        if M[j] <= (n + 1) * (1 + tol):
            break
        step = (M[j] - n - 1) / ((n + 1) * (M[j] - 1))
        u *= 1 - step
        u[j] += step
    return u


def _bounds(est, nu, d, logTK, t, c):
    width = c * np.sqrt(np.maximum(est, 0.0) * nu * d * logTK / t)
    return est - width, est + width


def _eliminate(X, theta_hat, alive, nu, d, logTK, t, c):
    lo, hi = _bounds(X @ theta_hat, nu, d, logTK, t, c)
    keep = alive & (hi >= lo[alive].max())
    if not keep.any():  # keep the arm with the best lower bound among survivors
        keep = np.zeros_like(alive)
        keep[np.flatnonzero(alive)[np.argmax(lo[alive])]] = True
    return keep


def lin_nash(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out):
    """LinNash with the paper's width constant 6."""
    return _lin_nash(Z, Y, theta, users, cands, T, reward_type, seed, out, 6.0)


def lin_nash_tuned(Z, Y, theta, users, cands, ugroups, G, T, sigma2, reward_type, p, kappa, seed, out):
    """LinNash with width constant 1: at our horizons the paper's constant 6
    keeps most arms alive, the analogue of using kappa = 1 for our rule."""
    return _lin_nash(Z, Y, theta, users, cands, T, reward_type, seed, out, 1.0)


def _lin_nash(Z, Y, theta, users, cands, T, reward_type, seed, out, c):
    if users.size != 1 or Z.shape != (1, 1) or reward_type != BERNOULLI:
        raise ValueError("lin_nash needs a fixed arm set with Bernoulli rewards")
    rng = np.random.default_rng(seed)
    X = Z[0, 0] * Y[cands[0]]
    means = X @ theta
    K, d = X.shape
    nu = 1.0  # rewards in [0, 1] are 1-sub-Poisson (Lemma 1)
    logTK = np.log(T * K)

    # Part I: Algorithm 1 (GenerateArmSequence)
    T1 = min(T, int(np.ceil(3.0 * np.sqrt(T * d * nu * logTK))))
    lam = d_optimal_design(X)
    u = mvee_weights(X)
    design = list(np.flatnonzero(lam > 0))
    need = {z: int(np.ceil(lam[z] * T1 / 3)) for z in design}
    count = dict.fromkeys(design, 0)
    seq = np.empty(T1, np.int64)
    rr = 0
    for i in range(T1):
        if rng.random() < 0.5 or not design:
            seq[i] = rng.choice(K, p=u)
        else:
            rr %= len(design)
            z = design[rr]
            seq[i] = z
            count[z] += 1
            if count[z] >= need[z]:
                design.pop(rr)
            else:
                rr += 1
    out[:T1] = means[seq]
    rewards = (rng.random(T1) < means[seq]).astype(np.float64)
    Xs = X[seq]
    theta_hat = np.linalg.pinv(Xs.T @ Xs) @ (Xs.T @ rewards)
    alive = _eliminate(X, theta_hat, np.ones(K, bool), nu, d, logTK, T1 / 3, c)

    # Part II: phased elimination with Nash confidence bounds
    t, Tp = T1, 2.0 * T1 / 3
    while t < T:
        idx = np.flatnonzero(alive)
        lam = d_optimal_design(X[idx])
        V, s = np.zeros((d, d)), np.zeros(d)
        for a, w in zip(idx, lam):
            if w <= 0 or t >= T:
                continue
            n = min(int(np.ceil(w * Tp)), T - t)
            out[t:t + n] = means[a]
            V += n * np.outer(X[a], X[a])
            s += rng.binomial(n, means[a]) * X[a]
            t += n
        theta_hat = np.linalg.pinv(V) @ s
        alive = _eliminate(X, theta_hat, alive, nu, d, logTK, Tp, c)
        Tp *= 2
    return T1
