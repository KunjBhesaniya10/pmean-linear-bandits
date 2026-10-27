"""Linear contextual bandit environments.

One encoding covers both the fixed-arm linear bandit and the contextual
(user x item) setting, so every kernel in `pmlb.linear.algorithms` takes
the same arguments:

    Z        (n_users, dz)  user / context features
    Y        (n_items, dy)  item features
    theta    (dz * dy,)     true parameter
    users    (R,)           user who arrives in round t is users[t % R]
    cands    (R, K)         candidate items shown in round t are cands[t % R]
    ugroups  (n_users, m)   the m protected groups (ids in 0..G-1) each user belongs to
    G                       number of groups

The feature of item i for user u is kron(Z[u], Y[i]), so the expected reward
is Z[u] @ W @ Y[i] with W = theta.reshape(dz, dy). A fixed-arm bandit is the
special case of one user with Z = [[1]], R = 1 and every item a candidate.

The context stream (users, cands) is fixed across Monte Carlo runs, so the
runner's average of `out` estimates E[<x_t, theta*>] round by round, and the
per-round optimum and group of the user are known in advance.
"""
from dataclasses import dataclass, field

import numpy as np

from ..envs import BERNOULLI, GAUSSIAN  # noqa: F401  (re-exported reward types)


@dataclass
class LinearEnv:
    Z: np.ndarray
    Y: np.ndarray
    theta: np.ndarray
    users: np.ndarray
    cands: np.ndarray
    ugroups: np.ndarray
    G: int
    sigma2: float  # sub-Gaussian variance proxy given to the algorithms
    reward_type: int
    group_names: list = field(default_factory=lambda: ["all"])
    group_attr: list = field(default_factory=lambda: ["all"])  # attribute each group id belongs to

    @property
    def d(self):
        return self.Z.shape[1] * self.Y.shape[1]

    @property
    def K(self):
        return self.cands.shape[1]

    def kernel_args(self, T):
        return (np.ascontiguousarray(self.Z, dtype=np.float64),
                np.ascontiguousarray(self.Y, dtype=np.float64),
                np.ascontiguousarray(self.theta, dtype=np.float64),
                np.ascontiguousarray(self.users, dtype=np.int64),
                np.ascontiguousarray(self.cands, dtype=np.int64),
                np.ascontiguousarray(self.ugroups, dtype=np.int64),
                int(self.G), int(T), float(self.sigma2), int(self.reward_type))

    def item_means(self, u):
        """Expected reward of every item for user u."""
        W = self.theta.reshape(self.Z.shape[1], self.Y.shape[1])
        return self.Y @ (W.T @ self.Z[u])

    def all_means(self):
        """(n_users, n_items) matrix of expected rewards."""
        W = self.theta.reshape(self.Z.shape[1], self.Y.shape[1])
        return self.Z @ W @ self.Y.T

    def optimal_rewards(self, T):
        """max over the round's candidates of the expected reward, for t = 0..T-1."""
        M = self.all_means()
        R = self.users.size
        best = M[self.users[:, None], self.cands].max(axis=1)
        return best[np.arange(T) % R]

    def round_groups(self, T):
        """(T, m) groups of the user who arrives in each round."""
        return self.ugroups[self.users[np.arange(T) % self.users.size]]


def fixed_arm_env(X, theta, sigma2, reward_type=GAUSSIAN):
    """Plain linear bandit with a fixed arm set X (K, d)."""
    X = np.asarray(X, dtype=np.float64)
    return LinearEnv(Z=np.ones((1, 1)), Y=X, theta=np.asarray(theta, dtype=np.float64),
                     users=np.zeros(1, np.int64), cands=np.arange(X.shape[0])[None, :],
                     ugroups=np.zeros((1, 1), np.int64), G=1, sigma2=sigma2, reward_type=reward_type)


def rare_direction_instance(n_common=96, common_high=0.75, seed=0):
    """Fixed-arm instance where the best arm lies along a rare feature direction.

    d = 6. The n_common arms span only the first four coordinates and earn
    0.25 to common_high. Four "niche" arms use the two rare coordinates: the optimum
    (reward 0.95), a near-zero decoy (0.03) and two mediocre ones. Uniform
    exploration over arms sees the rare directions in only 4 of every
    n_common + 4 pulls.
    """
    rng = np.random.default_rng(seed)
    theta = np.array([0.4, 0.3, 0.2, 0.1, 0.9, -0.6])
    common = np.zeros((n_common, 6))
    common[:, :4] = rng.dirichlet(np.ones(4), size=n_common)
    # rescale so the common rewards span [0.25, common_high]
    r = common @ theta
    common *= (0.25 + (common_high - 0.25) * (r - r.min()) / (r.max() - r.min()))[:, None] / r[:, None]
    niche = np.array([
        [0.125, 0, 0, 0, 1.0, 0.0],    # 0.95, the optimum
        [0.075, 0, 0, 0, 0.5, 0.75],   # 0.03, near-zero decoy
        [0.25, 0, 0, 0, 0.3, 0.2],     # 0.25
        [0.0, 0, 0, 0, 0.0, -0.5],     # 0.30, rare direction 6 alone
    ])
    X = np.vstack([common, niche])
    return X, theta


def random_instance(K=50, d=6, seed=0):
    """Benign control: K arms with i.i.d. positive features, theta* > 0, rewards in about [0.2, 1]."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.2, 1.0, d)
    theta /= np.linalg.norm(theta)
    X = rng.uniform(0.0, 1.0, (K, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, theta


def two_group_instance(T, minority_share=0.1, K=10, n_items=200, seed=0):
    """Contextual instance with a majority and a minority group.

    Users: z = [1, 1{minority}], so x = kron(z, y) has a shared block and a
    minority-only block that is non-zero only in minority rounds, a feature
    direction that is rare by construction. Items y in R^4: two "mainstream"
    coordinates that both groups like and two "niche" coordinates that only
    the minority likes; the minority-specific weights also make mainstream
    items worse for them, so a model fit on mostly majority data recommends
    the wrong items to the minority.
    """
    rng = np.random.default_rng(seed)
    Y = rng.dirichlet(np.full(4, 0.5), size=n_items)
    w_shared = np.array([0.8, 0.6, 0.15, 0.1])
    w_minority = np.array([-0.6, -0.45, 0.6, 0.7])
    theta = np.concatenate([w_shared, w_minority])
    Z = np.array([[1.0, 0.0], [1.0, 1.0]])
    users = (rng.random(T) < minority_share).astype(np.int64)
    cands = np.stack([rng.choice(n_items, K, replace=False) for _ in range(T)])
    return LinearEnv(Z=Z, Y=Y, theta=theta, users=users, cands=cands,
                     ugroups=np.array([[0], [1]]), G=2, sigma2=0.25, reward_type=BERNOULLI,
                     group_names=["majority", "minority"], group_attr=["group", "group"])
