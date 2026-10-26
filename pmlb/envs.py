"""Stochastic multi-armed bandit instances.

Reward sampling is a numba function keyed by an integer reward type so the
compiled algorithm kernels can call it directly.
"""
from dataclasses import dataclass

import numpy as np
from numba import njit

BERNOULLI = 0
GAUSSIAN = 1


@njit(cache=True)
def sample_reward(mean, sigma, reward_type):
    if reward_type == BERNOULLI:
        return 1.0 if np.random.rand() < mean else 0.0
    return np.random.normal(mean, sigma)


@dataclass
class MABInstance:
    means: np.ndarray
    reward_type: int
    sigma2: float  # sub-Gaussian variance proxy given to the algorithms

    @property
    def k(self):
        return self.means.size

    @property
    def mu_star(self):
        return float(self.means.max())

    @property
    def sigma(self):
        return float(np.sqrt(self.sigma2))


def bernoulli_instance(k, low=0.005, high=1.0, sigma2=1.0, rng=np.random):
    # sigma2 = 1 matches the reference code (the tight proxy for Bernoulli is 1/4).
    return MABInstance(rng.uniform(low, high, size=k), BERNOULLI, sigma2)


def gaussian_instance(k, low=10.0, high=1000.0, sigma2=400.0, rng=np.random):
    return MABInstance(rng.uniform(low, high, size=k), GAUSSIAN, sigma2)
