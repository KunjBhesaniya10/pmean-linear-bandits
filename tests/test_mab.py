import numpy as np
import pytest

from pmlb.envs import BERNOULLI, GAUSSIAN, MABInstance
from pmlb.mab import ALGORITHMS
from pmlb.runner import run_mab


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
def test_kernel_writes_valid_rewards_and_is_seeded(name):
    means = np.linspace(0.1, 0.9, 5)
    out1, out2 = np.empty(2000), np.empty(2000)
    tau1 = ALGORITHMS[name](means, 2000, 1.0, BERNOULLI, 0.0, 7, out1)
    tau2 = ALGORITHMS[name](means, 2000, 1.0, BERNOULLI, 0.0, 7, out2)
    assert np.isin(out1, means).all()
    assert tau1 == tau2 and np.array_equal(out1, out2)


@pytest.mark.parametrize("name", ["welfarist_ucb", "ucb"])
def test_learns_best_arm_on_easy_instance(name):
    inst = MABInstance(np.array([100.0, 300.0, 500.0, 900.0]), GAUSSIAN, 400.0)
    res = run_mab(name, inst, 20000, 0.0, 4, n_jobs=1)
    assert res.expected_rewards[-1000:].mean() > 0.99 * inst.mu_star


def test_welfarist_exploration_grows_as_p_decreases():
    inst = MABInstance(np.array([10.0, 50.0, 200.0, 400.0]), GAUSSIAN, 400.0)
    taus = [run_mab("welfarist_ucb", inst, 200000, p, 4, n_jobs=1).taus.mean() for p in (0, -2, -5)]
    assert taus[0] < taus[1] < taus[2]
