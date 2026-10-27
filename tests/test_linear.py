import numpy as np
import pandas as pd
import pytest

from pmlb.data.movielens import user_features
from pmlb.linear import ALGORITHMS
from pmlb.linear.algorithms import _update, _widths
from pmlb.linear.envs import GAUSSIAN, fixed_arm_env, random_instance, two_group_instance
from pmlb.runner import run_linear
from pmlb.welfare import contextual_p_mean_regret_curve, group_utilities


def _envs():
    X, theta = random_instance(K=8, d=3)
    return [fixed_arm_env(X, theta, 0.25, GAUSSIAN), two_group_instance(300, K=5, n_items=20)]


@pytest.mark.parametrize("name", sorted(ALGORITHMS))
@pytest.mark.parametrize("env_id", [0, 1])
def test_kernel_writes_candidate_means_and_is_seeded(name, env_id):
    env = _envs()[env_id]
    T = 300
    out1, out2 = np.empty(T), np.empty(T)
    args = env.kernel_args(T) + (0.0, 10.0)
    tau1 = ALGORITHMS[name](*args, 3, out1)
    tau2 = ALGORITHMS[name](*args, 3, out2)
    assert tau1 == tau2 and np.array_equal(out1, out2)
    M = env.all_means()
    for t in range(T):
        row = t % env.users.size
        assert np.isclose(M[env.users[row], env.cands[row]], out1[t]).any()


def test_sherman_morrison_and_widths_match_direct_linear_algebra():
    rng = np.random.default_rng(0)
    d = 5
    xs = rng.normal(size=(30, d))
    xs[:, 1] = 0.0  # zero entries exercise the sparse path
    rs = rng.normal(size=30)
    Vinv, b, th, tmp = np.eye(d), np.zeros(d), np.zeros(d), np.zeros(d)
    for x, r in zip(xs, rs):
        _update(Vinv, b, th, x, r, tmp)
    V = np.eye(d) + xs.T @ xs
    assert np.allclose(Vinv, np.linalg.inv(V))
    assert np.allclose(th, np.linalg.solve(V, xs.T @ rs))
    w = np.empty(30)
    _widths(xs, Vinv, w)
    assert np.allclose(w, np.sqrt(np.einsum("ai,ij,aj->a", xs, np.linalg.inv(V), xs)))


@pytest.mark.parametrize("name", ["linucb", "explore_then_linucb"])
def test_learns_best_arm_on_easy_instance(name):
    X = np.eye(4)
    env = fixed_arm_env(X, np.array([0.2, 0.4, 0.6, 1.0]), 0.01, GAUSSIAN)
    res = run_linear(name, env, 20000, 0.0, 4, n_jobs=1)
    assert res.expected_rewards[-1000:].mean() > 0.99


def test_exploration_grows_as_p_decreases():
    X = np.eye(4)
    env = fixed_arm_env(X, np.array([0.2, 0.4, 0.6, 1.0]), 0.04, GAUSSIAN)
    taus = [run_linear("explore_then_linucb", env, 100000, p, 4, n_jobs=1).taus.mean() for p in (0, -2, -5)]
    assert taus[0] < taus[1] < taus[2]


def test_per_group_stopping_lets_majority_exploit_first():
    env = two_group_instance(20000, minority_share=0.1)
    per_group = run_linear("explore_then_linucb", env, 20000, 0.0, 4, n_jobs=1).taus.mean()
    one_stop = run_linear("explore_then_linucb_global", env, 20000, 0.0, 4, n_jobs=1).taus.mean()
    assert per_group < one_stop


def test_group_utilities_and_contextual_regret_of_the_oracle():
    env = two_group_instance(500, K=5, n_items=20)
    rstar = env.optimal_rewards(500)
    u = group_utilities(rstar, rstar, env.round_groups(500), env.G)
    assert np.allclose(u, 1.0)
    assert np.allclose(contextual_p_mean_regret_curve(rstar, rstar, -2), 0.0)
    half = group_utilities(rstar / 2, rstar, env.round_groups(500), env.G)
    assert np.allclose(half, 0.5)


def test_movielens_user_features_one_hot():
    users = pd.DataFrame({"gender": ["F", "M"], "age": [1, 56], "occupation": [8, 0]})
    Z, ugroups, names, attrs = user_features(users)
    assert Z.shape == (2, 30) and np.allclose(Z.sum(1), 3)
    assert [names[g] for g in ugroups[0]] == ["female", "age <18", "farmer"]
    assert [names[g] for g in ugroups[1]] == ["male", "age 56+", "other"]
    assert attrs.count("occupation") == 21
