import numpy as np
import pytest

from pmlb.welfare import p_mean, p_mean_regret_curve, running_p_mean


def test_p_mean_special_cases():
    v = np.array([1.0, 4.0, 16.0])
    assert p_mean(v, 1) == pytest.approx(7.0)
    assert p_mean(v, 0) == pytest.approx(4.0)
    assert p_mean(v, -1) == pytest.approx(3 / (1 + 0.25 + 1 / 16))
    assert p_mean(v, -np.inf) == 1.0


def test_p_mean_is_monotone_in_p():
    v = np.random.default_rng(0).uniform(0.1, 1.0, 20)
    vals = [p_mean(v, p) for p in (-5, -1.5, -0.5, 0, 0.5, 1)]
    assert np.all(np.diff(vals) > 0)


@pytest.mark.parametrize("p", [1, 0.5, 0, -0.5, -1.5])
def test_running_p_mean_matches_prefix(p):
    r = np.random.default_rng(1).uniform(0.1, 1.0, 50)
    curve = running_p_mean(r, p)
    for t in (1, 7, 50):
        assert curve[t - 1] == pytest.approx(p_mean(r[:t], p))


def test_regret_zero_when_always_optimal():
    assert np.allclose(p_mean_regret_curve(np.full(10, 0.9), 0.9, -1.5), 0.0)
