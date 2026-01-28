import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from pysindy import SINDy  # adjust import to your package path


@pytest.fixture(scope="session")
def simple_systems():
    """Generate two simple 2D dynamical systems:
    System A: x' = -y, y' = x
    System B: x' = -2y, y' = 2x
    """
    t = np.linspace(0, 2 * np.pi, 50)
    x_a = np.stack([np.cos(t), np.sin(t)], axis=1)
    xdot_a = np.stack([-np.sin(t), np.cos(t)], axis=1)

    x_b = np.stack([np.cos(2 * t), np.sin(2 * t)], axis=1)
    xdot_b = np.stack([-2 * np.sin(2 * t), 2 * np.cos(2 * t)], axis=1)

    return (x_a, xdot_a), (x_b, xdot_b)


def test_metadata_routing_sample_weight(simple_systems):
    """Test that sample weights route correctly through SINDy fit().

    The expected coefficients are a convex combination of the systems'
    true coefficients, weighted by the number of trajectories and/or
    sample weights. This verifies that behind-the-scenes routing of
    sample_weight → optimizer.fit() works correctly.
    """
    (x_a, xdot_a), (x_b, xdot_b) = simple_systems

    # --- Build training trajectories ---
    # One system duplicated twice (implicit weighting)
    x_trajs = [x_a, x_a, x_b]
    xdot_trajs = [xdot_a, xdot_a, xdot_b]

    # --- Simple library and optimizer setup ---
    sindy = SINDy(optimizer=LinearRegression(fit_intercept=False))

    # --- Fit without explicit sample weights ---
    sindy.fit(x_trajs, t=0.1, x_dot=xdot_trajs)
    coef_unweighted = np.copy(sindy.optimizer.coef_)

    # --- Fit with sample weights to emphasize trajectory 3 (different system) ---
    sample_weight = [np.ones(len(x_a)), np.ones(len(x_a)), 10 * np.ones(len(x_b))]
    sindy.fit(x_trajs, t=0.1, x_dot=xdot_trajs, sample_weight=sample_weight)
    coef_weighted = np.copy(sindy.optimizer.coef_)

    # True systems differ by factor of 2
    ratio = np.mean(np.abs(coef_weighted / coef_unweighted))
    fail_msg = "Weighted coefficients should reflect stronger influence from system B"
    assert ratio > 1.5, fail_msg

    # 4. Convex combination logic sanity check
    sindy_A = SINDy(optimizer=LinearRegression(fit_intercept=False))
    sindy_A.fit([x_a], t=0.1, x_dot=[xdot_a])
    coef_A = np.copy(sindy_A.optimizer.coef_)

    sindy_B = SINDy(optimizer=LinearRegression(fit_intercept=False))
    sindy_B.fit([x_b], t=0.1, x_dot=[xdot_b])
    coef_B = np.copy(sindy_B.optimizer.coef_)

    expected_unweighted = (2 * coef_A + coef_B) / 3.0
    expected_weighted = (2 * coef_A + 10 * coef_B) / 12.0

    assert np.allclose(coef_unweighted, expected_unweighted, rtol=1e-2, atol=1e-6)
    assert np.allclose(coef_weighted, expected_weighted, rtol=1e-2, atol=1e-6)

    assert np.linalg.norm(coef_weighted - coef_B) < np.linalg.norm(
        coef_unweighted - coef_B
    )
