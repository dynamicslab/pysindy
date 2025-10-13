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
    X_trajs = [x_a, x_a, x_b]
    Xdot_trajs = [xdot_a, xdot_a, xdot_b]

    # --- Simple library and optimizer setup ---
    sindy = SINDy(optimizer=LinearRegression(fit_intercept=False))

    # --- Fit without explicit sample weights ---
    sindy.fit(X_trajs, t=0.1, x_dot=Xdot_trajs)
    coef_unweighted = np.copy(sindy.model.named_steps["model"].coef_)

    # --- Fit with sample weights to emphasize trajectory 3 (different system) ---
    sample_weight = [np.ones(len(x_a)), np.ones(len(x_a)), 10 * np.ones(len(x_b))]
    sindy.fit(X_trajs, t=0.1, x_dot=Xdot_trajs, sample_weight=sample_weight)
    coef_weighted = np.copy(sindy.model.named_steps["model"].coef_)

    # --- Assertions ---
    # 1. Shapes are consistent
    assert coef_weighted.shape == coef_unweighted.shape

    # 2. The coefficients must differ when weighting is applied
    assert not np.allclose(coef_weighted, coef_unweighted)

    # 3. Weighted model should bias toward system B coefficients
    #    since trajectory B had much higher weight
    # True systems differ by factor of 2
    ratio = np.mean(np.abs(coef_weighted / coef_unweighted))
    assert ratio > 1.05, "Weighted coefficients should reflect stronger influence from system B"

    # 4. Convex combination logic sanity check
    # Unweighted: (A + A + B)/3 = A * 2/3 + B * 1/3
    # Weighted:   (A + A + 10*B)/(12) ≈ A * 2/12 + B * 10/12
    # So weighted coefficients should be closer to B's dynamics
    assert np.linalg.norm(coef_weighted - 2 * coef_unweighted) < np.linalg.norm(coef_unweighted)
