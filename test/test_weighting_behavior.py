import numpy as np
from pysindy import SINDy
from pysindy.feature_library import IdentityLibrary
from pysindy.optimizers import STLSQ


def test_weighting_convex_combination():
    # Short, deterministic state sequence (same for all trajectories)
    n = 8
    t = np.linspace(0, 2 * np.pi, n)
    X = np.vstack([np.cos(t), np.sin(t)]).T  # shape (n, 2)

    # Two linear dynamics matrices
    M1 = np.array([[0.0, -1.0], [1.0, 0.0]])
    M2 = np.array([[0.0, -2.0], [2.0, 0.0]])

    # Build three trajectories: two copies of system M1 and one of M2
    x = [X.copy(), X.copy(), X.copy()]
    x_dot = [X @ M1.T, X @ M1.T, X @ M2.T]
    t_list = [t, t, t]

    # Per-trajectory constant sample weights (arrays)
    w1 = 1.0
    w2 = 0.5
    sw = [np.ones(n) * w1, np.ones(n) * w1, np.ones(n) * w2]

    # Use identity feature library so features are the state variables themselves
    lib = IdentityLibrary()
    # Use STLSQ configured to behave like ordinary least squares
    opt = STLSQ(threshold=0.0, alpha=0.0, max_iter=1, unbias=True)

    model = SINDy(optimizer=opt, feature_library=lib)
    model.fit(x, t_list, x_dot=x_dot, sample_weight=sw)

    coef = model.coefficients()

    # Expected convex combination: two trajectories with weight w1, one with weight w2
    # Effective weight for M1: 2 * w1, for M2: 1 * w2
    expected = (2 * w1 * M1 + w2 * M2) / (2 * w1 + w2)

    # Compare coefficients
    assert np.allclose(coef, expected, atol=1e-8, rtol=1e-6)
