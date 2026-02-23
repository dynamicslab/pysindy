import numpy as np
from scipy.integrate import odeint

import pysindy as ps
from pysindy.differentiation import FiniteDifference
from pysindy.optimizers import EvidenceGreedy

from pysindy.utils.bindy import TemporalNoisePropagation

import random
rng = np.random.default_rng(seed=42)

def lorenz(z, t):
    """Standard Lorenz system."""
    x, y, z_ = z
    return [
        10.0 * (y - x),
        x * (28.0 - z_) - y,
        x * y - 8.0 / 3.0 * z_,
    ]


def _lorenz_data(sigma_x):
    t = np.arange(0, 10, 0.01)
    x0 = np.array([-8.0, 8.0, 27.0], dtype=float)
    x = odeint(lorenz, x0, t)
    x = x + sigma_x * rng.normal(size=x.shape)
    return t, x0, x


def test_evidence_greedy_lorenz_example():
    """Check that EvidenceGreedy can reasonably recover the Lorenz dynamics.

    This mirrors the Lorenz example in the EvidenceGreedy docstring.
    """
    # Measurement Noise Level
    sigma_x = 1e-2

    # Time grid and data (same as in the docstring)
    t, x0, x = _lorenz_data(sigma_x)

    # t = np.arange(0, 10, 0.01, dtype=float)
    # x = np.ones((len(t), 3), dtype=float) 
    fd = FiniteDifference(
        order=6,
        d=1,
        axis=0,
        is_uniform=True,
        drop_endpoints=False,
        periodic=False,
    )

    # Compute noise variance in x_dot via temporal noise propagation
    sigma2 = TemporalNoisePropagation(fd, t, sigma_x)

    # EvidenceGreedy optimizer with the same hyperparameters as the docstring
    opt = EvidenceGreedy(alpha=1e-6, _sigma2=sigma2, max_iter=20, unbias = True, normalize_columns = True)

    model = ps.SINDy(optimizer=opt)
    model.fit(x, t[1] - t[0])
    model.print()

    print(model.optimizer.ind_.T)
    print(model.optimizer.coef_.T)


if __name__ == "__main__":
    test_evidence_greedy_lorenz_example()
