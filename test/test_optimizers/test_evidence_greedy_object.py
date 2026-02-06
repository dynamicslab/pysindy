import numpy as np
from scipy.integrate import odeint

import pysindy as ps
from pysindy.differentiation import FiniteDifference
from pysindy.optimizers import EvidenceGreedy


def lorenz(z, t):
    """Standard Lorenz system."""
    x, y, z_ = z
    return [
        10.0 * (y - x),
        x * (28.0 - z_) - y,
        x * y - 8.0 / 3.0 * z_,
    ]


def main():
    # Generate Lorenz data
    t = np.arange(0, 10.0, 0.01)
    dt = float(t[1] - t[0])
    x0 = np.array([-8.0, 8.0, 27.0], dtype=float)
    x = odeint(lorenz, x0, t)

    sigma_x = 1e-1
    x = x + sigma_x * np.random.normal(size=x.shape)

    # Differentiation method
    fd = FiniteDifference(
        order=4,
        d=1,
        axis=0,
        is_uniform=True,
        drop_endpoints=False,
        periodic=False,
    )

    # EvidenceGreedy optimizer
    
    opt = EvidenceGreedy(alpha=1e-6, max_iter=None, unbias=False, normalize_columns=True)

    # New wrapper object
    model = ps.BINDy(sigma_x)
    # model = ps.BINDy(sigma_x,
    #     optimizer=opt,
    #     differentiation_method=fd,
    #     feature_library=ps.PolynomialLibrary(degree=2, include_bias=True),
    # )

    print("\n=== BINDy (Lorenz) ===")
    # Fit using scalar dt
    model.fit(x, t=dt)

    print("\nRecovered equations:")
    model.print(precision=3)

if __name__ == "__main__":
    main()
