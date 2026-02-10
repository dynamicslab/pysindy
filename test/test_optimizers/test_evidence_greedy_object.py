import numpy as np
from scipy.integrate import odeint

import pysindy as ps


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
