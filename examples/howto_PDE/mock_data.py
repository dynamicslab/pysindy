from typing import Dict

import numpy as np
from utils import svd_info_from_snapshots

__all__ = [
    "get_burgers_data",
    "get_compressible_data",
]


def get_burgers_data() -> Dict:
    nx, nt = 64, 40
    x = np.linspace(-8.0, 8.0, nx)
    t = np.linspace(0.0, 10.0, nt)
    dt = float(t[1] - t[0])

    u = np.ones((nx, nt, 1))

    u_dot = np.zeros_like(u)
    svd = svd_info_from_snapshots(u[..., 0])

    return {
        "name": "burgers",
        "x": x,
        "t": t,
        "dt": dt,
        "u": u,
        "u_dot": u_dot,
        "svd": svd,
    }


def get_compressible_data(
    n: int = 16,
    nt: int = 30,
    l: float = 5.0,
    horizon: float = 0.5,
    mu: float = 1.0,
    rt: float = 1.0,
) -> Dict:
    x = np.linspace(0.0, l, n, endpoint=False)
    y = np.linspace(0.0, l, n, endpoint=False)
    t = np.linspace(0.0, horizon, nt)
    dt = float(t[1] - t[0])
    dx = float(x[1] - x[0]) if n > 1 else 1.0

    u = np.ones((n, n, nt, 3))
    u_dot = np.zeros_like(u)

    snapshots = np.transpose(u, (0, 1, 3, 2)).reshape(n * n * 3, nt)
    svd = svd_info_from_snapshots(snapshots)

    return {
        "name": "compressible_mock",
        "x": x,
        "y": y,
        "t": t,
        "dt": dt,
        "dx": dx,
        "mu": mu,
        "rt": rt,
        "u": u,
        "u_dot": u_dot,
        "svd": svd,
    }
