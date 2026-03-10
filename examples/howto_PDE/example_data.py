from typing import Dict

import numpy as np
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from utils import svd_info_from_snapshots

import pysindy as ps

__all__ = [
    "get_burgers_data",
    "get_compressible_data",
]


def get_burgers_data() -> Dict:
    data_path = "../data/burgers.mat"
    data = loadmat(data_path)
    t = np.ravel(data["t"])
    x = np.ravel(data["x"])
    u = np.real(data["usol"])

    dt = float(t[1] - t[0])
    u_dot = ps.FiniteDifference(axis=1)(u, t=t)
    svd = svd_info_from_snapshots(u)

    return {
        "name": "burgers",
        "x": x,
        "t": t,
        "dt": dt,
        "u": u[..., np.newaxis],
        "u_dot": u_dot[..., np.newaxis],
        "svd": svd,
    }


def _compressible_rhs(t, state, x, n, mu, rt):
    fields = state.reshape(n, n, 3)
    u = fields[:, :, 0]
    v = fields[:, :, 1]
    rho = fields[:, :, 2]

    fd_x1 = ps.FiniteDifference(d=1, axis=0, periodic=True)
    fd_y1 = ps.FiniteDifference(d=1, axis=1, periodic=True)
    fd_x2 = ps.FiniteDifference(d=2, axis=0, periodic=True)
    fd_y2 = ps.FiniteDifference(d=2, axis=1, periodic=True)

    ux, uy = fd_x1(u, x), fd_y1(u, x)
    vx, vy = fd_x1(v, x), fd_y1(v, x)
    uxx, uyy = fd_x2(u, x), fd_y2(u, x)
    vxx, vyy = fd_x2(v, x), fd_y2(v, x)
    px = fd_x1(rho * rt, x)
    py = fd_y1(rho * rt, x)

    out = np.zeros((n, n, 3))
    out[:, :, 0] = -(u * ux + v * uy) - (px - mu * (uxx + uyy)) / rho
    out[:, :, 1] = -(u * vx + v * vy) - (py - mu * (vxx + vyy)) / rho
    out[:, :, 2] = -(u * px / rt + v * py / rt + rho * ux + rho * vy)
    return out.reshape(3 * n * n)


def get_compressible_data(
    n: int = 24,
    nt: int = 40,
    l: float = 5.0,
    horizon: float = 0.5,
    mu: float = 1.0,
    rt: float = 1.0,
) -> Dict:
    t = np.linspace(0, horizon, nt)
    x = np.arange(0, n) * l / n
    y = np.arange(0, n) * l / n
    dx = float(x[1] - x[0])

    y0 = np.zeros((n, n, 3))
    y0[:, :, 0] = (
        -np.sin(2 * np.pi / l * x)[:, None] + 0.5 * np.cos(4 * np.pi / l * y)[None, :]
    )
    y0[:, :, 1] = (
        0.5 * np.cos(2 * np.pi / l * x)[:, None] - np.sin(4 * np.pi / l * y)[None, :]
    )
    y0[:, :, 2] = (
        1
        + 0.5 * np.cos(2 * np.pi / l * x)[:, None] * np.cos(4 * np.pi / l * y)[None, :]
    )

    sol = solve_ivp(
        _compressible_rhs,
        (t[0], t[-1]),
        y0=y0.reshape(3 * n * n),
        t_eval=t,
        args=(x, n, mu, rt),
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )

    u_obs = sol.y.reshape(n, n, 3, -1).transpose(0, 1, 3, 2)

    u_dot = ps.FiniteDifference(d=1, axis=2)(u_obs, t)
    snapshots = np.transpose(u_obs, (0, 1, 3, 2)).reshape(n * n * 3, nt)
    svd = svd_info_from_snapshots(snapshots)

    return {
        "x": x,
        "y": y,
        "t": t,
        "dt": float(t[1] - t[0]),
        "dx": dx,
        "mu": mu,
        "rt": rt,
        "u": u_obs,
        "u_dot": u_dot,
        "svd": svd,
    }
