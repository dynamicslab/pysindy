from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.figure import Figure
from sklearn.neighbors import KernelDensity


def plot_burgers_data_and_derivative(x, t, u, u_dot) -> Figure:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    X, T = np.meshgrid(x, t, indexing="ij")
    axs[0].pcolormesh(T, -X, u[..., 0])
    axs[0].set_title("Burgers u(x,t)")
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("x")
    axs[1].pcolormesh(T, -X, u_dot[..., 0])
    axs[1].set_title(r"Burgers $u_t$ from FD")
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("x")
    return fig


def plot_compressible_data_and_derivative(
    t, u, u_dot, time_index: Optional[int] = None
) -> None:
    if time_index is None:
        time_index = len(t) // 2
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    labels = ["u", "v", "rho"]
    for i, name in enumerate(labels):
        axs[0, i].imshow(u[:, :, time_index, i], cmap="seismic")
        axs[0, i].set_title(f"{name}(x,y,tmid)")
        axs[1, i].imshow(u_dot[:, :, time_index, i], cmap="seismic")
        axs[1, i].set_title(f"{name}_t(x,y,tmid)")


def plot_svd_energy(svd_info, title: str) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
    axs[0].semilogy(svd_info.singular_values / svd_info.singular_values[0], "o-")
    axs[0].set_title(f"{title}: singular values")
    axs[0].set_xlabel("mode")
    axs[1].plot(svd_info.energy, "o-")
    axs[1].axhline(0.99, color="k", linestyle="--")
    axs[1].axvline(svd_info.rank_99 - 1, color="r", linestyle="--")
    axs[1].set_title(f"{title}: cumulative energy")
    axs[1].set_xlabel("mode")


def plot_field_prediction(
    true_field: np.ndarray, pred_fields: dict[str, np.ndarray], title: str
) -> None:
    ncols = len(pred_fields) + 1
    fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 3), constrained_layout=True)
    axs[0].imshow(true_field, cmap="seismic")
    axs[0].set_title("True")
    for i, (name, pred) in enumerate(pred_fields.items(), start=1):
        axs[i].imshow(pred, cmap="seismic")
        axs[i].set_title(name)
    fig.suptitle(title)


def excess_information_loss(x_true: np.ndarray, x_sim: np.ndarray) -> float:
    kde = KernelDensity(kernel="gaussian").fit(x_true)
    base = kde.score_samples(x_true).sum()
    return float(base - kde.score_samples(x_sim).sum())


def print_trapping_diagnostics(model, name: str) -> None:
    eig_hist = np.asarray(getattr(model.optimizer, "PWeigs_history_", []))
    if eig_hist.size:
        print(f"{name} trapping eig max (last): {np.max(eig_hist[-1]):.4e}")


@dataclass
class SVDInfo:
    singular_values: np.ndarray
    energy: np.ndarray
    rank_99: int
    basis: np.ndarray
    time_series: np.ndarray


def svd_info_from_snapshots(snapshot_matrix: np.ndarray) -> SVDInfo:
    u, s, _ = np.linalg.svd(snapshot_matrix, full_matrices=False)
    denom = np.sum(s**2)
    if denom <= 0:
        energy = np.zeros_like(s)
        rank_99 = 1 if s.size else 0
    else:
        energy = np.cumsum(s**2) / denom
        rank_99 = int(np.searchsorted(energy, 0.99) + 1)
    rank_99 = min(max(rank_99, 1), u.shape[1]) if u.size else 0
    basis = u[:, :rank_99] if rank_99 > 0 else np.zeros((snapshot_matrix.shape[0], 0))
    coeffs = (
        (basis.T @ snapshot_matrix).T
        if rank_99 > 0
        else np.zeros((snapshot_matrix.shape[1], 0))
    )
    return SVDInfo(s, energy, rank_99, basis, coeffs)


def burgers_true_coefficients(feature_names: list[str]) -> list[dict[sp.Expr, float]]:
    x0 = feature_names[0]
    terms = {
        sp.parse_expr(f"{x0} * {x0}_1"): -1.0,
        sp.parse_expr(f"{x0}_11"): 0.1,
    }
    return [terms]


def compressible_true_coefficients(
    feature_names: list[str], mu: float, rt: float
) -> list[dict[sp.Expr, float]]:
    x0 = feature_names[0]
    x1 = feature_names[1]
    x2 = feature_names[2]
    return [
        {
            sp.parse_expr(f"{x0} * {x0}_1"): -1.0,
            sp.parse_expr(f"{x1} * {x0}_2"): -1.0,
            sp.parse_expr(f"{x2}**-1 * {x2}_1"): -float(rt),
            sp.parse_expr(f"{x2}**-1 * {x0}_11"): float(mu),
            sp.parse_expr(f"{x2}**-1 * {x0}_22"): float(mu),
        },
        {
            sp.parse_expr(f"{x0} * {x1}_1"): -1.0,
            sp.parse_expr(f"{x1} * {x1}_2"): -1.0,
            sp.parse_expr(f"{x2}**-1 * {x2}_2"): -float(rt),
            sp.parse_expr(f"{x2}**-1 * {x1}_11"): float(mu),
            sp.parse_expr(f"{x2}**-1 * {x1}_22"): float(mu),
        },
        {
            sp.parse_expr(f"{x0} * {x2}_1"): -1.0,
            sp.parse_expr(f"{x1} * {x2}_2"): -1.0,
            sp.parse_expr(f"{x2} * {x0}_1"): -1.0,
            sp.parse_expr(f"{x2} * {x1}_2"): -1.0,
        },
    ]
