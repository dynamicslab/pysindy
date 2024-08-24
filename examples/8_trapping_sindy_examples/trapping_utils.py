import warnings

import matplotlib.gridspec as gridspec
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
from sympy import limit
from sympy import Symbol

warnings.filterwarnings("ignore")
import pysindy as ps
import dysts.flows as flows
from dysts.analysis import sample_initial_conditions
import pymech.neksuite as nek

# Initialize quadratic SINDy library, with custom ordering
# to be consistent with the constraint
sindy_library = ps.PolynomialLibrary(include_bias=True)
sindy_library_no_bias = ps.PolynomialLibrary(include_bias=False)

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-15
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-10

# Define constants for loading in Von Karman data
nel = 2622  # Number of spectral elements
nGLL = 7  # Order of the spectral mesh


# define the objective function to be minimized by simulated annealing
def obj_function(m, L_obj, Q_obj, P_obj):
    lsv, sing_vals, rsv = np.linalg.svd(P_obj)
    P_rt = lsv @ np.diag(np.sqrt(sing_vals)) @ rsv
    P_rt_inv = lsv @ np.diag(np.sqrt(1 / sing_vals)) @ rsv
    mQ_full = np.tensordot(Q_obj, m, axes=([2], [0])) + np.tensordot(
        Q_obj, m, axes=([1], [0])
    )
    A_obj = L_obj + mQ_full
    As = (P_rt @ A_obj @ P_rt_inv + P_rt_inv @ A_obj.T @ P_rt) / 2
    eigvals, eigvecs = np.linalg.eigh(As)
    return eigvals[-1]

def get_trapping_radius(max_eigval, eps_Q, d):
    x = Symbol("x")
    delta = max_eigval**2 - 4 * eps_Q * np.linalg.norm(d, 2) / 3
    delta_func = max_eigval**2 - 4 * x * np.linalg.norm(d, 2) / 3
    rad_trap = 0
    rad_stab = 0
    if max_eigval < 0 and delta >= 0:
        y_trap = -(3 / (2 * x)) * (max_eigval + sp.sqrt(delta_func))
        y_stab = (3 / (2 * x)) * (-max_eigval + sp.sqrt(delta_func))
        rad_trap = limit(y_trap, x, eps_Q, dir="+")
        rad_stab = limit(y_stab, x, eps_Q, dir="+")
    return rad_trap, rad_stab


def check_local_stability(Xi, sindy_opt, mean_val):
    mod_matrix = sindy_opt.mod_matrix
    rt_mod_mat = sindy_opt.rt_mod_mat
    rt_inv_mod_mat = sindy_opt.rt_inv_mod_mat
    opt_m = sindy_opt.m_history_[-1]
    PC_tensor = sindy_opt.PC_
    PL_tensor_unsym = sindy_opt.PL_unsym_
    PL_tensor = sindy_opt.PL_
    PM_tensor = sindy_opt.PM_
    PQ_tensor = sindy_opt.PQ_
    mPM = np.tensordot(PM_tensor, opt_m, axes=([2], [0]))
    P_tensor = PL_tensor_unsym + mPM
    As = np.tensordot(P_tensor, Xi, axes=([3, 2], [0, 1]))
    As = (rt_mod_mat @ As @ rt_inv_mod_mat + rt_inv_mod_mat @ As.T @ rt_mod_mat) / 2
    eigvals, _ = np.linalg.eigh(As)
    print("optimal m: ", opt_m)
    print("As eigvals: ", np.sort(eigvals))
    max_eigval = np.sort(eigvals)[-1]
    C = np.tensordot(PC_tensor, Xi, axes=([2, 1], [0, 1]))
    L = np.tensordot(PL_tensor_unsym, Xi, axes=([3, 2], [0, 1]))
    Q = np.tensordot(
        mod_matrix, np.tensordot(PQ_tensor, Xi, axes=([4, 3], [0, 1])), axes=([1], [0])
    )
    Q = (Q + np.transpose(Q, [1, 2, 0]) + np.transpose(Q, [2, 0, 1]))
    Q = np.tensordot(
        rt_inv_mod_mat,
        np.tensordot(
            rt_inv_mod_mat,
            np.tensordot(
                rt_inv_mod_mat,
                Q,
                axes=([1], [0])
            ),
            axes=([0], [1])
        ),
        axes=([0], [2])
    )
    # Q = np.einsum("ya,abc,bd,cf", rt_inv_mod_mat, Q, rt_inv_mod_mat, rt_inv_mod_mat)
    eps_Q = np.sqrt(np.sum(Q ** 2))
    print(r'0.5 * |tilde{H}_0|_F = ', 0.5 * eps_Q)
    print(r'0.5 * |tilde{H}_0|_F^2 / beta = ', 0.5 * eps_Q ** 2 / sindy_opt.beta)
    Q = np.tensordot(PQ_tensor, Xi, axes=([4, 3], [0, 1]))
    d = C + np.dot(L, opt_m) + np.dot(np.tensordot(Q, opt_m, axes=([2], [0])), opt_m)
    d = rt_mod_mat @ d
    Rm, R_ls = get_trapping_radius(max_eigval, eps_Q, d)
    Reff = Rm / mean_val
    print("Estimate of trapping region size, Rm = ", Rm)
    if not np.isclose(mean_val, 1.0):
        print("Normalized trapping region size, Reff = ", Reff)
        print("Local stability size, R_ls= ", R_ls)
    return Rm, R_ls


# use optimal m, calculate and plot the stability radius when the third-order
# energy-preserving scheme slightly breaks
def make_trap_progress_plots(r, sindy_opt):
    mod_matrix = sindy_opt.mod_matrix
    PC_tensor = sindy_opt.PC_
    PL_tensor_unsym = sindy_opt.PL_unsym_
    PQ_tensor = sindy_opt.PQ_
    ms = sindy_opt.m_history_
    eigs = sindy_opt.PWeigs_history_
    coef_history = sindy_opt.history_
    rhos_plus = []
    rhos_minus = []
    for i in range(len(eigs)):
        if eigs[i][-1] < 0:
            Xi = coef_history[i]
            C = np.tensordot(PC_tensor, Xi, axes=([2, 1], [1, 0]))
            L = np.tensordot(PL_tensor_unsym, Xi, axes=([3, 2], [1, 0]))
            Q = np.tensordot(
                mod_matrix,
                np.tensordot(PQ_tensor, Xi, axes=([4, 3], [1, 0])),
                axes=([1], [0]),
            )
            Q_ep = (Q + np.transpose(Q, [1, 2, 0]) + np.transpose(Q, [2, 0, 1]))
            Qijk_permsum = np.tensordot(
                sindy_opt.rt_inv_mod_mat,
                np.tensordot(
                    sindy_opt.rt_inv_mod_mat,
                    np.tensordot(
                        sindy_opt.rt_inv_mod_mat,
                        Q_ep,
                        axes=([1], [0])
                    ),
                    axes=([0], [1])
                ),
                axes=([0], [2])
            )
            eps_Q = np.sqrt(np.sum(Qijk_permsum ** 2))
            Q = np.tensordot(PQ_tensor, Xi, axes=([4, 3], [1, 0]))
            d = (
                C
                + np.dot(L, ms[i])
                + np.dot(np.tensordot(Q, ms[i], axes=([2], [0])), ms[i])
            )
            d = sindy_opt.rt_mod_mat @ d
            delta = (
                eigs[i][-1] ** 2
                - 4 * eps_Q * np.linalg.norm(d, 2) / 3
            )
            if delta < 0:
                Rm = 0
                DA = 0
            else:
                Rm = -(3 / (2 * eps_Q)) * (
                    eigs[i][-1] + np.sqrt(delta)
                )
                DA = (3 / (2 * eps_Q)) * (
                    -eigs[i][-1] + np.sqrt(delta)
                )
            rhos_plus.append(DA)
            rhos_minus.append(Rm)
    try:
        x = np.arange(len(rhos_minus[1:]))
        plt.figure()
        plt.plot(x, rhos_minus[1:], "k--", label=r"$\rho_-$", linewidth=3)
        plt.plot(x, rhos_plus[1:], label=r"$\rho_+$", linewidth=3, color="k")
        ax = plt.gca()
        ax.fill_between(
            x, rhos_minus[1:], rhos_plus[1:], color="c", label=r"$\dot{K} < 0$"
        )
        ax.fill_between(
            x,
            rhos_plus[1:],
            np.ones(len(x)) * rhos_plus[-1] * 5,
            color="r",
            label="Possibly unstable",
        )
        ax.fill_between(
            x, np.zeros(len(x)), rhos_minus[1:], color="g", label=r"Trapping region"
        )
        plt.grid(True)
        plt.ylabel("Stability radius")
        plt.xlabel("Algorithm iteration")
        plt.legend(framealpha=1.0)
        plt.xlim(1, x[-1])
        plt.ylim(1, rhos_plus[-1] * 5)
        plt.xscale("log")
        plt.yscale("log")
    except IndexError:
        print(
            "The A^S matrix is not fully Hurwitz so will not plot the stability radii"
        )
    return rhos_minus, rhos_plus


# Plot first three modes in 3D for ground truth and SINDy prediction
def make_3d_plots(x_test, x_test_pred, filename):
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(8, 8))
    plt.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2], "r", label="true x")
    plt.plot(
        x_test_pred[:, 0], x_test_pred[:, 1], x_test_pred[:, 2], "k", label="pred x"
    )
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_axis_off()
    plt.legend(fontsize=14)


# Plot the SINDy fits of X and Xdot against the ground truth
def make_fits(r, t, xdot_test, xdot_test_pred, x_test, x_test_pred, filename):
    fig = plt.figure(figsize=(16, 8))
    spec = gridspec.GridSpec(ncols=2, nrows=r, figure=fig, hspace=0.0, wspace=0.0)
    for i in range(r):
        plt.subplot(spec[i, 0])  # r, 2, 2 * i + 2)
        plt.plot(t, xdot_test[:, i], "r", label=r"true $\dot{x}_" + str(i) + "$")
        plt.plot(t, xdot_test_pred[:, i], "k--", label=r"pred $\dot{x}_" + str(i) + "$")
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.legend(fontsize=12)
        if i == r - 1:
            plt.xlabel("t", fontsize=18)
        plt.subplot(spec[i, 1])
        plt.plot(t, x_test[:, i], "r", label=r"true $x_" + str(i) + "$")
        plt.plot(t, x_test_pred[:, i], "k--", label=r"pred $x_" + str(i) + "$")
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.legend(fontsize=12)
        if i == r - 1:
            plt.xlabel("t", fontsize=18)


# Plot errors between m_{k+1} and m_k and similarly for the model coefficients
def make_progress_plots(r, sindy_opt):
    W = np.asarray(sindy_opt.history_)
    M = np.asarray(sindy_opt.m_history_)
    dW = np.zeros(W.shape[0])
    dM = np.zeros(M.shape[0])
    for i in range(1, W.shape[0]):
        dW[i] = np.sum((W[i, :, :] - W[i - 1, :, :]) ** 2)
        dM[i] = np.sum((M[i, :] - M[i - 1, :]) ** 2)
    plt.figure()
    plt.semilogy(dW, label=r"Coefficient progress, $\|\xi_{k+1} - \xi_k\|_2^2$")
    plt.semilogy(dM, label=r"Vector m progress, $\|m_{k+1} - m_k\|_2^2$")
    plt.xlabel("Algorithm iterations", fontsize=16)
    plt.ylabel("Errors", fontsize=16)
    plt.legend(fontsize=14)
    PWeigs = np.asarray(sindy_opt.PWeigs_history_)
    plt.figure()
    for j in range(r):
        if np.all(PWeigs[:, j] > 0.0):
            plt.semilogy(PWeigs[:, j], label=r"diag($P\xi)_{" + str(j) + str(j) + "}$")
        else:
            plt.plot(PWeigs[:, j], label=r"diag($P\xi)_{" + str(j) + str(j) + "}$")
        plt.xlabel("Algorithm iterations", fontsize=16)
        plt.legend(fontsize=12)
        plt.ylabel(r"Eigenvalues of $P\xi$", fontsize=16)


def load_data(
    systems_list,
    all_properties,
    n=200,
    pts_per_period=20,
    random_bump=False,
    include_transients=False,
    n_trajectories=20,
):
    all_sols_train = dict()
    all_sols_test = dict()
    all_t_train = dict()
    all_t_test = dict()

    for i, equation_name in enumerate(systems_list):
        eq = getattr(flows, equation_name)()
        all_sols_train[equation_name] = []
        all_sols_test[equation_name] = []
        all_t_train[equation_name] = []
        all_t_test[equation_name] = []
        print(i, eq)

        for j in range(n_trajectories):
            ic_train, ic_test = sample_initial_conditions(
                eq, 2, traj_length=1000, pts_per_period=30
            )

            # Kick it off the attractor by random bump with, at most, 1% of the norm of the IC
            if random_bump:
                ic_train += (np.random.rand(len(ic_train)) - 0.5) * abs(ic_train) / 50
                ic_test += (np.random.rand(len(ic_test)) - 0.5) * abs(ic_test) / 50

            # Sample at roughly the smallest time scale!!
            if include_transients:
                pts_per_period = int(1 / (all_properties[equation_name]["dt"] * 10))
                n = pts_per_period * 10  # sample 10 periods at the largest time scale

            eq.ic = ic_train
            t_sol, sol = eq.make_trajectory(
                n,
                pts_per_period=pts_per_period,
                resample=True,
                return_times=True,
                standardize=False,
            )
            all_sols_train[equation_name].append(sol)
            all_t_train[equation_name].append(t_sol)
            eq.ic = ic_test
            t_sol, sol = eq.make_trajectory(
                n,
                pts_per_period=pts_per_period,
                resample=True,
                return_times=True,
                standardize=False,
            )
            all_sols_test[equation_name].append(sol)
            all_t_test[equation_name].append(t_sol)
    return all_sols_train, all_t_train, all_sols_test, all_t_test


# Make a bar plot of the distribution of SINDy coefficients
# and distribution of Galerkin coefficients for the von Karman street
def make_bar(galerkin9, L, Q, Lens, Qens):
    r = L.shape[0]
    bins = np.logspace(-11, 0, 50)
    plt.figure(figsize=(8, 4))
    plt.grid("True")
    galerkin_full = np.vstack(
        (
            galerkin9["L"].reshape(r**2, 1),
            galerkin9["Q"].reshape(len(galerkin9["Q"].flatten()), 1),
        )
    )
    plt.hist(np.abs(galerkin_full), bins=bins, label="POD-9 model")
    sindy_full = np.vstack(
        (L.reshape(r**2, 1), Q.reshape(len(galerkin9["Q"].flatten()), 1))
    )
    plt.hist(
        np.abs(sindy_full.flatten()),
        bins=bins,
        color="k",
        label="Trapping SINDy model (energy)",
    )
    sindy_full = np.vstack(
        (Lens.reshape(r**2, 1), Qens.reshape(len(galerkin9["Q"].flatten()), 1))
    )
    plt.hist(
        np.abs(sindy_full.flatten()),
        bins=bins,
        color="r",
        label="Trapping SINDy model (enstrophy)",
    )
    plt.xscale("log")
    plt.legend(fontsize=14)
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_yticks([0, 10, 20, 30])
    plt.xlabel("Coefficient values", fontsize=20)
    plt.ylabel("Number of coefficients", fontsize=20)
    plt.title("Histogram of coefficient values", fontsize=20)


# Helper function for reading and plotting the von Karman data
def get_velocity(file):
    field = nek.readnek(file)
    u = np.array(
        [
            field.elem[i].vel[0, 0, j, k]
            for i in range(nel)
            for j in range(nGLL)
            for k in range(nGLL)
        ]
    )
    v = np.array(
        [
            field.elem[i].vel[1, 0, j, k]
            for i in range(nel)
            for j in range(nGLL)
            for k in range(nGLL)
        ]
    )
    return u, v


# Helper function for reading and plotting the von Karman data
def get_vorticity(file):
    field = nek.readnek(file)
    vort = np.array(
        [
            field.elem[i].temp[0, 0, j, k]
            for i in range(nel)
            for j in range(nGLL)
            for k in range(nGLL)
        ]
    )
    return vort


# Define von Karman grid
nx = 400
ny = 200
xmesh = np.linspace(-5, 15, nx)
ymesh = np.linspace(-5, 5, ny)
XX, YY = np.meshgrid(xmesh, ymesh)


# Helper function for plotting the von Karman data
def interp(
    field, Cx, Cy, method="cubic", mask=(np.sqrt(XX**2 + YY**2) < 0.5).flatten("C")
):
    """
    field - 1D array of cell values
    Cx, Cy - cell x-y values
    X, Y - meshgrid x-y values
    grid - if exists, should be an ngrid-dim logical that will be set to zer
    """
    ngrid = len(XX.flatten())
    grid_field = np.squeeze(
        np.reshape(griddata((Cx, Cy), field, (XX, YY), method=method), (ngrid, 1))
    )
    if mask is not None:
        grid_field[mask] = 0
    return grid_field


# Helper function for plotting the von Karman data
def plot_field(field, clim=[-5, 5], label=None):
    """Plot cylinder field with masked circle"""
    im = plt.imshow(
        field,
        cmap="RdBu",
        vmin=clim[0],
        vmax=clim[1],
        origin="lower",
        extent=[-5, 15, -5, 5],
        interpolation="gaussian",
        label=label,
    )
    cyl = plt.Circle((0, 0), 0.5, edgecolor="k", facecolor="gray")
    plt.gcf().gca().add_artist(cyl)
    return im


# Initialize a function for general quadratic Galerkin models
def galerkin_model(a, L, Q):
    """RHS of POD-Galerkin model, for time integration"""
    return (L @ a) + np.einsum("ijk,j,k->i", Q, a, a)


# Plot the SINDy trajectory, trapping region, and ellipsoid where Kdot >= 0
def trapping_region(r, x_test_pred, Xi, sindy_opt, filename):

    # Need to compute A^S from the optimal m obtained from SINDy algorithm
    opt_m = sindy_opt.m_history_[-1]
    PL_tensor_unsym = sindy_opt.PL_unsym_
    PL_tensor = sindy_opt.PL_
    PQ_tensor = sindy_opt.PQ_
    mPQ = np.tensordot(opt_m, PQ_tensor, axes=([0], [0]))
    P_tensor = PL_tensor - mPQ
    As = np.tensordot(P_tensor, Xi, axes=([3, 2], [0, 1]))
    eigvals, eigvecs = np.linalg.eigh(As)
    print("optimal m: ", opt_m)
    print("As eigvals: ", eigvals)

    # Extract maximum eigenvalue, and compute radius of the trapping region
    max_eigval = np.sort(eigvals)[-1]

    # Should be using the unsymmetrized L
    L = np.tensordot(PL_tensor_unsym, Xi, axes=([3, 2], [0, 1]))
    Q = np.tensordot(PQ_tensor, Xi, axes=([4, 3], [0, 1]))
    d = np.dot(L, opt_m) + np.dot(np.tensordot(Q, opt_m, axes=([2], [0])), opt_m)
    Rm = np.linalg.norm(d) / np.abs(max_eigval)

    # Make 3D plot illustrating the trapping region
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, figsize=(8, 8))
    Y = np.zeros(x_test_pred.shape)
    Y = x_test_pred - opt_m * np.ones(x_test_pred.shape)

    Y = np.dot(eigvecs, Y.T).T
    plt.plot(
        Y[:, 0],
        Y[:, 1],
        Y[:, -1],
        "k",
        label="SINDy model prediction with new initial condition",
        alpha=1.0,
        linewidth=3,
    )
    h = np.dot(eigvecs, d)

    alpha = np.zeros(r)
    for i in range(r):
        if filename == "Von Karman" and (i == 2 or i == 3):
            h[i] = 0
        alpha[i] = np.sqrt(0.5) * np.sqrt(np.sum(h**2 / eigvals) / eigvals[i])

    shift_orig = h / (4.0 * eigvals)

    # draw sphere in eigencoordinate space, centered at 0
    u, v = np.mgrid[0 : 2 * np.pi : 40j, 0 : np.pi : 20j]
    x = Rm * np.cos(u) * np.sin(v)
    y = Rm * np.sin(u) * np.sin(v)
    z = Rm * np.cos(v)

    ax.plot_wireframe(
        x,
        y,
        z,
        color="b",
        label=r"Trapping region estimate, $B(m, R_m)$",
        alpha=0.5,
        linewidth=0.5,
    )
    ax.plot_surface(x, y, z, color="b", alpha=0.05)
    ax.view_init(elev=0.0, azim=30)

    # define ellipsoid
    rx, ry, rz = np.asarray([alpha[0], alpha[1], alpha[-1]])

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    # Add this piece so we can compare with the analytic Lorenz ellipsoid,
    # which is typically defined only with a shift in the "y" direction here.
    if filename == "Lorenz Attractor":
        shift_orig[0] = 0
        shift_orig[-1] = 0
    x = rx * np.outer(np.cos(u), np.sin(v)) - shift_orig[0]
    y = ry * np.outer(np.sin(u), np.sin(v)) - shift_orig[1]
    z = rz * np.outer(np.ones_like(u), np.cos(v)) - shift_orig[-1]

    # Plot ellipsoid
    ax.plot_wireframe(
        x,
        y,
        z,
        rstride=5,
        cstride=5,
        color="r",
        label="Ellipsoid of positive energy growth",
        alpha=1.0,
        linewidth=0.5,
    )

    if filename == "Lorenz Attractor":
        rho = 28.0
        beta = 8.0 / 3.0

        # define analytic ellipsoid in original Lorenz state space
        rx, ry, rz = [np.sqrt(beta * rho), np.sqrt(beta * rho**2), rho]

        # Set of all spherical angles:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        # ellipsoid in (x, y, z) coordinate to -> shifted by m
        x = rx * np.outer(np.cos(u), np.sin(v)) - opt_m[0]
        y = ry * np.outer(np.sin(u), np.sin(v)) - opt_m[1]
        z = rz * np.outer(np.ones_like(u), np.cos(v)) + rho - opt_m[-1]

        # Transform into eigencoordinate space
        xyz = np.tensordot(eigvecs, np.asarray([x, y, z]), axes=[1, 0])
        x = xyz[0, :, :]
        y = xyz[1, :, :]
        z = xyz[2, :, :]

        # Plot ellipsoid
        ax.plot_wireframe(
            x,
            y,
            z,
            rstride=4,
            cstride=4,
            color="g",
            label=r"Lorenz analytic ellipsoid",
            alpha=1.0,
            linewidth=1.5,
        )

    # Adjust plot features and save
    plt.legend(fontsize=16, loc="upper left")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_axis_off()


# Make Lissajou figures with ground truth and SINDy model
def make_lissajou(r, x_train, x_test, x_train_pred, x_test_pred, filename):
    fig = plt.figure(figsize=(8, 8))
    spec = gridspec.GridSpec(ncols=r, nrows=r, figure=fig, hspace=0.0, wspace=0.0)
    for i in range(r):
        for j in range(i, r):
            plt.subplot(spec[i, j])
            plt.plot(x_train[:, i], x_train[:, j], linewidth=1)
            plt.plot(x_train_pred[:, i], x_train_pred[:, j], "k--", linewidth=1)
            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                plt.ylabel(r"$x_" + str(i) + r"$", fontsize=18)
            if i == r - 1:
                plt.xlabel(r"$x_" + str(j) + r"$", fontsize=18)
        for j in range(i):
            plt.subplot(spec[i, j])
            plt.plot(x_test[:, j], x_test[:, i], "r", linewidth=1)
            plt.plot(x_test_pred[:, j], x_test_pred[:, i], "k--", linewidth=1)
            ax = plt.gca()
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                plt.ylabel(r"$x_" + str(i) + r"$", fontsize=18)
            if i == r - 1:
                plt.xlabel(r"$x_" + str(j) + r"$", fontsize=18)
