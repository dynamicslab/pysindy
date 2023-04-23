import warnings

import matplotlib.gridspec as gridspec
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
from sympy import limit
from sympy import Symbol

warnings.filterwarnings("ignore")

import dysts.flows as flows
from dysts.analysis import sample_initial_conditions


# define the objective function to be minimized by simulated annealing
def obj_function(m, L_obj, Q_obj, P_obj):
    mQ_full = np.tensordot(Q_obj, m, axes=([2], [0]))
    As = L_obj + P_obj @ mQ_full
    eigvals, eigvecs = np.linalg.eigh(As)
    return eigvals[-1]


# Define some setup and plotting functions
# Build the skew-symmetric nonlinearity constraints
def make_constraints(r):
    q = 0
    N = int((r**2 + 3 * r) / 2.0) + 1  # + 1 for constant term
    p = r + r * (r - 1) + int(r * (r - 1) * (r - 2) / 6.0)
    constraint_zeros = np.zeros(p)
    constraint_matrix = np.zeros((p, r * N))

    # Set coefficients adorning terms like a_i^3 to zero
    # [1, x, y, z, xy, xz, yz, x2, y2, z2, 1, ...]
    # [1 1 1 x x x y y y ...]
    for i in range(r):
        # constraint_matrix[q, r * (N - r) + i * (r + 1)] = 1.0
        constraint_matrix[q, r * (N - r) + i * (r + 1)] = 3.0
        q = q + 1

    # Set coefficients adorning terms like a_ia_j^2 to be antisymmetric
    for i in range(r):
        for j in range(i + 1, r):
            constraint_matrix[q, r * (N - r + j) + i] = 1.0
            constraint_matrix[
                q, r + r * (r + j - 1) + j + r * int(i * (2 * r - i - 3) / 2.0)
            ] = 1.0
            q = q + 1
    for i in range(r):
        for j in range(0, i):
            constraint_matrix[q, r * (N - r + j) + i] = 1.0
            constraint_matrix[
                q, r + r * (r + i - 1) + j + r * int(j * (2 * r - j - 3) / 2.0)
            ] = 1.0
            q = q + 1

    # Set coefficients adorning terms like a_ia_ja_k to be antisymmetric
    for i in range(r):
        for j in range(i + 1, r):
            for k in range(j + 1, r):
                constraint_matrix[
                    q, r + r * (r + k - 1) + i + r * int(j * (2 * r - j - 3) / 2.0)
                ] = (1 / 2.0)
                constraint_matrix[
                    q, r + r * (r + k - 1) + j + r * int(i * (2 * r - i - 3) / 2.0)
                ] = (1 / 2.0)
                constraint_matrix[
                    q, r + r * (r + j - 1) + k + r * int(i * (2 * r - i - 3) / 2.0)
                ] = (1 / 2.0)
                q = q + 1

    return constraint_zeros, constraint_matrix


# Use optimal m, and calculate eigenvalues(PW) to see if identified model is stable
def check_stability(r, Xi, mod_matrix, sindy_opt, mean_val):
    opt_m = sindy_opt.m_history_[-1]
    PC_tensor = sindy_opt.PC_
    PL_tensor_unsym = sindy_opt.PL_unsym_
    PL_tensor = sindy_opt.PL_
    PM_tensor = sindy_opt.PM_
    PQ_tensor = sindy_opt.PQ_
    mPM = np.tensordot(PM_tensor, opt_m, axes=([2], [0]))
    P_tensor = PL_tensor + mPM
    As = np.tensordot(P_tensor, Xi, axes=([3, 2], [0, 1]))
    As = mod_matrix @ As
    eigvals, eigvecs = np.linalg.eigh(As)
    print("optimal m: ", opt_m)
    print("As eigvals: ", np.sort(eigvals))
    max_eigval = np.sort(eigvals)[-1]
    C = np.tensordot(PC_tensor, Xi, axes=([2, 1], [0, 1]))
    L = np.tensordot(PL_tensor_unsym, Xi, axes=([3, 2], [0, 1]))
    Q = np.tensordot(PQ_tensor, Xi, axes=([4, 3], [0, 1]))
    d = C + np.dot(L, opt_m) + np.dot(np.tensordot(Q, opt_m, axes=([2], [0])), opt_m)
    d = mod_matrix @ d
    Rm = np.linalg.norm(d) / np.abs(max_eigval)
    Reff = Rm / mean_val
    print("Estimate of trapping region size, Rm = ", Rm)
    print("Normalized trapping region size, Reff = ", Reff)


def get_trapping_radius(max_eigval, eps_Q, r, d):
    x = Symbol("x")
    delta = max_eigval**2 - 4 * np.sqrt(r**3) * eps_Q * np.linalg.norm(d, 2) / 3
    delta_func = max_eigval**2 - 4 * np.sqrt(r**3) * x * np.linalg.norm(d, 2) / 3
    if delta < 0:
        rad_trap = 0
        rad_stab = 0
    else:
        y_trap = -(3 / (2 * np.sqrt(r**3) * x)) * (max_eigval + sp.sqrt(delta_func))
        y_stab = (3 / (2 * np.sqrt(r**3) * x)) * (-max_eigval + sp.sqrt(delta_func))
        rad_trap = limit(y_trap, x, eps_Q, dir="+")
        rad_stab = limit(y_stab, x, eps_Q, dir="+")
    return rad_trap, rad_stab


def check_stability_new(r, Xi, mod_matrix, sindy_opt, mean_val):
    opt_m = sindy_opt.m_history_[-1]
    PC_tensor = sindy_opt.PC_
    PL_tensor_unsym = sindy_opt.PL_unsym_
    PL_tensor = sindy_opt.PL_
    PM_tensor = sindy_opt.PM_
    PQ_tensor = sindy_opt.PQ_
    mPM = np.tensordot(PM_tensor, opt_m, axes=([2], [0]))
    P_tensor = PL_tensor + mPM
    As = np.tensordot(P_tensor, Xi, axes=([3, 2], [0, 1]))
    As = mod_matrix @ As
    eigvals, eigvecs = np.linalg.eigh(As)
    print("optimal m: ", opt_m)
    print("As eigvals: ", np.sort(eigvals))
    max_eigval = np.sort(eigvals)[-1]
    C = np.tensordot(PC_tensor, Xi, axes=([2, 1], [0, 1]))
    L = np.tensordot(PL_tensor_unsym, Xi, axes=([3, 2], [0, 1]))
    Q = np.tensordot(PQ_tensor, Xi, axes=([4, 3], [0, 1]))
    Q_sum = np.max(
        np.abs((Q + np.transpose(Q, [1, 2, 0]) + np.transpose(Q, [2, 0, 1])))
    )
    d = C + np.dot(L, opt_m) + np.dot(np.tensordot(Q, opt_m, axes=([2], [0])), opt_m)
    d = mod_matrix @ d
    eps_Q = np.max(np.abs(Q_sum))
    Rm, DA = get_trapping_radius(max_eigval, eps_Q, r, d)
    Reff = Rm / mean_val
    print("Estimate of trapping region size, Rm = ", Rm)
    print("Normalized trapping region size, Reff = ", Reff)
    print("Local stability size, DA = ", DA)
    return Rm, DA


# use optimal m, calculate and plot the stability radius when the third-order
# energy-preserving scheme slightly breaks
def make_DA_progress_plots(r, mod_matrix, sindy_opt):
    PC_tensor = sindy_opt.PC_
    PL_tensor_unsym = sindy_opt.PL_unsym_
    PQ_tensor = sindy_opt.PQ_
    ms = sindy_opt.m_history_
    eigs = sindy_opt.PWeigs_history_
    coef_history = sindy_opt.history_
    rhos = []
    for i in range(len(eigs)):
        if eigs[i][-1] < 0:
            # Q = np.tensordot(sindy_opt.PQ_, coef_history[i], axes=([4, 3], [1, 0]))
            # Q_sum = Q + np.transpose(Q, [1, 2, 0]) + np.transpose(Q, [2, 0, 1])
            C = np.tensordot(PC_tensor, coef_history[i], axes=([2, 1], [1, 0]))
            L = np.tensordot(PL_tensor_unsym, coef_history[i], axes=([3, 2], [1, 0]))
            Q = np.tensordot(PQ_tensor, coef_history[i], axes=([4, 3], [1, 0]))
            Q_sum = np.max(
                np.abs((Q + np.transpose(Q, [1, 2, 0]) + np.transpose(Q, [2, 0, 1])))
            )
            d = (
                C
                + np.dot(L, ms[i])
                + np.dot(np.tensordot(Q, ms[i], axes=([2], [0])), ms[i])
            )
            d = mod_matrix @ d
            eps_Q = np.max(np.abs(Q_sum))
            Rm = (3 / (2 * np.sqrt(r**2) * eps_Q)) * (
                np.sqrt(
                    eigs[i][-1] ** 2
                    - 4 * np.sqrt(r**3) * eps_Q * np.linalg.norm(d, 2) / 3
                )
                - eigs[i][-1]
            )
            if Rm > 0:
                rhos.append(Rm)
            else:
                rhos.append(0)
            # rhos.append(-eigs[i][-1] / np.max(np.abs(Q_sum)))
    print(np.linalg.norm(d, 2))
    plt.plot(rhos[1:])
    plt.grid(True)
    plt.ylabel("Stability radius")
    plt.xlabel("Algorithm iteration")
    plt.show()


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
    plt.show()


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

    plt.show()


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
