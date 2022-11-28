import time
import warnings

import dysts.flows as flows
import numpy as np
from dysts.analysis import sample_initial_conditions
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

import pysindy as ps


def plot_coef_errors(
    all_sols_train,
    best_normalized_coef_errors,
    xdot_rmse_errors,
    best_threshold_values,
    scale_list,
    systems_list,
    normalize_columns=True,
):
    # Count up number of systems that can be successfully identified to 10% total coefficient error
    num_attractors = len(systems_list)
    coef_summary = np.zeros(num_attractors)
    for i, attractor_name in enumerate(all_sols_train):
        coef_summary[i] = best_normalized_coef_errors[attractor_name][0] < 0.1

    print(
        "# of dynamical systems that have < 10% coefficient error in the fit, ",
        "when , error * 100, % Gaussian noise is added to every trajectory point ",
        int(np.sum(coef_summary)),
        " / ",
        len(systems_list),
    )

    plt.figure(figsize=(20, 2))
    for i, attractor_name in enumerate(all_sols_train):
        plt.scatter(
            i,
            best_normalized_coef_errors[attractor_name][0],
            c="r",
            label="Avg. normalized coef errors",
        )
        plt.scatter(
            i,
            abs(np.array(xdot_rmse_errors[attractor_name])),
            c="g",
            label="Avg. RMSE errors",
        )
        plt.scatter(
            i, best_threshold_values[attractor_name], c="b", label="Avg. best threshold"
        )
    plt.grid(True)
    plt.yscale("log")
    plt.plot(
        np.linspace(-0.5, num_attractors + 1, num_attractors),
        0.1 * np.ones(num_attractors),
        "k--",
        label="10% error",
    )
    plt.legend(
        ["10% normalized error", "$E_{coef}$", "$E_{RMSE}$", "Optimal threshold"],
        framealpha=1.0,
        ncol=4,
        fontsize=13,
    )
    ax = plt.gca()
    plt.xticks(np.arange(num_attractors), rotation="vertical", fontsize=16)
    plt.xlim(-0.5, num_attractors + 1)
    systems_list_cleaned = []
    for i, system in enumerate(systems_list):
        if system == "GuckenheimerHolmes":
            systems_list_cleaned.append("GuckenHolmes")
        elif system == "NuclearQuadrupole":
            systems_list_cleaned.append("NuclearQuad")
        elif system == "RabinovichFabrikant":
            systems_list_cleaned.append("RabFabrikant")
        elif system == "KawczynskiStrizhak":
            systems_list_cleaned.append("KawcStrizhak")
        elif system == "RikitakeDynamo":
            systems_list_cleaned.append("RikiDynamo")
        elif system == "ShimizuMorioka":
            systems_list_cleaned.append("ShMorioka")
        elif system == "HindmarshRose":
            systems_list_cleaned.append("Hindmarsh")
        elif system == "RayleighBenard":
            systems_list_cleaned.append("RayBenard")
        else:
            systems_list_cleaned.append(system)
    ax.set_xticklabels(np.array(systems_list_cleaned))
    if normalize_columns:
        plt.ylim(1e-4, 1e4)
    else:
        plt.ylim(1e-4, 1e1)
    plt.yticks(fontsize=20)
    plt.savefig("model_summary_without_added_noise_Algo3.pdf")

    # Repeat the plot, but reorder things by the amount of scale separation
    scale_sort = np.argsort(scale_list)
    scale_list_sorted = np.sort(scale_list)
    systems_list_sorted = np.array(systems_list)[scale_sort]
    cerrs = []
    rmse_errs = []
    plt.figure(figsize=(20, 2))
    for i, attractor_name in enumerate(systems_list_sorted):
        plt.scatter(
            i,
            best_normalized_coef_errors[attractor_name][0],
            c="r",
            label="Avg. normalized coef errors",
        )
        plt.scatter(
            i,
            abs(np.array(xdot_rmse_errors[attractor_name])),
            c="g",
            label="Avg. RMSE errors",
        )
        rmse_errs.append(abs(np.array(xdot_rmse_errors[attractor_name]))[0])
        cerrs.append(best_normalized_coef_errors[attractor_name][0])

    print(scale_list_sorted, rmse_errs)
    plt.grid(True)
    plt.yscale("log")
    plt.plot(
        np.linspace(-0.5, num_attractors + 1, num_attractors),
        0.1 * np.ones(num_attractors),
        "k--",
        label="10% error",
    )
    plt.legend(
        ["10% normalized error", "$E_{coef}$"],
        framealpha=1.0,
        ncol=4,
        fontsize=13,
        loc="upper left",
    )
    ax = plt.gca()
    plt.xticks(np.arange(num_attractors), rotation="vertical", fontsize=16)
    plt.xlim(-0.5, num_attractors + 1)
    ax.set_xticklabels(np.array(systems_list_cleaned)[scale_sort])
    # plt.ylim(1e-4, 1e1)
    plt.yticks(fontsize=20)
    plt.savefig("model_summary_scaleSeparation_without_added_noise_Algo3.pdf")

    from scipy.stats import linregress

    slope, intercept, r_value, p_value, std_err = linregress(
        scale_list_sorted, np.log(rmse_errs)
    )
    print(slope, intercept, r_value, p_value, std_err)
    print("R^2 value = ", r_value**2)

    plt.figure(figsize=(20, 2))
    for i, attractor_name in enumerate(systems_list_sorted):
        plt.scatter(
            scale_list_sorted[i],
            best_normalized_coef_errors[attractor_name][0],
            c="r",
            label="Avg. normalized coef errors",
        )
        plt.scatter(
            scale_list_sorted[i],
            abs(np.array(xdot_rmse_errors[attractor_name])),
            c="g",
            label="Avg. RMSE errors",
        )
    plt.plot(scale_list_sorted, np.exp(slope * scale_list_sorted + intercept), "k")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True)
    # plt.yscale('log')
    # plt.plot(np.linspace(-0.5, num_attractors + 1, num_attractors), 0.1 * np.ones(num_attractors), 'k--', label='10% error')
    plt.legend(
        ["Best linear feat", "$E_{coef}$"],
        loc="lower right",
        framealpha=1.0,
        ncol=4,
        fontsize=13,
    )
    ax = plt.gca()
    # plt.xticks(np.arange(num_attractors), rotation='vertical', fontsize=16)
    # plt.xlim(-0.5, num_attractors + 1)
    # ax.set_xticklabels(np.array(systems_list_cleaned)[scale_sort])
    # plt.ylim(1e-4, 1e1)
    plt.yticks(fontsize=20)
    plt.savefig("model_summary_scaleSeparation_without_added_noise.pdf")
    plt.show()


def plot_individual_coef_errors(
    all_sols_train,
    predicted_coefficients,
    true_coefficients,
    dimension_list,
    systems_list,
    models,
):
    poly_library = ps.PolynomialLibrary(degree=4)
    colors = ["r", "b", "g", "m"]
    labels = ["xdot", "ydot", "zdot", "wdot"]

    for i, system in enumerate(systems_list):
        x_train = all_sols_train[system]
        plt.figure(figsize=(20, 2))
        if dimension_list[i] == 3:
            feature_names = poly_library.fit(x_train).get_feature_names(["x", "y", "z"])
        else:
            feature_names = poly_library.fit(x_train).get_feature_names(
                ["x", "y", "z", "w"]
            )
        for k in range(dimension_list[i]):
            plt.grid(True)
            plt.scatter(
                feature_names,
                np.mean(np.array(predicted_coefficients[system])[:, k, :], 0),
                color=colors[k],
                label=labels[k],
                s=100,
            )
            plt.scatter(
                feature_names,
                np.array(true_coefficients[i][k, :]),
                color="k",
                label="True " + labels[k],
                s=50,
            )
        if dimension_list[i] == 3:
            plt.legend(loc="upper right", framealpha=1.0, ncol=6)
        else:
            plt.legend(loc="upper right", framealpha=1.0, ncol=8)
        plt.title(system)
        # plt.yscale('symlog', linthreshy=1e-3)
        plt.legend(loc="upper right", framealpha=1.0, ncol=6)
        print(system)
        models[i].print()


def load_data(
    systems_list,
    all_properties,
    n=200,
    pts_per_period=20,
    random_bump=False,
    include_transients=False,
    n_trajectories=1,
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
                print(ic_train)
                ic_train += (np.random.rand(len(ic_train)) - 0.5) * abs(ic_train) / 50
                ic_test += (np.random.rand(len(ic_test)) - 0.5) * abs(ic_test) / 50
                print(ic_train)

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


def make_test_trajectories(
    systems_list,
    all_properties,
    n=200,
    pts_per_period=20,
    random_bump=False,
    include_transients=False,
    approximate_center=0.0,  # approximate center of the attractor
    n_trajectories=20,
):
    all_sols_test = dict()
    all_t_test = dict()

    for i, equation_name in enumerate(systems_list):

        dimension = all_properties[equation_name]["embedding_dimension"]
        all_sols_test[equation_name] = np.zeros((n, n_trajectories, dimension))
        all_t_test[equation_name] = np.zeros((n, n_trajectories))

        eq = getattr(flows, equation_name)()
        print(i, eq)

        ic_test = sample_initial_conditions(
            eq, n_trajectories, traj_length=1000, pts_per_period=30
        )

        # Sample at roughly the smallest time scale!!
        if include_transients:
            pts_per_period = int(1 / (all_properties[equation_name]["dt"] * 10))
            n = pts_per_period * 10  # sample 10 periods at the largest time scale

        # Kick it off the attractor by random bump with, at most, 25% of the norm of the IC
        for j in range(n_trajectories):
            if random_bump:
                ic_test[j, :] += (
                    (np.random.rand(len(ic_test[j, :])) - 0.5) * abs(ic_test[j, :]) / 10
                )
            eq.ic = ic_test[j, :]
            t_sol, sol = eq.make_trajectory(
                n,
                pts_per_period=pts_per_period,
                resample=True,
                return_times=True,
                standardize=False,
            )
            all_sols_test[equation_name][:, j, :] = sol
            all_t_test[equation_name][:, j] = t_sol
    return all_sols_test, all_t_test


def normalized_RMSE(x_dot_true, x_dot_pred):
    return np.linalg.norm(x_dot_true - x_dot_pred, ord=2) / np.linalg.norm(
        x_dot_true, ord=2
    )


def AIC_c(x_dot_true, x_dot_pred, coef_pred):
    N = x_dot_true.shape[0]
    k = np.count_nonzero(coef_pred)
    if (N - k - 1) == 0:
        return N * np.log(np.linalg.norm(x_dot_true - x_dot_pred, ord=2) ** 2) + 2 * k
    else:
        return (
            N * np.log(np.linalg.norm(x_dot_true - x_dot_pred, ord=2) ** 2)
            + 2 * k
            + 2 * k * (k + 1) / (N - k - 1)
        )


def total_coefficient_error_normalized(xi_true, xi_pred):
    return np.linalg.norm(xi_true - xi_pred, ord=2) / np.linalg.norm(xi_true, ord=2)


def total_mean_coefficient_error_normalized(xi_true, xi_pred):
    return np.mean(abs(xi_true - xi_pred) / abs(xi_true))


def coefficient_errors(xi_true, xi_pred):
    errors = np.zeros(xi_true.shape)
    for i in range(xi_true.shape[0]):
        for j in range(xi_true.shape[1]):
            if np.isclose(xi_true[i, j], 0.0):
                errors[i, j] = abs(xi_true[i, j] - xi_pred[i, j])
            else:
                errors[i, j] = abs(xi_true[i, j] - xi_pred[i, j]) / xi_true[i, j]
    return errors


def total_coefficient_error(xi_true, xi_pred):
    errors = np.zeros(xi_true.shape)
    for i in range(xi_true.shape[0]):
        for j in range(xi_true.shape[1]):
            errors[i, j] = xi_true[i, j] - xi_pred[i, j]
    return np.linalg.norm(errors, ord=2)


def success_rate(xi_true, xi_pred):
    print("to do")


def Pareto_scan_ensembling(
    systems_list,
    dimension_list,
    true_coefficients,
    all_sols_train,
    all_t_train,
    all_sols_test,
    all_t_test,
    normalize_columns=False,
    error_level=0,  # as a percent of the RMSE of the training data
    tol_iter=300,
    n_models=10,
    n_subset=40,
    replace=False,
    weak_form=False,
    algorithm="STLSQ",
):
    """
    Stitch all the training trajectories together and then subsample
    them to make n_models SINDy models. Pareto optimal is determined
    by computing the minimum average RMSE error in x_dot.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    # define data structure for records
    xdot_rmse_errors = {}
    xdot_coef_errors = {}
    predicted_coefficients = {}
    best_threshold_values = {}
    best_normalized_coef_errors = {}
    AIC = {}

    # initialize structures
    num_attractors = len(systems_list)
    for system in systems_list:
        xdot_rmse_errors[system] = list()
        xdot_coef_errors[system] = list()
        predicted_coefficients[system] = list()
        best_threshold_values[system] = list()
        best_normalized_coef_errors[system] = list()
        AIC[system] = list()

    # iterate over all systems and noise levels
    if normalize_columns:
        dtol = 1e-2  # threshold values will be higher if feature library is normalized
    else:
        dtol = 1e-6

    max_iter = 1000
    if not weak_form:
        poly_library = ps.PolynomialLibrary(degree=4)
    else:
        library_functions = [
            lambda x: x,
            lambda x: x * x,
            lambda x, y: x * y,
            lambda x: x * x * x,
            lambda x, y: x * x * y,
            lambda x, y: x * y * y,
            lambda x, y, z: x * y * z,
            lambda x: x * x * x * x,
            lambda x, y: x * x * x * y,
            lambda x, y: x * y * y * y,
            lambda x, y: x * x * y * y,
            lambda x, y, z: x * y * y * z,
            lambda x, y, z: x * y * z * z,
            lambda x, y, z: x * x * y * z,
            lambda x, y, z, w: x * y * z * w,
        ]
        library_function_names = [
            lambda x: x,
            lambda x: x + "^2",
            lambda x, y: x + " " + y,
            lambda x: x + "^3",
            lambda x, y: x + "^2 " + y,
            lambda x, y: x + " " + y + "^2",
            lambda x, y, z: x + " " + y + " " + z,
            lambda x: x + "^4",
            lambda x, y: x + "^3 " + y,
            lambda x, y: x + " " + y + "^3",
            lambda x, y: x + "^2 " + y + "^2",
            lambda x, y, z: x + " " + y + "^2 " + z,
            lambda x, y, z: x + " " + y + " " + z + "^2",
            lambda x, y, z: x + "^2 " + y + " " + z,
            lambda x, y, z, w: x + " " + y + " " + z + " " + w,
        ]
    models = []
    x_dot_tests = []
    x_dot_test_preds = []
    condition_numbers = np.zeros(num_attractors)
    t_start = time.time()

    for i, attractor_name in enumerate(systems_list):
        print(i, " / ", num_attractors, ", System = ", attractor_name)

        x_train = np.copy(all_sols_train[attractor_name])
        x_train_list = []
        t_train_list = []
        x_test_list = []
        t_test_list = []
        for j in range(len(x_train)):
            rmse = mean_squared_error(
                x_train[j], np.zeros(x_train[j].shape), squared=False
            )
            x_train_noisy = x_train[j] + np.random.normal(
                0, rmse / 100.0 * error_level, x_train[j].shape
            )
            x_train_list.append(x_train_noisy)
            x_test_list.append(all_sols_test[attractor_name][j])
            t_train_list.append(all_t_train[attractor_name][j])
            t_test_list.append(all_t_test[attractor_name][j])
        # x_test = all_sols_test[attractor_name]
        # t_test = all_t_test[attractor_name]
        if dimension_list[i] == 3:
            input_names = ["x", "y", "z"]
        else:
            input_names = ["x", "y", "z", "w"]

        # feature_names = poly_library.fit(x_train).get_feature_names(input_names)

        # Critical step for the weak form -- change the grid to match the system!
        if weak_form:
            poly_library = ps.WeakPDELibrary(
                library_functions=library_functions,
                function_names=library_function_names,
                spatiotemporal_grid=all_t_train[attractor_name][0],
                is_uniform=True,
                include_bias=True,
                K=200,
            )

        if algorithm == "MIOSR":
            # Sweep a Pareto front
            (
                coef_best,
                err_best,
                err_rmse,
                coef_history,
                err_history,
                threshold_best,
                model,
                condition_numbers[i],
            ) = rudy_algorithm_miosr(
                x_train_list,
                x_test_list,
                t_train_list,
                ode_lib=poly_library,
                alpha=1e-5,
                normalize_columns=normalize_columns,
                t_test=t_test_list,
                input_names=input_names,
                ensemble=True,
                n_models=n_models,
                n_subset=n_subset,
                replace=replace,
            )
        elif algorithm == "SR3":
            # Sweep a Pareto front
            (
                coef_best,
                err_best,
                err_rmse,
                coef_history,
                err_history,
                threshold_best,
                model,
                condition_numbers[i],
            ) = rudy_algorithm_sr3(
                x_train_list,
                x_test_list,
                t_train_list,
                ode_lib=poly_library,
                dtol=dtol,
                optimizer_max_iter=max_iter,
                tol_iter=tol_iter,
                change_factor=1.1,
                normalize_columns=normalize_columns,
                t_test=t_test_list,
                input_names=input_names,
                ensemble=True,
                n_models=n_models,
                n_subset=n_subset,
                replace=replace,
            )
        elif algorithm == "Lasso":
            # Sweep a Pareto front
            (
                coef_best,
                err_best,
                err_rmse,
                coef_history,
                err_history,
                threshold_best,
                model,
                condition_numbers[i],
            ) = rudy_algorithm_lasso(
                x_train_list,
                x_test_list,
                t_train_list,
                ode_lib=poly_library,
                dtol=dtol,
                optimizer_max_iter=max_iter,
                tol_iter=tol_iter,
                change_factor=1.1,
                normalize_columns=normalize_columns,
                t_test=t_test_list,
                input_names=input_names,
                ensemble=True,
                n_models=n_models,
                n_subset=n_subset,
                replace=replace,
            )
        else:
            # Sweep a Pareto front
            (
                coef_best,
                err_best,
                err_rmse,
                coef_history,
                err_history,
                threshold_best,
                model,
                condition_numbers[i],
            ) = rudy_algorithm2(
                x_train_list,
                x_test_list,
                t_train_list,
                ode_lib=poly_library,
                dtol=dtol,
                optimizer_max_iter=max_iter,
                tol_iter=tol_iter,
                change_factor=1.1,
                alpha=1e-5,
                normalize_columns=normalize_columns,
                t_test=t_test_list,
                input_names=input_names,
                ensemble=True,
                n_models=n_models,
                n_subset=n_subset,
                replace=replace,
            )

        x_dot_test = model.differentiate(
            x_test_list, t=t_test_list, multiple_trajectories=True
        )
        x_dot_test_pred = model.predict(x_test_list, multiple_trajectories=True)
        models.append(model)
        x_dot_tests.append(x_dot_test)
        x_dot_test_preds.append(x_dot_test_pred)
        best_threshold_values[attractor_name].append(threshold_best)
        xdot_rmse_errors[attractor_name].append(err_rmse)
        AIC[attractor_name].append(err_best)
        xdot_coef_errors[attractor_name].append(
            coefficient_errors(true_coefficients[i], np.mean(coef_best, axis=0))
        )
        predicted_coefficients[attractor_name].append(coef_best)
        best_normalized_coef_errors[attractor_name].append(
            total_coefficient_error_normalized(
                true_coefficients[i], np.mean(coef_best, axis=0)
            )
        )
    t_end = time.time()
    print("Total time = ", t_end - t_start)
    return (
        xdot_rmse_errors,
        xdot_coef_errors,
        AIC,
        x_dot_tests,
        x_dot_test_preds,
        predicted_coefficients,
        best_threshold_values,
        best_normalized_coef_errors,
        models,
        condition_numbers,
    )


def rudy_algorithm2(
    x_train,
    x_test,
    t_train,
    t_test,
    ode_lib,
    dtol,
    alpha=1e-5,
    tol_iter=25,
    change_factor=2,
    normalize_columns=True,
    optimizer_max_iter=20,
    input_names=["x", "y", "z"],
    ensemble=False,
    n_models=10,
    n_subset=40,
    replace=False,
):
    """
    # Algorithm to scan over threshold values during Ridge Regression, and select
    # highest performing model on the test set
    """

    n_trajectories = np.array(x_test).shape[0]
    n_state = np.array(x_test).shape[2]
    if isinstance(ode_lib, ps.WeakPDELibrary):
        n_time = ode_lib.K
    else:
        n_time = np.array(x_test).shape[1]

    # Do an initial least-squares fit to get an initial guess of the coefficients
    # start with initial guess that all coefs are zero
    optimizer = ps.EnsembleOptimizer(
        opt=ps.STLSQ(
            threshold=0,
            alpha=alpha,
            max_iter=optimizer_max_iter,
            normalize_columns=normalize_columns,
            ridge_kw={"tol": 1e-10},
        ),
        bagging=ensemble,
        n_models=n_models,
        n_subset=n_subset,
        replace=replace,
        # ensemble_aggregator=np.mean
    )

    # Compute initial model
    model = ps.SINDy(
        feature_library=ode_lib, optimizer=optimizer, feature_names=input_names
    )
    model.fit(
        x_train,
        t=t_train,
        quiet=True,
        multiple_trajectories=True,
    )
    condition_number = np.linalg.cond(optimizer.Theta_)

    # Set the L0 penalty based on the condition number of Theta
    coef_best = np.array(optimizer.coef_list)
    optimizer.coef_ = np.mean(coef_best, axis=0)
    model_best = model

    # For each model, compute x_dot_test and compute the RMSE error
    error_new = np.zeros(n_models)
    error_best = np.zeros(n_models)
    error_rmse_new = np.zeros(n_models)
    error_rmse = np.zeros(n_models)

    for i in range(n_models):
        optimizer.coef_ = coef_best[i, :, :]
        x_dot_test = model.differentiate(x_test, t=t_test, multiple_trajectories=True)
        x_dot_test_pred = model.predict(x_test, multiple_trajectories=True)
        dx_test = np.array(x_dot_test).reshape(n_trajectories * n_time, n_state)
        dx_pred = np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state)
        error_best[i] = AIC_c(dx_test, dx_pred, coef_best[i, :, :])
        error_rmse[i] = normalized_RMSE(
            dx_test,
            dx_pred,
        )

    coef_history_ = np.zeros(
        (n_models, coef_best.shape[1], coef_best.shape[2], 1 + tol_iter)
    )
    error_history_ = np.zeros((n_models, 1 + tol_iter))
    coef_history_[:, :, :, 0] = coef_best
    error_history_[:, 0] = error_best
    tol = dtol
    threshold_best = tol

    # Loop over threshold values, note needs some coding
    # if not using STLSQ optimizer
    for i in range(tol_iter):
        optimizer = ps.EnsembleOptimizer(
            opt=ps.STLSQ(
                threshold=tol,
                alpha=alpha,
                max_iter=optimizer_max_iter,
                normalize_columns=normalize_columns,
                ridge_kw={"tol": 1e-10},
            ),
            bagging=ensemble,
            n_models=n_models,
            n_subset=n_subset,
            replace=replace,
            # ensemble_aggregator=np.mean
        )
        model = ps.SINDy(
            feature_library=ode_lib, optimizer=optimizer, feature_names=input_names
        )
        model.fit(
            x_train,
            t=t_train,
            quiet=True,
            multiple_trajectories=True,
        )

        # For each model, compute x_dot_test and compute the RMSE error
        coef_new = np.array(optimizer.coef_list)
        if np.isclose(np.sum(coef_new), 0.0):
            break

        for j in range(n_models):
            optimizer.coef_ = np.copy(coef_new[j, :, :])
            model.optimizer.coef_ = np.copy(coef_new[j, :, :])
            x_dot_test = model.differentiate(
                x_test, t=t_test, multiple_trajectories=True
            )
            # x_dot_test_pred = model.predict(x_test, multiple_trajectories=True)
            x_dot_test_pred = model.predict(x_test, multiple_trajectories=True)
            # x_dot_test_pred = optimizer.Theta_ @ coef_new[j, :, :].T
            error_new[j] = AIC_c(
                np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
                np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
                coef_new[j, :, :],
            )
            error_rmse_new[j] = normalized_RMSE(
                np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
                np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
            )
            # print(j, error_new[j], coef_new[j, :, :])
        # print(i, error_new)

        coef_history_[:, :, :, i + 1] = coef_new
        error_history_[:, i + 1] = error_new

        # If error improves, set the new best coefficients
        # Note < not <= since if all coefficients are zero,
        # this would still keep increasing the threshold!
        if np.mean(error_new) < np.mean(error_best):
            error_best = np.copy(error_new)
            error_rmse = np.copy(error_rmse_new)
            coef_best = np.copy(coef_new)
            threshold_best = tol
            model.optimizer.coef_ = np.median(coef_new, axis=0)
            # model.optimizer.coef_ = model.optimizer.coef_[abs(model.optimizer.coef_) > 1e-2]
            model_best = model
        dtol = dtol * change_factor
        tol += dtol

    return (
        coef_best,
        error_best,
        error_rmse,
        coef_history_,
        error_history_,
        threshold_best,
        model_best,
        condition_number,
    )


def rudy_algorithm_lasso(
    x_train,
    x_test,
    t_train,
    t_test,
    ode_lib,
    dtol,
    alpha=1e-5,
    tol_iter=25,
    change_factor=2,
    normalize_columns=True,
    optimizer_max_iter=20,
    input_names=["x", "y", "z"],
    ensemble=False,
    n_models=10,
    n_subset=40,
    replace=False,
):
    """
    # Algorithm to scan over threshold values during Ridge Regression, and select
    # highest performing model on the test set
    """

    n_trajectories = np.array(x_test).shape[0]
    n_state = np.array(x_test).shape[2]
    if isinstance(ode_lib, ps.WeakPDELibrary):
        n_time = ode_lib.K
    else:
        n_time = np.array(x_test).shape[1]

    # Do an initial least-squares fit to get an initial guess of the coefficients
    # start with initial guess that all coefs are zero
    optimizer = ps.EnsembleOptimizer(
        opt=Lasso(
            alpha=0, max_iter=optimizer_max_iter, fit_intercept=False
        ),  # currently ignoring normalize_columns parameter
        bagging=ensemble,
        n_models=n_models,
        n_subset=n_subset,
        replace=replace,
        # ensemble_aggregator=np.mean
    )

    # Compute initial model
    model = ps.SINDy(
        feature_library=ode_lib, optimizer=optimizer, feature_names=input_names
    )
    model.fit(
        x_train,
        t=t_train,
        quiet=True,
        multiple_trajectories=True,
    )
    condition_number = np.linalg.cond(optimizer.Theta_)

    # Set the L0 penalty based on the condition number of Theta
    coef_best = np.array(optimizer.coef_list)
    optimizer.coef_ = np.mean(coef_best, axis=0)
    model_best = model

    # For each model, compute x_dot_test and compute the RMSE error
    error_new = np.zeros(n_models)
    error_best = np.zeros(n_models)
    error_rmse_new = np.zeros(n_models)
    error_rmse = np.zeros(n_models)

    for i in range(n_models):
        optimizer.coef_ = coef_best[i, :, :]
        x_dot_test = model.differentiate(x_test, t=t_test, multiple_trajectories=True)
        x_dot_test_pred = model.predict(x_test, multiple_trajectories=True)
        dx_test = np.array(x_dot_test).reshape(n_trajectories * n_time, n_state)
        dx_pred = np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state)
        error_best[i] = AIC_c(dx_test, dx_pred, coef_best[i, :, :])
        error_rmse[i] = normalized_RMSE(
            dx_test,
            dx_pred,
        )

    coef_history_ = np.zeros(
        (n_models, coef_best.shape[1], coef_best.shape[2], 1 + tol_iter)
    )
    error_history_ = np.zeros((n_models, 1 + tol_iter))
    coef_history_[:, :, :, 0] = coef_best
    error_history_[:, 0] = error_best
    tol = dtol
    threshold_best = tol

    # Loop over threshold values, note needs some coding
    # if not using STLSQ optimizer
    for i in range(tol_iter):
        optimizer = ps.EnsembleOptimizer(
            opt=Lasso(alpha=tol, max_iter=optimizer_max_iter, fit_intercept=False),
            bagging=ensemble,
            n_models=n_models,
            n_subset=n_subset,
            replace=replace,
            # ensemble_aggregator=np.mean
        )
        model = ps.SINDy(
            feature_library=ode_lib, optimizer=optimizer, feature_names=input_names
        )
        model.fit(
            x_train,
            t=t_train,
            quiet=True,
            multiple_trajectories=True,
        )

        # For each model, compute x_dot_test and compute the RMSE error
        coef_new = np.array(optimizer.coef_list)
        if np.isclose(np.sum(coef_new), 0.0):
            break

        for j in range(n_models):
            optimizer.coef_ = np.copy(coef_new[j, :, :])
            model.optimizer.coef_ = np.copy(coef_new[j, :, :])
            x_dot_test = model.differentiate(
                x_test, t=t_test, multiple_trajectories=True
            )
            # x_dot_test_pred = model.predict(x_test, multiple_trajectories=True)
            x_dot_test_pred = model.predict(x_test, multiple_trajectories=True)
            # x_dot_test_pred = optimizer.Theta_ @ coef_new[j, :, :].T
            error_new[j] = AIC_c(
                np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
                np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
                coef_new[j, :, :],
            )
            error_rmse_new[j] = normalized_RMSE(
                np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
                np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
            )
            # print(j, error_new[j], coef_new[j, :, :])
        # print(i, error_new)

        coef_history_[:, :, :, i + 1] = coef_new
        error_history_[:, i + 1] = error_new

        # If error improves, set the new best coefficients
        # Note < not <= since if all coefficients are zero,
        # this would still keep increasing the threshold!
        if np.mean(error_new) < np.mean(error_best):
            error_best = np.copy(error_new)
            error_rmse = np.copy(error_rmse_new)
            coef_best = np.copy(coef_new)
            threshold_best = tol
            model.optimizer.coef_ = np.median(coef_new, axis=0)
            # model.optimizer.coef_ = model.optimizer.coef_[abs(model.optimizer.coef_) > 1e-2]
            model_best = model
        dtol = dtol * change_factor
        tol += dtol

    return (
        coef_best,
        error_best,
        error_rmse,
        coef_history_,
        error_history_,
        threshold_best,
        model_best,
        condition_number,
    )


def rudy_algorithm_sr3(
    x_train,
    x_test,
    t_train,
    t_test,
    ode_lib,
    dtol,
    tol_iter=25,
    change_factor=2,
    normalize_columns=True,
    optimizer_max_iter=20,
    input_names=["x", "y", "z"],
    ensemble=False,
    n_models=10,
    n_subset=40,
    replace=False,
):
    """
    # Algorithm to scan over threshold values during Ridge Regression, and select
    # highest performing model on the test set
    """

    n_trajectories = np.array(x_test).shape[0]
    n_state = np.array(x_test).shape[2]
    if isinstance(ode_lib, ps.WeakPDELibrary):
        n_time = ode_lib.K
    else:
        n_time = np.array(x_test).shape[1]

    # Do an initial least-squares fit to get an initial guess of the coefficients
    # start with initial guess that all coefs are zero
    optimizer = ps.EnsembleOptimizer(
        opt=ps.SR3(
            threshold=0,
            max_iter=optimizer_max_iter,
            normalize_columns=normalize_columns,
            nu=0.1,
        ),
        bagging=ensemble,
        n_models=n_models,
        n_subset=n_subset,
        replace=replace,
        # ensemble_aggregator=np.mean
    )

    # Compute initial model
    model = ps.SINDy(
        feature_library=ode_lib, optimizer=optimizer, feature_names=input_names
    )
    model.fit(
        x_train,
        t=t_train,
        quiet=True,
        multiple_trajectories=True,
    )
    condition_number = np.linalg.cond(optimizer.Theta_)

    # Set the L0 penalty based on the condition number of Theta
    coef_best = np.array(optimizer.coef_list)
    optimizer.coef_ = np.mean(coef_best, axis=0)
    model_best = model

    # For each model, compute x_dot_test and compute the RMSE error
    error_new = np.zeros(n_models)
    error_best = np.zeros(n_models)
    error_rmse_new = np.zeros(n_models)
    error_rmse = np.zeros(n_models)

    for i in range(n_models):
        optimizer.coef_ = coef_best[i, :, :]
        x_dot_test = model.differentiate(x_test, t=t_test, multiple_trajectories=True)
        x_dot_test_pred = model.predict(x_test, multiple_trajectories=True)
        error_best[i] = AIC_c(
            np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
            np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
            coef_best[i, :, :],
        )
        error_rmse[i] = normalized_RMSE(
            np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
            np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
        )

    coef_history_ = np.zeros(
        (n_models, coef_best.shape[1], coef_best.shape[2], 1 + tol_iter)
    )
    error_history_ = np.zeros((n_models, 1 + tol_iter))
    coef_history_[:, :, :, 0] = coef_best
    error_history_[:, 0] = error_best
    tol = dtol
    threshold_best = tol

    # Loop over threshold values, note needs some coding
    # if not using STLSQ optimizer
    for i in range(tol_iter):
        optimizer = ps.EnsembleOptimizer(
            opt=ps.SR3(
                threshold=tol,
                max_iter=optimizer_max_iter,
                normalize_columns=normalize_columns,
                nu=0.1,
            ),
            bagging=ensemble,
            n_models=n_models,
            n_subset=n_subset,
            replace=replace,
            # ensemble_aggregator=np.mean
        )
        model = ps.SINDy(
            feature_library=ode_lib, optimizer=optimizer, feature_names=input_names
        )
        model.fit(
            x_train,
            t=t_train,
            quiet=True,
            multiple_trajectories=True,
        )

        # For each model, compute x_dot_test and compute the RMSE error
        coef_new = np.array(optimizer.coef_list)
        if np.isclose(np.sum(coef_new), 0.0):
            break

        for j in range(n_models):
            optimizer.coef_ = np.copy(coef_new[j, :, :])
            model.optimizer.coef_ = np.copy(coef_new[j, :, :])
            x_dot_test = model.differentiate(
                x_test, t=t_test, multiple_trajectories=True
            )
            # x_dot_test_pred = model.predict(x_test, multiple_trajectories=True)
            x_dot_test_pred = model.predict(x_test, multiple_trajectories=True)
            # x_dot_test_pred = optimizer.Theta_ @ coef_new[j, :, :].T
            error_new[j] = AIC_c(
                np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
                np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
                coef_new[j, :, :],
            )
            error_rmse_new[j] = normalized_RMSE(
                np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
                np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
            )
            # print(j, error_new[j], coef_new[j, :, :])
        # print(i, error_new)

        coef_history_[:, :, :, i + 1] = coef_new
        error_history_[:, i + 1] = error_new

        # If error improves, set the new best coefficients
        # Note < not <= since if all coefficients are zero,
        # this would still keep increasing the threshold!
        if np.mean(error_new) < np.mean(error_best):
            error_best = np.copy(error_new)
            error_rmse = np.copy(error_rmse_new)
            coef_best = np.copy(coef_new)
            threshold_best = tol
            model.optimizer.coef_ = np.median(coef_new, axis=0)
            # model.optimizer.coef_ = model.optimizer.coef_[abs(model.optimizer.coef_) > 1e-2]
            model_best = model
        dtol = dtol * change_factor
        tol += dtol

    return (
        coef_best,
        error_best,
        error_rmse,
        coef_history_,
        error_history_,
        threshold_best,
        model_best,
        condition_number,
    )


def rudy_algorithm_miosr(
    x_train,
    x_test,
    t_train,
    t_test,
    ode_lib,
    alpha=1e-5,
    normalize_columns=True,
    input_names=["x", "y", "z"],
    ensemble=False,
    n_models=10,
    n_subset=40,
    replace=False,
):
    """
    # Algorithm to scan over threshold values during Ridge Regression, and select
    # highest performing model on the test set
    """

    n_trajectories = np.array(x_test).shape[0]
    n_state = np.array(x_test).shape[2]
    if isinstance(ode_lib, ps.WeakPDELibrary):
        n_time = ode_lib.K
    else:
        n_time = np.array(x_test).shape[1]

    # Do an initial least-squares fit to get an initial guess of the coefficients
    # start with initial guess that all coefs are zero
    optimizer = ps.EnsembleOptimizer(
        opt=ps.MIOSR(
            target_sparsity=1,
            alpha=alpha,
            normalize_columns=normalize_columns,
            regression_timeout=100,
        ),
        bagging=ensemble,
        n_models=n_models,
        n_subset=n_subset,
        replace=replace,
        # ensemble_aggregator=np.mean
    )

    # Compute initial model
    model = ps.SINDy(
        feature_library=ode_lib, optimizer=optimizer, feature_names=input_names
    )
    model.fit(
        x_train,
        t=t_train,
        quiet=True,
        multiple_trajectories=True,
    )
    condition_number = np.linalg.cond(optimizer.Theta_)
    print(np.shape(optimizer.Theta_))
    tol_iter = np.shape(optimizer.Theta_)[1] - 1

    # Set the L0 penalty based on the condition number of Theta
    coef_best = np.array(optimizer.coef_list)
    optimizer.coef_ = np.mean(coef_best, axis=0)
    model_best = model

    # For each model, compute x_dot_test and compute the RMSE error
    error_new = np.zeros(n_models)
    error_best = np.zeros(n_models)
    error_rmse_new = np.zeros(n_models)
    error_rmse = np.zeros(n_models)

    for i in range(n_models):
        optimizer.coef_ = coef_best[i, :, :]
        x_dot_test = model.differentiate(x_test, t=t_test, multiple_trajectories=True)
        x_dot_test_pred = model.predict(x_test, multiple_trajectories=True)
        error_rmse[i] = normalized_RMSE(
            np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
            np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
        )
        error_best[i] = AIC_c(
            np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
            np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
            coef_best[i, :, :],
        )

    coef_history_ = np.zeros(
        (n_models, coef_best.shape[1], coef_best.shape[2], 1 + tol_iter)
    )
    error_history_ = np.zeros((n_models, 1 + tol_iter))
    coef_history_[:, :, :, 0] = coef_best
    error_history_[:, 0] = error_best
    sparsity_best = 0

    # Loop over threshold values, note needs some coding
    # if not using STLSQ optimizer
    for i in range(tol_iter):
        # print(i)
        optimizer = ps.EnsembleOptimizer(
            opt=ps.MIOSR(
                target_sparsity=i + 1,
                alpha=alpha,
                normalize_columns=normalize_columns,
                regression_timeout=5,
            ),
            bagging=ensemble,
            n_models=n_models,
            n_subset=n_subset,
            replace=replace,
            # ensemble_aggregator=np.mean
        )
        model = ps.SINDy(
            feature_library=ode_lib, optimizer=optimizer, feature_names=input_names
        )
        # t_start = time.time()
        model.fit(
            x_train,
            t=t_train,
            quiet=True,
            multiple_trajectories=True,
        )
        # t_end = time.time()
        # print(t_end - t_start)

        # For each model, compute x_dot_test and compute the RMSE error
        coef_new = np.array(optimizer.coef_list)
        if np.isclose(np.sum(coef_new), 0.0):
            break

        for j in range(n_models):
            optimizer.coef_ = np.copy(coef_new[j, :, :])
            model.optimizer.coef_ = np.copy(coef_new[j, :, :])
            x_dot_test = model.differentiate(
                x_test, t=t_test, multiple_trajectories=True
            )
            # x_dot_test_pred = model.predict(x_test, multiple_trajectories=True)
            x_dot_test_pred = model.predict(x_test, multiple_trajectories=True)
            # x_dot_test_pred = optimizer.Theta_ @ coef_new[j, :, :].T
            error_new[j] = AIC_c(
                np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
                np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
                coef_new[j, :, :],
            )
            error_rmse_new[j] = normalized_RMSE(
                np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
                np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
            )
            # print(j, error_new[j], coef_new[j, :, :])
        # print(i, error_new)

        coef_history_[:, :, :, i + 1] = coef_new
        error_history_[:, i + 1] = error_new

        # If error improves, set the new best coefficients
        # Note < not <= since if all coefficients are zero,
        # this would still keep increasing the threshold!
        if np.mean(error_new) < np.mean(error_best):
            error_best = np.copy(error_new)
            error_rmse = np.copy(error_rmse_new)
            coef_best = np.copy(coef_new)
            sparsity_best = i
            model.optimizer.coef_ = np.median(coef_new, axis=0)
            # model.optimizer.coef_ = model.optimizer.coef_[abs(model.optimizer.coef_) > 1e-2]
            model_best = model

    return (
        coef_best,
        error_best,
        error_rmse,
        coef_history_,
        error_history_,
        sparsity_best,
        model_best,
        condition_number,
    )
