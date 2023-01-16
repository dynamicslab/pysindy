import time
import warnings

import dysts.flows as flows
import numpy as np
from dysts.analysis import sample_initial_conditions
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

import pysindy as ps


def load_data(
    systems_list,
    all_properties,
    n=200,
    pts_per_period=20,
    random_bump=False,
    include_transients=False,
    n_trajectories=20,
):
    """
    Function for generating n_trajectories of training and testing data
    for each dynamical system in the systems_list.

    Parameters
    ----------
    systems_list : list of strings, shape (num_systems)
        List of the dynamical systems.
    all_properties : dictionary of dictionaries
        Dictionary containing the parameters for each dynamical system.
    n : integer, optional (default 200)
        Number of points to generate for each testing trajectory.
    pts_per_period : integer, optional (default 20)
        Number of points sample each "period" of the dynamical system.
    random_bump : bool, optional (default False)
        Whether to start with an initial condition slightly off of the
        chaotic attractor.
    include_transients : bool, optional (default False)
        Whether to disregard the pts_per_period parameter and sample at
        high-resolution by doing so proportional to the smallest
        significant timescale defined through the 'dt'
        parameter in the all_properties variable.
    n_trajectories : integer, optional (default 20)
        The number of trajectories to make.

    Returns
    ---------
    all_sols_train : dictionary of 3D numpy arrays,
            shape (n_trajectories, num_sample_points, dimension_list[i])
        Dictionary containing all the training trajectories for each
        dynamical system, each entry has shape
        (n_trajectories, num_sample_points, dimension_list[i]).
    all_t_train : dictionary of 2D numpy arrays,
            shape (n_trajectories, num_sample_points)
        Dictionary containing all the training trajectory timebases
        for each dynamical system, each entry has shape
        (n_trajectories, num_sample_points).
    all_sols_test : dictionary of 3D numpy arrays,
            shape (n_trajectories, num_sample_points, dimension_list[i])
        Dictionary containing all the testing trajectories for each
        dynamical system, each entry has shape
        (n_trajectories, num_sample_points, dimension_list[i]).
    all_t_test : dictionary of 2D numpy arrays,
            shape (n_trajectories, num_sample_points)
        Dictionary containing all the testing trajectory timebases
        for each dynamical system, each entry has shape
        (n_trajectories, num_sample_points).
    """
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


def make_test_trajectories(
    systems_list,
    all_properties,
    n=200,
    pts_per_period=20,
    random_bump=False,
    include_transients=False,
    n_trajectories=20,
):
    """
    Function for generating n_trajectories of testing data
    for each dynamical system in the systems_list.

    Parameters
    ----------
    systems_list : list of strings, shape (num_systems)
        List of the dynamical systems.
    all_properties : dictionary of dictionaries
        Dictionary containing the parameters for each dynamical system.
    n : integer, optional (default 200)
        Number of points to generate for each testing trajectory.
    pts_per_period : integer, optional (default 20)
        Number of points sample each "period" of the dynamical system.
    random_bump : bool, optional (default False)
        Whether to start with an initial condition slightly off of the
        chaotic attractor.
    include_transients : bool, optional (default False)
        Whether to disregard the pts_per_period parameter and sample at
        high-resolution by doing so proportional to the smallest
        significant timescale defined through the 'dt'
        parameter in the all_properties variable.
    n_trajectories : integer, optional (default 20)
        The number of trajectories to make.

    Returns
    ---------
    all_sols_test : dictionary of 3D numpy arrays,
            shape (n_trajectories, num_sample_points, dimension_list[i])
        Dictionary containing all the testing trajectories for each
        dynamical system, each entry has shape
        (n_trajectories, num_sample_points, dimension_list[i]).
    all_t_test : dictionary of 2D numpy arrays,
            shape (n_trajectories, num_sample_points)
        Dictionary containing all the testing trajectory timebases
        for each dynamical system, each entry has shape
        (n_trajectories, num_sample_points).
    """
    all_sols_test = dict()
    all_t_test = dict()

    for i, equation_name in enumerate(systems_list):

        dimension = all_properties[equation_name]["embedding_dimension"]
        all_sols_test[equation_name] = np.zeros((n, n_trajectories, dimension))
        all_t_test[equation_name] = np.zeros((n, n_trajectories))

        eq = getattr(flows, equation_name)()
        # print(i, eq)

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
    """
    Compute the normalized RMSE error between the Xdot from the real data
    and the Xdot from a SINDy model. Usually done only for a set of
    testing trajectories.

    Parameters
    ----------
    x_dot_true : 2D numpy array of floats,
                 shape (num_sample_points, state_size)
        True x_dot trajectory(s).
    x_dot_pred : 2D numpy array of floats,
                 shape (num_sample_points, state_size)
        Predicted x_dot trajectory(s).

    Returns
    ---------
    errors : float
        Total normalized X_dot RMSE errors.
    """
    errors = np.linalg.norm(x_dot_true - x_dot_pred, ord=2) / np.linalg.norm(
        x_dot_true, ord=2
    )
    return errors


def AIC_c(x_dot_true, x_dot_pred, xi_pred):
    """
    Compute the finite-sample-corrected AIC criteria, as in
    Mangan & Brunton 2017.

    Parameters
    ----------
    x_dot_true : 2D numpy array of floats,
                 shape (num_sample_points, state_size)
        True x_dot trajectory(s).
    x_dot_pred : 2D numpy array of floats,
                 shape (num_sample_points, state_size)
        Predicted x_dot trajectory(s).

    xi_pred : 2D numpy array of floats, shape (state_size, n_features)
        Predicted equation coefficients.

    Returns
    ---------
    AIC : float
        Finite-sample-size-corrected Akaike Information Criteria.
    """
    N = x_dot_true.shape[0]
    k = np.count_nonzero(xi_pred)
    if (N - k - 1) == 0:
        AIC = N * np.log(np.linalg.norm(x_dot_true - x_dot_pred, ord=2) ** 2) + 2 * k
    else:
        AIC = (
            N * np.log(np.linalg.norm(x_dot_true - x_dot_pred, ord=2) ** 2)
            + 2 * k
            + 2 * k * (k + 1) / (N - k - 1)
        )
    return AIC


def total_coefficient_error_normalized(xi_true, xi_pred):
    """
    Compute the TOTAL normalized coefficient error between the true
    coefficients of the underlying equations (assuming they are known)
    and the coefficients identified by the SINDy model.

    Parameters
    ----------
    xi_true : 2D numpy array of floats, shape (state_size, n_features)
        True equation coefficients.
    xi_pred : 2D numpy array of floats, shape (state_size, n_features)
        Predicted equation coefficients.

    Returns
    ---------
    errors : float
        Total normalized coefficient error.
    """
    errors = np.linalg.norm(xi_true - xi_pred, ord=2) / np.linalg.norm(xi_true, ord=2)
    return errors


def coefficient_errors(xi_true, xi_pred):
    """
    Compute the INDIVIDUAL normalized coefficient errors between the true
    coefficients of the underlying equations (assuming they are known)
    and the coefficients identified by the SINDy model.

    Parameters
    ----------
    xi_true : 2D numpy array of floats, shape (state_size, n_features)
        True equation coefficients.
    xi_pred : 2D numpy array of floats, shape (state_size, n_features)
        Predicted equation coefficients.

    Returns
    ---------
    errors : 2D numpy array of floats, shape (state_size, n_features)
        Normalized coefficient errors for each term in the equations.
    """
    errors = np.zeros(xi_true.shape)
    for i in range(xi_true.shape[0]):
        for j in range(xi_true.shape[1]):
            if np.isclose(xi_true[i, j], 0.0):
                errors[i, j] = abs(xi_true[i, j] - xi_pred[i, j])
            else:
                errors[i, j] = abs(xi_true[i, j] - xi_pred[i, j]) / xi_true[i, j]
    return errors


def success_rate(xi_true, xi_pred):
    """
    Compute the success or recovery rate, i.e. 0 or 1 is returned for each
    identified coefficient that matches the correct coefficient to some
    error tolerance threshold.

    Parameters
    ----------
    xi_true : 2D numpy array of floats, shape (state_size, n_features)
        True equation coefficients.
    xi_pred : 2D numpy array of floats, shape (state_size, n_features)
        Predicted equation coefficients.

    Returns
    ---------
    success_rate : 2D numpy array of bools, shape (state_size, n_features)
        Success/recovery rate for each coefficient in the governing
        equations.
    """
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
    noise_level=0,
    tol_iter=300,
    n_models=10,
    n_subset=40,
    replace=False,
    weak_form=False,
    algorithm="STLSQ",
    strong_rmse=False,
    K=200,
):
    """
    Very general function for performing hyperparameter scans. This
    function stiches all the training trajectories together and then
    subsamples them to make n_models SINDy models.
    The Pareto optimal model is
    determined by computing the minimum average AIC metric.

    Parameters
    ----------
    systems_list : list of strings, shape (num_systems)
        List of the dynamical systems.
    dimension_list : list or numpy array of integers, shape (num_systems)
        List of the state space dimension of each dynamical system.
    true_coefficients : list of 2D numpy arrays,
            shape (num_systems, dimension_list[i], n_features)
        List of the true coefficient matrices of each dynamical system.
    all_sols_train : dictionary of 3D numpy arrays,
            shape (n_trajectories, num_sample_points, dimension_list[i])
        Dictionary containing all the training trajectories for each
        dynamical system, each entry has shape
        (n_trajectories, num_sample_points, dimension_list[i]).
    all_t_train : dictionary of 2D numpy arrays,
            shape (n_trajectories, num_sample_points)
        Dictionary containing all the training trajectory timebases
        for each dynamical system, each entry has shape
        (n_trajectories, num_sample_points).
    all_sols_test : dictionary of 3D numpy arrays,
            shape (n_trajectories, num_sample_points, dimension_list[i])
        Dictionary containing all the testing trajectories for each
        dynamical system, each entry has shape
        (n_trajectories, num_sample_points, dimension_list[i]).
    all_t_test : dictionary of 2D numpy arrays,
            shape (n_trajectories, num_sample_points)
        Dictionary containing all the testing trajectory timebases
        for each dynamical system, each entry has shape
        (n_trajectories, num_sample_points).
    normalize_columns : bool, optional (default False)
        Flag to normalize the columns in the SINDy feature library.
    noise_level : float, optional (default 0.0)
        Amount (standard deviation) of zero-mean Gaussian noise to add
        to every point in all the training data. This number should be
        interpreted as a percent of the RMSE of the training data.
    tol_iter : integer, optional (default 300)
        Number of hyperparameter values to try during the Pareto scan.
    n_models : integer, optional (default 10)
        Number of models to generate when building the ensemble of models.
    n_subset : integer, optional (default 40)
        Number of time points to subsample when building the ensemble
        of models.
    replace : bool, optional (default False)
        Whether to subsample with replacement or not.
    weak_form : bool, optional (default False)
        Whether to use the weak formulation of the SINDy library.
    algorithm : string, optional (default STLSQ)
        The SINDy optimization algorithm that is used in the hyperparameter
        scan.
    strong_rmse : bool, optional (default False)
        If weak_form = False, this parameter does nothing.
        If weak_form = True and strong_rmse = True use the original RMSE error,
        calculated without the weak form, for determining the
        hyperparameter scan. If weak_form = True and strong_rmse = False, use the weak
        form of the RMSE error for determining the hyperparameter scan.
    K : int, optional (default 200)
        If weak_form = False, this parameter does nothing.
        If weak_form = True, this determines the number of points in the weak form
        version of the regression problem, so K increasing improves the performance.

    Returns
    -------
    xdot_rmse_errors : dictionary of 1D numpy arrays, shape (n_models)
        Normalized RMSE errors between the true and predicted Xdot values
        on all the testing data, using the Pareto-optimal SINDy model.
    xdot_coef_errors : dictionary of 1D numpy arrays, shape (n_models)
        Normalized coefficient errors between the true and predicted
        coefficients, using the Pareto-optimal SINDy model.
    AIC : dictionary of 1D numpy arrays, shape (n_models)
        Finite-sample-size-corrected AIC values computed using the
        Pareto-optimal SINDy model.
    x_dot_tests : list of 3D numpy arrays, each of shape
                  (n_trajectories, num_sample_points, dimension_list[i]).
        List of all the x_dot testing trajectories.
    x_dot_test_preds : list of 3D numpy arrays, each of shape
                  (n_trajectories, num_sample_points, dimension_list[i]).
        List of all the x_dot trajectories predicted using the Pareto
        optimal SINDy model.
    predicted_coefficients : dictionary of 3D numpy arrays, each of shape
                  (n_models, dimension_list[i], n_features).
        Coefficients determined for each system from the
        Pareto-optimal model.
    best_threshold_values : dictionary of floats, shape (num_systems)
        Best hyperparameter value determined by the hyperparameter scan.
        For STLSQ this is the l0 threshold, for MIOSR this is the number
        of nonzero terms.
    models : list of PySINDy models, shape (num_systems)
        Best SINDy models for each dynamical system, determined by
        the hyperparameter scan.
    condition_numbers : numpy array of floats, shape (num_systems)
        Condition number of the PySINDy feature library, using the
        full training data, for each system.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    # define data structure for records
    xdot_rmse_errors = {}
    xdot_coef_errors = {}
    predicted_coefficients = {}
    best_threshold_values = {}
    AIC = {}

    # initialize structures
    num_attractors = len(systems_list)
    for system in systems_list:
        xdot_rmse_errors[system] = list()
        xdot_coef_errors[system] = list()
        predicted_coefficients[system] = list()
        best_threshold_values[system] = list()
        AIC[system] = list()

    # beware ... initial hyperparameter value is initial_hyperparam
    # and this is hard-coded here!
    if normalize_columns:
        initial_hyperparam = (
            1e-2  # threshold values will be higher if feature library is normalized
        )
    else:
        initial_hyperparam = 1e-6

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
                0, rmse / 100.0 * noise_level, x_train[j].shape
            )
            x_train_list.append(x_train_noisy)
            x_test_list.append(all_sols_test[attractor_name][j])
            t_train_list.append(all_t_train[attractor_name][j])
            t_test_list.append(all_t_test[attractor_name][j])

        if dimension_list[i] == 3:
            input_names = ["x", "y", "z"]
        else:
            input_names = ["x", "y", "z", "w"]

        # Critical step for the weak form -- change the grid for each system!
        if weak_form:
            poly_library = ps.WeakPDELibrary(
                library_functions=library_functions,
                function_names=library_function_names,
                spatiotemporal_grid=all_t_train[attractor_name][0],
                is_uniform=True,
                include_bias=True,
                K=K,
            )

        # pre-calculate the test trajectory derivatives and library matrices
        x_dot_test = [
            poly_library.calc_trajectory(
                ps.FiniteDifference(axis=-2),
                ps.AxesArray(x_test_list[i], axes={"ax_time": 0, "ax_coord": 1}),
                t_test_list[i],
            )
            for i in range(len(x_test_list))
        ]
        mats = poly_library.fit_transform(
            [ps.AxesArray(xt, axes={"ax_time": 0, "ax_coord": 1}) for xt in x_test_list]
        )
        order = np.arange(mats[0].shape[-1])
        if isinstance(poly_library, ps.WeakPDELibrary):
            lib2 = ps.PolynomialLibrary(degree=4).fit(
                [
                    ps.AxesArray(xt, axes={"ax_time": 0, "ax_coord": 1})
                    for xt in x_test_list
                ]
            )
            poly_library.fit(
                [
                    ps.AxesArray(xt, axes={"ax_time": 0, "ax_coord": 1})
                    for xt in x_test_list
                ]
            )
            order = np.array(
                [
                    np.where(name == np.array(lib2.get_feature_names()))[0]
                    for name in poly_library.get_feature_names()
                ]
            )[:, 0]
            print(order)
            if strong_rmse:
                x_dot_test = [
                    lib2.calc_trajectory(
                        ps.FiniteDifference(axis=-2),
                        ps.AxesArray(
                            x_test_list[i], axes={"ax_time": 0, "ax_coord": 1}
                        ),
                        t_test_list[i],
                    )
                    for i in range(len(x_test_list))
                ]
                mats = lib2.fit_transform(
                    [
                        ps.AxesArray(xt, axes={"ax_time": 0, "ax_coord": 1})
                        for xt in x_test_list
                    ]
                )
                mats = [mat[:, order] for mat in mats]

        # Sweep a Pareto front depending on the algorithm
        if algorithm == "MIOSR":
            (
                coef_best,
                err_best,
                err_rmse,
                coef_history,
                AIC_history,
                threshold_best,
                model,
                condition_numbers[i],
            ) = hyperparameter_scan_miosr(
                x_train_list,
                x_test_list,
                t_train_list,
                t_test_list,
                x_dot_test,
                mats,
                ode_lib=poly_library,
                alpha=1e-5,
                normalize_columns=normalize_columns,
                input_names=input_names,
                n_models=n_models,
                n_subset=n_subset,
                replace=replace,
            )
        elif "SR3" in algorithm:
            if algorithm == "SR3 ($\nu = 0.1$)":
                nu = 0.1
            else:
                nu = 1.0
            (
                coef_best,
                err_best,
                err_rmse,
                coef_history,
                AIC_history,
                threshold_best,
                model,
                condition_numbers[i],
            ) = hyperparameter_scan_sr3(
                x_train_list,
                x_test_list,
                t_train_list,
                t_test_list,
                x_dot_test,
                mats,
                ode_lib=poly_library,
                initial_hyperparam=initial_hyperparam,
                max_iter=max_iter,
                tol_iter=tol_iter,
                change_factor=1.1,
                normalize_columns=normalize_columns,
                input_names=input_names,
                n_models=n_models,
                n_subset=n_subset,
                replace=replace,
                nu=nu,
            )
        elif algorithm == "Lasso":
            (
                coef_best,
                err_best,
                err_rmse,
                coef_history,
                AIC_history,
                threshold_best,
                model,
                condition_numbers[i],
            ) = hyperparameter_scan_lasso(
                x_train_list,
                x_test_list,
                t_train_list,
                t_test_list,
                x_dot_test,
                mats,
                ode_lib=poly_library,
                initial_hyperparam=initial_hyperparam,
                max_iter=max_iter,
                tol_iter=tol_iter,
                change_factor=1.1,
                normalize_columns=normalize_columns,
                input_names=input_names,
                n_models=n_models,
                n_subset=n_subset,
                replace=replace,
            )
        else:
            (
                coef_best,
                err_best,
                err_rmse,
                coef_history,
                AIC_history,
                threshold_best,
                model,
                condition_numbers[i],
            ) = hyperparameter_scan_stlsq(
                x_train_list,
                x_test_list,
                t_train_list,
                t_test_list,
                x_dot_test,
                mats,
                ode_lib=poly_library,
                initial_hyperparam=initial_hyperparam,
                max_iter=max_iter,
                tol_iter=tol_iter,
                change_factor=1.1,
                alpha=1e-5,
                normalize_columns=normalize_columns,
                input_names=input_names,
                n_models=n_models,
                n_subset=n_subset,
                replace=replace,
            )
        # Using the Pareto-optimal model, compute true x_dot (with median ensemble aggregator)
        x_dot_test_pred = [np.median(coef_best, axis=0).dot(mat.T).T for mat in mats]

        models.append(model)
        x_dot_tests.append(x_dot_test)
        x_dot_test_preds.append(x_dot_test_pred)
        best_threshold_values[attractor_name].append(threshold_best)
        xdot_rmse_errors[attractor_name].append(err_rmse)
        AIC[attractor_name].append(err_best)
        xdot_coef_errors[attractor_name].append(
            coefficient_errors(
                true_coefficients[i], np.mean(coef_best, axis=0)[:, np.argsort(order)]
            )
        )
        predicted_coefficients[attractor_name].append(
            coef_best[:, :, np.argsort(order)]
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
        models,
        condition_numbers,
    )


def hyperparameter_scan_stlsq(
    x_train,
    x_test,
    t_train,
    t_test,
    x_dot_test,
    mats,
    ode_lib,
    initial_hyperparam,
    alpha=1e-5,
    tol_iter=25,
    change_factor=2,
    normalize_columns=True,
    max_iter=20,
    input_names=["x", "y", "z"],
    n_models=10,
    n_subset=40,
    replace=False,
):
    """
    Algorithm to scan over threshold values during STLSQ with Ridge
    Regression, and then select the highest performing model on the
    test set by computing the AIC.

    Parameters
    ----------
    x_train : 3D numpy array,
              shape (n_trajectories, num_sample_points, dimension_list[i])
        All the training trajectories for a dynamical system.
    t_train : 2D numpy array,
              shape (n_trajectories, num_sample_points)
        All the training trajectory timebases for a dynamical system.
    x_test : 3D numpy array,
              shape (n_trajectories, num_sample_points, dimension_list[i])
        All the testing trajectories for a dynamical system.
    t_test : 2D numpy array,
              shape (n_trajectories, num_sample_points)
        All the testing trajectory timebases for a dynamical system.
    ode_lib : PySINDy library
        Pre-defined PySINDy library to use for the SINDy fits.
    initial_hyperparam : float
        Initial value for the hyperparameter before the scan begins. This
        value should be very small, because it will only increase
        substantially during the hyperparameter scan.
    alpha: float, optional (default 1e-5)
        Hyperparameter determining the strength of ridge regularization
        in the STLSQ optimizer. Not optimized in this code.
    tol_iter : integer, optional (default 300)
        Number of hyperparameter values to try during the Pareto scan.
    change_factor : float, optional (default = 2)
        During each step of the hyperparameter scan, the next value of the
        hyperparameter = hyperparameter * change_factor.
    normalize_columns : bool, optional (default False)
        Flag to normalize the columns in the SINDy feature library.
    max_iter : integer, optional, (default 20)
        Maximum number of iterations to perform with the optimizer, at
        each fixed value of the hyperparameter (i.e. during every
        model fit).
    input_names : list of strings, optional, shape (state_size),
                  (default ["x", "y", "z"])
        List of strings representing variable names to use for the
        SINDy models.
    n_models : integer, optional (default 10)
        Number of models to generate when building the ensemble of models.
    n_subset : integer, optional (default 40)
        Number of time points to subsample when building the ensemble
        of models.
    replace : bool, optional (default False)
        Whether to subsample with replacement or not.

    Returns
    -------
    coef_best : 1D numpy arrays, shape (n_models)
        Normalized coefficient errors between the true and predicted
        coefficients, using the Pareto-optimal SINDy model.
    AIC_best : 1D numpy arrays, shape (n_models)
        Finite-sample-size-corrected AIC values computed using the
        Pareto-optimal SINDy model.
    error_rmse : 1D numpy arrays, shape (n_models)
        Normalized RMSE errors between the true and predicted Xdot values
        on all the testing data, using the Pareto-optimal SINDy model.
    xdot_coef_errors : dictionary of 1D numpy arrays, shape (n_models)
        Normalized coefficient errors between the true and predicted
        coefficients, using the Pareto-optimal SINDy model.
    x_dot_tests : list of 3D numpy arrays, each of shape
                  (n_trajectories, num_sample_points, dimension_list[i]).
        List of all the x_dot testing trajectories.
    x_dot_test_preds : list of 3D numpy arrays, each of shape
                  (n_trajectories, num_sample_points, dimension_list[i]).
        List of all the x_dot trajectories predicted using the Pareto
        optimal SINDy model.
    predicted_coefficients : dictionary of 3D numpy arrays, each of shape
                  (n_models, dimension_list[i], n_features).
        Coefficients determined for each system from the
        Pareto-optimal model.
    coef_history_ : list of coefficient values
            shape (n_models, dimension_list[i], n_features, tol_iter)
        Coefficients determined at each step of the hyperparameter scan.
    AIC_history_ : list of AIC values, shape (tol_iter)
        Average AIC determined at each step of the hyperparameter scan.
    threshold_best : float
        Best hyperparameter value determined by the hyperparameter scan.
        For STLSQ this is the l0 threshold, for MIOSR this is the number
        of nonzero terms.
    model_best : PySINDy model
        Best SINDy model for a dynamical system, determined by
        the hyperparameter scan.
    condition_number : float
        Condition number of the PySINDy feature library.
    """

    n_trajectories = np.array(x_test).shape[0]
    n_state = np.array(x_test).shape[2]
    n_time = np.array(x_dot_test).shape[1]

    # Do an initial least-squares fit to get an initial guess of the coefficients
    # start with initial guess that all coefs are zero
    optimizer = ps.EnsembleOptimizer(
        opt=ps.STLSQ(
            threshold=0,
            alpha=alpha,
            max_iter=max_iter,
            normalize_columns=normalize_columns,
            ridge_kw={"tol": 1e-10},
        ),
        bagging=True,
        n_models=n_models,
        n_subset=n_subset,
        replace=replace,
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

    coef_best = np.array(optimizer.coef_list)
    optimizer.coef_ = np.mean(coef_best, axis=0)
    model_best = model

    # For each model, compute x_dot_test and compute the RMSE error
    AIC_new = np.zeros(n_models)
    AIC_best = np.zeros(n_models)
    error_rmse_new = np.zeros(n_models)
    error_rmse = np.zeros(n_models)

    for i in range(n_models):
        x_dot_test_pred = [coef_best[i].dot(mat.T).T for mat in mats]
        dx_test = np.array(x_dot_test).reshape(n_trajectories * n_time, n_state)
        dx_pred = np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state)
        AIC_best[i] = AIC_c(dx_test, dx_pred, coef_best[i, :, :])
        error_rmse[i] = normalized_RMSE(
            dx_test,
            dx_pred,
        )
    coef_history_ = np.zeros(
        (n_models, coef_best.shape[1], coef_best.shape[2], 1 + tol_iter)
    )
    AIC_history_ = np.zeros((n_models, 1 + tol_iter))
    coef_history_[:, :, :, 0] = coef_best
    AIC_history_[:, 0] = AIC_best
    tol = initial_hyperparam
    threshold_best = tol

    # Loop over threshold values, note needs some coding
    # if not using STLSQ optimizer
    for i in range(tol_iter):
        optimizer = ps.EnsembleOptimizer(
            opt=ps.STLSQ(
                threshold=tol,
                alpha=alpha,
                max_iter=max_iter,
                normalize_columns=normalize_columns,
                ridge_kw={"tol": 1e-10},
            ),
            bagging=True,
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
            x_dot_test_pred = [coef_new[j].dot(mat.T).T for mat in mats]

            AIC_new[j] = AIC_c(
                np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
                np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
                coef_new[j, :, :],
            )
            error_rmse_new[j] = normalized_RMSE(
                np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
                np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
            )

        coef_history_[:, :, :, i + 1] = coef_new
        AIC_history_[:, i + 1] = AIC_new

        # If error improves, set the new best coefficients
        # Note < not <= since if all coefficients are zero,
        # this would still keep increasing the threshold!
        if np.mean(AIC_new) < np.mean(AIC_best):
            AIC_best = np.copy(AIC_new)
            error_rmse = np.copy(error_rmse_new)
            coef_best = np.copy(coef_new)
            threshold_best = tol
            model.optimizer.coef_ = np.median(coef_new, axis=0)
            model_best = model
        initial_hyperparam = initial_hyperparam * change_factor
        tol += initial_hyperparam

    return (
        coef_best,
        AIC_best,
        error_rmse,
        coef_history_,
        AIC_history_,
        threshold_best,
        model_best,
        condition_number,
    )


def hyperparameter_scan_lasso(
    x_train,
    x_test,
    t_train,
    t_test,
    x_dot_test,
    mats,
    ode_lib,
    initial_hyperparam,
    tol_iter=300,
    change_factor=2,
    normalize_columns=True,
    max_iter=20,
    input_names=["x", "y", "z"],
    n_models=10,
    n_subset=40,
    replace=False,
):
    """
    Algorithm to scan over threshold values using the Lasso,
    and then select the highest performing model on the
    test set by computing the AIC.

    Parameters
    ----------
    x_train : 3D numpy array,
              shape (n_trajectories, num_sample_points, dimension_list[i])
        All the training trajectories for a dynamical system.
    t_train : 2D numpy array,
              shape (n_trajectories, num_sample_points)
        All the training trajectory timebases for a dynamical system.
    x_test : 3D numpy array,
              shape (n_trajectories, num_sample_points, dimension_list[i])
        All the testing trajectories for a dynamical system.
    t_test : 2D numpy array,
              shape (n_trajectories, num_sample_points)
        All the testing trajectory timebases for a dynamical system.
    ode_lib : PySINDy library
        Pre-defined PySINDy library to use for the SINDy fits.
    initial_hyperparam : float
        Initial value for the hyperparameter before the scan begins. This
        value should be very small, because it will only increase
        substantially during the hyperparameter scan.
    tol_iter : integer, optional (default 300)
        Number of hyperparameter values to try during the Pareto scan.
    change_factor : float, optional (default = 2)
        During each step of the hyperparameter scan, the next value of the
        hyperparameter = hyperparameter * change_factor.
    normalize_columns : bool, optional (default False)
        Flag to normalize the columns in the SINDy feature library.
    max_iter : integer, optional, (default 20)
        Maximum number of iterations to perform with the optimizer, at
        each fixed value of the hyperparameter (i.e. during every
        model fit).
    input_names : list of strings, optional, shape (state_size),
                  (default ["x", "y", "z"])
        List of strings representing variable names to use for the
        SINDy models.
    n_models : integer, optional (default 10)
        Number of models to generate when building the ensemble of models.
    n_subset : integer, optional (default 40)
        Number of time points to subsample when building the ensemble
        of models.
    replace : bool, optional (default False)
        Whether to subsample with replacement or not.

    Returns
    -------
    coef_best : 1D numpy arrays, shape (n_models)
        Normalized coefficient errors between the true and predicted
        coefficients, using the Pareto-optimal SINDy model.
    AIC_best : 1D numpy arrays, shape (n_models)
        Finite-sample-size-corrected AIC values computed using the
        Pareto-optimal SINDy model.
    error_rmse : 1D numpy arrays, shape (n_models)
        Normalized RMSE errors between the true and predicted Xdot values
        on all the testing data, using the Pareto-optimal SINDy model.
    xdot_coef_errors : dictionary of 1D numpy arrays, shape (n_models)
        Normalized coefficient errors between the true and predicted
        coefficients, using the Pareto-optimal SINDy model.
    x_dot_tests : list of 3D numpy arrays, each of shape
                  (n_trajectories, num_sample_points, dimension_list[i]).
        List of all the x_dot testing trajectories.
    x_dot_test_preds : list of 3D numpy arrays, each of shape
                  (n_trajectories, num_sample_points, dimension_list[i]).
        List of all the x_dot trajectories predicted using the Pareto
        optimal SINDy model.
    predicted_coefficients : dictionary of 3D numpy arrays, each of shape
                  (n_models, dimension_list[i], n_features).
        Coefficients determined for each system from the
        Pareto-optimal model.
    coef_history_ : list of coefficient values
            shape (n_models, dimension_list[i], n_features, tol_iter)
        Coefficients determined at each step of the hyperparameter scan.
    AIC_history_ : list of AIC values, shape (tol_iter)
        Average AIC determined at each step of the hyperparameter scan.
    threshold_best : float
        Best hyperparameter value determined by the hyperparameter scan.
        For STLSQ this is the l0 threshold, for MIOSR this is the number
        of nonzero terms.
    model_best : PySINDy model
        Best SINDy model for a dynamical system, determined by
        the hyperparameter scan.
    condition_number : float
        Condition number of the PySINDy feature library.
    """
    n_trajectories = np.array(x_test).shape[0]
    n_state = np.array(x_test).shape[2]
    n_time = np.array(x_dot_test).shape[1]

    # Do an initial least-squares fit to get an initial guess of the coefficients
    # start with initial guess that all coefs are zero
    optimizer = ps.EnsembleOptimizer(
        opt=Lasso(
            alpha=0, max_iter=max_iter, fit_intercept=False
        ),  # currently ignoring normalize_columns parameter
        bagging=True,
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
    AIC_new = np.zeros(n_models)
    AIC_best = np.zeros(n_models)
    error_rmse_new = np.zeros(n_models)
    error_rmse = np.zeros(n_models)

    for i in range(n_models):
        x_dot_test_pred = [coef_best[i].dot(mat.T).T for mat in mats]

        dx_test = np.array(x_dot_test).reshape(n_trajectories * n_time, n_state)
        dx_pred = np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state)
        AIC_best[i] = AIC_c(dx_test, dx_pred, coef_best[i, :, :])
        error_rmse[i] = normalized_RMSE(
            dx_test,
            dx_pred,
        )

    coef_history_ = np.zeros(
        (n_models, coef_best.shape[1], coef_best.shape[2], 1 + tol_iter)
    )
    AIC_history_ = np.zeros((n_models, 1 + tol_iter))
    coef_history_[:, :, :, 0] = coef_best
    AIC_history_[:, 0] = AIC_best
    tol = initial_hyperparam
    threshold_best = tol

    # Loop over threshold values, note needs some coding
    # if not using STLSQ optimizer
    for i in range(tol_iter):
        optimizer = ps.EnsembleOptimizer(
            opt=Lasso(alpha=tol, max_iter=max_iter, fit_intercept=False),
            bagging=True,
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
            x_dot_test_pred = [coef_new[j].dot(mat.T).T for mat in mats]
            AIC_new[j] = AIC_c(
                np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
                np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
                coef_new[j, :, :],
            )
            error_rmse_new[j] = normalized_RMSE(
                np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
                np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
            )

        coef_history_[:, :, :, i + 1] = coef_new
        AIC_history_[:, i + 1] = AIC_new

        # If error improves, set the new best coefficients
        # Note < not <= since if all coefficients are zero,
        # this would still keep increasing the threshold!
        if np.mean(AIC_new) < np.mean(AIC_best):
            AIC_best = np.copy(AIC_new)
            error_rmse = np.copy(error_rmse_new)
            coef_best = np.copy(coef_new)
            threshold_best = tol
            model.optimizer.coef_ = np.median(coef_new, axis=0)
            model_best = model
        initial_hyperparam = initial_hyperparam * change_factor
        tol += initial_hyperparam

    return (
        coef_best,
        AIC_best,
        error_rmse,
        coef_history_,
        AIC_history_,
        threshold_best,
        model_best,
        condition_number,
    )


def hyperparameter_scan_sr3(
    x_train,
    x_test,
    t_train,
    t_test,
    x_dot_test,
    mats,
    ode_lib,
    initial_hyperparam,
    tol_iter=300,
    change_factor=2,
    normalize_columns=True,
    max_iter=20,
    input_names=["x", "y", "z"],
    n_models=10,
    n_subset=40,
    replace=False,
    nu=1.0,
):
    """
    Algorithm to scan over threshold values with SR3, and then select
    the highest performing model on the test set by computing the AIC.

    Parameters
    ----------
    x_train : 3D numpy array,
              shape (n_trajectories, num_sample_points, dimension_list[i])
        All the training trajectories for a dynamical system.
    t_train : 2D numpy array,
              shape (n_trajectories, num_sample_points)
        All the training trajectory timebases for a dynamical system.
    x_test : 3D numpy array,
              shape (n_trajectories, num_sample_points, dimension_list[i])
        All the testing trajectories for a dynamical system.
    t_test : 2D numpy array,
              shape (n_trajectories, num_sample_points)
        All the testing trajectory timebases for a dynamical system.
    ode_lib : PySINDy library
        Pre-defined PySINDy library to use for the SINDy fits.
    initial_hyperparam : float
        Initial value for the hyperparameter before the scan begins. This
        value should be very small, because it will only increase
        substantially during the hyperparameter scan.
    tol_iter : integer, optional (default 300)
        Number of hyperparameter values to try during the Pareto scan.
    change_factor : float, optional (default = 2)
        During each step of the hyperparameter scan, the next value of the
        hyperparameter = hyperparameter * change_factor.
    normalize_columns : bool, optional (default False)
        Flag to normalize the columns in the SINDy feature library.
    max_iter : integer, optional, (default 20)
        Maximum number of iterations to perform with the optimizer, at
        each fixed value of the hyperparameter (i.e. during every
        model fit).
    input_names : list of strings, optional, shape (state_size),
                  (default ["x", "y", "z"])
        List of strings representing variable names to use for the
        SINDy models.
    n_models : integer, optional (default 10)
        Number of models to generate when building the ensemble of models.
    n_subset : integer, optional (default 40)
        Number of time points to subsample when building the ensemble
        of models.
    replace : bool, optional (default False)
        Whether to subsample with replacement or not.
    nu : float, optional (default 1.0)
        SR3 hyperparameter to determine the strength of the relaxation.

    Returns
    -------
    coef_best : 1D numpy arrays, shape (n_models)
        Normalized coefficient errors between the true and predicted
        coefficients, using the Pareto-optimal SINDy model.
    AIC_best : 1D numpy arrays, shape (n_models)
        Finite-sample-size-corrected AIC values computed using the
        Pareto-optimal SINDy model.
    error_rmse : 1D numpy arrays, shape (n_models)
        Normalized RMSE errors between the true and predicted Xdot values
        on all the testing data, using the Pareto-optimal SINDy model.
    xdot_coef_errors : dictionary of 1D numpy arrays, shape (n_models)
        Normalized coefficient errors between the true and predicted
        coefficients, using the Pareto-optimal SINDy model.
    x_dot_tests : list of 3D numpy arrays, each of shape
                  (n_trajectories, num_sample_points, dimension_list[i]).
        List of all the x_dot testing trajectories.
    x_dot_test_preds : list of 3D numpy arrays, each of shape
                  (n_trajectories, num_sample_points, dimension_list[i]).
        List of all the x_dot trajectories predicted using the Pareto
        optimal SINDy model.
    predicted_coefficients : dictionary of 3D numpy arrays, each of shape
                  (n_models, dimension_list[i], n_features).
        Coefficients determined for each system from the
        Pareto-optimal model.
    coef_history_ : list of coefficient values
            shape (n_models, dimension_list[i], n_features, tol_iter)
        Coefficients determined at each step of the hyperparameter scan.
    AIC_history_ : list of AIC values, shape (tol_iter)
        Average AIC determined at each step of the hyperparameter scan.
    threshold_best : float
        Best hyperparameter value determined by the hyperparameter scan.
        For STLSQ this is the l0 threshold, for MIOSR this is the number
        of nonzero terms.
    model_best : PySINDy model
        Best SINDy model for a dynamical system, determined by
        the hyperparameter scan.
    condition_number : float
        Condition number of the PySINDy feature library.
    """
    n_trajectories = np.array(x_test).shape[0]
    n_state = np.array(x_test).shape[2]
    n_time = np.array(x_dot_test).shape[1]

    # Do an initial least-squares fit to get an initial guess of the coefficients
    # start with initial guess that all coefs are zero
    optimizer = ps.EnsembleOptimizer(
        opt=ps.SR3(
            threshold=0,
            max_iter=max_iter,
            normalize_columns=normalize_columns,
            nu=nu,
        ),
        bagging=True,
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
    AIC_new = np.zeros(n_models)
    AIC_best = np.zeros(n_models)
    error_rmse_new = np.zeros(n_models)
    error_rmse = np.zeros(n_models)

    for i in range(n_models):
        x_dot_test_pred = [coef_best[i].dot(mat.T).T for mat in mats]
        AIC_best[i] = AIC_c(
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
    AIC_history_ = np.zeros((n_models, 1 + tol_iter))
    coef_history_[:, :, :, 0] = coef_best
    AIC_history_[:, 0] = AIC_best
    tol = initial_hyperparam
    threshold_best = tol

    # Loop over threshold values, note needs some coding
    # if not using STLSQ optimizer
    for i in range(tol_iter):
        optimizer = ps.EnsembleOptimizer(
            opt=ps.SR3(
                threshold=tol,
                max_iter=max_iter,
                normalize_columns=normalize_columns,
                nu=nu,
            ),
            bagging=True,
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
            x_dot_test_pred = [coef_new[j].dot(mat.T).T for mat in mats]
            AIC_new[j] = AIC_c(
                np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
                np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
                coef_new[j, :, :],
            )
            error_rmse_new[j] = normalized_RMSE(
                np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
                np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
            )

        coef_history_[:, :, :, i + 1] = coef_new
        AIC_history_[:, i + 1] = AIC_new

        # If error improves, set the new best coefficients
        # Note < not <= since if all coefficients are zero,
        # this would still keep increasing the threshold!
        if np.mean(AIC_new) < np.mean(AIC_best):
            AIC_best = np.copy(AIC_new)
            error_rmse = np.copy(error_rmse_new)
            coef_best = np.copy(coef_new)
            threshold_best = tol
            model.optimizer.coef_ = np.median(coef_new, axis=0)
            model_best = model
        initial_hyperparam = initial_hyperparam * change_factor
        tol += initial_hyperparam

    return (
        coef_best,
        AIC_best,
        error_rmse,
        coef_history_,
        AIC_history_,
        threshold_best,
        model_best,
        condition_number,
    )


def hyperparameter_scan_miosr(
    x_train,
    x_test,
    t_train,
    t_test,
    x_dot_test,
    mats,
    ode_lib,
    alpha=1e-5,
    normalize_columns=True,
    input_names=["x", "y", "z"],
    n_models=10,
    n_subset=40,
    replace=False,
):
    """
    Algorithm to scan over sparsity values during the mixed-integer
    optimization algorithm (MIOSR) with optional Ridge
    Regression, and then select the highest performing model on the
    test set by computing the AIC. Note that there is no max_iter,
    tol_iter, or change_factor parameters, as in other scans. This is
    because MIOSR does not have an explicit threshold, instead the user
    chooses how many nonzero terms to use. So the hyperparameter scan is
    done by simplying trying to fit the model with all the coefficients
    nonzero, all the way until all the coefficients are zero.

    Parameters
    ----------
    x_train : 3D numpy array,
              shape (n_trajectories, num_sample_points, dimension_list[i])
        All the training trajectories for a dynamical system.
    t_train : 2D numpy array,
              shape (n_trajectories, num_sample_points)
        All the training trajectory timebases for a dynamical system.
    x_test : 3D numpy array,
              shape (n_trajectories, num_sample_points, dimension_list[i])
        All the testing trajectories for a dynamical system.
    t_test : 2D numpy array,
              shape (n_trajectories, num_sample_points)
        All the testing trajectory timebases for a dynamical system.
    ode_lib : PySINDy library
        Pre-defined PySINDy library to use for the SINDy fits.
    initial_hyperparam : float
        Initial value for the hyperparameter before the scan begins. This
        value should be very small, because it will only increase
        substantially during the hyperparameter scan.
    alpha: float, optional (default 1e-5)
        Hyperparameter determining the strength of ridge regularization
        in the STLSQ optimizer. Not optimized in this code.
    normalize_columns : bool, optional (default False)
        Flag to normalize the columns in the SINDy feature library.
    input_names : list of strings, optional, shape (state_size),
                  (default ["x", "y", "z"])
        List of strings representing variable names to use for the
        SINDy models.
    n_models : integer, optional (default 10)
        Number of models to generate when building the ensemble of models.
    n_subset : integer, optional (default 40)
        Number of time points to subsample when building the ensemble
        of models.
    replace : bool, optional (default False)
        Whether to subsample with replacement or not.

    Returns
    -------
    coef_best : 1D numpy arrays, shape (n_models)
        Normalized coefficient errors between the true and predicted
        coefficients, using the Pareto-optimal SINDy model.
    AIC_best : 1D numpy arrays, shape (n_models)
        Finite-sample-size-corrected AIC values computed using the
        Pareto-optimal SINDy model.
    error_rmse : 1D numpy arrays, shape (n_models)
        Normalized RMSE errors between the true and predicted Xdot values
        on all the testing data, using the Pareto-optimal SINDy model.
    xdot_coef_errors : dictionary of 1D numpy arrays, shape (n_models)
        Normalized coefficient errors between the true and predicted
        coefficients, using the Pareto-optimal SINDy model.
    x_dot_tests : list of 3D numpy arrays, each of shape
                  (n_trajectories, num_sample_points, dimension_list[i]).
        List of all the x_dot testing trajectories.
    x_dot_test_preds : list of 3D numpy arrays, each of shape
                  (n_trajectories, num_sample_points, dimension_list[i]).
        List of all the x_dot trajectories predicted using the Pareto
        optimal SINDy model.
    predicted_coefficients : dictionary of 3D numpy arrays, each of shape
                  (n_models, dimension_list[i], n_features).
        Coefficients determined for each system from the
        Pareto-optimal model.
    coef_history_ : list of coefficient values
            shape (n_models, dimension_list[i], n_features, tol_iter)
        Coefficients determined at each step of the hyperparameter scan.
    AIC_history_ : list of AIC values, shape (tol_iter)
        Average AIC determined at each step of the hyperparameter scan.
    threshold_best : float
        Best hyperparameter value determined by the hyperparameter scan.
        For STLSQ this is the l0 threshold, for MIOSR this is the number
        of nonzero terms.
    model_best : PySINDy model
        Best SINDy model for a dynamical system, determined by
        the hyperparameter scan.
    condition_number : float
        Condition number of the PySINDy feature library.
    """
    n_trajectories = np.array(x_test).shape[0]
    n_state = np.array(x_test).shape[2]
    n_time = np.array(x_dot_test).shape[1]

    # Do an initial least-squares fit to get an initial guess of the coefficients
    # start with initial guess that all coefs are zero
    optimizer = ps.EnsembleOptimizer(
        opt=ps.MIOSR(
            target_sparsity=1,
            alpha=alpha,
            normalize_columns=normalize_columns,
            regression_timeout=100,
        ),
        bagging=True,
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
    tol_iter = np.shape(optimizer.Theta_)[1] - 1

    # Set the L0 penalty based on the condition number of Theta
    coef_best = np.array(optimizer.coef_list)
    optimizer.coef_ = np.mean(coef_best, axis=0)
    model_best = model

    # For each model, compute x_dot_test and compute the RMSE error
    AIC_new = np.zeros(n_models)
    AIC_best = np.zeros(n_models)
    error_rmse_new = np.zeros(n_models)
    error_rmse = np.zeros(n_models)

    for i in range(n_models):
        x_dot_test_pred = [coef_best[i].dot(mat.T).T for mat in mats]
        error_rmse[i] = normalized_RMSE(
            np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
            np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
        )
        AIC_best[i] = AIC_c(
            np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
            np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
            coef_best[i, :, :],
        )

    coef_history_ = np.zeros(
        (n_models, coef_best.shape[1], coef_best.shape[2], 1 + tol_iter)
    )
    AIC_history_ = np.zeros((n_models, 1 + tol_iter))
    coef_history_[:, :, :, 0] = coef_best
    AIC_history_[:, 0] = AIC_best
    sparsity_best = 0

    # Loop over threshold values, note needs some coding
    # if not using STLSQ optimizer
    for i in range(tol_iter):
        optimizer = ps.EnsembleOptimizer(
            opt=ps.MIOSR(
                target_sparsity=i + 1,
                alpha=alpha,
                normalize_columns=normalize_columns,
                regression_timeout=5,
            ),
            bagging=True,
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
            x_dot_test_pred = [coef_new[j].dot(mat.T).T for mat in mats]
            AIC_new[j] = AIC_c(
                np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
                np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
                coef_new[j, :, :],
            )
            error_rmse_new[j] = normalized_RMSE(
                np.array(x_dot_test).reshape(n_trajectories * n_time, n_state),
                np.array(x_dot_test_pred).reshape(n_trajectories * n_time, n_state),
            )

        coef_history_[:, :, :, i + 1] = coef_new
        AIC_history_[:, i + 1] = AIC_new

        # If error improves, set the new best coefficients
        # Note < not <= since if all coefficients are zero,
        # this would still keep increasing the threshold!
        if np.mean(AIC_new) < np.mean(AIC_best):
            AIC_best = np.copy(AIC_new)
            error_rmse = np.copy(error_rmse_new)
            coef_best = np.copy(coef_new)
            sparsity_best = i
            model.optimizer.coef_ = np.median(coef_new, axis=0)
            model_best = model

    return (
        coef_best,
        AIC_best,
        error_rmse,
        coef_history_,
        AIC_history_,
        sparsity_best,
        model_best,
        condition_number,
    )


def weakform_reorder_coefficients(systems_list, dimension_list, true_coefficients):
    """
    This function reorders the true model coefficients if using the weak
    formulation, in order to compare with the weak formulation of the SINDy
    library, which is in a different order than the PolynomialLibrary
    with degree = 4.
    """
    # reordering to use if system is 3D
    reorder1 = np.array(
        [
            0,
            1,
            2,
            3,
            4,
            7,
            9,
            5,
            6,
            8,
            10,
            16,
            19,
            11,
            12,
            17,
            13,
            15,
            18,
            14,
            20,
            30,
            34,
            21,
            22,
            31,
            26,
            29,
            33,
            23,
            25,
            32,
            27,
            28,
            24,
        ],
        dtype=int,
    )

    # reordering to use if system is 4D
    reorder2 = np.array(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            9,
            12,
            14,
            6,
            7,
            8,
            10,
            11,
            13,
            15,
            25,
            31,
            34,
            16,
            17,
            18,
            26,
            27,
            32,
            19,
            22,
            24,
            28,
            30,
            33,
            20,
            21,
            23,
            29,
            35,
            55,
            65,
            69,
            36,
            37,
            38,
            56,
            57,
            66,
            45,
            51,
            54,
            61,
            64,
            68,
            39,
            42,
            44,
            58,
            60,
            67,
            46,
            47,
            52,
            62,
            48,
            50,
            53,
            63,
            40,
            41,
            43,
            59,
            49,
        ],
        dtype=int,
    )
    for i, system in enumerate(systems_list):
        if dimension_list[i] == 3:
            true_coefficients[i] = true_coefficients[i][:, reorder1]
        else:
            true_coefficients[i] = true_coefficients[i][:, reorder2]
    return true_coefficients
