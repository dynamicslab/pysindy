from itertools import repeat
from typing import Sequence

import numpy as np
from numpy.random import choice
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import bisect
from sklearn.base import MultiOutputMixin
from sklearn.utils.validation import check_array

# Define a special object for the default value of t in
# validate_input. Normally we would set the default
# value of t to be None, but it is possible for the user
# to pass in None, in which case validate_input performs
# no checks on t.
T_DEFAULT = object()


def validate_input(x, t=T_DEFAULT):
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be array-like")
    elif x.ndim == 1:
        x = x.reshape(-1, 1)
    check_array(x, ensure_2d=False, allow_nd=True)

    if t is not T_DEFAULT:
        if t is None:
            raise ValueError("t must be a scalar or array-like.")
        # Apply this check if t is a scalar
        elif np.ndim(t) == 0 and (isinstance(t, int) or isinstance(t, float)):
            if t <= 0:
                raise ValueError("t must be positive")
        # Only apply these tests if t is array-like
        elif isinstance(t, np.ndarray):
            if not len(t) == x.shape[-2]:
                raise ValueError("Length of t should match x.shape[-2].")
            if not np.all(t[:-1] < t[1:]):
                raise ValueError("Values in t should be in strictly increasing order.")
        else:
            raise ValueError("t must be a scalar or array-like.")

    if x.ndim != 2:
        x_new = x.reshape(x.size // x.shape[-1], x.shape[-1])
    else:
        x_new = x
    return x_new


def validate_control_variables(
    x, u, multiple_trajectories=False, trim_last_point=False, return_array=True
):
    """
    Ensure that control variables u are compatible with the data x.
    If ``return_array`` and ``multiple_trajectories`` are True, convert u from a list
    into an array (of concatenated list entries).
    """
    if multiple_trajectories:
        if not isinstance(x, Sequence):
            raise ValueError("x must be a list when multiple_trajectories is True")
        if not isinstance(u, Sequence):
            raise ValueError("u must be a list when multiple_trajectories is True")
        if len(x) != len(u):
            raise ValueError(
                "x and u must be lists of the same length when "
                "multiple_trajectories is True"
            )

        u_arr = [_check_control_shape(xi, ui, trim_last_point) for xi, ui in zip(x, u)]

        if return_array:
            u_arr = np.vstack(u_arr)

    else:
        u_arr = _check_control_shape(x, u, trim_last_point)

    return u_arr


def _check_control_shape(x, u, trim_last_point):
    """
    Convert control variables u to np.array(dtype=float64) and compare
    its shape against x. Assumes x is array-like.
    """
    try:
        u = np.array(u, dtype="float64")
    except TypeError as e:
        raise e(
            "control variables u could not be converted to np.ndarray(dtype=float64)"
        )
    if np.ndim(u) == 0:
        u = u[np.newaxis]
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    elif u.ndim != 2:
        u = u.reshape(u.size // u.shape[-1], u.shape[-1])
    if len(x) != u.shape[0]:
        raise ValueError(
            "control variables u must have same number of rows as x. "
            "u has {} rows and x has {} rows".format(u.shape[0], len(x))
        )
    return u[:-1] if trim_last_point else u


def drop_nan_rows(x, x_dot):
    x = x[~np.isnan(x_dot).any(axis=1)]
    x_dot = x_dot[~np.isnan(x_dot).any(axis=1)]
    return x, x_dot


def drop_random_rows(
    x,
    x_dot,
    n_subset,
    replace,
    feature_library,
    pde_library_flag,
    multiple_trajectories,
):
    num_trajectories = feature_library.num_trajectories
    # Can't choose random n_subset points if data is from a PDE
    # (and therefore is spatially local).
    # Need to unfold it and just choose n_subset from the temporal slices
    if pde_library_flag is not None:
        if pde_library_flag == "WeakPDE":
            # Weak form needs uniform, ascending grid, so cannot replace
            replace = False
            s = [slice(None, None)] * feature_library.spatiotemporal_grid.ndim
            s[-2] = 0
            s[-1] = slice(None, -1)
            spatial_grid = feature_library.spatiotemporal_grid[tuple(s)]
            temporal_grid = feature_library.grid_pts[-1]
            num_time = len(temporal_grid)
            dims = spatial_grid.shape[:-1]
        else:
            # new bagging for arbitrary samples
            n_samples = x.shape[0]
            spatial_grid = feature_library.spatial_grid
            dims = spatial_grid.shape[:-1]
            if len(dims) > 0:
                num_time = n_samples // np.product(dims) // num_trajectories
            else:
                num_time = n_samples // num_trajectories

        n_features = x.shape[1]
        if n_subset > num_time:
            n_subset = num_time
        rand_inds = np.sort(choice(range(num_time), n_subset, replace=replace))

        if len(dims) > 0:
            x_shaped = np.reshape(
                x, np.concatenate([dims, [num_time * num_trajectories], [n_features]])
            )
        else:
            x_shaped = np.reshape(
                x, np.concatenate([[num_time * num_trajectories], [n_features]])
            )
        s0 = [slice(dim) for dim in x_shaped.shape]

        rand_inds_total = []
        for i in range(num_trajectories):
            rand_inds_total.append(rand_inds + num_time * i)
        s0[len(dims)] = rand_inds_total

        if len(dims) > 0:
            x_new = np.reshape(
                x_shaped[tuple(s0)],
                (np.product(dims) * n_subset * num_trajectories, x.shape[1]),
            )
        else:
            x_new = np.reshape(
                x_shaped[tuple(s0)], (n_subset * num_trajectories, x.shape[1])
            )

        if pde_library_flag == "WeakPDE":
            spatiotemporal_grid = feature_library.spatiotemporal_grid
            s1 = [slice(None)] * len(spatiotemporal_grid.shape)
            s1[-2] = rand_inds
            new_spatiotemporal_grid = spatiotemporal_grid[tuple(s1)]
            feature_library.spatiotemporal_grid = new_spatiotemporal_grid
            feature_library._set_up_grids()
            s0[len(dims)] = rand_inds
            if multiple_trajectories:
                x_dot_new = [
                    convert_u_dot_integral(xi[tuple(s0)], feature_library)
                    for xi in feature_library.old_x
                ]
                x_dot_new = np.vstack(x_dot_new)
            else:
                x_dot_new = convert_u_dot_integral(
                    feature_library.old_x[tuple(s0)], feature_library
                )
        else:
            x_dot_shaped = np.reshape(
                x_dot,
                np.concatenate([dims, [num_time * num_trajectories], [n_features]]),
            )
            x_dot_new = np.reshape(
                x_dot_shaped[tuple(s0)],
                (np.product(dims) * n_subset * num_trajectories, x.shape[1]),
            )
    else:
        # Choose random n_subset points to use
        rand_inds = np.sort(choice(range(np.shape(x)[0]), n_subset, replace=replace))
        x_new = x[rand_inds, :]
        x_dot_new = x_dot[rand_inds, :]

    return x_new, x_dot_new


def reorder_constraints(c, n_features, output_order="row"):
    """Reorder constraint matrix."""
    ret = c.copy()

    if ret.ndim == 1:
        ret = ret.reshape(1, -1)

    n_targets = ret.shape[1] // n_features
    shape = (n_targets, n_features)

    if output_order == "row":
        for i in range(ret.shape[0]):
            ret[i] = ret[i].reshape(shape).flatten(order="F")
    else:
        for i in range(ret.shape[0]):
            ret[i] = ret[i].reshape(shape, order="F").flatten()

    return ret


def prox_l0(x, threshold):
    """Proximal operator for L0 regularization."""
    return x * (np.abs(x) > threshold)


def prox_weighted_l0(x, thresholds):
    """Proximal operator for weighted l0 regularization."""
    y = np.zeros(np.shape(x))
    transp_thresholds = thresholds.T
    for i in range(transp_thresholds.shape[0]):
        for j in range(transp_thresholds.shape[1]):
            y[i, j] = x[i, j] * (np.abs(x[i, j]) > transp_thresholds[i, j])
    return y


def prox_l1(x, threshold):
    """Proximal operator for L1 regularization."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def prox_weighted_l1(x, thresholds):
    """Proximal operator for weighted l1 regularization."""
    return np.sign(x) * np.maximum(np.abs(x) - thresholds, np.zeros(x.shape))


def prox_l2(x, threshold):
    """Proximal operator for ridge regularization."""
    return 2 * threshold * x


def prox_weighted_l2(x, thresholds):
    """Proximal operator for ridge regularization."""
    return 2 * thresholds * x


# TODO: replace code block with proper math block
def prox_cad(x, lower_threshold):
    """
    Proximal operator for CAD regularization

    .. code ::

        prox_cad(z, a, b) =
            0                    if |z| < a
            sign(z)(|z| - a)   if a < |z| <= b
            z                    if |z| > b

    Entries of :math:`x` smaller than a in magnitude are set to 0,
    entries with magnitudes larger than b are untouched,
    and entries in between have soft-thresholding applied.

    For simplicity we set :math:`b = 5*a` in this implementation.
    """
    upper_threshold = 5 * lower_threshold
    return prox_l0(x, upper_threshold) + prox_l1(x, lower_threshold) * (
        np.abs(x) < upper_threshold
    )


def get_prox(regularization):
    prox = {
        "l0": prox_l0,
        "weighted_l0": prox_weighted_l0,
        "l1": prox_l1,
        "weighted_l1": prox_weighted_l1,
        "l2": prox_l2,
        "weighted_l2": prox_weighted_l2,
        "cad": prox_cad,
    }
    if regularization.lower() in prox.keys():
        return prox[regularization.lower()]
    else:
        raise NotImplementedError("{} has not been implemented".format(regularization))


def get_regularization(regularization):
    if regularization.lower() == "l0":
        return lambda x, lam: lam * np.count_nonzero(x)
    elif regularization.lower() == "weighted_l0":
        return lambda x, lam: np.sum(lam[np.nonzero(x)])
    elif regularization.lower() == "l1":
        return lambda x, lam: lam * np.sum(np.abs(x))
    elif regularization.lower() == "weighted_l1":
        return lambda x, lam: np.sum(np.abs(lam @ x))
    elif regularization.lower() == "l2":
        return lambda x, lam: lam * np.sum(x ** 2)
    elif regularization.lower() == "weighted_l2":
        return lambda x, lam: np.sum(lam @ x ** 2)
    elif regularization.lower() == "cad":  # dummy function
        return lambda x, lam: 0
    else:
        raise NotImplementedError("{} has not been implemented".format(regularization))


def capped_simplex_projection(trimming_array, trimming_fraction):
    """Projection of trimming_array onto the capped simplex"""
    a = np.min(trimming_array) - 1.0
    b = np.max(trimming_array) - 0.0

    def f(x):
        return (
            np.sum(np.maximum(np.minimum(trimming_array - x, 1.0), 0.0))
            - (1.0 - trimming_fraction) * trimming_array.size
        )

    x = bisect(f, a, b)

    return np.maximum(np.minimum(trimming_array - x, 1.0), 0.0)


def print_model(
    coef,
    input_features,
    errors=None,
    intercept=None,
    error_intercept=None,
    precision=3,
    pm="±",
):
    """
    Args:
        coef:
        input_features:
        errors:
        intercept:
        sigma_intercept:
        precision:
        pm:
    Returns:
    """

    def term(c, sigma, name):
        rounded_coef = np.round(c, precision)
        if rounded_coef == 0 and sigma is None:
            return ""
        elif sigma is None:
            return f"{c:.{precision}f} {name}"
        elif rounded_coef == 0 and np.round(sigma, precision) == 0:
            return ""
        else:
            return f"({c:.{precision}f} {pm} {sigma:.{precision}f}) {name}"

    errors = errors if errors is not None else repeat(None)
    components = [term(c, e, i) for c, e, i in zip(coef, errors, input_features)]
    eq = " + ".join(filter(bool, components))

    if not eq or intercept or error_intercept is not None:
        intercept = intercept or 0
        intercept_str = term(intercept, error_intercept, "").strip()
        if eq and intercept_str:
            eq += " + "
            eq += intercept_str
        elif not eq:
            eq = f"{intercept:.{precision}f}"
    return eq


def equations(pipeline, input_features=None, precision=3, input_fmt=None):
    input_features = pipeline.steps[0][1].get_feature_names(input_features)
    if input_fmt:
        input_features = [input_fmt(i) for i in input_features]
    coef = pipeline.steps[-1][1].coef_
    intercept = pipeline.steps[-1][1].intercept_
    if np.isscalar(intercept):
        intercept = intercept * np.ones(coef.shape[0])
    return [
        print_model(
            coef[i], input_features, intercept=intercept[i], precision=precision
        )
        for i in range(coef.shape[0])
    ]


def supports_multiple_targets(estimator):
    """Checks whether estimator supports multiple targets."""
    if isinstance(estimator, MultiOutputMixin):
        return True
    try:
        return estimator._more_tags()["multioutput"]
    except (AttributeError, KeyError):
        return False


def convert_u_dot_integral(u, weak_pde_library):
    """
    Takes a full set of spatiotemporal fields u(x, t) and finds the weak
    form of u_dot using a pre-defined weak pde library.
    """
    K = weak_pde_library.K
    gdim = weak_pde_library.grid_ndim
    u_dot_integral = np.zeros((K, u.shape[-1]))
    deriv_orders = np.zeros(gdim)
    deriv_orders[-1] = 1
    w_diff = -weak_pde_library._smooth_ppoly(deriv_orders)
    for j in range(u.shape[-1]):
        u_interp = RegularGridInterpolator(
            tuple(weak_pde_library.grid_pts), np.take(u, j, axis=-1)
        )
        for k in range(K):
            u_new = u_interp(np.take(weak_pde_library.XT, k, axis=0))
            u_dot_integral_temp = trapezoid(
                w_diff[k] * u_new,
                x=weak_pde_library.xtgrid_k[k, :, 0],
                axis=0,
            )
            for i in range(1, gdim):
                u_dot_integral_temp = trapezoid(
                    u_dot_integral_temp, x=weak_pde_library.xtgrid_k[k, :, i], axis=0
                )
            u_dot_integral[k, j] = u_dot_integral_temp
    return u_dot_integral
