from itertools import repeat
from typing import Sequence

import numpy as np
from scipy.optimize import bisect
from sklearn.base import MultiOutputMixin
from sklearn.utils.validation import check_array

# Define a special object for the default value of t in
# validate_input. Normally we would set the default
# value of t to be None, but it is possible for the user
# to pass in None, in which case validate_input performs
# no checks on t.
T_DEFAULT = object()


def flatten_2d_tall(x):
    return x.reshape(x.size // x.shape[-1], x.shape[-1])


def validate_input(x, t=T_DEFAULT):
    """Forces input data to have compatible dimensions, if possible.

    Args:
        x: array of input data (measured coordinates across time)
        t: time values for measurements.

    Returns:
        x as 2D array, with time dimension on first axis and coordinate
        index on second axis.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be array-like")
    elif x.ndim == 1:
        x = x.reshape(-1, 1)
    x_new = flatten_2d_tall(x)

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

    return x_new


def validate_no_reshape(x, t=T_DEFAULT):
    """Check types and numerical sensibility of arguments.

    Args:
        x: array of input data (measured coordinates across time)
        t: time values for measurements.

    Returns:
        x as 2D array, with time dimension on first axis and coordinate
        index on second axis.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError("Input value must be array-like")
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
    return x


def validate_control_variables(x, u, trim_last_point=False):
    """Ensure that control variables u are compatible with the data x.

    Trims last control variable timepoint if set to True
    """
    if not isinstance(x, Sequence):
        raise ValueError("x must be a list when multiple_trajectories is True")
    if not isinstance(u, Sequence):
        raise ValueError("u must be a list when multiple_trajectories is True")
    if len(x) != len(u):
        raise ValueError(
            "x and u must be lists of the same length when "
            "multiple_trajectories is True"
        )

    def _check_control_shape(x, u, trim_last_point):
        """
        Compare shape of control variable u against x.
        """
        if u.shape[u.ax_time] != x.shape[x.ax_time]:
            raise ValueError(
                "control variables u must have same number of rows as x. "
                "u has {} rows and x has {} rows".format(u.shape[0], len(x))
            )
        return u[:-1] if trim_last_point else u

    u_arr = [_check_control_shape(xi, ui, trim_last_point) for xi, ui in zip(x, u)]

    return u_arr


def drop_nan_samples(x, y):
    """Drops samples from x and y where there is either has a nan value"""
    x_non_sample_axes = tuple(ax for ax in range(x.ndim) if ax != x.ax_sample)
    y_non_sample_axes = tuple(ax for ax in range(y.ndim) if ax != y.ax_sample)
    x_good_samples = (~np.isnan(x)).any(axis=x_non_sample_axes)
    y_good_samples = (~np.isnan(y)).any(axis=y_non_sample_axes)
    good_sample_ind = np.nonzero(x_good_samples & y_good_samples)[0]
    x = x.take(good_sample_ind, axis=x.ax_sample)
    y = y.take(good_sample_ind, axis=y.ax_sample)
    return x, y


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
        return lambda x, lam: lam * np.sum(x**2)
    elif regularization.lower() == "weighted_l2":
        return lambda x, lam: np.sum(lam @ x**2)
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
