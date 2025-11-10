import warnings
from functools import wraps
from typing import Callable
from typing import Sequence
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import bisect
from sklearn.base import MultiOutputMixin
from sklearn.utils.validation import check_array

from ._axes import AxesArray

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


def validate_no_reshape(x, t: Union[float, np.ndarray, object] = T_DEFAULT):
    """Check types and numerical sensibility of arguments.

    Args:
        x: array of input data (measured coordinates across time)
        t: time values for measurements.

    Returns:
        x as 2D array, with time dimension on first axis and coordinate
        index on second axis.
    """
    if not hasattr(x, "shape"):
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
        elif hasattr(t, "shape"):
            if not len(t) == x.shape[-2]:
                raise ValueError("Length of t should match x.shape[-2].")
            if not np.all(t[:-1] < t[1:]):
                raise ValueError("Values in t should be in strictly increasing order.")
        else:
            raise ValueError("t must be a scalar or array-like.")
    return x


def validate_control_variables(x: Sequence[AxesArray], u: Sequence[AxesArray]) -> None:
    """Ensure that control variables u are compatible with the data x.

    Args:
        x: trajectories of system variables
        u: trajectories of control variables
    """
    if not isinstance(x, Sequence):
        raise ValueError("x must be a Sequence")
    if not isinstance(u, Sequence):
        raise ValueError("u must be a Sequence")
    if len(x) != len(u):
        raise ValueError("x and u must be the same length")

    def _check_control_shape(x, u):
        """
        Compare shape of control variable u against x.
        """
        if u.n_time != x.n_time:
            raise ValueError(
                "control variables u must have same number of time points as x. "
                f"u has {u.n_time} time points and x has {x.n_time} time points"
            )
        return u

    u_arr = [_check_control_shape(xi, ui) for xi, ui in zip(x, u)]

    return u_arr


def drop_nan_samples(x, y):
    """Drops samples from x and y where either has a nan value"""
    x_non_sample_axes = tuple(ax for ax in range(x.ndim) if ax != x.ax_sample)
    y_non_sample_axes = tuple(ax for ax in range(y.ndim) if ax != y.ax_sample)
    x_good_samples = (~np.isnan(x)).any(axis=x_non_sample_axes)
    y_good_samples = (~np.isnan(y)).any(axis=y_non_sample_axes)
    good_sample_ind = np.nonzero(x_good_samples & y_good_samples)[0]
    x = x.take(good_sample_ind, axis=x.ax_sample)
    y = y.take(good_sample_ind, axis=y.ax_sample)
    return x, y


def reorder_constraints(arr, n_features, output_order="feature"):
    """Switch between 'feature' and 'target' constraint order."""
    warnings.warn("Target format constraints are deprecated.", stacklevel=2)
    n_constraints = arr.shape[0] if arr.ndim > 1 else 1
    n_tgt = arr.size // n_features // n_constraints
    if output_order == "feature":
        starting_shape = (n_constraints, n_tgt, n_features)
    else:
        starting_shape = (n_constraints, n_features, n_tgt)

    return arr.reshape(starting_shape).transpose([0, 2, 1]).reshape((n_constraints, -1))


def _validate_prox_and_reg_inputs(func):
    """Add guard code to ensure weight and argument have compatible shape/type

    Decorates prox and regularization functions.
    """

    @wraps(func)
    def wrapper(x, regularization_weight):
        if isinstance(regularization_weight, np.ndarray):
            weight_shape = regularization_weight.shape
            if weight_shape != x.shape:
                raise ValueError(
                    f"Invalid shape for 'regularization_weight': "
                    f"{weight_shape}. Must be the same shape as x: {x.shape}."
                )
        elif not isinstance(regularization_weight, (int, float)):
            raise ValueError("'regularization_weight' must be a scalar")
        return func(x, regularization_weight)

    return wrapper


def get_prox(
    regularization: str,
) -> Callable[
    [NDArray[np.float64], Union[float, NDArray[np.float64]]], NDArray[np.float64]
]:
    """
    Args:
    -----
    regularization: 'l0' | 'weighted_l0' | 'l1' | 'weighted_l1' | 'l2' | 'weighted_l2'

    Returns:
    --------
    proximal_operator: (x: np.array, reg_weight: float | np.array) -> np.array
        A function that takes an input array x and a regularization weight,
        which can be either a scalar or array of the same shape,
        and returns an array of the same shape
    """

    prox = {
        "l0": _prox_l0,
        "weighted_l0": _prox_l0,
        "l1": _prox_l1,
        "weighted_l1": _prox_l1,
        "l2": _prox_l2,
        "weighted_l2": _prox_l2,
    }
    regularization = regularization.lower()
    return prox[regularization]


@_validate_prox_and_reg_inputs
def _prox_l0(
    x: NDArray[np.float64],
    regularization_weight: Union[float, NDArray[np.float64]],
):
    threshold = np.sqrt(2 * regularization_weight)
    return x * (np.abs(x) > threshold)


@_validate_prox_and_reg_inputs
def _prox_l1(
    x: NDArray[np.float64],
    regularization_weight: Union[float, NDArray[np.float64]],
):

    return np.sign(x) * np.maximum(np.abs(x) - regularization_weight, 0)


@_validate_prox_and_reg_inputs
def _prox_l2(
    x: NDArray[np.float64],
    regularization_weight: Union[float, NDArray[np.float64]],
):
    return x / (1 + 2 * regularization_weight)


@_validate_prox_and_reg_inputs
def _regularization_l0(
    x: NDArray[np.float64],
    regularization_weight: Union[float, NDArray[np.float64]],
):
    return np.sum(regularization_weight * (x != 0))


@_validate_prox_and_reg_inputs
def _regularization_l1(
    x: NDArray[np.float64],
    regularization_weight: Union[float, NDArray[np.float64]],
):
    return np.sum(regularization_weight * np.abs(x))


@_validate_prox_and_reg_inputs
def _regularization_l2(
    x: NDArray[np.float64],
    regularization_weight: Union[float, NDArray[np.float64]],
):
    return np.sum(regularization_weight * x**2)


def get_regularization(
    regularization: str,
) -> Callable[[NDArray[np.float64], Union[float, NDArray[np.float64]]], float]:
    """
    Args:
    -----
    regularization: 'l0' | 'weighted_l0' | 'l1' | 'weighted_l1' | 'l2' | 'weighted_l2'

    Returns:
    --------
    regularization_function: (x: np.array, reg_weight: float | np.array) -> np.array
        A function that takes an input array x and a regularization weight,
        which can be either a scalar or array of the same shape,
        and returns a float
    """

    regularization_fn = {
        "l0": _regularization_l0,
        "weighted_l0": _regularization_l0,
        "l1": _regularization_l1,
        "weighted_l1": _regularization_l1,
        "l2": _regularization_l2,
        "weighted_l2": _regularization_l2,
    }
    regularization = regularization.lower()
    return regularization_fn[regularization]


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


def supports_multiple_targets(estimator):
    """Checks whether estimator supports multiple targets."""
    if isinstance(estimator, MultiOutputMixin):
        return True
    try:
        return estimator._more_tags()["multioutput"]
    except (AttributeError, KeyError):
        return False
