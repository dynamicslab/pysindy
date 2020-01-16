from itertools import repeat
from functools import wraps

import numpy as np
from sklearn.utils.validation import check_array

# Define a special object for the default value of t in
# validate_input. Normally we would set the default
# value of t to be None, but it is possile for the user
# to pass in None, in which case validate_input performs
# no checks on t.
T_DEFAULT = object()


def validate_input(x, t=T_DEFAULT):
    if not isinstance(x, np.ndarray):
        raise ValueError("x must be array-like")
    elif x.ndim == 1:
        x = x.reshape(-1, 1)
    check_array(x)

    if t is not T_DEFAULT:
        if t is None:
            raise ValueError("t must be a scalar or array-like.")
        # Apply this check if t is a scalar
        elif np.ndim(t) == 0:
            if t <= 0:
                raise ValueError("t must be positive")
        # Only apply these tests if t is array-like
        elif isinstance(t, np.ndarray):
            if not len(t) == x.shape[0]:
                raise ValueError("Length of t should match x.shape[0].")
            if not np.all(t[:-1] < t[1:]):
                raise ValueError("Values in t should be in strictly increasing order.")
        else:
            raise ValueError("t must be a scalar or array-like.")

    return x


def drop_nan_rows(x, x_dot):
    x = x[~np.isnan(x_dot).any(axis=1)]
    x_dot = x_dot[~np.isnan(x_dot).any(axis=1)]
    return x, x_dot


def debug(func):
    """Decorator which prints function signature and return value
    whenever it is called
    """

    @wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = ["{}={}".format(k, v) for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print("Calling {}({})".format(func.__name__, signature))
        value = func(*args, **kwargs)
        print("{} returned {}".format(func.__name__, value))
        return value

    return wrapper_debug


def prox_l0(x, threshold):
    """Proximal operator for l0 regularization
    """
    return x * (np.abs(x) > threshold)


def prox_l1(x, threshold):
    """Proximal operator for l1 regularization
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def prox_cad(x, lower_threshold):
    """
    Proximal operator for CAD regularization
    prox_cad(z, a, b) =
        0                  if |z| < a
        sign(z)(|z| - a)   if a < |z| <= b
        z                  if |z| > b

    Entries of x smaller than a in magnitude are set to 0,
    entries with magnitudes larger than b are untouched,
    and entries in between have soft-thresholding applied.

    For simplicity we set b = 5*a in this implementation.
    """
    upper_threshold = 5 * lower_threshold
    return prox_l0(x, upper_threshold) + prox_l1(x, lower_threshold) * (
        np.abs(x) < upper_threshold
    )


def get_prox(regularization):
    if regularization.lower() == "l0":
        return prox_l0
    elif regularization.lower() == "l1":
        return prox_l1
    elif regularization.lower() == "cad":
        return prox_cad
    else:
        raise NotImplementedError("{} has not been implemented".format(regularization))


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


def equation(pipeline, input_features=None, precision=3, input_fmt=None):
    input_features = pipeline.steps[0][1].get_feature_names(input_features)
    if input_fmt:
        input_features = [input_fmt(i) for i in input_features]
    coef = pipeline.steps[-1][1].coef_
    intercept = pipeline.steps[-1][1].intercept_
    return print_model(coef, input_features, intercept=intercept, precision=precision)
