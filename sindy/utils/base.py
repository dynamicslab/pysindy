import warnings
from itertools import repeat
from functools import wraps

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import FitFailedWarning
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ridge_regression
from sklearn.linear_model.base import _rescale_data
from sklearn.linear_model.base import LinearModel
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y


def validate_input(x):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    check_array(x)
    return x


def drop_nan_rows(x, x_dot):
    x = x[~np.isnan(x_dot).any(axis=1)]
    x_dot = x_dot[~np.isnan(x_dot).any(axis=1)]
    return x, x_dot


def debug(func):
    """Print the function signature and return value"""
    @wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = ["{}={}".format(k, v) for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(
            "Calling {}({})".format(func.__name__, signature)
        )
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
    return (
        prox_l0(x, upper_threshold)
        + prox_l1(x, lower_threshold)
        * (np.abs(x) < upper_threshold)
    )


def get_prox(regularization):
    if regularization.lower() == 'l0':
        return prox_l0
    elif regularization.lower() == 'l1':
        return prox_l1
    elif regularization.lower() == 'cad':
        return prox_cad
    else:
        raise NotImplementedError(
            '{} has not been implemented'.format(regularization)
        )


def print_model(coef, input_features, errors=None, intercept=None, error_intercept=None, precision=3, pm="±"):
    """

    Args:
        coef:
        input_features:
        errors:
        intercept:
        error_intercept:
        precision:
        pm:

    Returns:

    """

    def term(coef, sigma, name):
        rounded_coef = np.round(coef, precision)
        if name == "1":
            name = ""
        if rounded_coef == 0 and sigma is None:
            return ""
        elif sigma is None:
            return f"{coef:.{precision}f} {name}"
        elif rounded_coef == 0 and np.round(sigma, precision) == 0:
            return ""
        else:
            return f"({coef:.{precision}f} {pm} {sigma:.{precision}f}) {name}"

    errors = errors if errors is not None else repeat(None)
    components = map(term, coef, errors, input_features)
    eq = " + ".join(filter(bool, components))

    if not eq or intercept or error_intercept is not None:
        intercept = intercept or 0
        if eq:
            eq += " + "
        eq += term(intercept, error_intercept, "").strip() or f"{intercept:.{precision}f}"

    return eq


def equation(pipeline, input_features=None, precision=3, input_fmt=None):
    input_features = pipeline.steps[0][1].get_feature_names(input_features)
    if input_fmt:
        input_features = [input_fmt(i) for i in input_features]
    coef = pipeline.steps[-1][1].coef_
    intercept = pipeline.steps[-1][1].intercept_
    return print_model(coef, input_features, intercept=intercept, precision=precision)


class RationalFunctionMixin:
    def _transform(self, x, y):
        return np.hstack((x, y.reshape(-1, 1) * x))

    def fit(self, x, y, **kwargs):
        # x, y = check_X_y(x, y, multi_output=False)
        super().fit(self._transform(x, y), y, **kwargs)
        self._arrange_coef()
        return self

    def _arrange_coef(self):
        nom = len(self.coef_) // 2
        self.coef_nominator_ = self.coef_[:nom]
        self.coef_denominator_ = -self.coef_[nom:]

    def predict(self, x):
        check_is_fitted(self, "coef_")
        x = check_array(x)
        return (self.intercept_ + x @ self.coef_nominator_) / (1 + x @ self.coef_denominator_)

    def print_model(self, input_features=None):
        input_features = input_features or ["x_{}".format(i) for i in range(len(self.coef_nominator_))]
        nominator = print_model(self.coef_nominator_, input_features)
        if self.intercept_:
            nominator += "+ {}".format(self.intercept_)
        if np.any(self.coef_denominator_):
            denominator = print_model(self.coef_denominator_, input_features, 1)
            model = "(" + nominator + ") / (" + denominator + ")"
        else:
            model = nominator
        return model


class PrintMixin:
    def print_model(self, input_features=None, precision=3):
        input_features = input_features or ["x_{}".format(i) for i in range(len(self.coef_))]
        return print_model(self.coef_, input_features, intercept=self.intercept_, precision=precision)
