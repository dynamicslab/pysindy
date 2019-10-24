import warnings
from itertools import repeat

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
