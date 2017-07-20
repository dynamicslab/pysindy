import numpy as np
from itertools import count
import warnings

from sklearn.base import RegressorMixin
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.linear_model.base import LinearModel, _rescale_data
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sparsereg.util import cardinality


def _print_model(coef, input_features, intercept=None):
    model = "+".join("{}*{}".format(c, n) for c, n in zip(coef, input_features) if c)
    if intercept:
        model += " + {}".format(intercept)
    return model


def equation(pipeline, input_features=None):
    input_features = input_features or pipeline.steps[0][1].get_feature_names()
    coef = pipeline.steps[-1][1].coef_
    intercept = pipeline.steps[-1][1].intercept_
    return _print_model(coef, input_features, intercept)


class RationalFunctionMixin():
    def _transform(self, x, y):
        return np.hstack((x, y.reshape(-1, 1) * x))

    def fit(self, x, y):
        x, y = check_X_y(x, y, multi_output=False)
        super().fit(self._transform(x, y), y)
        l = len(self.coef_)//2
        self.coef_nominator_ = self.coef_[:l]
        self.coef_denominator_ = -self.coef_[l:]
        return self

    def predict(self, x):
        check_is_fitted(self, "coef_")
        x = check_array(x)
        return (self.intercept_ + x @ self.coef_nominator_) / (1 + x @ self.coef_denominator_)

    def print_model(self, input_features=None):
        input_features = input_features or ["x_{}".format(i) for i in range(len(self.coef_nominator_))]
        nominator = _print_model(self.coef_nominator_, input_features)
        if self.intercept_:
            nominator += "+ {}".format(self.intercept_)
        if np.any(self.coef_denominator_):
            denominator = _print_model(self.coef_denominator_, input_features, 1)
            model = "(" + nominator + ") / (" + denominator + ")"
        else:
            model = nominator
        return model


class PrintMixin:
    def print_model(self, input_features=None):
        input_features = input_features or ["x_{}".format(i) for i in range(len(self.coef_))]
        return _print_model(self.coef_, input_features, self.intercept_)


def _sparse_coefficients(dim, ind, coef, threshold):
    c = np.zeros(dim)
    c[ind] = coef
    big_ind = abs(c) > threshold
    c[~big_ind] = 0
    return c, big_ind


def _regress(x, y, alpha):
    if alpha != 0:
        coefs = np.linalg.lstsq(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)[0]
    else:
        coefs = np.linalg.lstsq(x, y)[0]
    return coefs


class STRidge(LinearModel, RegressorMixin):
    def __init__(self, threshold=0.01, alpha=0.1, max_iter=100, normalize=True, fit_intercept=True, copy_x=True, unbias=True):
        self.threshold = threshold
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_x
        self.alpha = alpha
        self.unbias = unbias

    def fit(self, x_, y, sample_weight=None):
        alpha = self.alpha
        n_samples, n_features = x_.shape

        X, y = check_X_y(x_, y, accept_sparse=[], y_numeric=True, multi_output=False)

        x, y, X_offset, y_offset, X_scale = self._preprocess_data(
            x_, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=None)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            x, y = _rescale_data(x, y, sample_weight)

        ind = np.ones(n_features, dtype=bool)
        coefs = _regress(x, y, alpha)
        new_coefs, ind = _sparse_coefficients(n_features, ind, coefs, self.threshold)
        self.iters = 0
        if self.threshold > 0:
            for _ in range(1, self.max_iter):
                if np.count_nonzero(ind) == 0:
                    warnings.warn("Sparsity parameter is too big ({}) and eliminated all coeficients".format(self.threshold))
                    coefs = np.zeros_like(coefs)
                    break

                new_coefs = _regress(x[:, ind], y, alpha)
                new_coefs, ind = _sparse_coefficients(n_features, ind, new_coefs, self.threshold)
                self.iters += 1
                if np.allclose(new_coefs, coefs) and np.count_nonzero(ind) != 0:
                    break
                coefs = new_coefs
            else:
                warnings.warn("SINDy did not converge after {} iterations.".format(self.max_iter), ConvergenceWarning)

            if self.unbias and self.alpha > 0 and np.any(ind):
                coefs = _regress(x[:, ind], y, 0)  # unbias
                coefs, _ = _sparse_coefficients(n_features, ind, coefs, self.threshold)
                self.iters += 1

        self.coef_ = coefs
        self._set_intercept(X_offset, y_offset, X_scale)
        self.coef_[abs(self.coef_) < np.finfo(float).eps] = 0
        return self

    def pprint(self, names=None):
        fmt = "{}*{}".format
        names = names or ("x_{}".format(i) for i in count())
        expr = "+".join(fmt(c, n) for c, n in zip(self.coef_, names) if c != 0) or "0"
        return expr

    @property
    def complexity(self):
        return np.count_nonzero(self.coef_)


class BoATS(LinearModel, RegressorMixin):
    def __init__(self, alpha=0.01, sigma=0.01, n=10, copy_X=True, fit_intercept=True, normalize=True):
        self.alpha = alpha
        self.sigma = sigma
        self.n = n
        self.copy_X = copy_X
        self.fit_intercept = fit_intercept
        self.normalize = normalize


    def fit(self, x_, y, sample_weight=None):
        n_samples, n_features = x_.shape

        X, y = check_X_y(x_, y, accept_sparse=[], y_numeric=True, multi_output=False)

        x, y, X_offset, y_offset, X_scale = self._preprocess_data(
            x_, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=None)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            x, y = _rescale_data(x, y, sample_weight)

        coefs, intercept = fit_with_noise(x, y, self.sigma, self.alpha, self.n,)
        self.intercept_ = intercept
        self.coef_ = coefs
        self._set_intercept(X_offset, y_offset, X_scale)
        return self


def fit_with_noise(x, y, sigma, alpha, n, lmc=LinearRegression):
    size = y.shape
    n_samples, n_features = x.shape
    beta_0 = np.mean([lmc().fit(x, y + np.random.normal(size=size, scale=sigma)).coef_ for _ in range(n)], axis=0)
    beta_init = lmc().fit(x, y).coef_

    beta_sel = beta_init.copy()
    mask = np.abs(beta_init) < alpha * np.abs(beta_0)
    if np.all(mask):
        raise FitFailedWarning("alpha too high: {}".format(alpha))
    beta_sel[mask] = 0
    model = lmc().fit(x[:, ~mask], y)
    beta_sel[~mask] = model.coef_

    return beta_sel, model.intercept_
