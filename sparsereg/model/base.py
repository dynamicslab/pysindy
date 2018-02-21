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
    if intercept or not model:
        model += " + {}".format(intercept)
    return model


def equation(pipeline, input_features=None, precision=3):
    input_features = pipeline.steps[0][1].get_feature_names(input_features)
    coef = np.round(pipeline.steps[-1][1].coef_, precision)
    intercept = np.round(pipeline.steps[-1][1].intercept_, precision)
    return _print_model(coef, input_features, intercept)


class RationalFunctionMixin:
    def _transform(self, x, y):
        return np.hstack((x, y.reshape(-1, 1) * x))

    def fit(self, x, y, **kwargs):
        #x, y = check_X_y(x, y, multi_output=False)
        super().fit(self._transform(x, y), y, **kwargs)
        self._arrange_coef()
        return self

    def _arrange_coef(self):
        l = len(self.coef_)//2
        self.coef_nominator_ = self.coef_[:l]
        self.coef_denominator_ = -self.coef_[l:]

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


class STRidge(LinearModel, RegressorMixin):
    def __init__(self, threshold=0.01, alpha=0.1, max_iter=100, normalize=True,
                fit_intercept=True, threshold_intercept=False, copy_x=True, unbias=True, _rcond=-1):
        self.threshold = threshold
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.threshold_intercept = threshold_intercept
        self.normalize = normalize
        self.copy_X = copy_x
        self.alpha = alpha
        self.unbias = unbias
        self._rcond= _rcond

        self.history_ = []

    def _sparse_coefficients(self, dim, ind, coef):
        c = np.zeros(dim)
        c[ind] = coef
        big_ind = np.abs(c) >= self.threshold
        c[~big_ind] = 0
        self.history_.append(c)
        return c, big_ind

    def _regress(self, x, y, alpha):
        if alpha != 0:
            coef = np.linalg.lstsq(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y, rcond=self._rcond)[0]
        else:
            coef = np.linalg.lstsq(x, y, rcond=self._rcond)[0]
        self.iters += 1
        return coef

    def _no_change(self):
        this_coef = self.history_[-1]
        if len(self.history_) > 1:
            last_coef = self.history_[-2]
        else:
            last_coef = np.zeros_like(this_coef)
        return all(bool(i) == bool(j) for i,j in zip(this_coef, last_coef))

    def _reduce(self, x, y):
        """Iterates the thresholding. Assumes an initial guess is saved in self.coef_ and self.ind_"""
        ind = self.ind_
        n_samples, n_features = x.shape
        n_features_selected = sum(ind)

        for _ in range(self.iters, self.max_iter):
            if np.count_nonzero(ind) == 0:
                warnings.warn("Sparsity parameter is too big ({}) and eliminated all coeficients".format(self.threshold))
                coef = np.zeros_like(ind, dtype=float)
                break

            coef = self._regress(x[:, ind], y, self.alpha)
            coef, ind = self._sparse_coefficients(n_features, ind, coef)

            if sum(ind) == n_features_selected or self._no_change():
                # could not (further) select important features
                break
        else:
            warnings.warn("STRidge._reduce did not converge after {} iterations.".format(self.max_iter), ConvergenceWarning)
            try:
                coef
            except NameError:
                coef = self.coef_
                warnings.warn("STRidge._reduce has no iterations left to determine coef", ConvergenceWarning)
        self.coef_ = coef
        self.ind_ = ind

    def _unbias(self, x, y):
        if np.any(self.ind_):
            coef = self._regress(x[:, self.ind_], y, 0)
            self.coef_, self.ind_ = self._sparse_coefficients(x.shape[1], self.ind_, coef)

    def fit(self, x_, y, sample_weight=None):
        x_, y = check_X_y(x_, y, accept_sparse=[], y_numeric=True, multi_output=False)

        x, y, X_offset, y_offset, X_scale = self._preprocess_data(
            x_, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=sample_weight)

        if sample_weight is not None:
            x, y = _rescale_data(x, y, sample_weight)

        self.iters = 0
        self.ind_ = np.ones(x.shape[1], dtype=bool) # initial guess
        if self.threshold > 0:
            self._reduce(x, y)
        else:
            self.coef_ = self._regress(x[:, self.ind_], y, self.alpha)

        if self.unbias and self.alpha >= 0:
            self._unbias(x, y)

        self._set_intercept(X_offset, y_offset, X_scale)
        if self.threshold_intercept and abs(self.intercept_) < self.threshold:
            self.intercept_ = 0
        return self

    @property
    def complexity(self):
        return np.count_nonzero(self.coef_) + np.count_nonzero([abs(self.intercept_) >= self.threshold])


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

        coefs, intercept = fit_with_noise(x, y, self.sigma, self.alpha, self.n)
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
