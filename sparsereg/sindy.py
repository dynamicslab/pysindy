from itertools import count
import warnings

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel, check_X_y, _rescale_data
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning

from .util import normalize, cardinality


def _sparse_coefficients(dim, ind, coef, knob):
    c = np.zeros(dim)
    c[ind] = coef
    big_ind = abs(c) > knob
    c[~big_ind] = 0
    return c, big_ind


def _regress(x, y, l1):
    if l1 != 0:
        coefs = np.linalg.lstsq(x.T @ x + l1 * np.eye(x.shape[1]), x.T @ y)[0]
    else:
        coefs = np.linalg.lstsq(x, y)[0]
    return coefs


class SINDy(LinearModel, RegressorMixin):
    def __init__(self, knob=0.01, l1=0.1, max_iter=100, normalize=True, fit_intercept=True, copy_x=True):
        self.knob = knob
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_x
        self.l1 = l1

    def fit(self, x_, y, sample_weight=None):
        l1 = self.l1
        n_samples, n_features = x_.shape

        X, y = check_X_y(x_, y, accept_sparse=[], y_numeric=True, multi_output=False)

        x, y, X_offset, y_offset, X_scale = self._preprocess_data(
            x_, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=None)

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            x, y = _rescale_data(x, y, sample_weight)

        ind = np.ones(n_features, dtype=bool)
        coefs = _regress(x, y, l1)
        new_coefs, ind = _sparse_coefficients(n_features, ind, coefs, self.knob)

        for _ in range(self.max_iter):
            if np.count_nonzero(ind) == 0:
                raise FitFailedWarning("Sparsity parameter is too big ({}) and eliminated all coeficients".format(self.knob))

            new_coefs = _regress(x[:, ind], y, l1)
            new_coefs, ind = _sparse_coefficients(n_features, ind, new_coefs, self.knob)
            if np.allclose(new_coefs, coefs) and np.count_nonzero(ind) != 0:
                break
            coefs = new_coefs
        else:
            warnings.warn("SINDy did not converge after {} iterations.".format(max_iter), ConvergenceWarning)

        coefs = _regress(x[:, ind], y, 0)  # unbias
        coefs, _ = _sparse_coefficients(n_features, ind, coefs, self.knob)
        self.coef_ = coefs
        self._set_intercept(X_offset, y_offset, X_scale)
        self.intercept_ = self.intercept_ if abs(self.intercept_) > self.knob else 0
        return self

    def pprint(self, names=None):
        fmt = "{}*{}".format
        names = names or ("x_{}".format(i) for i in count())
        expr =  "+".join(fmt(c, n) for c, n in zip(self.coef_, names) if c != 0) or "0"
        return expr

    @property
    def complexity(self):
        return np.count_nonzero(self.coef_)
