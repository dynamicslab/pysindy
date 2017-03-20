from itertools import count

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel, check_X_y

from .util import normalize


def _sparse_coefficients(dim, ind, coef, knob):
    c = np.zeros(dim)
    c[ind] = coef
    big_ind = abs(c) > knob
    c[~big_ind] = 0
    return c, big_ind


class SINDy(LinearModel, RegressorMixin):
    def __init__(self, knob=0.01, max_iter=100, normalize=True, copy_x=False):
        self.knob = knob
        self.max_iter = max_iter
        self.fit_intercept = False
        self.normalize = normalize
        self.copy_X = copy_x

    def fit(self, x_, y):

        n_samples, n_features = x_.shape

        X, y = check_X_y(x_, y, accept_sparse=[], y_numeric=True, multi_output=False)

        x, y, X_offset, y_offset, X_scale = self._preprocess_data(
            x_, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=None)


        ind = np.ones(n_features, dtype=bool)
        coefs = np.linalg.lstsq(x, y)[0]
        new_coefs, ind = _sparse_coefficients(n_features, ind, coefs, self.knob)

        if np.count_nonzero(ind) > 0:
            for _ in range(self.max_iter):
                new_coefs = np.linalg.lstsq(x[:, ind], y)[0]
                new_coefs, ind = _sparse_coefficients(n_features, ind, new_coefs, self.knob)
                if np.allclose(new_coefs, coefs) or np.count_nonzero(ind) == 0:
                    break
                coefs = new_coefs
            else:
                pass # put warning here
        else:
            pass # put warning here

        self._set_intercept(X_offset, y_offset, X_scale)

        self.coef_ = coefs
        return self

    def pprint(self, names=None):
        fmt = "{}*{}".format
        names = names or ("x_{}".format(i) for i in count())
        expr =  "+".join(fmt(c, n) for c, n in zip(self.coef_, names) if c != 0) or "0"
        return expr
