from itertools import count

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score


def _sparse_coefficients(dim, ind, coef, knob):
    c = np.zeros(dim)
    c[ind] = coef
    big_ind = abs(c) > knob
    c[~big_ind] = 0
    return c, big_ind


class SINDy(BaseEstimator):
    def __init__(self, knob=0.01, max_iter=100, normalize=2):
        self.knob = knob
        self.max_iter = max_iter
        self.normalize = normalize

    def fit(self, x_, y):

        n_samples, n_features = x_.shape

        if self.normalize:
            x, m = normalize(x, order=self.normalize)
        else:
            x, m = x_, np.ones(n_features)
        
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

        self.coef_ = m*coefs
        return self

    def predict(self, x):
        return x @ self.coef_

    def score(self, x, y):
        yhat = self.predict(x)
        return r2_score(y, yhat)

    def pprint(self, names=None):
        fmt = "{}*{}".format
        names = names or ("x_{}".format(i) for i in count())
        expr =  "+".join(fmt(c, n) for c, n in zip(self.coef_, names) if c != 0) or "0"
        return expr
