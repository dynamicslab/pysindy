import numpy as np

from sklearn.base import BaseEstimator
from sklearn.exceptions import FitFailedWarning


def exclude_by_variability(coef, threshold):
    """
    Variability measurement is given by f = std(c) / mean(c).
    If the model is sensitive to c, than f is big.

    :params coef: array of coeficients of linear model (n_runs, single_coef_shape)
    :param threshold:
    """
    with np.errstate(invalid="ignore"):
        variability = np.std(coef, axis=0) / np.mean(coef, axis=0)
    variability[np.isnan(variability)] = 0

    new_coef = np.mean(coef, axis=0)
    mask = abs(variability) > threshold
    new_coef[mask] = 0
    return new_coef


class ReducedLinearModel(BaseEstimator):
    def __init__(self, mask, lm):
        self.mask = mask
        self.lm = lm

    def fit(self, x, y):
        mask = self.mask
        if not x.shape[1] == mask.shape[0]:
            raise FitFailedWarning

        self.lm = self.lm.fit(x[:, mask], y)
        self.coef_ = np.zeros(shape=mask.shape)
        self.coef_[mask] = self.lm.coef_
        return self

    def predict(self, x):
        return self.lm.predict(x[:, self.mask])

    def scores(self, x, y):
        return self.lm.scores(x[:, self.mask], y)


def fit_with_noise(x, y, lm, sigma=0.05, n=10):

    scale = np.max(np.abs(y)) * sigma
    size = y.shape

    n_samples, n_features = x.shape

    c = [lm.fit(x, y + np.random.normal(size=size, scale=scale)).coef_ for _ in range(n)]
    coef_ = exclude_by_variability(c, sigma*10)

    mask = coef_ != 0
    if not np.any(mask):
        raise FitFailedWarning

    model = ReducedLinearModel(mask, lm).fit(x, y)
    return model
