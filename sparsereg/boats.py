import numpy as np
from sklearn.base import RegressorMixin
from sklearn.exceptions import FitFailedWarning
from sklearn.linear_model.base import LinearModel, check_X_y, _rescale_data
from sklearn.linear_model import LinearRegression


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
    print(beta_0, beta_init)
    if np.all(mask):
        raise FitFailedWarning("alpha too high: {}".format(alpha))
    beta_sel[mask] = 0
    model = lmc().fit(x[:, ~mask], y)
    beta_sel[~mask] = model.coef_

    return beta_sel, model.intercept_
