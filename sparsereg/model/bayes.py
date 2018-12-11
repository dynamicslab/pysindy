import warnings

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model.base import LinearModel
from sklearn.utils.validation import check_X_y

from sparsereg.model.base import PrintMixin

eps = np.finfo(np.float64).eps


def scale_sigma(est, X_offset, X_scale):
    if est.fit_intercept:
        std_intercept = np.sqrt(np.abs(X_offset @ np.diag(est.sigma_).T))
    else:
        std_intercept = 0
    sigma = np.diag(est.sigma_) / (X_scale + eps)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        std_coef = np.sqrt(sigma)
    return std_intercept, std_coef


class JMAP(LinearModel, RegressorMixin, PrintMixin):
    def __init__(
        self,
        ae0=1e-6,
        be0=1e-6,
        af0=1e-6,
        bf0=1e-6,
        max_iter=300,
        tol=1e-3,
        normalize=False,
        fit_intercept=True,
        copy_X=True,
    ):
        self.max_iter = max_iter
        self.normalize = normalize
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.tol = tol
        self.ae0 = ae0
        self.be0 = be0
        self.af0 = af0
        self.bf0 = bf0
        warnings.warn(
            f"Consider using sklearn.linear_model.BayesianRidge instead of {self.__class__.__name__}."
        )

    def fit(self, x, y):

        x, y = check_X_y(x, y, accept_sparse=[], y_numeric=True, multi_output=False)  # boilerplate

        x, y, X_offset, y_offset, X_scale = self._preprocess_data(
            x, y, fit_intercept=self.fit_intercept, normalize=self.normalize, copy=self.copy_X
        )

        fh, vf, ve, sigma = jmap(
            y, x, self.ae0, self.be0, self.af0, self.bf0, max_iter=self.max_iter, tol=self.tol
        )
        self.X_offset_ = X_offset
        self.X_scale_ = X_scale

        self.sigma_ = sigma
        self.ve_ = ve
        self.vf_ = vf
        self.coef_ = fh
        self.alpha_ = 1.0 / np.mean(ve)
        self.lambda_ = 1.0 / np.mean(vf)
        self.std_intercept_, self.std_coef_ = scale_sigma(self, X_offset, X_scale)
        self._set_intercept(X_offset, y_offset, X_scale)
        return self

    def predict(self, X, return_std=False):
        y_mean = self._decision_function(X)
        if return_std is False:
            return y_mean
        else:
            if self.normalize:
                X = (X - self.X_offset_) / self.X_scale_
            sigmas_squared_data = ((X @ self.sigma_) * X).sum(axis=1)
            y_std = np.sqrt(sigmas_squared_data + (1.0 / self.alpha_))
            return y_mean, y_std


def _converged(fhs, tol=0.1):
    if len(fhs) < 2:
        return False
    rtol = np.sum(np.abs((fhs[-1] - fhs[-2]) / fhs[-1]))
    return rtol <= tol


def jmap(g, H, ae0, be0, af0, bf0, max_iter=1000, tol=1e-4, rcond=None, observer=None):
    """Maximum a posteriori estimator for g = H @ f + e

    p(g | f) = normal(H f, ve I)
    p(ve) = inverse_gauss(ae0, be0)
    p(f | vf) = normal(0, vf I)
    p(vf) = inverse_gauss(af0, bf0)

    JMAP: maximizes p(f,ve,vf|g) = p(g | f) p(f | vf) p(ve) p(vf) / p(g)
                    with respect to f, ve and vf

    Original Author: Ali Mohammad-Djafari, April 2015

    Args:
        g:
        H:
        ae0:
        be0:
        af0:
        bf0:
        max_iter:
        rcond:

    Returns:

    """

    n_features, n_samples = H.shape

    HtH = H.T @ H
    Htg = H.T @ g
    ve0 = be0 / ae0
    vf0 = bf0 / af0
    lambda_ = ve0 / vf0
    fh, *_ = np.linalg.lstsq(HtH + lambda_ * np.eye(n_samples, n_samples), Htg, rcond=rcond)

    fhs = [fh]

    for _ in range(max_iter):
        dg = g - H @ fh

        ae = ae0 + 0.5
        be = be0 + 0.5 * dg ** 2
        ve = be / ae + eps
        iVe = np.diag(1 / ve)

        af = af0 + 0.5
        bf = bf0 + 0.5 * fh ** 2
        vf = bf / af + eps
        iVf = np.diag(1.0 / vf)

        HR = H.T @ iVe @ H + iVf
        fh, *_ = np.linalg.lstsq(HR, H.T @ iVe @ g, rcond=rcond)

        fhs.append(fh)
        if observer is not None:
            observer(fh, vf, ve)
        if _converged(fhs, tol=tol):
            break

    else:
        warnings.warn(f"jmap did not converge after {max_iter} iterations.", ConvergenceWarning)

    # sigma = np.diag(np.diag(np.linalg.inv(HR)))
    sigma = np.linalg.inv(HR)
    return fh, vf, ve, sigma
