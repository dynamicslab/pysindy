import warnings

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model.base import _rescale_data
from sklearn.linear_model.base import LinearModel
from sklearn.utils.validation import check_X_y

from sparsereg.model.base import PrintMixin


# todo use meta class for boilerplate
class JMAP(LinearModel, RegressorMixin, PrintMixin):
    def __init__(
        self,
        ae0=0.1,
        be0=0.1,
        af0=0.1,
        bf0=0.1,
        max_iter=1000,
        tol=1e-4,
        normalize=False,
        fit_intercept=True,
        copy_X=True,
    ):
        self.max_iter = max_iter
        self.normalize = normalize
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.tol = tol  # boilerplate
        self.ae0 = ae0
        self.be0 = be0
        self.af0 = af0
        self.bf0 = bf0

    def fit(self, x, y, sample_weight=None):

        x, y = check_X_y(x, y, accept_sparse=[], y_numeric=True, multi_output=False)  # boilerplate

        x, y, X_offset, y_offset, X_scale = self._preprocess_data(
            x,
            y,
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            copy=self.copy_X,
            sample_weight=sample_weight,
        )  # boilerplate

        if sample_weight is not None:
            x, y = _rescale_data(x, y, sample_weight)  # boilerplate

        fh, vf, ve, sigma2 = jmap(
            y, x, self.ae0, self.be0, self.af0, self.bf0, max_iter=self.max_iter, tol=self.tol
        )
        self.sigma2_ = sigma2
        self.vf_ = vf
        self.ve_ = ve
        self.coef_ = fh
        self._set_intercept(X_offset, y_offset, X_scale)
        return self


def _converged(fhs, tol=0.1):
    if len(fhs) < 2:
        return False
    rtol = np.mean(np.abs((fhs[-1] - fhs[-2]) / fhs[-1]))
    return rtol <= tol


def jmap(g, H, ae0, be0, af0, bf0, max_iter=1000, tol=1e-4, rcond=None, observer=None):
    """Maximum a posteriori estimator for g = H @ f + e

    p(g | f) = N(f | H f, ve I)
    p(ve) = IG(ve | ae0, be0)
    p(f | vf) = N(f | 0, vf I)
    p(vf) = IG(ve | ae0, be0)

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
        gh = H @ fh
        dg = g - gh

        ae = ae0 + 0.5
        be = be0 + 0.5 * dg ** 2
        ve = be / ae
        iVe = np.diag(1 / ve)

        af = af0 + 0.5
        bf = bf0 + 0.5 * fh ** 2
        vf = bf / af
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

    sigma2 = np.diag(np.linalg.inv(HR))
    return fh, vf, ve, sigma2
