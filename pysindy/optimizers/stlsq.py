import warnings

import numpy as np
from sklearn.linear_model import ridge_regression
from sklearn.exceptions import ConvergenceWarning

from pysindy.optimizers import BaseOptimizer


class STLSQ(BaseOptimizer):
    """
    Sequentially thresholded least squares algorithm.

    Attempts to minimize the objective function
    ||y - Xw||^2_2 + alpha * ||w||^2_2 by iteratively performing
    least squares and masking out elements of the weight that are
    below a given threshold.

    Parameters
    ----------
    threshold : float, optional (default 0.1)
        Minimum magnitude for a coefficient in the weight vector.
        Coefficients with magnitude below the threshold are set
        to zero.

    alpha : float, optional (default 0)
        Optional L2 (ridge) regularization on the weight vector.

    max_iter : int, optional (default 20)
        Maximum iterations of the optimization algorithm.

    ridge_kw : dict, optional
        Optional keyword arguments to pass to the ridge regression.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s)

    ind_ : array, shape (n_features,) or (n_targets, n_features)
        Array of 0s and 1s indicating which coefficients of the
        weight vector have not been masked out.
    """

    def __init__(
        self, threshold=0.1, alpha=0.0, max_iter=20, ridge_kw=None, **kwargs
    ):
        super(STLSQ, self).__init__(**kwargs)

        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        if alpha < 0:
            raise ValueError("alpha cannot be negative")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")

        self.threshold = threshold
        self.alpha = alpha
        self.max_iter = max_iter
        self.ridge_kw = ridge_kw

    def _sparse_coefficients(self, dim, ind, coef, threshold):
        """Perform thresholding of the weight vector(s)
        """
        c = np.zeros(dim)
        c[ind] = coef
        big_ind = np.abs(c) >= threshold
        c[~big_ind] = 0
        self.history_.append(c)
        return c, big_ind

    def _regress(self, x, y):
        """Perform the ridge regression
        """
        kw = self.ridge_kw or {}
        coef = ridge_regression(x, y, self.alpha, **kw)
        self.iters += 1
        return coef

    def _no_change(self):
        """Check if the coefficient mask has changed after thresholding
        """
        this_coef = self.history_[-1]
        if len(self.history_) > 1:
            last_coef = self.history_[-2]
        else:
            last_coef = np.zeros_like(this_coef)
        return all(bool(i) == bool(j) for i, j in zip(this_coef, last_coef))

    def _reduce(self, x, y):
        """
        Iterates the thresholding. Assumes an initial guess is saved in
        self.coef_ and self.ind_
        """
        ind = self.ind_
        n_samples, n_features = x.shape
        n_features_selected = sum(ind)

        for _ in range(self.max_iter):
            if np.count_nonzero(ind) == 0:
                warnings.warn(
                    """Sparsity parameter is too big ({}) and eliminated all
                    coeficients""".format(
                        self.threshold
                    )
                )
                coef = np.zeros_like(ind, dtype=float)
                break

            coef = self._regress(x[:, ind], y)
            coef, ind = self._sparse_coefficients(
                n_features, ind, coef, self.threshold
            )

            if sum(ind) == n_features_selected or self._no_change():
                # could not (further) select important features
                break
        else:
            warnings.warn(
                "STLSQ._reduce did not converge after {} iterations.".format(
                    self.max_iter
                ),
                ConvergenceWarning,
            )
            try:
                coef
            except NameError:
                coef = self.coef_
                warnings.warn(
                    "STLSQ._reduce has no iterations left to determine coef",
                    ConvergenceWarning,
                )
        self.coef_ = coef
        self.ind_ = ind
