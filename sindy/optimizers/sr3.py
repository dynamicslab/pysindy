import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.exceptions import ConvergenceWarning

from sindy.optimizers import BaseOptimizer
from sindy.utils import get_prox


class SR3(BaseOptimizer):
    def __init__(
        self,
        threshold=0.1,
        nu=1.0,
        tol=1e-5,
        thresholder='l0',
        max_iter=30,
        **kwargs
    ):
        super(SR3, self).__init__(**kwargs)

        if threshold < 0:
            raise ValueError('threshold cannot be negative')
        if nu <= 0:
            raise ValueError('nu must be positive')
        if tol <= 0:
            raise ValueError('tol must be positive')
        if max_iter <= 0:
            raise ValueError('max_iter must be positive')

        self.threshold = threshold
        self.nu = nu
        self.tol = tol
        self.thresholder = thresholder
        self.prox = get_prox(thresholder)
        self.max_iter = max_iter

    def _update_full_coef(self, cho, x_transpose_y, coef_sparse):
        b = x_transpose_y + coef_sparse / self.nu
        coef_full = cho_solve(cho, b)
        self.iters += 1
        return coef_full

    def _update_sparse_coef(self, coef_full):
        coef_sparse = self.prox(coef_full, self.threshold)
        self.history_.append(coef_sparse)
        return coef_sparse

    def _convergence_criterion(self):
        this_coef = self.history_[-1]
        if len(self.history_) > 1:
            last_coef = self.history_[-2]
        else:
            last_coef = np.zeros_like(this_coef)
        return np.sum((this_coef - last_coef)**2)

    def _reduce(self, x, y):
        """
        Iterates the thresholding. Assumes an initial guess
        is saved in self.coef_ and self.ind_
        """
        coef_sparse = self.coef_
        n_samples, n_features = x.shape

        # Precompute some objects for upcoming least-squares solves.
        # Assumes that self.nu is fixed throughout optimization procedure.
        cho = cho_factor(
            np.dot(x.T, x) + np.diag(np.full(x.shape[1], 1.0 / self.nu))
        )
        x_transpose_y = np.dot(x.T, y)

        for _ in range(self.max_iter):
            coef_full = self._update_full_coef(
                cho,
                x_transpose_y,
                coef_sparse
            )
            coef_sparse = self._update_sparse_coef(coef_full)

            if self._convergence_criterion() < self.tol:
                # Could not (further) select important features
                break
        else:
            warnings.warn(
                "SR3._reduce did not converge after {} iterations.".format(
                    self.max_iter
                ),
                ConvergenceWarning,
            )

        self.coef_ = coef_sparse
        self.coef_full_ = coef_full
