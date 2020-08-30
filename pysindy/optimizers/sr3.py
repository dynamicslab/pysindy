import warnings

import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from scipy.optimize import bisect
from sklearn.exceptions import ConvergenceWarning

from pysindy.optimizers import BaseOptimizer
from pysindy.utils import get_prox


class SR3(BaseOptimizer):
    """
    Sparse relaxed regularized regression.

    Attempts to minimize the objective function

    .. math::

        0.5\\|y-Xw\\|^2_2 + \\lambda \\times R(v)
        + (0.5 / \\nu)\\|w-v\\|^2_2

    where :math:`R(v)` is a regularization function. See the following reference
    for more details:

        Zheng, Peng, et al. "A unified framework for sparse relaxed
        regularized regression: Sr3." IEEE Access 7 (2018): 1404-1423.

    Parameters
    ----------
    threshold : float, optional (default 0.1)
        Determines the strength of the regularization. When the
        regularization function R is the l0 norm, the regularization
        is equivalent to performing hard thresholding, and lambda
        is chosen to threshold at the value given by this parameter.
        This is equivalent to choosing lambda = threshold^2 / (2 * nu).

    nu : float, optional (default 1)
        Determines the level of relaxation. Decreasing nu encourages
        w and v to be close, whereas increasing nu allows the
        regularized coefficients v to be farther from w.

    tol : float, optional (default 1e-5)
        Tolerance used for determining convergence of the optimization
        algorithm.

    thresholder : string, optional (default 'l0')
        Regularization function to use. Currently implemented options
        are 'l0' (l0 norm), 'l1' (l1 norm), and 'cad' (clipped
        absolute deviation).

    trimming_fraction : float, optional (default 0.0)
        Fraction of the data samples to trim during fitting. Should
        be a float between 0.0 and 1.0. If 0.0, trimming is not
        performed.

    trimming_step_size : float, optional (default 1.0)
        Step size to use in the trimming optimization procedure.

    max_iter : int, optional (default 30)
        Maximum iterations of the optimization algorithm.

    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Regularized weight vector(s). This is the v in the objective
        function.

    coef_full_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s) that are not subjected to the regularization.
        This is the w in the objective function.

    history_ : list
        History of sparse coefficients. ``history_[k]`` contains the
        sparse coefficients (v in the optimization objective function)
        at iteration k.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import SR3
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = SR3(threshold=0.1, nu=1)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -10.004 1 + 10.004 x0
    x1' = 27.994 1 + -0.993 x0 + -1.000 1 x1
    x2' = -2.662 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        threshold=0.1,
        nu=1.0,
        tol=1e-5,
        thresholder="l0",
        trimming_fraction=0.0,
        trimming_step_size=1.0,
        max_iter=30,
        normalize=False,
        fit_intercept=False,
        copy_X=True,
    ):
        super(SR3, self).__init__(
            max_iter=max_iter,
            normalize=normalize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
        )

        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        if nu <= 0:
            raise ValueError("nu must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if (trimming_fraction < 0) or (trimming_fraction > 1):
            raise ValueError("trimming fraction must be between 0 and 1")

        self.threshold = threshold
        self.nu = nu
        self.tol = tol
        self.thresholder = thresholder
        self.prox = get_prox(thresholder)
        if trimming_fraction == 0.0:
            self.use_trimming = False
        else:
            self.use_trimming = True
        self.trimming_fraction = trimming_fraction
        self.trimming_step_size = trimming_step_size

    def enable_trimming(self, trimming_fraction):
        self.use_trimming = True
        self.trimming_fraction = trimming_fraction

    def disable_trimming(self):
        self.use_trimming = False
        self.trimming_fraction = None

    def _update_full_coef(self, cho, x_transpose_y, coef_sparse):
        """Update the unregularized weight vector"""
        b = x_transpose_y + coef_sparse / self.nu
        coef_full = cho_solve(cho, b)
        self.iters += 1
        return coef_full

    def _update_sparse_coef(self, coef_full):
        """Update the regularized weight vector"""
        coef_sparse = self.prox(coef_full, self.threshold)
        self.history_.append(coef_sparse.T)
        return coef_sparse

    def _update_trimming_array(self, coef_full, trimming_array, trimming_grad):
        trimming_array = trimming_array - self.trimming_step_size * trimming_grad
        trimming_array = self.cSimplexProj(trimming_array, self.trimming_fraction)
        self.history_trimming_.append(trimming_array)
        return trimming_array

    def _trimming_grad(self, x, y, coef_full, trimming_array):
        """gradient for the trimming variable"""
        R2 = (y - x.dot(coef_full)) ** 2
        return 0.5 * np.sum(R2, axis=1)

    def _convergence_criterion(self):
        """Calculate the convergence criterion for the optimization"""
        this_coef = self.history_[-1]
        if len(self.history_) > 1:
            last_coef = self.history_[-2]
        else:
            last_coef = np.zeros_like(this_coef)
        err_coef = np.sqrt(np.sum((this_coef - last_coef) ** 2)) / self.nu
        if self.use_trimming:
            this_trimming_array = self.history_trimming_[-1]
            if len(self.history_trimming_) > 1:
                last_trimming_array = self.history_trimming_[-2]
            else:
                last_trimming_array = np.zeros_like(this_trimming_array)
            err_trimming = (
                np.sqrt(np.sum((this_trimming_array - last_trimming_array) ** 2))
                / self.trimming_step_size
            )
            return err_coef + err_trimming
        return err_coef

    def _reduce(self, x, y):
        """
        Iterates the thresholding. Assumes an initial guess
        is saved in self.coef_ and self.ind_
        """
        coef_sparse = self.coef_.T
        n_samples, n_features = x.shape

        if self.use_trimming:
            coef_full = coef_sparse.copy()
            trimming_array = np.repeat(1.0 - self.trimming_fraction, n_samples)
            self.history_trimming_ = [trimming_array]

        # Precompute some objects for upcoming least-squares solves.
        # Assumes that self.nu is fixed throughout optimization procedure.
        cho = cho_factor(np.dot(x.T, x) + np.diag(np.full(x.shape[1], 1.0 / self.nu)))
        x_transpose_y = np.dot(x.T, y)

        for _ in range(self.max_iter):
            if self.use_trimming:
                x_weighted = x * trimming_array.reshape(n_samples, 1)
                cho = cho_factor(
                    np.dot(x_weighted.T, x)
                    + np.diag(np.full(x.shape[1], 1.0 / self.nu))
                )
                x_transpose_y = np.dot(x_weighted.T, y)
                trimming_grad = 0.5 * np.sum((y - x.dot(coef_full)) ** 2, axis=1)
            coef_full = self._update_full_coef(cho, x_transpose_y, coef_sparse)
            coef_sparse = self._update_sparse_coef(coef_full)
            if self.use_trimming:
                trimming_array = self._update_trimming_array(
                    coef_full, trimming_array, trimming_grad
                )

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

        self.coef_ = coef_sparse.T
        self.coef_full_ = coef_full.T
        if self.use_trimming:
            self.trimming_array = trimming_array

    @staticmethod
    def cSimplexProj(trimming_array, trimming_fraction):
        """projected onto the capped simplex"""
        a = np.min(trimming_array) - 1.0
        b = np.max(trimming_array) - 0.0

        def f(x):
            return (
                np.sum(np.maximum(np.minimum(trimming_array - x, 1.0), 0.0))
                - (1.0 - trimming_fraction) * trimming_array.size
            )

        x = bisect(f, a, b)

        return np.maximum(np.minimum(trimming_array - x, 1.0), 0.0)
