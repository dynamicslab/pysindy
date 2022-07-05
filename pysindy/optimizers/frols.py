import warnings

import numpy as np
from sklearn.linear_model import ridge_regression

from .base import BaseOptimizer


class FROLS(BaseOptimizer):
    """Forward Regression Orthogonal Least-Squares (FROLS) optimizer.

    Attempts to minimize the objective function
    :math:`\\|y - Xw\\|^2_2 + \\alpha \\|w\\|^2_2`
    by iteractively selecting the most correlated
    function in the library. This is a greedy algorithm.

    See the following reference for more details:

        Billings, Stephen A. Nonlinear system identification:
        NARMAX methods in the time, frequency, and spatio-temporal domains.
        John Wiley & Sons, 2013.

    Parameters
    ----------
    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    normalize_columns : boolean, optional (default False)
        Normalize the columns of x (the SINDy library terms) before regression
        by dividing by the L2-norm. Note that the 'normalize' option in sklearn
        is deprecated in sklearn versions >= 1.0 and will be removed.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    kappa : float, optional (default None)
        If passed, compute the MSE errors with an extra L0 term with
        strength equal to kappa times the condition number of Theta.

    max_iter : int, optional (default 10)
        Maximum iterations of the optimization algorithm. This determines
        the number of nonzero terms chosen by the FROLS algorithm.

    alpha : float, optional (default 0.05)
        Optional L2 (ridge) regularization on the weight vector.

    ridge_kw : dict, optional (default None)
        Optional keyword arguments to pass to the ridge regression.

    verbose : bool, optional (default False)
        If True, prints out the different error terms every
        iteration.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s).

    history_ : list
        History of ``coef_``. ``history_[k]`` contains the values of
        ``coef_`` at iteration k of FROLS.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import FROLS
    >>> lorenz = lambda z,t : [10 * (z[1] - z[0]),
    >>>                        z[0] * (28 - z[2]) - z[1],
    >>>                        z[0] * z[1] - 8 / 3 * z[2]]
    >>> t = np.arange(0, 2, .002)
    >>> x = odeint(lorenz, [-8, 8, 27], t)
    >>> opt = FROLS(threshold=.1, alpha=.5)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1] - t[0])
    >>> model.print()
    x0' = -9.999 1 + 9.999 x0
    x1' = 27.984 1 + -0.996 x0 + -1.000 1 x1
    x2' = -2.666 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        normalize_columns=False,
        fit_intercept=False,
        copy_X=True,
        kappa=None,
        max_iter=10,
        alpha=0.05,
        ridge_kw=None,
        verbose=False,
    ):
        super(FROLS, self).__init__(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            max_iter=max_iter,
            normalize_columns=normalize_columns,
        )
        self.alpha = alpha
        self.ridge_kw = ridge_kw
        self.kappa = kappa
        if self.max_iter <= 0:
            raise ValueError("Max iteration must be > 0")
        self.verbose = verbose

    def _normed_cov(self, a, b):
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            try:
                return np.vdot(a, b) / np.vdot(a, a)
            except RuntimeWarning:
                raise ValueError(
                    "Trying to orthogonalize linearly dependent columns created NaNs"
                )

    def _select_function(self, x, y, sigma, skip=[]):
        n_features = x.shape[1]
        g = np.zeros(n_features)  # Coefficients to orthogonalized functions
        ERR = np.zeros(n_features)  # Error Reduction Ratio at this step
        for m in range(n_features):
            if m not in skip:
                g[m] = self._normed_cov(x[:, m], y)
                ERR[m] = (
                    abs(g[m]) ** 2 * np.real(np.vdot(x[:, m], x[:, m])) / sigma
                )  # Error reduction

        best_idx = np.argmax(
            ERR
        )  # Select best function by maximum Error Reduction Ratio

        # Return index of best function, along with ERR and coefficient
        return best_idx, ERR[best_idx], g[best_idx]

    def _orthogonalize(self, vec, Q):
        """
        Orthogonalize vec with respect to columns of Q
        """
        Qs = vec.copy()
        s = Q.shape[1]
        for r in range(s):
            Qs -= self._normed_cov(Q[:, r], Qs) * Q[:, r]
        return Qs

    def _reduce(self, x, y):
        """
        Performs at most n_feature iterations of the
        greedy Forward Regression Orthogonal Least Squares (FROLS) algorithm
        """
        n_samples, n_features = x.shape
        n_targets = y.shape[1]
        cond_num = np.linalg.cond(x)
        if self.kappa is not None:
            l0_penalty = self.kappa * cond_num
        else:
            l0_penalty = 0.0

        # Print initial values for each term in the optimization
        if self.verbose:
            row = [
                "Iteration",
                "Index",
                "|y - Xw|^2",
                "a * |w|_2",
                "|w|_0",
                "b * |w|_0",
                "Total: |y-Xw|^2+a*|w|_2+b*|w|_0",
            ]
            print(
                "{: >10} ... {: >5} ... {: >10} ... {: >10} ... {: >5}"
                " ... {: >10} ... {: >10}".format(*row)
            )
            print(
                " Note that these error metrics are related but not the same"
                " as the loss function being minimized by FROLS!"
            )

        # History of selected functions: [iteration x output x coefficients]
        self.history_ = np.zeros((n_features, n_targets, n_features), dtype=x.dtype)
        # Error Reduction Ratio: [iteration x output]
        self.ERR_global = np.zeros((n_features, n_targets))
        for k in range(n_targets):
            # Initialize arrays
            g_global = np.zeros(
                n_features, dtype=x.dtype
            )  # Coefficients for selected (orthogonal) functions
            L = np.zeros(
                n_features, dtype=int
            )  # Order of selection, i.e. L[0] is the first, L[1] second...
            A = np.zeros(
                (n_features, n_features), dtype=x.dtype
            )  # Used for inversion to original function set (A @ coef = g)

            # Orthogonal function libraries
            Q = np.zeros_like(x)  # Global library (built over time)
            Qs = np.zeros_like(x)  # Same, but for each step
            sigma = np.real(np.vdot(y[:, k], y[:, k]))  # Variance of the signal
            for i in range(n_features):
                for m in range(n_features):
                    if m not in L[:i]:
                        # Orthogonalize with respect to already selected functions
                        Qs[:, m] = self._orthogonalize(x[:, m], Q[:, :i])

                L[i], self.ERR_global[i, k], g_global[i] = self._select_function(
                    Qs, y[:, k], sigma, L[:i]
                )

                # Store transformation from original functions to orthogonal library
                for j in range(i):
                    A[j, i] = self._normed_cov(Q[:, j], x[:, L[i]])
                A[i, i] = 1.0
                Q[:, i] = Qs[:, L[i]].copy()
                Qs *= 0.0

                # Invert orthogonal coefficient vector
                # to get coefficients for original functions
                # at this step of the iteration
                if i != 0:
                    kw = self.ridge_kw or {}
                    alpha = ridge_regression(A[:i, :i], g_global[:i], self.alpha, **kw)
                else:
                    alpha = []

                # Coefficients for the library functions
                coef_k = np.zeros_like(g_global)
                coef_k[L[:i]] = alpha
                coef_k[abs(coef_k) < 1e-10] = 0

                self.history_[i, k, :] = np.copy(coef_k)

                if i >= self.max_iter:
                    break
                if self.verbose:
                    coef = self.history_[i, k, :]
                    R2 = np.sum((y[:, k] - np.dot(x, coef).T) ** 2)
                    L2 = self.alpha * np.sum(coef**2)
                    L0 = np.count_nonzero(coef)
                    row = [i, k, R2, L2, L0, l0_penalty * L0, R2 + L2 + l0_penalty * L0]
                    print(
                        "{0:10d} ... {1:5d} ... {2:10.4e} ... {3:10.4e}"
                        " ... {4:5d} ... {5:10.4e} ... {6:10.4e}".format(*row)
                    )

        # Function selection: L2 error for output k at iteration i is given by
        # sum(ERR_global[k, :i]), and the number of nonzero coefficients is (i+1)
        l2_err = np.cumsum(self.ERR_global, axis=1)
        l0_norm = np.arange(1, n_features + 1)[:, None]
        self.loss_ = l2_err + l0_penalty * l0_norm  # Save for debugging

        # For each output, choose the minimum L0-penalized loss function
        self.coef_ = np.zeros((n_targets, n_features), dtype=x.dtype)
        loss_min = np.argmin(
            self.loss_[: self.max_iter + 1, :], axis=0
        )  # Minimum loss for this output function
        for k in range(n_targets):
            self.coef_[k, :] = self.history_[loss_min[k], k, :]
