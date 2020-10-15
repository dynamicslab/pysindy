import warnings

import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from sklearn.exceptions import ConvergenceWarning

from ..utils import capped_simplex_projection
from ..utils import get_prox
from ..utils import get_regularization
from .base import BaseOptimizer


class ConstrainedSR3(BaseOptimizer):
    """
    Sparse relaxed regularized regression with linear inequality constraints.

    Attempts to minimize the objective function

    .. math::

        0.5\\|y-Xw\\|^2_2 + \\lambda \\times R(v)
        + (0.5 / nu)\\|w-v\\|^2_2

        subject to

    .. math::

        Cw <= d

    over v and w where :math:`R(v)` is a regularization function, C is a
    constraint matrix, and d is a vector of values. See the following
    reference for more details:

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
        are 'l0' (l0 norm), 'l1' (l1 norm), 'cad' (clipped
        absolute deviation), 'weighted_l0' (weighted l0 norm), and
        'weighted_l1' (weighted l1 norm).

    max_iter : int, optional (default 30)
        Maximum iterations of the optimization algorithm.

    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    constraint_lhs : 2D numpy array, shape (n_constraints, n_features * n_targets)
        The left hand side matrix C of Cx <= d.

    constraint_rhs : 1D numpy array, shape (n_constraints,)
        The right hand side vector d of Cx <= d

    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    initial_guess : np.ndarray, shape (n_features) or (n_targets, n_features), \
                optional (default None)
        Initial guess for coefficients ``coef_``, (v in the mathematical equations)
        If None, least-squares is used to obtain an initial guess.

    unbias : boolean, optional (default True)
        Whether to perform an extra step of unregularized linear regression to unbias
        the coefficients for the identified support.
        For example, if `STLSQ(alpha=0.1)` is used then the learned coefficients will
        be biased toward 0 due to the L2 regularization.
        Setting `unbias=True` will trigger an additional step wherein the nonzero
        coefficients learned by the `STLSQ` object will be updated using an
        unregularized least-squares fit.
        `unbias` is automatically set to False if a constraint is used.

    thresholds : np.ndarray, shape (n_targets, n_features), optional \
            (default None)
        Array of thresholds for each library function coefficient.
        Each row corresponds to a measurement variable and each column
        to a function from the feature library.
        Recall that SINDy seeks a matrix :math:`\\Xi` such that
        :math:`\\dot{X} \\approx \\Theta(X)\\Xi`.
        ``thresholds[i, j]`` should specify the threshold to be used for the
        (j + 1, i + 1) entry of :math:`\\Xi`. That is to say it should give the
        threshold to be used for the (j + 1)st library function in the equation
        for the (i + 1)st measurement variable.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Regularized weight vector(s). This is the v in the objective
        function.

    coef_full_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s) that are not subjected to the regularization.
        This is the w in the objective function.
    """

    def __init__(
        self,
        threshold=0.1,
        nu=1.0,
        tol=1e-5,
        thresholder="l0",
        max_iter=30,
        trimming_fraction=0.0,
        trimming_initialization=None,
        trimming_step_size=1.0,
        constraint_lhs=None,
        constraint_rhs=None,
        normalize=False,
        fit_intercept=False,
        copy_X=True,
        initial_guess=None,
        thresholds=None,
    ):
        super(ConstrainedSR3, self).__init__(
            max_iter=max_iter,
            initial_guess=initial_guess,
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
        if thresholder[:8].lower() == "weighted" and thresholds is None:
            raise ValueError(
                "weighted thresholder requires the thresholds parameter to be used"
            )
        if thresholder[:8].lower() != "weighted" and thresholds is not None:
            raise ValueError(
                "The thresholds argument cannot be used without a weighted thresholder,"
                " e.g. thresholder='weighted_l0'"
            )
        if thresholds is not None and np.any(thresholds < 0):
            raise ValueError("thresholds cannot contain negative entries")

        self.threshold = threshold
        self.thresholds = thresholds
        self.nu = nu
        self.tol = tol
        self.thresholder = thresholder
        self.prox = get_prox(thresholder)
        self.reg = get_regularization(thresholder)
        self.use_constraints = (constraint_lhs is not None) and (
            constraint_rhs is not None
        )

        if self.use_constraints:
            self.n_constraints = constraint_lhs.shape[0]
            self.constraint_lhs = constraint_lhs
            self.constraint_rhs = constraint_rhs
            self.unbias = False

        if trimming_fraction == 0.0:
            self.use_trimming = False
        else:
            self.use_trimming = True
            self.trimming_fraction = trimming_fraction
            self.trimming_initialization = trimming_initialization
            self.trimming_step_size = trimming_step_size

    def enable_trimming(self, trimming_fraction):
        """
        Enable the trimming of potential outliers.

        Parameters
        ----------
        trimming_fraction: float
            The fraction of samples to be trimmed.
            Must be between 0 and 1.
        """
        self.use_trimming = True
        self.trimming_fraction = trimming_fraction

    def disable_trimming(self):
        """Disable trimming of potential outliers."""
        self.use_trimming = False
        self.trimming_fraction = None

    def _update_full_coef(self, cho, x_transpose_y, coef_sparse):
        """Update the unregularized weight vector"""
        b = x_transpose_y + coef_sparse / self.nu
        coef_full = cho_solve(cho, b)
        self.iters += 1
        return coef_full

    def _update_full_coef_constraints(self, H, x_transpose_y, coef_sparse):
        g = x_transpose_y + coef_sparse / self.nu
        inv1 = np.linalg.inv(H)
        inv1_mod = np.kron(inv1, np.eye(coef_sparse.shape[1]))
        inv2 = np.linalg.inv(
            self.constraint_lhs.dot(inv1_mod).dot(self.constraint_lhs.T)
        )

        rhs = g.flatten() + self.constraint_lhs.T.dot(inv2).dot(
            self.constraint_rhs - self.constraint_lhs.dot(inv1_mod).dot(g.flatten())
        )
        rhs = rhs.reshape(g.shape)
        return inv1.dot(rhs)

    def _update_sparse_coef(self, coef_full):
        """Update the regularized weight vector"""
        coef_sparse = np.zeros(np.shape(coef_full))
        if self.thresholds is None:
            coef_sparse = self.prox(coef_full, self.threshold)
        else:
            coef_sparse = self.prox(coef_full, self.thresholds)
        self.history_.append(coef_sparse.T)
        return coef_sparse

    def _update_trimming_array(self, coef_full, trimming_array, trimming_grad):
        trimming_array = trimming_array - self.trimming_step_size * trimming_grad
        trimming_array = capped_simplex_projection(
            trimming_array, self.trimming_fraction
        )
        self.history_trimming_.append(trimming_array)
        return trimming_array

    def _trimming_grad(self, x, y, coef_full, trimming_array):
        """Gradient for the trimming variable"""
        R2 = (y - x.dot(coef_full)) ** 2
        return 0.5 * np.sum(R2, axis=1)

    def _objective(self, x, y, coef_full, coef_sparse, trimming_array=None):
        """Objective function"""
        R2 = (y - np.dot(x, coef_full)) ** 2
        D2 = (coef_full - coef_sparse) ** 2
        if self.use_trimming:
            assert trimming_array is not None
            R2 *= trimming_array.reshape(x.shape[0], 1)
        if self.thresholds is None:
            return (
                0.5 * np.sum(R2)
                + self.reg(coef_full, 0.5 * self.threshold ** 2 / self.nu)
                + 0.5 * np.sum(D2) / self.nu
            )
        else:
            return (
                0.5 * np.sum(R2)
                + self.reg(coef_full, 0.5 * self.thresholds ** 2 / self.nu)
                + 0.5 * np.sum(D2) / self.nu
            )

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
        Perform at most ``self.max_iter`` iterations of the SR3 algorithm
        with inequality constraints.

        Assumes initial guess for coefficients is stored in ``self.coef_``.
        """
        coef_sparse = self.coef_.T
        n_samples, n_features = x.shape

        if self.use_trimming:
            coef_full = coef_sparse.copy()
            assert self.trimming_fraction != 0
            if self.trimming_initialization is None:
                self.trimming_initialization = np.repeat(
                    self.trimming_fraction, n_samples
                )
            assert (
                np.abs(
                    np.sum(self.trimming_initialization)
                    - self.trimming_fraction * n_samples
                )
                <= 1e-6 * n_samples
            )
            assert np.all(
                (self.trimming_initialization >= 0.0)
                & (self.trimming_initialization <= 1.0)
            )
            trimming_array = self.trimming_initialization.copy()
            self.history_trimming_ = [trimming_array]

        # Precompute some objects for upcoming least-squares solves.
        # Assumes that self.nu is fixed throughout optimization procedure.
        H = np.dot(x.T, x) + np.diag(np.full(x.shape[1], 1.0 / self.nu))
        x_transpose_y = np.dot(x.T, y)
        if not self.use_constraints:
            cho = cho_factor(H)

        obj_his = []
        for _ in range(self.max_iter):
            if self.use_trimming:
                x_weighted = x * trimming_array.reshape(n_samples, 1)
                H = np.dot(x_weighted.T, x) + np.diag(
                    np.full(x.shape[1], 1.0 / self.nu)
                )
                x_transpose_y = np.dot(x_weighted.T, y)
                if not self.use_constraints:
                    cho = cho_factor(H)
                trimming_grad = 0.5 * np.sum((y - x.dot(coef_full)) ** 2, axis=1)
            if self.use_constraints:
                coef_full = self._update_full_coef_constraints(
                    H, x_transpose_y, coef_sparse
                )
            else:
                coef_full = self._update_full_coef(cho, x_transpose_y, coef_sparse)
            coef_sparse = self._update_sparse_coef(coef_full)
            if self.use_trimming:
                trimming_array = self._update_trimming_array(
                    coef_full, trimming_array, trimming_grad
                )

            if self.use_trimming:
                obj_his.append(
                    self._objective(x, y, coef_full, coef_sparse, trimming_array)
                )
            else:
                obj_his.append(self._objective(x, y, coef_full, coef_sparse))
            if self._convergence_criterion() < self.tol:
                # TODO: Update this for trimming/constraints
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
        self.obj_his = obj_his
