import warnings
from typing import Union

import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from sklearn.exceptions import ConvergenceWarning

from ..utils import capped_simplex_projection
from ..utils import get_prox
from ..utils import get_regularization
from .base import BaseOptimizer


class SR3(BaseOptimizer):
    """
    Sparse relaxed regularized regression.

    Attempts to minimize the objective function

    .. math::

        0.5\\|y-Xw\\|^2_2 + \\lambda R(u)
        + (0.5 / \\nu)\\|w-u\\|^2_2

    where :math:`R(u)` is a regularization function.
    See the following references for more details:

        Zheng, Peng, et al. "A unified framework for sparse relaxed
        regularized regression: SR3." IEEE Access 7 (2018): 1404-1423.

        Champion, K., Zheng, P., Aravkin, A. Y., Brunton, S. L., & Kutz, J. N.
        (2020). A unified sparse optimization framework to learn parsimonious
        physics-informed models from data. IEEE Access, 8, 169259-169271.

    Parameters
    ----------
    reg_weight_lam : float or np.ndarray[float], shape (n_targets, n_features) \
        optional (default 0.005)
        Determines the strength of the regularization. When the
        regularization function R is the l0 norm, the regularization
        is equivalent to performing hard thresholding.
        Use the method calculate_l0_weight to calculate the weight from the threshold.

        When using weighted regularization, this is the array of weights
        for each library function coefficient.
        Each row corresponds to a measurement variable and each column
        to a function from the feature library.
        Recall that SINDy seeks a matrix :math:`\\Xi` such that
        :math:`\\dot{X} \\approx \\Theta(X)\\Xi`.
        ``reg_weight_lam[i, j]`` should specify the weight to be used for the
        (j + 1, i + 1) entry of :math:`\\Xi`. That is to say it should give the
        weight to be used for the (j + 1)st library function in the equation
        for the (i + 1)st measurement variable.

    regularizer : string, optional (default 'L0')
        Regularization function to use. Currently implemented options
        are 'L0' (L0 norm), 'L1' (L1 norm) and 'L2' (L2 norm).
        Note by 'L2 norm' we really mean
        the squared L2 norm, i.e. ridge regression

    relax_coeff_nu : float, optional (default 1)
        Determines the level of relaxation. Decreasing nu encourages
        w and v to be close, whereas increasing nu allows the
        regularized coefficients v to be farther from w.

    tol : float, optional (default 1e-5)
        Tolerance used for determining convergence of the optimization
        algorithm.

    trimming_fraction : float, optional (default 0.0)
        Fraction of the data samples to trim during fitting. Should
        be a float between 0.0 and 1.0. If 0.0, trimming is not
        performed.

    trimming_step_size : float, optional (default 1.0)
        Step size to use in the trimming optimization procedure.

    max_iter : int, optional (default 30)
        Maximum iterations of the optimization algorithm.

    initial_guess : np.ndarray, shape (n_features) or (n_targets, n_features), \
            optional (default None)
        Initial guess for coefficients ``coef_``.
        If None, least-squares is used to obtain an initial guess.

    normalize_columns : boolean, optional (default False)
        Normalize the columns of x (the SINDy library terms) before regression
        by dividing by the L2-norm. Note that the 'normalize' option in sklearn
        is deprecated in sklearn versions >= 1.0 and will be removed.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    verbose : bool, optional (default False)
        If True, prints out the different error terms every
        max_iter / 10 iterations.

    unbias: bool (default False)
        See base class for definition.  Most options are incompatible
        with unbiasing.

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
    >>> lorenz = lambda z,t : [10 * (z[1] - z[0]),
    >>>                        z[0] * (28 - z[2]) - z[1],
    >>>                        z[0] * z[1] - 8 / 3 * z[2]]
    >>> t = np.arange(0, 2, .002)
    >>> x = odeint(lorenz, [-8, 8, 27], t)
    >>> opt = SR3(reg_weight_lam=0.1, relax_coeff_nu=1)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1] - t[0])
    >>> model.print()
    x0' = -10.004 1 + 10.004 x0
    x1' = 27.994 1 + -0.993 x0 + -1.000 1 x1
    x2' = -2.662 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        reg_weight_lam=0.005,
        regularizer="L0",
        relax_coeff_nu=1.0,
        tol=1e-5,
        trimming_fraction=0.0,
        trimming_step_size=1.0,
        max_iter=30,
        copy_X=True,
        initial_guess=None,
        normalize_columns=False,
        verbose=False,
        unbias=False,
    ):
        super(SR3, self).__init__(
            max_iter=max_iter,
            initial_guess=initial_guess,
            copy_X=copy_X,
            normalize_columns=normalize_columns,
            unbias=unbias,
        )
        if relax_coeff_nu <= 0:
            raise ValueError("relax_coeff_nu must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")
        if (trimming_fraction < 0) or (trimming_fraction > 1):
            raise ValueError("trimming fraction must be between 0 and 1")
        if trimming_fraction > 0 and unbias:
            raise ValueError(
                "Unbiasing counteracts some of the effects of trimming.  Either set"
                " unbias=False or trimming_fraction=0.0"
            )
        if regularizer.lower() not in (
            "l0",
            "l1",
            "l2",
            "weighted_l0",
            "weighted_l1",
            "weighted_l2",
        ):
            raise NotImplementedError(
                "Please use a valid thresholder, l0, l1, l2, "
                "weighted_l0, weighted_l1, weighted_l2."
            )
        if regularizer[:8].lower() == "weighted" and not isinstance(
            reg_weight_lam, np.ndarray
        ):
            raise ValueError(
                "weighted regularization requires the reg_weight_lam parameter "
                "to be a 2d array"
            )
        if np.any(reg_weight_lam < 0):
            raise ValueError("reg_weight_lam cannot contain negative entries")
        if isinstance(reg_weight_lam, np.ndarray):
            reg_weight_lam = reg_weight_lam.T
        self.reg_weight_lam = reg_weight_lam
        self.relax_coeff_nu = relax_coeff_nu
        self.tol = tol
        self.regularizer = regularizer
        self.reg = get_regularization(regularizer)
        self.prox = get_prox(regularizer)
        if trimming_fraction == 0.0:
            self.use_trimming = False
        else:
            self.use_trimming = True
        self.trimming_fraction = trimming_fraction
        self.trimming_step_size = trimming_step_size
        self.verbose = verbose

    @staticmethod
    def calculate_l0_weight(
        threshold: Union[float, np.ndarray[np.float64]], relax_coeff_nu: float
    ):
        """
        Calculate the L0 regularizer weight that is equivalent to a known L0 threshold

        See Appendix S1 of the following paper for more details.
            Champion, K., Zheng, P., Aravkin, A. Y., Brunton, S. L., & Kutz, J. N.
            (2020). A unified sparse optimization framework to learn parsimonious
            physics-informed models from data. IEEE Access, 8, 169259-169271.
        """
        return (threshold**2) / (2 * relax_coeff_nu)

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

    def _objective(self, x, y, q, coef_full, coef_sparse, trimming_array=None):
        """Objective function"""
        if q != 0:
            print_ind = q % (self.max_iter // 10.0)
        else:
            print_ind = q
        R2 = (y - np.dot(x, coef_full)) ** 2
        D2 = (coef_full - coef_sparse) ** 2
        if self.use_trimming:
            assert trimming_array is not None
            R2 *= trimming_array.reshape(x.shape[0], 1)
        regularization = self.reg(coef_full, self.reg_weight_lam)
        if print_ind == 0 and self.verbose:
            row = [
                q,
                np.sum(R2),
                np.sum(D2) / self.relax_coeff_nu,
                regularization,
                np.sum(R2) + np.sum(D2) + regularization,
            ]
            print(
                "{0:10d} ... {1:10.4e} ... {2:10.4e} ... {3:10.4e}"
                " ... {4:10.4e}".format(*row)
            )
        return (
            0.5 * np.sum(R2)
            + 0.5 * regularization
            + 0.5 * np.sum(D2) / self.relax_coeff_nu
        )

    def _update_full_coef(self, cho, x_transpose_y, coef_sparse):
        """Update the unregularized weight vector"""
        b = x_transpose_y + coef_sparse / self.relax_coeff_nu
        coef_full = cho_solve(cho, b)
        self.iters += 1
        return coef_full

    def _update_sparse_coef(self, coef_full):
        """Update the regularized weight vector"""
        coef_sparse = self.prox(coef_full, self.reg_weight_lam * self.relax_coeff_nu)
        return coef_sparse

    def _update_trimming_array(self, coef_full, trimming_array, trimming_grad):
        trimming_array = trimming_array - self.trimming_step_size * trimming_grad
        trimming_array = capped_simplex_projection(
            trimming_array, self.trimming_fraction
        )
        self.history_trimming_.append(trimming_array)
        return trimming_array

    def _convergence_criterion(self):
        """Calculate the convergence criterion for the optimization"""
        this_coef = self.history_[-1]
        if len(self.history_) > 1:
            last_coef = self.history_[-2]
        else:
            last_coef = np.zeros_like(this_coef)
        err_coef = np.sqrt(np.sum((this_coef - last_coef) ** 2)) / self.relax_coeff_nu
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
        Perform at most ``self.max_iter`` iterations of the SR3 algorithm.

        Assumes initial guess for coefficients is stored in ``self.coef_``.
        """
        if self.initial_guess is not None:
            self.coef_ = self.initial_guess

        coef_sparse = self.coef_.T
        n_samples, n_features = x.shape

        coef_full = coef_sparse.copy()
        if self.use_trimming:
            trimming_array = np.repeat(1.0 - self.trimming_fraction, n_samples)
            self.history_trimming_ = [trimming_array]
        else:
            trimming_array = None

        # Precompute some objects for upcoming least-squares solves.
        # Assumes that self.nu is fixed throughout optimization procedure.
        cho = cho_factor(
            np.dot(x.T, x) + np.diag(np.full(x.shape[1], 1.0 / self.relax_coeff_nu))
        )
        x_transpose_y = np.dot(x.T, y)

        # Print initial values for each term in the optimization
        if self.verbose:
            row = [
                "Iteration",
                "|y - Xw|^2",
                "|w-u|^2/v",
                "R(u)",
                "Total Error: |y-Xw|^2 + |w-u|^2/v + R(u)",
            ]
            print(
                "{: >10} ... {: >10} ... {: >10} ... {: >10} ... {: >10}".format(*row)
            )
        objective_history = []

        for k in range(self.max_iter):
            if self.use_trimming:
                x_weighted = x * trimming_array.reshape(n_samples, 1)
                cho = cho_factor(
                    np.dot(x_weighted.T, x)
                    + np.diag(np.full(x.shape[1], 1.0 / self.relax_coeff_nu))
                )
                x_transpose_y = np.dot(x_weighted.T, y)
                trimming_grad = 0.5 * np.sum((y - x.dot(coef_full)) ** 2, axis=1)
            coef_full = self._update_full_coef(cho, x_transpose_y, coef_sparse)
            coef_sparse = self._update_sparse_coef(coef_full)
            self.history_.append(coef_sparse.T)
            if self.use_trimming:
                trimming_array = self._update_trimming_array(
                    coef_full, trimming_array, trimming_grad
                )
            objective_history.append(
                self._objective(x, y, k, coef_full, coef_sparse, trimming_array)
            )
            if self._convergence_criterion() < self.tol:
                # Could not (further) select important features
                break
        else:
            warnings.warn(
                f"SR3 did not converge after {self.max_iter} iterations.",
                ConvergenceWarning,
            )
        self.coef_ = coef_sparse.T
        self.coef_full_ = coef_full.T
        if self.use_trimming:
            self.trimming_array = trimming_array
        self.objective_history = objective_history
