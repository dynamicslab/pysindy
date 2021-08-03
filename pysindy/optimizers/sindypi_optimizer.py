import warnings

import cvxpy as cp
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from ..utils import get_regularization
from .sr3 import SR3


class SINDyPIoptimizer(SR3):
    """
    SINDy-PI optimizer

    Attempts to minimize the objective function

    .. math::

        0.5\\|X-Xw\\|^2_2 + \\lambda \\times R(v)

    over w where :math:`R(v)` is a regularization function. See the following
    reference for more details:

        Kaheman, Kadierdan, J. Nathan Kutz, and Steven L. Brunton. SINDy-PI:
        a robust algorithm for parallel implicit sparse identification of
        nonlinear dynamics.
        Proceedings of the Royal Society A 476.2242 (2020): 20200279.

    Parameters
    ----------
    threshold : float, optional (default 0.1)
        Determines the strength of the regularization. When the
        regularization function R is the l0 norm, the regularization
        is equivalent to performing hard thresholding, and lambda
        is chosen to threshold at the value given by this parameter.
        This is equivalent to choosing lambda = threshold^2 / (2 * nu).

    tol : float, optional (default 1e-5)
        Tolerance used for determining convergence of the optimization
        algorithm.

    thresholder : string, optional (default 'l1')
        Regularization function to use. Currently implemented options
        are 'l1' (l1 norm) and 'weighted_l1' (weighted l1 norm).

    max_iter : int, optional (default 10000)
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

    unbias : boolean
        Whether to perform an extra step of unregularized linear regression
        to unbias the coefficients for the identified support.
        ``unbias`` is automatically set to False if a constraint is used and
        is otherwise left uninitialized.
    """

    def __init__(
        self,
        threshold=0.1,
        tol=1e-5,
        thresholder="l1",
        max_iter=10000,
        normalize=False,
        fit_intercept=False,
        copy_X=True,
        thresholds=None,
    ):
        super(SINDyPIoptimizer, self).__init__(
            threshold=threshold,
            tol=tol,
            thresholder=thresholder,
            max_iter=max_iter,
            normalize=normalize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
        )

        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if threshold < 0.0:
            raise ValueError("threshold must not be negative")
        if thresholder != "l1":
            raise ValueError(
                "l0 and other nonconvex regularizers are not implemented "
                " in current version of SINDy-PI"
            )
        if thresholder[:8].lower() == "weighted" and thresholds is None:
            raise ValueError("weighted thresholder requires the thresholds parameter")
        if thresholder[:8].lower() != "weighted" and thresholds is not None:
            raise ValueError(
                "The thresholds argument cannot be used without a weighted"
                " thresholder, e.g. thresholder='weighted_l0'"
            )
        if thresholds is not None and np.any(thresholds < 0):
            raise ValueError("thresholds cannot contain negative entries")

        self.thresholds = thresholds
        self.reg = get_regularization(thresholder)
        self.unbias = False
        self.Theta = None

    def _set_threshold(self, threshold):
        self.threshold = threshold

    def _update_full_coef_constraints(self, x):
        N = x.shape[1]
        xi = cp.Variable((N, N))
        if self.thresholds is None:
            cost = cp.sum_squares(x - x @ xi) + self.threshold * cp.norm1(xi)
        else:
            cost = cp.sum_squares(x - x @ xi) + cp.norm1(self.thresholds @ xi)
        prob = cp.Problem(
            cp.Minimize(cost),
            [cp.diag(xi) == np.zeros(N)],
        )

        # Beware: hard-coding the tolerances
        prob.solve(
            max_iter=self.max_iter, verbose=True, eps_abs=self.tol, eps_rel=self.tol
        )

        if xi.value is None:
            warnings.warn(
                "Infeasible solve, try changing your library",
                ConvergenceWarning,
            )
        return xi.value

    def _update_parallel_coef_constraints(self, x):
        N = x.shape[1]
        xi_final = np.zeros((N, N))

        # Todo: parallelize this for loop with Multiprocessing/joblib
        for i in range(N):
            xi = cp.Variable(N)
            if self.thresholds is None:
                cost = cp.sum_squares(x[:, i] - x @ xi) + self.threshold * cp.norm1(xi)
            else:
                cost = cp.sum_squares(x[:, i] - x @ xi) + cp.norm1(self.thresholds @ xi)
            prob = cp.Problem(
                cp.Minimize(cost),
                [xi[i] == 0.0],
            )
            prob.solve(verbose=True, eps_abs=self.tol, eps_rel=self.tol)
            if xi.value is None:
                warnings.warn(
                    "Infeasible solve on iteration " + str(i) + ", try "
                    "changing your library",
                    ConvergenceWarning,
                )
            xi_final[:, i] = xi.value
        return xi_final

    def _objective(self, x, coefs):
        """Objective function"""
        R2 = (x - x @ coefs) ** 2
        if self.thresholds is None:
            return np.sum(R2) + self.reg(coefs, self.threshold)
        else:
            return np.sum(R2) + self.reg(coefs, self.thresholds)

    def _reduce(self, x, y):
        """
        Perform at most ``self.max_iter`` iterations of the SINDy-PI
        optimization problem, using CVXPY.
        """
        self.Theta = x  # Need to save the library for post-processing
        n_samples, n_features = x.shape
        objective_history = []
        # coefs = self._update_full_coef_constraints(x)
        coefs = self._update_parallel_coef_constraints(x)
        objective_history.append(self._objective(x, coefs))
        self.coef_ = coefs
        self.objective_history = objective_history
