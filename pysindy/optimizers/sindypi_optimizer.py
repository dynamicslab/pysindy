import warnings

import cvxpy as cp
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from ..utils import get_regularization
from .sr3 import SR3


class SINDyPIoptimizer(SR3):
    """
    Sparse relaxed regularized regression with linear equality constraints.

    Attempts to minimize the objective function

    .. math::

        0.5\\|y-Xw\\|^2_2 + \\lambda \\times R(v)
        + (0.5 / nu)\\|w-v\\|^2_2

        \\text{subject to}

    .. math::

        Cw = d

    over v and w where :math:`R(v)` is a regularization function, C is a
    constraint matrix, and d is a vector of values. See the following
    reference for more details:

        Champion, Kathleen, et al. "A unified sparse optimization framework
        to learn parsimonious physics-informed models from data."
        arXiv preprint arXiv:1906.10612 (2019).

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

    initial_guess : np.ndarray, shape (n_features) or (n_targets, n_features), \
                optional (default None)
        Initial guess for coefficients ``coef_``, (v in the mathematical equations)
        If None, least-squares is used to obtain an initial guess.

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
        initial_guess=None,
        thresholds=None,
    ):
        super(SINDyPIoptimizer, self).__init__(
            threshold=threshold,
            tol=tol,
            thresholder=thresholder,
            max_iter=max_iter,
            initial_guess=initial_guess,
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

    def _update_full_coef_constraints(self, x, coef_full):
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

        # Beware: hard-coding the tolerances sometimes
        prob.solve(max_iter=self.max_iter, verbose=True, eps_abs=1e-6, eps_rel=1e-6)

        if xi.value is None:
            warnings.warn(
                "Infeasible solve, try changing your library",
                ConvergenceWarning,
            )
            return None
        coef_sparse = (xi.value).reshape(coef_full.shape)
        return coef_sparse

    def _objective(self, x, y, coef_full):
        """Objective function"""
        R2 = (x - np.dot(x, coef_full)) ** 2
        if self.thresholds is None:
            return np.sum(R2) + self.reg(coef_full, self.threshold)
        else:
            return np.sum(R2) + self.reg(coef_full, self.thresholds)

    def _reduce(self, x, y):
        """
        Perform at most ``self.max_iter`` iterations of the SR3 algorithm
        with inequality constraints.

        Set coefficients randomly
        """
        self.Theta = x
        n_samples, n_features = x.shape
        coef_full = np.random.rand(n_features, n_features)
        objective_history = []
        coef_full = self._update_full_coef_constraints(x, coef_full)
        objective_history.append(self._objective(x, y, coef_full))

        self.coef_full_ = coef_full.T
        self.coef_ = coef_full.T
        print(coef_full)
        self.objective_history = objective_history
