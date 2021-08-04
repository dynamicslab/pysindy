import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from scipy.integrate import solve_ivp

from ..utils import get_regularization
from pysindy.differentiation import FiniteDifference
from .sr3 import SR3


class SINDyModifiedoptimizer(SR3):
    """
    Modified SINDy optimizer

    Attempts to minimize the objective function

    .. math::

        0.5\\|X-Xw\\|^2_2 + \\lambda \\times R(v)

    over w where :math:`R(v)` is a regularization function. See the following
    reference for more details:

        Kaheman, Kadierdan, Steven L. Brunton, and J. Nathan Kutz.
        Automatic differentiation to simultaneously identify nonlinear dynamics
        and extract noise probability distributions from data.
        arXiv preprint arXiv:2009.08810 (2020).

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
    
    differentiation_method : differentiation object, optional
        Method for differentiating the data. This must be a class extending
        :class:`pysindy.differentiation_methods.base.BaseDifferentiation` class.
        The default option is centered difference.

    soft_start : boolean, optional (default False) 
        If false, initialize the noise as all zeros. If true, 
        calculate the noise in the signal y. 

    c : float, optional (default = 0.9)
        The value of the weights in the e_s error term in the
        modified SINDy optimization.

    feature_library : feature library object
        Feature library object used to specify candidate right-hand side features.
        This must be a class extending
        :class:`pysindy.feature_library.base.BaseFeatureLibrary`.
        This is required for modified SINDy because we need to recompute
        the feature library during each iteration.

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
        differentiation_method=None,
        t=None,
        soft_start=False,
        q=3,
        c=0.9,
        feature_library=None,
    ):
        super(SINDyModifiedoptimizer, self).__init__(
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
        if differentiation_method is None:
            differentiation_method = FiniteDifference(drop_endpoints=False)
        self.differentiation_method = differentiation_method
        self.thresholds = thresholds
        self.reg = get_regularization(thresholder)
        self.unbias = False
        self.soft_start = soft_start
        self.q = q 
        self.c = c
        self.t = t
        if feature_library is None:
            raise ValueError("Modified SINDy requires a specified feature library"
                " because the feature library is updated every algorithm iteration"
            )
        self.feature_library = feature_library
    
    def _set_threshold(self, threshold):
        self.threshold = threshold

    def _update_coefs(self, Theta, N, ydot):
        n_samples = N.shape[0]
        # At each timestep j, compute esj
        # es = np.zeros(n_samples)
        #for j in range(n_samples):
        #    for i in range(self.q):
        #        es[j] += self.c ** (abs(i) - 1) * (yji - nji - Fji) ** 2
        #solve_ivp(rhs, (
        coefs = 0
        N_new = 0 
        return coefs, N_new

    def _objective(self, x, ydot, N, coefs):
        """Objective function"""
        e_d = (ydot - x @ coefs) ** 2
        # e_sj = 0
        if self.thresholds is None:
            return np.sum(e_d) + self.reg(coefs, self.threshold)
        else:
            return np.sum(e_d) + self.reg(coefs, self.thresholds)

    def _noise(self, y, lam=20):
        n_samples, n_targets = y.shape
        D = np.zeros((n_samples, n_samples))
        D[0, :4] = [2, -5, 4, -1]
        D[n_samples - 1, n_samples - 4:] = [-1, 4, -5, 2]

        for i in range(1, n_samples - 1):
            D[i, i] = -2
            D[i, i + 1] = 1
            D[i, i - 1] = 1
        D2 = D.dot(D)
        X_smooth = np.vstack([np.linalg.solve(np.eye(
                              n_samples) + lam * D2.T.dot(D2),
                              Y[j, :].reshape(n_samples, 1)).reshape(
                              1, n_samples) for j in range(n_targets)])
        N = y - X_smooth
        return N, X_smooth

    def _reduce(self, x, y):
        """
        Perform at most ``self.max_iter`` iterations of the modified SINDy
        optimization problem.

        Note here that the variable 'y' should be the measurement matrix
        X, not Xdot!
        """
        n_samples, n_features = x.shape
        if self.soft_start:
            N, y_smooth = self._noise(y)
            Theta = (self.feature_library).transform(y_smooth)
        else:
            N = np.zeros(y.shape)
            y_smooth = y
            Theta = x
        coefs = self.coef_
        if self.t is None:
            ydot = self.differentiation_method(y_smooth)
        else:
            ydot = self.differentiation_method(y_smooth, self.t)
        objective_history.append(self._objective(Theta, ydot, N, coefs))
        objective_history = []
        for _ in range(self.max_iter):
            coefs, N = self._update_coefs(Theta, N, ydot)
            y_smooth = y - N
            if self.t is None:
                ydot = self.differentiation_method(y_smooth)
            else:
                ydot = self.differentiation_method(y_smooth, self.t)
            Theta = (self.feature_library).transform(y_smooth)
            objective_history.append(self._objective(Theta, ydot, N, coefs))
        self.coef_ = coefs
        self.N_ = N
        self.objective_history = objective_history
