import warnings

import cvxpy as cp
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from .sr3 import SR3


class SINDyPI(SR3):
    """
    SINDy-PI optimizer

    Attempts to minimize the objective function

    .. math::

        0.5\\|X-Xw\\|^2_2 + \\lambda R(w)

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
        are 'l1' (l1 norm), 'weighted_l1' (weighted l1 norm), l2, and
        'weighted_l2' (weighted l2 norm)

    max_iter : int, optional (default 10000)
        Maximum iterations of the optimization algorithm.

    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    normalize_columns : boolean, optional (default False)
        This parameter normalizes the columns of Theta before the
        optimization is done. This tends to standardize the columns
        to similar magnitudes, often improving performance.

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

    model_subset : np.ndarray, shape(n_models), optional (default None)
        List of indices to compute models for. If list is not provided,
        the default is to compute SINDy-PI models for all possible
        candidate functions. This can take a long time for 4D systems
        or larger.

    verbose_cvxpy : bool, optional (default False)
        Boolean flag which is passed to CVXPY solve function to indicate if
        output should be verbose or not. Only relevant for optimizers that
        use the CVXPY package in some capabity.

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
        fit_intercept=False,
        copy_X=True,
        thresholds=None,
        model_subset=None,
        normalize_columns=False,
        verbose_cvxpy=False,
    ):
        super(SINDyPI, self).__init__(
            threshold=threshold,
            thresholds=thresholds,
            tol=tol,
            thresholder=thresholder,
            max_iter=max_iter,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            normalize_columns=normalize_columns,
        )

        if (
            thresholder.lower() != "l1"
            and thresholder.lower() != "l2"
            and thresholder.lower() != "weighted_l1"
            and thresholder.lower() != "weighted_l2"
        ):
            raise ValueError(
                "l0 and other nonconvex regularizers are not implemented "
                " in current version of SINDy-PI"
            )

        self.unbias = False
        self.verbose_cvxpy = verbose_cvxpy
        if model_subset is not None:
            if not isinstance(model_subset, list):
                raise ValueError("Model subset must be in list format.")
            subset_integers = [
                model_ind for model_ind in model_subset if isinstance(model_ind, int)
            ]
            if subset_integers != model_subset:
                raise ValueError("Model subset list must consist of integers.")

        self.model_subset = model_subset

    def _update_parallel_coef_constraints(self, x):
        """
        Solves each of the model fits separately, which can in principle be
        parallelized. Unfortunately most parallel Python packages don't give
        huge speedups. Instead, we allow the user to only solve a subset of
        the models with the parameter model_subset.
        """
        n_features = x.shape[1]
        xi_final = np.zeros((n_features, n_features))

        # Todo: parallelize this for loop with Multiprocessing/joblib
        if self.model_subset is None:
            self.model_subset = range(n_features)
        elif np.max(np.abs(self.model_subset)) >= n_features:
            raise ValueError(
                "A value in model_subset is larger than the number "
                "of features in the candidate library"
            )
        for i in self.model_subset:
            print("Model ", i)
            xi = cp.Variable(n_features)
            # Note that norm choice below must be convex,
            # so thresholder must be L1 or L2
            if (self.thresholder).lower() in ("l1", "weighted_l1"):
                if self.thresholds is None:
                    cost = cp.sum_squares(x[:, i] - x @ xi) + self.threshold * cp.norm1(
                        xi
                    )
                else:
                    cost = cp.sum_squares(x[:, i] - x @ xi) + cp.norm1(
                        self.thresholds[i, :] @ xi
                    )
            if (self.thresholder).lower() in ("l2", "weighted_l2"):
                if self.thresholds is None:
                    cost = (
                        cp.sum_squares(x[:, i] - x @ xi)
                        + self.threshold * cp.norm2(xi) ** 2
                    )
                else:
                    cost = (
                        cp.sum_squares(x[:, i] - x @ xi)
                        + cp.norm2(self.thresholds[i, :] @ xi) ** 2
                    )
            prob = cp.Problem(
                cp.Minimize(cost),
                [xi[i] == 0.0],
            )
            try:
                prob.solve(
                    max_iter=self.max_iter,
                    eps_abs=self.tol,
                    eps_rel=self.tol,
                    verbose=self.verbose_cvxpy,
                )
                if xi.value is None:
                    warnings.warn(
                        "Infeasible solve on iteration " + str(i) + ", try "
                        "changing your library",
                        ConvergenceWarning,
                    )
                xi_final[:, i] = xi.value
            # Annoying error coming from L2 norm switching to use the ECOS
            # solver, which uses "max_iters" instead of "max_iter", and
            # similar semantic changes for the other variables.
            except TypeError:
                prob.solve(
                    max_iters=self.max_iter,
                    abstol=self.tol,
                    reltol=self.tol,
                    verbose=self.verbose_cvxpy,
                )
                if xi.value is None:
                    warnings.warn(
                        "Infeasible solve on iteration " + str(i) + ", try "
                        "changing your library",
                        ConvergenceWarning,
                    )
                xi_final[:, i] = xi.value
            except cp.error.SolverError:
                print("Solver failed on model ", str(i), ", setting coefs to zeros")
                xi_final[:, i] = np.zeros(n_features)
        return xi_final

    def _reduce(self, x, y):
        """
        Perform at most ``self.max_iter`` iterations of the SINDy-PI
        optimization problem, using CVXPY.
        """
        coef = self._update_parallel_coef_constraints(x)
        self.coef_ = coef.T
