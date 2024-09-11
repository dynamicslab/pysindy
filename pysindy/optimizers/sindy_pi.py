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
    regularizer : string, optional (default 'l1')
        Regularization function to use. Currently implemented options
        are 'l1' (l1 norm), 'weighted_l1' (weighted l1 norm), l2, and
        'weighted_l2' (weighted l2 norm)

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

    unbias: bool
        Required to be false, maintained for supertype compatibility
    """

    def __init__(
        self,
        reg_weight_lam=0.1,
        regularizer="l1",
        tol=1e-5,
        max_iter=10000,
        copy_X=True,
        model_subset=None,
        normalize_columns=False,
        verbose_cvxpy=False,
        unbias=False,
    ):
        super().__init__(
            reg_weight_lam=reg_weight_lam,
            regularizer=regularizer,
            tol=tol,
            max_iter=max_iter,
            copy_X=copy_X,
            normalize_columns=normalize_columns,
            unbias=unbias,
        )

        if regularizer.lower() not in ["l1", "l2", "weighted_l1", "weighted_l2"]:
            raise ValueError(
                "l0 and other nonconvex regularizers are not implemented "
                " in current version of SINDy-PI"
            )

        if self.unbias:
            raise ValueError("SINDyPI is incompatible with an unbiasing step")
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
            # so regularizer must be L1 or L2
            regularizer = self.regularizer.lower()
            lam = self.reg_weight_lam
            if regularizer == "l1":
                cost = cp.sum_squares(x[:, i] - x @ xi) + cp.sum(
                    cp.multiply(lam, cp.abs(xi))
                )
            elif regularizer == "weighted_l1":
                cost = cp.sum_squares(x[:, i] - x @ xi) + cp.sum(
                    cp.multiply(lam[:, i], cp.abs(xi))
                )
            elif regularizer == "l2":
                cost = cp.sum_squares(x[:, i] - x @ xi) + cp.sum(
                    cp.multiply(lam, xi**2)
                )
            elif regularizer == "weighted_l2":
                cost = cp.sum_squares(x[:, i] - x @ xi) + cp.sum(
                    cp.multiply(lam[:, i], xi**2)
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
                        "Infeasible solve on iteration "
                        + str(i)
                        + ", try changing your library",
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
