import warnings
from copy import deepcopy
from typing import Optional
from typing import Tuple

try:
    import cvxpy as cp

    cvxpy_flag = True
except ImportError:
    cvxpy_flag = False
    pass
import numpy as np
from scipy.linalg import cho_factor
from sklearn.exceptions import ConvergenceWarning

from ..utils import reorder_constraints
from .sr3 import SR3


class ConstrainedSR3(SR3):
    """
    Sparse relaxed regularized regression with linear (in)equality constraints.

    Attempts to minimize the objective function

    .. math::

        0.5\\|y-Xw\\|^2_2 + \\lambda R(u)
        + (0.5 / \\nu)\\|w-u\\|^2_2

    .. math::

        \\text{subject to } Cw = d

    over u and w, where :math:`R(u)` is a regularization function, C is a
    constraint matrix, and d is a vector of values. See the following
    reference for more details:

        Champion, Kathleen, et al. "A unified sparse optimization framework
        to learn parsimonious physics-informed models from data."
        IEEE Access 8 (2020): 169259-169271.

        Zheng, Peng, et al. "A unified framework for sparse relaxed
        regularized regression: Sr3." IEEE Access 7 (2018): 1404-1423.

    Parameters
    ----------
    constraint_lhs : numpy ndarray, optional (default None)
        Shape should be (n_constraints, n_features * n_targets),
        The left hand side matrix C of Cw <= d (Or Cw = d for equality
        constraints). There should be one row per constraint.

    constraint_rhs : numpy ndarray, shape (n_constraints,), optional (default None)
        The right hand side vector d of Cw <= d.

    constraint_order : string, optional (default "target")
        The format in which the constraints ``constraint_lhs`` were passed.
        Must be one of "target" or "feature".
        "target" indicates that the constraints are grouped by target:
        i.e. the first ``n_features`` columns
        correspond to constraint coefficients on the library features
        for the first target (variable), the next ``n_features`` columns to
        the library features for the second target (variable), and so on.
        "feature" indicates that the constraints are grouped by library
        feature: the first ``n_targets`` columns correspond to the first
        library feature, the next ``n_targets`` columns to the second library
        feature, and so on.

    inequality_constraints : bool, optional (default False)
        If True, CVXPY methods are used to solve the problem.

    verbose_cvxpy : bool, optional (default False)
        Boolean flag which is passed to CVXPY solve function to indicate if
        output should be verbose or not. Only relevant for optimizers that
        use the CVXPY package in some capabity.

    See base class for additional arguments

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

    objective_history_ : list
        History of the value of the objective at each step. Note that
        the trapping SINDy problem is nonconvex, meaning that this value
        may increase and decrease as the algorithm works.
    """

    def __init__(
        self,
        reg_weight_lam=0.005,
        regularizer="l0",
        relax_coeff_nu=1.0,
        tol=1e-5,
        max_iter=30,
        trimming_fraction=0.0,
        trimming_step_size=1.0,
        constraint_lhs=None,
        constraint_rhs=None,
        constraint_order="target",
        normalize_columns=False,
        copy_X=True,
        initial_guess=None,
        equality_constraints=False,
        inequality_constraints=False,
        constraint_separation_index: Optional[bool] = None,
        verbose=False,
        verbose_cvxpy=False,
        unbias=False,
    ):
        super().__init__(
            reg_weight_lam=reg_weight_lam,
            regularizer=regularizer,
            relax_coeff_nu=relax_coeff_nu,
            tol=tol,
            trimming_fraction=trimming_fraction,
            trimming_step_size=trimming_step_size,
            max_iter=max_iter,
            initial_guess=initial_guess,
            copy_X=copy_X,
            normalize_columns=normalize_columns,
            verbose=verbose,
            unbias=unbias,
        )

        self.verbose_cvxpy = verbose_cvxpy
        self.constraint_lhs = constraint_lhs
        self.constraint_rhs = constraint_rhs
        self.constraint_order = constraint_order
        self.use_constraints = (constraint_lhs is not None) or (
            constraint_rhs is not None
        )

        if (
            self.use_constraints
            and not equality_constraints
            and not inequality_constraints
        ):
            warnings.warn(
                "constraint_lhs and constraint_rhs passed to the optimizer, "
                " but user did not specify if the constraints were equality or"
                " inequality constraints. Assuming equality constraints."
            )
            equality_constraints = True

        if self.use_constraints:
            if constraint_order not in ("feature", "target"):
                raise ValueError(
                    "constraint_order must be either 'feature' or 'target'"
                )
            if unbias:
                raise ValueError(
                    "Constraints are incompatible with an unbiasing step.  Set"
                    " unbias=False"
                )

        if inequality_constraints and not cvxpy_flag:
            raise ValueError(
                "Cannot use inequality constraints without cvxpy installed."
            )

        if inequality_constraints and not self.use_constraints:
            raise ValueError(
                "Use of inequality constraints requires constraint_lhs and "
                "constraint_rhs."
            )

        if inequality_constraints and regularizer.lower() not in (
            "l1",
            "l2",
            "weighted_l1",
            "weighted_l2",
        ):
            raise ValueError(
                "Use of inequality constraints requires a convex regularizer."
            )
        self.inequality_constraints = inequality_constraints
        self.equality_constraints = equality_constraints
        if self.use_constraints and constraint_separation_index is None:
            if self.inequality_constraints and not self.equality_constraints:
                constraint_separation_index = len(constraint_lhs)
            elif self.equality_constraints and not self.inequality_constraints:
                constraint_separation_index = 0
            else:
                raise ValueError(
                    "If passing both inequality and equality constraints, must specify"
                    " constraint_separation_index."
                )
        self.constraint_separation_index = constraint_separation_index

    def _update_full_coef_constraints(self, H, x_transpose_y, coef_sparse):
        g = x_transpose_y + coef_sparse / self.relax_coeff_nu
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

    @staticmethod
    def _calculate_penalty(
        regularization: str, regularization_weight, xi: cp.Variable
    ) -> cp.Expression:
        """
        Args:
        -----
        regularization: 'l0' | 'weighted_l0' | 'l1' | 'weighted_l1' |
                        'l2' | 'weighted_l2'
        regularization_weight: float | np.array, can be a scalar
                               or an array of the same shape as xi
        xi: cp.Variable

        Returns:
        --------
        cp.Expression
        """
        regularization = regularization.lower()
        if regularization == "l1":
            return regularization_weight * cp.sum(cp.abs(xi))
        elif regularization == "weighted_l1":
            return cp.sum(cp.multiply(regularization_weight, cp.abs(xi)))
        elif regularization == "l2":
            return regularization_weight * cp.sum(xi**2)
        elif regularization == "weighted_l2":
            return cp.sum(cp.multiply(regularization_weight, xi**2))

    def _create_var_and_part_cost(
        self, var_len: int, x_expanded: np.ndarray, y: np.ndarray
    ) -> Tuple[cp.Variable, cp.Expression]:
        xi = cp.Variable(var_len)
        cost = cp.sum_squares(x_expanded @ xi - y.flatten())
        penalty = self._calculate_penalty(
            self.regularizer, np.ravel(self.reg_weight_lam), xi
        )
        return xi, cost + penalty

    def _update_coef_cvxpy(self, xi, cost, var_len, coef_prev, tol):
        if self.use_constraints:
            constraints = []
            if self.equality_constraints:
                constraints.append(
                    self.constraint_lhs[self.constraint_separation_index :, :] @ xi
                    == self.constraint_rhs[self.constraint_separation_index :],
                )
            if self.inequality_constraints:
                constraints.append(
                    self.constraint_lhs[: self.constraint_separation_index, :] @ xi
                    <= self.constraint_rhs[: self.constraint_separation_index]
                )
            prob = cp.Problem(cp.Minimize(cost), constraints)
        else:
            prob = cp.Problem(cp.Minimize(cost))

        prob_clone = deepcopy(prob)
        try:
            prob.solve(
                max_iter=self.max_iter,
                eps_abs=tol,
                eps_rel=tol,
                verbose=self.verbose_cvxpy,
            )
        except cp.error.SolverError:
            try:
                prob = prob_clone
                prob.solve(max_iter=self.max_iter, verbose=self.verbose_cvxpy)
                xi = prob.variables()[0]
            except cp.error.SolverError:
                warnings.warn("Solver failed, setting coefs to zeros")
                xi.value = np.zeros(var_len)

        if xi.value is None:
            warnings.warn(
                "Infeasible solve, probably an issue with the regularizer "
                " or the constraint that was used.",
                ConvergenceWarning,
            )
            return None
        coef_new = (xi.value).reshape(coef_prev.shape)
        return coef_new

    def _reduce(self, x, y):
        """
        Perform at most ``self.max_iter`` iterations of the SR3 algorithm
        with inequality constraints.

        Assumes initial guess for coefficients is stored in ``self.coef_``.
        """
        if self.initial_guess is not None:
            self.coef_ = self.initial_guess

        coef_sparse = self.coef_.T
        coef_full = coef_sparse.copy()
        n_samples, n_features = x.shape
        n_targets = y.shape[1]

        if self.use_trimming:
            trimming_array = np.repeat(1.0 - self.trimming_fraction, n_samples)
            self.history_trimming_ = [trimming_array]

        if self.use_constraints and self.constraint_order.lower() == "target":
            self.constraint_lhs = reorder_constraints(self.constraint_lhs, n_features)

        # Precompute some objects for upcoming least-squares solves.
        # Assumes that self.relax_coeff_nu is fixed throughout optimization procedure.
        H = np.dot(x.T, x) + np.diag(np.full(x.shape[1], 1.0 / self.relax_coeff_nu))
        x_transpose_y = np.dot(x.T, y)
        if not self.use_constraints:
            cho = cho_factor(H)
        if self.inequality_constraints:
            # Precompute some objects for optimization
            x_expanded = np.zeros((n_samples, n_targets, n_features, n_targets))
            for i in range(n_targets):
                x_expanded[:, i, :, i] = x
            x_expanded = np.reshape(
                x_expanded, (n_samples * n_targets, n_targets * n_features)
            )

        # Print initial values for each term in the optimization
        if self.verbose:
            row = [
                "Iteration",
                "|y - Xw|^2",
                "|w-u|^2/v",
                "R(u)",
                "Total Error: |y - Xw|^2 + |w - u|^2 / v + R(u)",
            ]
            print(
                "{: >10} ... {: >10} ... {: >10} ... {: >10} ... {: >10}".format(*row)
            )

        objective_history = []
        if self.inequality_constraints:
            var_len = coef_sparse.shape[0] * coef_sparse.shape[1]
            xi, cost = self._create_var_and_part_cost(var_len, x_expanded, y)
            coef_sparse = self._update_coef_cvxpy(
                xi, cost, var_len, coef_sparse, self.tol
            )
            objective_history.append(self._objective(x, y, 0, coef_full, coef_sparse))
        else:
            for k in range(self.max_iter):
                if self.use_trimming:
                    x_weighted = x * trimming_array.reshape(n_samples, 1)
                    H = np.dot(x_weighted.T, x) + np.diag(
                        np.full(x.shape[1], 1.0 / self.relax_coeff_nu)
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
                self.history_.append(np.copy(coef_sparse).T)

                if self.use_trimming:
                    trimming_array = self._update_trimming_array(
                        coef_full, trimming_array, trimming_grad
                    )

                    objective_history.append(
                        self._objective(x, y, k, coef_full, coef_sparse, trimming_array)
                    )
                else:
                    objective_history.append(
                        self._objective(x, y, k, coef_full, coef_sparse)
                    )
                if self._convergence_criterion() < self.tol:
                    # TODO: Update this for trimming/constraints
                    break
            else:
                warnings.warn(
                    f"ConstrainedSR3 did not converge after {self.max_iter}"
                    " iterations.",
                    ConvergenceWarning,
                )
        if self.use_constraints and self.constraint_order.lower() == "target":
            self.constraint_lhs = reorder_constraints(
                self.constraint_lhs, n_features, output_order="target"
            )
        self.coef_ = coef_sparse.T
        self.coef_full_ = coef_full.T
        if self.use_trimming:
            self.trimming_array = trimming_array
        self.objective_history = objective_history
