import warnings

try:
    import cvxpy as cp

    cvxpy_flag = True
except ImportError:
    cvxpy_flag = False
    pass
import numpy as np
from scipy.linalg import cho_factor
from sklearn.exceptions import ConvergenceWarning

from ..utils import get_regularization
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
        are 'l0' (l0 norm), 'l1' (l1 norm), 'l2' (l2 norm), 'cad' (clipped
        absolute deviation), 'weighted_l0' (weighted l0 norm),
        'weighted_l1' (weighted l1 norm), and 'weighted_l2' (weighted l2 norm).

    max_iter : int, optional (default 30)
        Maximum iterations of the optimization algorithm.

    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    constraint_lhs : numpy ndarray, optional (default None)
        Shape should be (n_constraints, n_features * n_targets),
        The left hand side matrix C of Cw <= d.
        There should be one row per constraint.

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

    normalize_columns : boolean, optional (default False)
        Normalize the columns of x (the SINDy library terms) before regression
        by dividing by the L2-norm. Note that the 'normalize' option in sklearn
        is deprecated in sklearn versions >= 1.0 and will be removed. Note that
        this parameter is incompatible with the constraints!

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    initial_guess : np.ndarray, optional (default None)
        Shape should be (n_features) or (n_targets, n_features).
        Initial guess for coefficients ``coef_``, (v in the mathematical equations)
        If None, least-squares is used to obtain an initial guess.

    thresholds : np.ndarray, shape (n_targets, n_features), optional (default None)
        Array of thresholds for each library function coefficient.
        Each row corresponds to a measurement variable and each column
        to a function from the feature library.
        Recall that SINDy seeks a matrix :math:`\\Xi` such that
        :math:`\\dot{X} \\approx \\Theta(X)\\Xi`.
        ``thresholds[i, j]`` should specify the threshold to be used for the
        (j + 1, i + 1) entry of :math:`\\Xi`. That is to say it should give the
        threshold to be used for the (j + 1)st library function in the equation
        for the (i + 1)st measurement variable.

    inequality_constraints : bool, optional (default False)
        If True, CVXPY methods are used to solve the problem.

    verbose : bool, optional (default False)
        If True, prints out the different error terms every
        max_iter / 10 iterations.

    verbose_cvxpy : bool, optional (default False)
        Boolean flag which is passed to CVXPY solve function to indicate if
        output should be verbose or not. Only relevant for optimizers that
        use the CVXPY package in some capabity.

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
        nu=1.0,
        tol=1e-5,
        thresholder="l0",
        max_iter=30,
        trimming_fraction=0.0,
        trimming_step_size=1.0,
        constraint_lhs=None,
        constraint_rhs=None,
        constraint_order="target",
        normalize_columns=False,
        fit_intercept=False,
        copy_X=True,
        initial_guess=None,
        thresholds=None,
        equality_constraints=False,
        inequality_constraints=False,
        constraint_separation_index=0,
        verbose=False,
        verbose_cvxpy=False,
    ):
        super(ConstrainedSR3, self).__init__(
            threshold=threshold,
            nu=nu,
            tol=tol,
            thresholder=thresholder,
            thresholds=thresholds,
            trimming_fraction=trimming_fraction,
            trimming_step_size=trimming_step_size,
            max_iter=max_iter,
            initial_guess=initial_guess,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            normalize_columns=normalize_columns,
            verbose=verbose,
        )

        self.verbose_cvxpy = verbose_cvxpy
        self.reg = get_regularization(thresholder)
        self.constraint_lhs = constraint_lhs
        self.constraint_rhs = constraint_rhs
        self.constraint_order = constraint_order
        self.use_constraints = (constraint_lhs is not None) and (
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
            self.equality_constraints = True

        if self.use_constraints:
            if constraint_order not in ("feature", "target"):
                raise ValueError(
                    "constraint_order must be either 'feature' or 'target'"
                )

            self.unbias = False

        if inequality_constraints and not cvxpy_flag:
            raise ValueError(
                "Cannot use inequality constraints without cvxpy installed."
            )

        if inequality_constraints and not self.use_constraints:
            raise ValueError(
                "Use of inequality constraints requires constraint_lhs and "
                "constraint_rhs."
            )

        if inequality_constraints and thresholder.lower() not in (
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
        self.constraint_separation_index = constraint_separation_index

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

    def _update_coef_cvxpy(self, x, y, coef_sparse):
        xi = cp.Variable(coef_sparse.shape[0] * coef_sparse.shape[1])
        cost = cp.sum_squares(x @ xi - y.flatten())
        if self.thresholder.lower() == "l1":
            cost = cost + self.threshold * cp.norm1(xi)
        elif self.thresholder.lower() == "weighted_l1":
            cost = cost + cp.norm1(np.ravel(self.thresholds) @ xi)
        elif self.thresholder.lower() == "l2":
            cost = cost + self.threshold * cp.norm2(xi)
        elif self.thresholder.lower() == "weighted_l2":
            cost = cost + cp.norm2(np.ravel(self.thresholds) @ xi)
        if self.use_constraints:
            if self.inequality_constraints and self.equality_constraints:
                # Process inequality constraints then equality constraints
                prob = cp.Problem(
                    cp.Minimize(cost),
                    [
                        self.constraint_lhs[: self.constraint_separation_index, :] @ xi
                        <= self.constraint_rhs[: self.constraint_separation_index],
                        self.constraint_lhs[self.constraint_separation_index :, :] @ xi
                        == self.constraint_rhs[self.constraint_separation_index :],
                    ],
                )
            elif self.inequality_constraints:
                prob = cp.Problem(
                    cp.Minimize(cost),
                    [self.constraint_lhs @ xi <= self.constraint_rhs],
                )
            else:
                prob = cp.Problem(
                    cp.Minimize(cost),
                    [self.constraint_lhs @ xi == self.constraint_rhs],
                )
        else:
            prob = cp.Problem(cp.Minimize(cost))

        # default solver is OSQP here but switches to ECOS for L2
        try:
            prob.solve(
                max_iter=self.max_iter,
                eps_abs=self.tol,
                eps_rel=self.tol,
                verbose=self.verbose_cvxpy,
            )
        # Annoying error coming from L2 norm switching to use the ECOS
        # solver, which uses "max_iters" instead of "max_iter", and
        # similar semantic changes for the other variables.
        except TypeError:
            try:
                prob.solve(abstol=self.tol, reltol=self.tol, verbose=self.verbose_cvxpy)
            except cp.error.SolverError:
                print("Solver failed, setting coefs to zeros")
                xi.value = np.zeros(coef_sparse.shape[0] * coef_sparse.shape[1])
        except cp.error.SolverError:
            print("Solver failed, setting coefs to zeros")
            xi.value = np.zeros(coef_sparse.shape[0] * coef_sparse.shape[1])

        if xi.value is None:
            warnings.warn(
                "Infeasible solve, probably an issue with the regularizer "
                " or the constraint that was used.",
                ConvergenceWarning,
            )
            return None
        coef_new = (xi.value).reshape(coef_sparse.shape)
        return coef_new

    def _update_sparse_coef(self, coef_full):
        """Update the regularized weight vector"""
        if self.thresholds is None:
            return super(ConstrainedSR3, self)._update_sparse_coef(coef_full)
        else:
            coef_sparse = self.prox(coef_full, self.thresholds.T)
        self.history_.append(coef_sparse.T)
        return coef_sparse

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

        if self.thresholds is None:
            regularization = self.reg(coef_full, self.threshold**2 / self.nu)
            if print_ind == 0 and self.verbose:
                row = [
                    q,
                    np.sum(R2),
                    np.sum(D2) / self.nu,
                    regularization,
                    np.sum(R2) + np.sum(D2) + regularization,
                ]
                print(
                    "{0:10d} ... {1:10.4e} ... {2:10.4e} ... {3:10.4e}"
                    " ... {4:10.4e}".format(*row)
                )
            return 0.5 * np.sum(R2) + 0.5 * regularization + 0.5 * np.sum(D2) / self.nu
        else:
            regularization = self.reg(coef_full, self.thresholds**2 / self.nu)
            if print_ind == 0 and self.verbose:
                row = [
                    q,
                    np.sum(R2),
                    np.sum(D2) / self.nu,
                    regularization,
                    np.sum(R2) + np.sum(D2) + regularization,
                ]
                print(
                    "{0:10d} ... {1:10.4e} ... {2:10.4e} ... {3:10.4e}"
                    " ... {4:10.4e}".format(*row)
                )
            return 0.5 * np.sum(R2) + 0.5 * regularization + 0.5 * np.sum(D2) / self.nu

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
        # Assumes that self.nu is fixed throughout optimization procedure.
        H = np.dot(x.T, x) + np.diag(np.full(x.shape[1], 1.0 / self.nu))
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
            coef_sparse = self._update_coef_cvxpy(x_expanded, y, coef_sparse)
            objective_history.append(self._objective(x, y, 0, coef_full, coef_sparse))
        else:
            for k in range(self.max_iter):
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
                    "SR3._reduce did not converge after {} iterations.".format(
                        self.max_iter
                    ),
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
