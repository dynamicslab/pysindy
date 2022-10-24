import warnings

import cvxpy as cp
import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from sklearn.exceptions import ConvergenceWarning

from ..utils import reorder_constraints
from .sr3 import SR3


class TrappingSR3(SR3):
    """
    Trapping variant of sparse relaxed regularized regression.
    This optimizer can be used to identify systems with globally
    stable (bounded) solutions.

    Attempts to minimize one of two related objective functions

    .. math::

        0.5\\|y-Xw\\|^2_2 + \\lambda R(w)
        + 0.5\\|Pw-A\\|^2_2/\\eta + \\delta_0(Cw-d)
        + \\delta_{\\Lambda}(A)

    or

    .. math::

        0.5\\|y-Xw\\|^2_2 + \\lambda R(w)
        + \\delta_0(Cw-d)
        + 0.5 * maximumeigenvalue(A)/\\eta

    where :math:`R(w)` is a regularization function, which must be convex,
    :math:`\\delta_0` is an indicator function that provides a hard constraint
    of CW = d, and :math:\\delta_{\\Lambda} is a term to project the :math:`A`
    matrix onto the space of negative definite matrices.
    See the following references for more details:

        Kaptanoglu, Alan A., et al. "Promoting global stability in
        data-driven models of quadratic nonlinear dynamics."
        arXiv preprint arXiv:2105.01843 (2021).

        Zheng, Peng, et al. "A unified framework for sparse relaxed
        regularized regression: Sr3." IEEE Access 7 (2018): 1404-1423.

        Champion, Kathleen, et al. "A unified sparse optimization framework
        to learn parsimonious physics-informed models from data."
        IEEE Access 8 (2020): 169259-169271.

    Parameters
    ----------
    evolve_w : bool, optional (default True)
        If false, don't update w and just minimize over (m, A)

    threshold : float, optional (default 0.1)
        Determines the strength of the regularization. When the
        regularization function R is the L0 norm, the regularization
        is equivalent to performing hard thresholding, and lambda
        is chosen to threshold at the value given by this parameter.
        This is equivalent to choosing lambda = threshold^2 / (2 * nu).

    eta : float, optional (default 1.0e20)
        Determines the strength of the stability term ||Pw-A||^2 in the
        optimization. The default value is very large so that the
        algorithm default is to ignore the stability term. In this limit,
        this should be approximately equivalent to the ConstrainedSR3 method.

    alpha_m : float, optional (default eta * 0.1)
        Determines the step size in the prox-gradient descent over m.
        For convergence, need alpha_m <= eta / ||w^T * PQ^T * PQ * w||.
        Typically 0.01 * eta <= alpha_m <= 0.1 * eta.

    alpha_A : float, optional (default eta)
        Determines the step size in the prox-gradient descent over A.
        For convergence, need alpha_A <= eta, so typically
        alpha_A = eta is used.

    gamma : float, optional (default 0.1)
        Determines the negative interval that matrix A is projected onto.
        For most applications gamma = 0.1 - 1.0 works pretty well.

    tol : float, optional (default 1e-5)
        Tolerance used for determining convergence of the optimization
        algorithm over w.

    tol_m : float, optional (default 1e-5)
        Tolerance used for determining convergence of the optimization
        algorithm over m.

    thresholder : string, optional (default 'L1')
        Regularization function to use. For current trapping SINDy,
        only the L1 and L2 norms are implemented. Note that other convex norms
        could be straightforwardly implemented, but L0 requires
        reformulation because of nonconvexity.

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

    eps_solver : float, optional (default 1.0e-7)
        If threshold != 0, this specifies the error tolerance in the
        CVXPY (OSQP) solve. Default is 1.0e-3 in OSQP.

    relax_optim : bool, optional (default True)
        If relax_optim = True, use the relax-and-split method. If False,
        try a direct minimization on the largest eigenvalue.

    inequality_constraints : bool, optional (default False)
        If True, relax_optim must be false or relax_optim = True
        AND threshold != 0, so that the CVXPY methods are used.

    max_iter : int, optional (default 30)
        Maximum iterations of the optimization algorithm.

    accel : bool, optional (default False)
        Whether or not to use accelerated prox-gradient descent for (m, A).

    m0 : np.ndarray, shape (n_targets), optional (default None)
        Initial guess for vector m in the optimization. Otherwise
        each component of m is randomly initialized in [-1, 1].

    A0 : np.ndarray, shape (n_targets, n_targets), optional (default None)
        Initial guess for vector A in the optimization. Otherwise
        A is initialized as A = diag(gamma).

    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    normalize_columns : boolean, optional (default False)
        Normalize the columns of x (the SINDy library terms) before regression
        by dividing by the L2-norm. Note that the 'normalize' option in sklearn
        is deprecated in sklearn versions >= 1.0 and will be removed.

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

    history_ : list
        History of sparse coefficients. ``history_[k]`` contains the
        sparse coefficients (v in the optimization objective function)
        at iteration k.

    objective_history_ : list
        History of the value of the objective at each step. Note that
        the trapping SINDy problem is nonconvex, meaning that this value
        may increase and decrease as the algorithm works.

    A_history_ : list
        History of the auxiliary variable A that approximates diag(PW).

    m_history_ : list
        History of the shift vector m that determines the origin of the
        trapping region.

    PW_history_ : list
        History of PW = A^S, the quantity we are attempting to make
        negative definite.

    PWeigs_history_ : list
        History of diag(PW), a list of the eigenvalues of A^S at
        each iteration. Tracking this allows us to ascertain if
        A^S is indeed being pulled towards the space of negative
        definite matrices.

    PL_unsym_ : np.ndarray, shape (n_targets, n_targets, n_targets, n_features)
        Unsymmetrized linear coefficient part of the P matrix in ||Pw - A||^2

    PL_ : np.ndarray, shape (n_targets, n_targets, n_targets, n_features)
        Linear coefficient part of the P matrix in ||Pw - A||^2

    PQ_ : np.ndarray, shape (n_targets, n_targets,
                            n_targets, n_targets, n_features)
        Quadratic coefficient part of the P matrix in ||Pw - A||^2

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import TrappingSR3
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = TrappingSR3(threshold=0.1)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -10.004 1 + 10.004 x0
    x1' = 27.994 1 + -0.993 x0 + -1.000 1 x1
    x2' = -2.662 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        evolve_w=True,
        threshold=0.1,
        eps_solver=1e-7,
        relax_optim=True,
        inequality_constraints=False,
        eta=None,
        alpha_A=None,
        alpha_m=None,
        gamma=-0.1,
        tol=1e-5,
        tol_m=1e-5,
        thresholder="l1",
        thresholds=None,
        max_iter=30,
        accel=False,
        normalize_columns=False,
        fit_intercept=False,
        copy_X=True,
        m0=None,
        A0=None,
        objective_history=None,
        constraint_lhs=None,
        constraint_rhs=None,
        constraint_order="target",
        verbose=False,
        verbose_cvxpy=False,
    ):
        super(TrappingSR3, self).__init__(
            threshold=threshold,
            max_iter=max_iter,
            normalize_columns=normalize_columns,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            thresholder=thresholder,
            thresholds=thresholds,
            verbose=verbose,
        )
        if thresholder.lower() not in ("l1", "l2", "weighted_l1", "weighted_l2"):
            raise ValueError("Regularizer must be (weighted) L1 or L2")
        if eta is None:
            warnings.warn(
                "eta was not set, so defaulting to eta = 1e20 "
                "with alpha_m = 1e-2 * eta, alpha_A = eta. Here eta is so "
                "large that the stability term in the optimization "
                "will be ignored."
            )
            eta = 1e20
            alpha_m = 1e18
            alpha_A = 1e20
        else:
            if alpha_m is None:
                alpha_m = eta * 1e-2
            if alpha_A is None:
                alpha_A = eta
        if eta <= 0:
            raise ValueError("eta must be positive")
        if alpha_m < 0 or alpha_m > eta:
            raise ValueError("0 <= alpha_m <= eta")
        if alpha_A < 0 or alpha_A > eta:
            raise ValueError("0 <= alpha_A <= eta")
        if gamma >= 0:
            raise ValueError("gamma must be negative")
        if tol <= 0 or tol_m <= 0 or eps_solver <= 0:
            raise ValueError("tol and tol_m must be positive")
        if not evolve_w and not relax_optim:
            raise ValueError("If doing direct solve, must evolve w")
        if inequality_constraints and relax_optim and threshold == 0.0:
            raise ValueError(
                "Ineq. constr. -> threshold!=0 + relax_optim=True or relax_optim=False."
            )
        if inequality_constraints and not evolve_w:
            raise ValueError(
                "Use of inequality constraints requires solving for xi (evolve_w=True)."
            )

        self.evolve_w = evolve_w
        self.eps_solver = eps_solver
        self.relax_optim = relax_optim
        self.inequality_constraints = inequality_constraints
        self.m0 = m0
        self.A0 = A0
        self.alpha_A = alpha_A
        self.alpha_m = alpha_m
        self.eta = eta
        self.gamma = gamma
        self.tol = tol
        self.tol_m = tol_m
        self.accel = accel
        self.verbose_cvxpy = verbose_cvxpy
        self.A_history_ = []
        self.m_history_ = []
        self.PW_history_ = []
        self.PWeigs_history_ = []
        self.history_ = []
        self.objective_history = objective_history
        self.unbias = False
        self.use_constraints = (constraint_lhs is not None) and (
            constraint_rhs is not None
        )

        if self.use_constraints:
            if constraint_order not in ("feature", "target"):
                raise ValueError(
                    "constraint_order must be either 'feature' or 'target'"
                )

            self.constraint_lhs = constraint_lhs
            self.constraint_rhs = constraint_rhs
            self.unbias = False
            self.constraint_order = constraint_order

    def _set_Ptensors(self, r):
        """Make the projection tensors used for the algorithm."""
        N = int((r**2 + 3 * r) / 2.0)

        # delta_{il}delta_{jk}
        PL_tensor = np.zeros((r, r, r, N))
        PL_tensor_unsym = np.zeros((r, r, r, N))
        for i in range(r):
            for j in range(r):
                for k in range(r):
                    for kk in range(N):
                        if i == k and j == kk:
                            PL_tensor_unsym[i, j, k, kk] = 1.0

        # Now symmetrize PL
        for i in range(r):
            for j in range(N):
                PL_tensor[:, :, i, j] = 0.5 * (
                    PL_tensor_unsym[:, :, i, j] + PL_tensor_unsym[:, :, i, j].T
                )

        # if j == k, delta_{il}delta_{N-r+j,n}
        # if j != k, delta_{il}delta_{r+j+k-1,n}
        PQ_tensor = np.zeros((r, r, r, r, N))
        for i in range(r):
            for j in range(r):
                for k in range(r):
                    for kk in range(r):
                        for n in range(N):
                            if (j == k) and (n == N - r + j) and (i == kk):
                                PQ_tensor[i, j, k, kk, n] = 1.0
                            if (j != k) and (n == r + j + k - 1) and (i == kk):
                                PQ_tensor[i, j, k, kk, n] = 1 / 2

        return PL_tensor_unsym, PL_tensor, PQ_tensor

    def _bad_PL(self, PL):
        """Check if PL tensor is properly defined"""
        tol = 1e-10
        return np.any((np.transpose(PL, [1, 0, 2, 3]) - PL) > tol)

    def _bad_PQ(self, PQ):
        """Check if PQ tensor is properly defined"""
        tol = 1e-10
        return np.any((np.transpose(PQ, [0, 2, 1, 3, 4]) - PQ) > tol)

    def _check_P_matrix(self, r, n_features, N):
        """Check if P tensor is properly defined"""
        # If these tensors are not passed, or incorrect shape, assume zeros
        if self.PQ_ is None:
            self.PQ_ = np.zeros((r, r, r, r, n_features))
            warnings.warn(
                "The PQ tensor (a requirement for the stability promotion) was"
                " not set, so setting this tensor to all zeros. "
            )
        elif (self.PQ_).shape != (r, r, r, r, n_features) and (self.PQ_).shape != (
            r,
            r,
            r,
            r,
            N,
        ):
            self.PQ_ = np.zeros((r, r, r, r, n_features))
            warnings.warn(
                "The PQ tensor (a requirement for the stability promotion) was"
                " initialized with incorrect dimensions, "
                "so setting this tensor to all zeros "
                "(with the correct dimensions). "
            )
        if self.PL_ is None:
            self.PL_ = np.zeros((r, r, r, n_features))
            warnings.warn(
                "The PL tensor (a requirement for the stability promotion) was"
                " not set, so setting this tensor to all zeros. "
            )
        elif (self.PL_).shape != (r, r, r, n_features) and (self.PL_).shape != (
            r,
            r,
            r,
            N,
        ):
            self.PL_ = np.zeros((r, r, r, n_features))
            warnings.warn(
                "The PL tensor (a requirement for the stability promotion) was"
                " initialized with incorrect dimensions, "
                "so setting this tensor to all zeros "
                "(with the correct dimensions). "
            )

        # Check if the tensor symmetries are properly defined
        if self._bad_PL(self.PL_):
            raise ValueError("PL tensor was passed but the symmetries are not correct")
        if self._bad_PQ(self.PQ_):
            raise ValueError("PQ tensor was passed but the symmetries are not correct")

        # If PL/PQ finite and correct, so trapping theorem is being used,
        # then make sure library is quadratic and correct shape
        if (np.any(self.PL_ != 0.0) or np.any(self.PQ_ != 0.0)) and n_features != N:
            print(
                "The feature library is the wrong shape or not quadratic, "
                "so please correct this if you are attempting to use the "
                "trapping algorithm with the stability term included. Setting "
                "PL and PQ tensors to zeros for now."
            )
            self.PL_ = np.zeros((r, r, r, n_features))
            self.PQ_ = np.zeros((r, r, r, r, n_features))

    def _update_coef_constraints(self, H, x_transpose_y, P_transpose_A, coef_sparse):
        """Solves the coefficient update analytically if threshold = 0"""
        g = x_transpose_y + P_transpose_A / self.eta
        inv1 = np.linalg.pinv(H, rcond=1e-15)
        inv2 = np.linalg.pinv(
            self.constraint_lhs.dot(inv1).dot(self.constraint_lhs.T), rcond=1e-15
        )

        rhs = g.flatten() + self.constraint_lhs.T.dot(inv2).dot(
            self.constraint_rhs - self.constraint_lhs.dot(inv1).dot(g.flatten())
        )
        rhs = rhs.reshape(g.shape)
        return inv1.dot(rhs)

    def _update_A(self, A_old, PW):
        """Update the symmetrized A matrix"""
        eigvals, eigvecs = np.linalg.eigh(A_old)
        eigPW, eigvecsPW = np.linalg.eigh(PW)
        r = A_old.shape[0]
        A = np.diag(eigvals)
        for i in range(r):
            if eigvals[i] > self.gamma:
                A[i, i] = self.gamma
        return eigvecsPW @ A @ np.linalg.inv(eigvecsPW)

    def _convergence_criterion(self):
        """Calculate the convergence criterion for the optimization over w"""
        this_coef = self.history_[-1]
        if len(self.history_) > 1:
            last_coef = self.history_[-2]
        else:
            last_coef = np.zeros_like(this_coef)
        err_coef = np.sqrt(np.sum((this_coef - last_coef) ** 2))
        return err_coef

    def _m_convergence_criterion(self):
        """Calculate the convergence criterion for the optimization over m"""
        return np.sum(np.abs(self.m_history_[-2] - self.m_history_[-1]))

    def _objective(self, x, y, coef_sparse, A, PW, q):
        """Objective function"""
        # Compute the errors
        R2 = (y - np.dot(x, coef_sparse)) ** 2
        A2 = (A - PW) ** 2
        L1 = self.threshold * np.sum(np.abs(coef_sparse.flatten()))

        # convoluted way to print every max_iter / 10 iterations
        if self.verbose and q % max(1, self.max_iter // 10) == 0:
            row = [
                q,
                0.5 * np.sum(R2),
                0.5 * np.sum(A2) / self.eta,
                L1,
                0.5 * np.sum(R2) + 0.5 * np.sum(A2) / self.eta + L1,
            ]
            print(
                "{0:10d} ... {1:10.4e} ... {2:10.4e} ... {3:10.4e}"
                " ... {4:10.4e}".format(*row)
            )
        return 0.5 * np.sum(R2) + 0.5 * np.sum(A2) / self.eta + L1

    def _solve_sparse_relax_and_split(self, r, N, x_expanded, y, Pmatrix, A, coef_prev):
        """Solve coefficient update with CVXPY if threshold != 0"""
        xi = cp.Variable(N * r)
        cost = cp.sum_squares(x_expanded @ xi - y.flatten())
        if self.thresholder.lower() == "l1":
            cost = cost + self.threshold * cp.norm1(xi)
        elif self.thresholder.lower() == "weighted_l1":
            cost = cost + cp.norm1(np.ravel(self.thresholds) @ xi)
        elif self.thresholder.lower() == "l2":
            cost = cost + self.threshold * cp.norm2(xi) ** 2
        elif self.thresholder.lower() == "weighted_l2":
            cost = cost + cp.norm2(np.ravel(self.thresholds) @ xi) ** 2
        cost = cost + cp.sum_squares(Pmatrix @ xi - A.flatten()) / self.eta
        if self.use_constraints:
            if self.inequality_constraints:
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
                eps_abs=self.eps_solver,
                eps_rel=self.eps_solver,
                verbose=self.verbose_cvxpy,
            )
        # Annoying error coming from L2 norm switching to use the ECOS
        # solver, which uses "max_iters" instead of "max_iter", and
        # similar semantic changes for the other variables.
        except TypeError:
            try:
                prob.solve(
                    abstol=self.eps_solver,
                    reltol=self.eps_solver,
                    verbose=self.verbose_cvxpy,
                )
            except cp.error.SolverError:
                print("Solver failed, setting coefs to zeros")
                xi.value = np.zeros(N * r)
        except cp.error.SolverError:
            print("Solver failed, setting coefs to zeros")
            xi.value = np.zeros(N * r)

        if xi.value is None:
            warnings.warn(
                "Infeasible solve, increase/decrease eta",
                ConvergenceWarning,
            )
            return None
        coef_sparse = (xi.value).reshape(coef_prev.shape)
        return coef_sparse

    def _solve_m_relax_and_split(self, r, N, m_prev, m, A, coef_sparse, tk_previous):
        """
        If using the relaxation formulation of trapping SINDy, solves the
        (m, A) algorithm update.
        """
        # prox-grad for (A, m)
        # Accelerated prox gradient descent
        if self.accel:
            tk = (1 + np.sqrt(1 + 4 * tk_previous**2)) / 2.0
            m_partial = m + (tk_previous - 1.0) / tk * (m - m_prev)
            tk_previous = tk
            mPQ = np.tensordot(m_partial, self.PQ_, axes=([0], [0]))
        else:
            mPQ = np.tensordot(m, self.PQ_, axes=([0], [0]))
        p = self.PL_ - mPQ
        PW = np.tensordot(p, coef_sparse, axes=([3, 2], [0, 1]))
        PQW = np.tensordot(self.PQ_, coef_sparse, axes=([4, 3], [0, 1]))
        A_b = (A - PW) / self.eta
        PQWT_PW = np.tensordot(PQW, A_b, axes=([2, 1], [0, 1]))
        if self.accel:
            m_new = m_partial - self.alpha_m * PQWT_PW
        else:
            m_new = m_prev - self.alpha_m * PQWT_PW
        m_current = m_new

        # Update A
        A_new = self._update_A(A - self.alpha_A * A_b, PW)
        return m_current, m_new, A_new, tk_previous

    def _solve_nonsparse_relax_and_split(self, H, xTy, P_transpose_A, coef_prev):
        """Update for the coefficients if threshold = 0."""
        if self.use_constraints:
            coef_sparse = self._update_coef_constraints(
                H, xTy, P_transpose_A, coef_prev
            ).reshape(coef_prev.shape)
        else:
            cho = cho_factor(H)
            coef_sparse = cho_solve(cho, xTy + P_transpose_A / self.eta).reshape(
                coef_prev.shape
            )
        return coef_sparse

    def _solve_direct_cvxpy(self, r, N, x_expanded, y, Pmatrix, coef_prev):
        """
        If using the direct formulation of trapping SINDy, solves the
        entire problem in CVXPY regardless of the threshold value.
        Note that this is a convex-composite (i.e. technically nonconvex)
        problem, solved in CVXPY, so convergence/quality guarantees are
        not available here!
        """
        xi = cp.Variable(N * r)
        cost = cp.sum_squares(x_expanded @ xi - y.flatten())
        if self.thresholder.lower() == "l1":
            cost = cost + self.threshold * cp.norm1(xi)
        elif self.thresholder.lower() == "weighted_l1":
            cost = cost + cp.norm1(np.ravel(self.thresholds) @ xi)
        elif self.thresholder.lower() == "l2":
            cost = cost + self.threshold * cp.norm2(xi) ** 2
        elif self.thresholder.lower() == "weighted_l2":
            cost = cost + cp.norm2(np.ravel(self.thresholds) @ xi) ** 2
        cost = cost + cp.lambda_max(cp.reshape(Pmatrix @ xi, (r, r))) / self.eta
        if self.use_constraints:
            if self.inequality_constraints:
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

        # default solver is SCS here
        try:
            prob.solve(eps=self.eps_solver, verbose=self.verbose_cvxpy)
        # Annoying error coming from L2 norm switching to use the ECOS
        # solver, which uses "max_iters" instead of "max_iter", and
        # similar semantic changes for the other variables.
        except TypeError:
            prob.solve(
                abstol=self.eps_solver,
                reltol=self.eps_solver,
                verbose=self.verbose_cvxpy,
            )
        except cp.error.SolverError:
            print("Solver failed, setting coefs to zeros")
            xi.value = np.zeros(N * r)

        if xi.value is None:
            print("Infeasible solve, increase/decrease eta")
            return None, None
        coef_sparse = (xi.value).reshape(coef_prev.shape)

        if np.all(self.PL_ == 0) and np.all(self.PQ_ == 0):
            return np.zeros(r), coef_sparse  # no optimization over m
        else:
            m_cp = cp.Variable(r)
            L = np.tensordot(self.PL_, coef_sparse, axes=([3, 2], [0, 1]))
            Q = np.reshape(
                np.tensordot(self.PQ_, coef_sparse, axes=([4, 3], [0, 1])), (r, r * r)
            )
            Ls = 0.5 * (L + L.T).flatten()
            cost_m = cp.lambda_max(cp.reshape(Ls - m_cp @ Q, (r, r)))
            prob_m = cp.Problem(cp.Minimize(cost_m))

            # default solver is SCS here
            prob_m.solve(eps=self.eps_solver, verbose=self.verbose_cvxpy)

            m = m_cp.value
            if m is None:
                print("Infeasible solve over m, increase/decrease eta")
                return None, coef_sparse
            return m, coef_sparse

    def _reduce(self, x, y):
        """
        Perform at most ``self.max_iter`` iterations of the
        TrappingSR3 algorithm.
        Assumes initial guess for coefficients is stored in ``self.coef_``.
        """

        n_samples, n_features = x.shape
        r = y.shape[1]
        N = int((r**2 + 3 * r) / 2.0)

        # Define PL and PQ tensors, only relevant if the stability term in
        # trapping SINDy is turned on.
        self.PL_unsym_, self.PL_, self.PQ_ = self._set_Ptensors(r)
        # make sure dimensions/symmetries are correct
        self._check_P_matrix(r, n_features, N)

        # Set initial coefficients
        if self.use_constraints and self.constraint_order.lower() == "target":
            self.constraint_lhs = reorder_constraints(
                self.constraint_lhs, n_features, output_order="target"
            )
        coef_sparse = self.coef_.T

        # Print initial values for each term in the optimization
        if self.verbose:
            row = [
                "Iteration",
                "Data Error",
                "Stability Error",
                "L1 Error",
                "Total Error",
            ]
            print(
                "{: >10} ... {: >10} ... {: >10} ... {: >10}"
                " ... {: >10}".format(*row)
            )

        # initial A
        if self.A0 is not None:
            A = self.A0
        elif np.any(self.PQ_ != 0.0):
            A = np.diag(self.gamma * np.ones(r))
        else:
            A = np.diag(np.zeros(r))
        self.A_history_.append(A)

        # initial guess for m
        if self.m0 is not None:
            m = self.m0
        else:
            np.random.seed(1)
            m = (np.random.rand(r) - np.ones(r)) * 2
        self.m_history_.append(m)

        # Precompute some objects for optimization
        x_expanded = np.zeros((n_samples, r, n_features, r))
        for i in range(r):
            x_expanded[:, i, :, i] = x
        x_expanded = np.reshape(x_expanded, (n_samples * r, r * n_features))
        xTx = np.dot(x_expanded.T, x_expanded)
        xTy = np.dot(x_expanded.T, y.flatten())

        # if using acceleration
        tk_prev = 1
        m_prev = m

        # Begin optimization loop
        objective_history = []
        for k in range(self.max_iter):

            # update P tensor from the newest m
            mPQ = np.tensordot(m, self.PQ_, axes=([0], [0]))
            p = self.PL_ - mPQ
            Pmatrix = p.reshape(r * r, r * n_features)

            # update w
            coef_prev = coef_sparse
            if self.evolve_w:
                if self.relax_optim:
                    if self.threshold > 0.0:
                        coef_sparse = self._solve_sparse_relax_and_split(
                            r, n_features, x_expanded, y, Pmatrix, A, coef_prev
                        )
                    else:
                        pTp = np.dot(Pmatrix.T, Pmatrix)
                        H = xTx + pTp / self.eta
                        P_transpose_A = np.dot(Pmatrix.T, A.flatten())
                        coef_sparse = self._solve_nonsparse_relax_and_split(
                            H, xTy, P_transpose_A, coef_prev
                        )
                else:
                    m, coef_sparse = self._solve_direct_cvxpy(
                        r, n_features, x_expanded, y, Pmatrix, coef_prev
                    )

            # If problem over xi becomes infeasible, break out of the loop
            if coef_sparse is None:
                coef_sparse = coef_prev
                break

            if self.relax_optim:
                m_prev, m, A, tk_prev = self._solve_m_relax_and_split(
                    r, n_features, m_prev, m, A, coef_sparse, tk_prev
                )

            # If problem over m becomes infeasible, break out of the loop
            if m is None:
                m = m_prev
                break
            self.history_.append(coef_sparse.T)
            PW = np.tensordot(p, coef_sparse, axes=([3, 2], [0, 1]))

            # (m,A) update finished, append the result
            self.m_history_.append(m)
            self.A_history_.append(A)
            eigvals, eigvecs = np.linalg.eig(PW)
            self.PW_history_.append(PW)
            self.PWeigs_history_.append(np.sort(eigvals))

            # update objective
            objective_history.append(self._objective(x, y, coef_sparse, A, PW, k))

            if (
                self._m_convergence_criterion() < self.tol_m
                and self._convergence_criterion() < self.tol
            ):
                # Could not (further) select important features
                break
        if k == self.max_iter - 1:
            warnings.warn(
                "TrappingSR3._reduce did not converge after {} iters.".format(
                    self.max_iter
                ),
                ConvergenceWarning,
            )

        self.coef_ = coef_sparse.T
        self.objective_history = objective_history
