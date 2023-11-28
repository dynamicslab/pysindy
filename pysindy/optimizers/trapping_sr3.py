import warnings
from itertools import combinations_with_replacement as combo_wr
from itertools import product
from typing import Tuple

import cvxpy as cp
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from sklearn.exceptions import ConvergenceWarning

from ..utils import reorder_constraints
from .constrained_sr3 import ConstrainedSR3


class TrappingSR3(ConstrainedSR3):
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

    Parameters
    ----------
    evolve_w :
        If false, don't update w and just minimize over (m, A)

    eta :
        Determines the strength of the stability term ||Pw-A||^2 in the
        optimization. The default value is very large so that the
        algorithm default is to ignore the stability term. In this limit,
        this should be approximately equivalent to the ConstrainedSR3 method.

    eps_solver :
        If threshold != 0, this specifies the error tolerance in the
        CVXPY (OSQP) solve. Default 1.0e-7 (Default is 1.0e-3 in OSQP.)

    relax_optim :
        If relax_optim = True, use the relax-and-split method. If False,
        try a direct minimization on the largest eigenvalue.

    inequality_constraints : bool, optional (default False)
        If True, relax_optim must be false or relax_optim = True
        AND threshold != 0, so that the CVXPY methods are used.

    alpha_A :
        Determines the step size in the prox-gradient descent over A.
        For convergence, need alpha_A <= eta, so default
        alpha_A = eta is used.

    alpha_m :
        Determines the step size in the prox-gradient descent over m.
        For convergence, need alpha_m <= eta / ||w^T * PQ^T * PQ * w||.
        Typically 0.01 * eta <= alpha_m <= 0.1 * eta.  (default eta * 0.1)

    gamma :
        Determines the negative interval that matrix A is projected onto.
        For most applications gamma = 0.1 - 1.0 works pretty well.

    tol_m :
        Tolerance used for determining convergence of the optimization
        algorithm over m.

    thresholder :
        Regularization function to use. For current trapping SINDy,
        only the L1 and L2 norms are implemented. Note that other convex norms
        could be straightforwardly implemented, but L0 requires
        reformulation because of nonconvexity. (default 'L1')

    accel :
        Whether or not to use accelerated prox-gradient descent for (m, A).
        (default False)

    m0 :
        Initial guess for trap center in the optimization. Default None
        initializes vector elements randomly in [-1, 1]. shape (n_targets)

    A0 :
        Initial guess for vector A in the optimization.  Shape (n_targets, n_targets)
        Default None, meaning A is initialized as A = diag(gamma).

    Attributes
    ----------
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

    objective_history_: list
        History of the objective value at each iteration

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
        *,
        eta: float | None = None,
        eps_solver: float = 1e-7,
        relax_optim: bool = True,
        inequality_constraints=False,
        alpha_A: float | None = None,
        alpha_m: float | None = None,
        gamma: float = -0.1,
        tol_m: float = 1e-5,
        thresholder: str = "l1",
        accel: bool = False,
        m0: NDArray | None = None,
        A0: NDArray | None = None,
        **kwargs,
    ):
        super().__init__(
            thresholder=thresholder,
            **kwargs,
        )
        if self.thresholder.lower() not in ("l1", "l2", "weighted_l1", "weighted_l2"):
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
        if self.tol <= 0 or tol_m <= 0 or eps_solver <= 0:
            raise ValueError("tol and tol_m must be positive")
        if inequality_constraints and relax_optim and self.threshold == 0.0:
            raise ValueError(
                "Ineq. constr. -> threshold!=0 + relax_optim=True or relax_optim=False."
            )

        self.eps_solver = eps_solver
        self.relax_optim = relax_optim
        self.inequality_constraints = inequality_constraints
        self.m0 = m0
        self.A0 = A0
        self.alpha_A = alpha_A
        self.alpha_m = alpha_m
        self.eta = eta
        self.gamma = gamma
        self.tol_m = tol_m
        self.accel = accel

    def _set_Ptensors(
        self, n_targets: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make the projection tensors used for the algorithm."""
        N = int((n_targets**2 + 3 * n_targets) / 2.0)

        # delta_{il}delta_{jk}
        PL_tensor_unsym = np.zeros((n_targets, n_targets, n_targets, N))
        for i, j in combo_wr(range(n_targets), 2):
            PL_tensor_unsym[i, j, i, j] = 1.0

        # Now symmetrize PL
        PL_tensor = (PL_tensor_unsym + np.transpose(PL_tensor_unsym, [1, 0, 2, 3])) / 2

        # if j == k, delta_{il}delta_{N-r+j,n}
        # if j != k, delta_{il}delta_{r+j+k-1,n}
        PQ_tensor = np.zeros((n_targets, n_targets, n_targets, n_targets, N))
        for (i, j, k, kk), n in product(combo_wr(range(n_targets), 4), range(N)):
            if (j == k) and (n == N - n_targets + j) and (i == kk):
                PQ_tensor[i, j, k, kk, n] = 1.0
            if (j != k) and (n == n_targets + j + k - 1) and (i == kk):
                PQ_tensor[i, j, k, kk, n] = 1 / 2

        return PL_tensor_unsym, PL_tensor, PQ_tensor

    @staticmethod
    def _check_P_matrix(
        n_tgts: int, n_feat: int, n_feat_expected: int, PL: np.ndarray, PQ: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Check if P tensor is properly defined"""
        if (
            PQ is None
            or PL is None
            or PQ.shape != (n_tgts, n_tgts, n_tgts, n_tgts, n_feat)
            or PL.shape != (n_tgts, n_tgts, n_tgts, n_feat)
            or n_feat != n_feat_expected  # library is not quadratic/incorrect shape
        ):
            PL = np.zeros((n_tgts, n_tgts, n_tgts, n_feat))
            PQ = np.zeros((n_tgts, n_tgts, n_tgts, n_tgts, n_feat))
            warnings.warn(
                "PQ and PL tensors not defined, wrong shape, or incompatible with "
                "feature library shape.  Ensure feature library is quadratic. "
                "Setting tensors to zero"
            )
        if not np.allclose(
            np.transpose(PL, [1, 0, 2, 3]), PL, atol=1e-10
        ) or not np.allclose(np.transpose(PQ, [0, 2, 1, 3, 4]), PQ, atol=1e-10):
            raise ValueError("PQ/PL tensors were passed but have the wrong symmetry")
        return PL, PQ

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

    def _solve_m_relax_and_split(self, m_prev, m, A, coef_sparse, tk_previous):
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

    def _solve_m_direct(self, n_tgts, coef_sparse):
        """
        If using the direct formulation of trapping SINDy, solves the
        entire problem in CVXPY regardless of the threshold value.
        Note that this is a convex-composite (i.e. technically nonconvex)
        problem, solved in CVXPY, so convergence/quality guarantees are
        not available here!
        """

        if np.all(self.PL_ == 0) and np.all(self.PQ_ == 0):
            return np.zeros(n_tgts), coef_sparse  # no optimization over m
        else:
            m_cp = cp.Variable(n_tgts)
            L = np.tensordot(self.PL_, coef_sparse, axes=([3, 2], [0, 1]))
            Q = np.reshape(
                np.tensordot(self.PQ_, coef_sparse, axes=([4, 3], [0, 1])),
                (n_tgts, n_tgts * n_tgts),
            )
            Ls = 0.5 * (L + L.T).flatten()
            cost_m = cp.lambda_max(cp.reshape(Ls - m_cp @ Q, (n_tgts, n_tgts)))
            prob_m = cp.Problem(cp.Minimize(cost_m))

            # default solver is SCS here
            prob_m.solve(eps=self.eps_solver, verbose=self.verbose_cvxpy)

            m = m_cp.value
            if m is None:
                print("Infeasible solve over m, increase/decrease eta")
                return None
            return m

    def _reduce(self, x, y):
        """
        Perform at most ``self.max_iter`` iterations of the
        TrappingSR3 algorithm.
        Assumes initial guess for coefficients is stored in ``self.coef_``.
        """
        self.A_history_ = []
        self.m_history_ = []
        self.PW_history_ = []
        self.PWeigs_history_ = []
        self.history_ = []
        n_samples, n_features = x.shape
        n_tgts = y.shape[1]
        n_feat_expected = int((n_tgts**2 + 3 * n_tgts) / 2.0)

        # Only relevant if the stability term is turned on.
        self.PL_unsym_, self.PL_, self.PQ_ = self._set_Ptensors(n_tgts)
        # make sure dimensions/symmetries are correct
        self.PL_, self.PQ_ = self._check_P_matrix(
            n_tgts, n_features, n_feat_expected, self.PL_, self.PQ_
        )

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
                "{: >10} ... {: >10} ... {: >10} ... {: >10} ... {: >10}".format(*row)
            )

        # initial A
        if self.A0 is not None:
            A = self.A0
        elif np.any(self.PQ_ != 0.0):
            A = np.diag(self.gamma * np.ones(n_tgts))
        else:
            A = np.diag(np.zeros(n_tgts))
        self.A_history_.append(A)

        # initial guess for m
        if self.m0 is not None:
            trap_ctr = self.m0
        else:
            np.random.seed(1)
            trap_ctr = (np.random.rand(n_tgts) - np.ones(n_tgts)) * 2
        self.m_history_.append(trap_ctr)

        # Precompute some objects for optimization
        x_expanded = np.zeros((n_samples, n_tgts, n_features, n_tgts))
        for i in range(n_tgts):
            x_expanded[:, i, :, i] = x
        x_expanded = np.reshape(x_expanded, (n_samples * n_tgts, n_tgts * n_features))
        xTx = np.dot(x_expanded.T, x_expanded)
        xTy = np.dot(x_expanded.T, y.flatten())

        # if using acceleration
        tk_prev = 1
        trap_prev_ctr = trap_ctr

        # Begin optimization loop
        self.objective_history_ = []
        for k in range(self.max_iter):

            # update P tensor from the newest trap center
            mPQ = np.tensordot(trap_ctr, self.PQ_, axes=([0], [0]))
            p = self.PL_ - mPQ
            Pmatrix = p.reshape(n_tgts * n_tgts, n_tgts * n_features)

            coef_prev = coef_sparse
            if self.relax_optim:
                if self.threshold > 0.0:
                    # sparse relax_and_split
                    xi, cost = self._create_var_and_part_cost(
                        n_features * n_tgts, x_expanded, y
                    )
                    cost = cost + cp.sum_squares(Pmatrix @ xi - A.flatten()) / self.eta
                    coef_sparse = self._update_coef_cvxpy(
                        xi, cost, n_tgts * n_features, coef_prev, self.eps_solver
                    )
                else:
                    pTp = np.dot(Pmatrix.T, Pmatrix)
                    H = xTx + pTp / self.eta
                    P_transpose_A = np.dot(Pmatrix.T, A.flatten())
                    coef_sparse = self._solve_nonsparse_relax_and_split(
                        H, xTy, P_transpose_A, coef_prev
                    )
                trap_prev_ctr, trap_ctr, A, tk_prev = self._solve_m_relax_and_split(
                    trap_prev_ctr, trap_ctr, A, coef_sparse, tk_prev
                )
            else:
                var_len = n_tgts * n_features
                xi, cost = self._create_var_and_part_cost(var_len, x_expanded, y)
                cost += (
                    cp.lambda_max(cp.reshape(Pmatrix @ xi, (n_tgts, n_tgts))) / self.eta
                )
                coef_sparse = self._update_coef_cvxpy(
                    xi, cost, var_len, coef_prev, self.eps_solver
                )
                trap_ctr = self._solve_m_direct(n_tgts, coef_sparse)

            # If problem over xi becomes infeasible, break out of the loop
            if coef_sparse is None:
                coef_sparse = coef_prev
                break

            # If problem over m becomes infeasible, break out of the loop
            if trap_ctr is None:
                trap_ctr = trap_prev_ctr
                break
            self.history_.append(coef_sparse.T)
            PW = np.tensordot(p, coef_sparse, axes=([3, 2], [0, 1]))

            # (m,A) update finished, append the result
            self.m_history_.append(trap_ctr)
            self.A_history_.append(A)
            eigvals, eigvecs = np.linalg.eig(PW)
            self.PW_history_.append(PW)
            self.PWeigs_history_.append(np.sort(eigvals))

            # update objective
            self.objective_history_.append(self._objective(x, y, coef_sparse, A, PW, k))

            if (
                self._m_convergence_criterion() < self.tol_m
                and self._convergence_criterion() < self.tol
            ):
                break
        else:
            warnings.warn(
                f"TrappingSR3 did not converge after {self.max_iter} iters.",
                ConvergenceWarning,
            )

        self.coef_ = coef_sparse.T
