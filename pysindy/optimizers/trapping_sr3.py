import warnings
from itertools import combinations as combo_nr
from itertools import product
from itertools import repeat
from math import comb
from typing import cast
from typing import NewType
from typing import Tuple
from typing import Union

import cvxpy as cp
import numpy as np
from numpy import intp
from numpy.typing import NBitBase
from numpy.typing import NDArray
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from sklearn.exceptions import ConvergenceWarning

from ..feature_library.polynomial_library import n_poly_features
from ..feature_library.polynomial_library import PolynomialLibrary
from ..utils import reorder_constraints
from .constrained_sr3 import ConstrainedSR3

AnyFloat = np.dtype[np.floating[NBitBase]]
Int1D = np.ndarray[tuple[int], np.dtype[np.int_]]
Float2D = np.ndarray[tuple[int, int], AnyFloat]
Float4D = np.ndarray[tuple[int, int, int, int], AnyFloat]
Float5D = np.ndarray[tuple[int, int, int, int, int], AnyFloat]
FloatND = NDArray[np.floating[NBitBase]]
NFeat = NewType("NFeat", int)
NTarget = NewType("NTarget", int)


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
        _n_tgts: int = None,
        _include_bias: bool = False,
        _interaction_only: bool = False,
        eta: Union[float, None] = None,
        eps_solver: float = 1e-7,
        relax_optim: bool = True,
        alpha_A: Union[float, None] = None,
        alpha_m: Union[float, None] = None,
        gamma: float = -0.1,
        tol_m: float = 1e-5,
        thresholder: str = "l1",
        accel: bool = False,
        m0: Union[NDArray, None] = None,
        A0: Union[NDArray, None] = None,
        **kwargs,
    ):
        # n_tgts, constraints, etc are data-dependent parameters and belong in
        # _reduce/fit ().  The following is a hack until we refactor how
        # constraints are applied in ConstrainedSR3 and MIOSR
        self._include_bias = _include_bias
        self._interaction_only = _interaction_only
        self._n_tgts = _n_tgts
        if _n_tgts is None:
            warnings.warn(
                "Trapping Optimizer initialized without _n_tgts.  It will likely"
                " be unable to fit data"
            )
            _n_tgts = 1
        if _include_bias:
            raise ValueError(
                "Currently not able to include bias until PQ matrices are modified"
            )
        if hasattr(kwargs, "constraint_separation_index"):
            constraint_separation_index = kwargs["constraint_separation_index"]
        elif kwargs.get("inequality_constraints", False):
            constraint_separation_index = kwargs["constraint_lhs"].shape[0]
        else:
            constraint_separation_index = 0
        constraint_rhs, constraint_lhs = _make_constraints(
            _n_tgts, include_bias=_include_bias
        )
        constraint_order = kwargs.pop("constraint_order", "feature")
        if constraint_order == "target":
            constraint_lhs = np.transpose(constraint_lhs, [0, 2, 1])
        constraint_lhs = np.reshape(constraint_lhs, (constraint_lhs.shape[0], -1))
        try:
            constraint_lhs = np.concatenate(
                (kwargs.pop("constraint_lhs"), constraint_lhs), 0
            )
            constraint_rhs = np.concatenate(
                (kwargs.pop("constraint_rhs"), constraint_rhs), 0
            )
        except KeyError:
            pass

        super().__init__(
            constraint_lhs=constraint_lhs,
            constraint_rhs=constraint_rhs,
            constraint_separation_index=constraint_separation_index,
            constraint_order=constraint_order,
            equality_constraints=True,
            thresholder=thresholder,
            **kwargs,
        )
        self.eps_solver = eps_solver
        self.relax_optim = relax_optim
        self.m0 = m0
        self.A0 = A0
        self.alpha_A = alpha_A
        self.alpha_m = alpha_m
        self.eta = eta
        self.gamma = gamma
        self.tol_m = tol_m
        self.accel = accel
        self.__post_init_guard()

    def __post_init_guard(self):
        """Conduct initialization post-init, as required by scikitlearn API"""
        if self.thresholder.lower() not in ("l1", "l2", "weighted_l1", "weighted_l2"):
            raise ValueError("Regularizer must be (weighted) L1 or L2")
        if self.eta is None:
            warnings.warn(
                "eta was not set, so defaulting to eta = 1e20 "
                "with alpha_m = 1e-2 * eta, alpha_A = eta. Here eta is so "
                "large that the stability term in the optimization "
                "will be ignored."
            )
            self.eta = 1e20
            self.alpha_m = 1e18
            self.alpha_A = 1e20
        else:
            if self.alpha_m is None:
                self.alpha_m = self.eta * 1e-2
            if self.alpha_A is None:
                self.alpha_A = self.eta
        if self.eta <= 0:
            raise ValueError("eta must be positive")
        if self.alpha_m < 0 or self.alpha_m > self.eta:
            raise ValueError("0 <= alpha_m <= eta")
        if self.alpha_A < 0 or self.alpha_A > self.eta:
            raise ValueError("0 <= alpha_A <= eta")
        if self.gamma >= 0:
            raise ValueError("gamma must be negative")
        if self.tol <= 0 or self.tol_m <= 0 or self.eps_solver <= 0:
            raise ValueError("tol and tol_m must be positive")
        if self.inequality_constraints and self.relax_optim and self.threshold == 0.0:
            raise ValueError(
                "Ineq. constr. -> threshold!=0 + relax_optim=True or relax_optim=False."
            )

    def set_params(self, **kwargs):
        super().set_params(**kwargs)
        self.__post_init_guard

    @staticmethod
    def _build_PL(polyterms: list[tuple[int, Int1D]]) -> tuple[Float4D, Float4D]:
        r"""Build the matrix that projects out the linear terms of a library

        Coefficients in each polynomial equation :math:`i\in \{1,\dots, r\}` can
        be stored in an array arranged as written out on paper (e.g.
        :math:` f_i(x) = a^i_0 + a^i_1 x_1, a^i_2 x_1x_2, \dots a^i_N x_r^2`) or
        in a series of matrices :math:`E \in \mathbb R^r`,
        :math:`L\in \mathbb R^{r\times r}`, and (without loss of generality) in
        :math:`Q\in \mathbb R^{r \times r \times r}, where each
        :math:`Q^{(i)}_{j,k}` is symmetric in the last two indexes.

        This function builds the projection tensor for extracting the linear
        terms :math:`L` from a set of coefficients in the first representation.
        The function also calculates the projection tensor for extracting the
        symmetrized version of L

        Args:
            polyterms: the ordering and meaning of terms in the equations.  Each
                entry represents a term in the equation and comprises its index
                and an array of exponents for each variable

        Returns:
            Two 4th order tensors, the first one symmetric in the first two
            indexes.
        """
        n_targets, n_features, lin_terms, _, _ = _build_lib_info(polyterms)
        PL_tensor_unsym = np.zeros((n_targets, n_targets, n_targets, n_features))
        tgts = range(n_targets)
        for j in range(n_targets):
            PL_tensor_unsym[tgts, j, tgts, lin_terms[j]] = 1
        PL_tensor = (PL_tensor_unsym + np.transpose(PL_tensor_unsym, [1, 0, 2, 3])) / 2
        return cast(Float4D, PL_tensor), cast(Float4D, PL_tensor_unsym)

    @staticmethod
    def _build_PQ(polyterms: list[tuple[int, Int1D]]) -> Float5D:
        r"""Build the matrix that projects out the quadratic terms of a library

        Coefficients in each polynomial equation :math:`i\in \{1,\dots, r\}` can
        be stored in an array arranged as written out on paper (e.g.
        :math:` f_i(x) = a^i_0 + a^i_1 x_1, a^i_2 x_1x_2, \dots a^i_N x_r^2`) or
        in a series of matrices :math:`E \in \mathbb R^r`,
        :math:`L\in \mathbb R^{r\times r}`, and (without loss of generality) in
        :math:`Q\in \mathbb R^{r \times r \times r}, where each
        :math:`Q^{(i)}_{j,k}` is symmetric in the last two indexes.

        This function builds the projection tensor for extracting the quadratic
        forms :math:`Q` from a set of coefficients in the first representation.

        Args:
            polyterms: the ordering and meaning of terms in the equations.  Each
                entry represents a term in the equation and comprises its index
                and an array of exponents for each variable

        Returns:
            5th order tensor symmetric in second and third indexes.
        """
        n_targets, n_features, _, pure_terms, mixed_terms = _build_lib_info(polyterms)
        PQ = np.zeros((n_targets, n_targets, n_targets, n_targets, n_features))
        tgts = range(n_targets)
        for j, k in product(*repeat(range(n_targets), 2)):
            if j == k:
                PQ[tgts, j, k, tgts, pure_terms[j]] = 1.0
            if j != k:
                PQ[tgts, j, k, tgts, mixed_terms[frozenset({j, k})]] = 1 / 2
        return cast(Float5D, PQ)

    def _set_Ptensors(
        self, n_targets: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make the projection tensors used for the algorithm."""
        lib = PolynomialLibrary(2, include_bias=self._include_bias).fit(
            np.zeros((1, n_targets))
        )
        polyterms = [(t_ind, exps) for t_ind, exps in enumerate(lib.powers_)]

        PL_tensor, PL_tensor_unsym = self._build_PL(polyterms)
        PQ_tensor = self._build_PQ(polyterms)

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
        n_samples, n_tgts = y.shape
        n_features = n_poly_features(
            n_tgts,
            2,
            include_bias=self._include_bias,
            interaction_only=self._interaction_only,
        )
        var_len = n_features * n_tgts

        # Only relevant if the stability term is turned on.
        self.PL_unsym_, self.PL_, self.PQ_ = self._set_Ptensors(n_tgts)
        # make sure dimensions/symmetries are correct
        self.PL_, self.PQ_ = self._check_P_matrix(
            n_tgts, n_features, n_features, self.PL_, self.PQ_
        )

        # Set initial coefficients
        if self.use_constraints and self.constraint_order.lower() == "target":
            self.constraint_lhs = reorder_constraints(
                self.constraint_lhs, n_features, output_order="feature"
            )
        coef_sparse: np.ndarray[tuple[NFeat, NTarget], AnyFloat] = self.coef_.T

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
                    coef_sparse = self._update_coef_sparse_rs(
                        var_len, x_expanded, y, Pmatrix, A, coef_prev
                    )
                else:
                    coef_sparse = self._update_coef_nonsparse_rs(
                        Pmatrix, A, coef_prev, xTx, xTy
                    )
                trap_prev_ctr, trap_ctr, A, tk_prev = self._solve_m_relax_and_split(
                    trap_prev_ctr, trap_ctr, A, coef_sparse, tk_prev
                )
            else:
                coef_sparse = self._update_coef_direct(
                    var_len, x_expanded, y, Pmatrix, coef_prev, n_tgts
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

    def _update_coef_sparse_rs(self, var_len, x_expanded, y, Pmatrix, A, coef_prev):
        xi, cost = self._create_var_and_part_cost(var_len, x_expanded, y)
        cost = cost + cp.sum_squares(Pmatrix @ xi - A.flatten()) / self.eta
        return self._update_coef_cvxpy(xi, cost, var_len, coef_prev, self.eps_solver)

    def _update_coef_nonsparse_rs(self, Pmatrix, A, coef_prev, xTx, xTy):
        pTp = np.dot(Pmatrix.T, Pmatrix)
        H = xTx + pTp / self.eta
        P_transpose_A = np.dot(Pmatrix.T, A.flatten())
        return self._solve_nonsparse_relax_and_split(H, xTy, P_transpose_A, coef_prev)

    def _update_coef_direct(self, var_len, x_expanded, y, Pmatrix, coef_prev, n_tgts):
        xi, cost = self._create_var_and_part_cost(var_len, x_expanded, y)
        cost += cp.lambda_max(cp.reshape(Pmatrix @ xi, (n_tgts, n_tgts))) / self.eta
        return self._update_coef_cvxpy(xi, cost, var_len, coef_prev, self.eps_solver)


def _make_constraints(n_tgts: int, **kwargs):
    """Create constraints for the Quadratic terms in TrappingSR3.

    These are the constraints from equation 5 of the Trapping SINDy paper.

    Args:
        n_tgts: number of coordinates or modes for which you're fitting an ODE.
        kwargs: Keyword arguments to PolynomialLibrary such as
            ``include_bias``.

    Returns:
        A tuple of the constraint zeros, and a constraint matrix to multiply
        by the coefficient matrix of Polynomial terms. Number of constraints is
        ``n_tgts + 2 * math.comb(n_tgts, 2) + math.comb(n_tgts, 3)``.
        Constraint matrix is of shape ``(n_constraint, n_feature, n_tgt)``.
        To get "feature" order constraints, use
        ``np.reshape(constraint_matrix, (n_constraints, -1))``.
        To get "target" order constraints, transpose axis 1 and 2 before
        reshaping.
    """
    lib = PolynomialLibrary(2, **kwargs).fit(np.zeros((1, n_tgts)))
    terms = [(t_ind, exps) for t_ind, exps in enumerate(lib.powers_)]
    _, n_terms, linear_terms, pure_terms, mixed_terms = _build_lib_info(terms)
    # index of tgt -> index of its pure quadratic term
    pure_terms = {np.argmax(exps): t_ind for t_ind, exps in terms if max(exps) == 2}
    # two indexes of tgts -> index of their mixed quadratic term
    mixed_terms = {
        frozenset(np.argwhere(exponent == 1).flatten()): t_ind
        for t_ind, exponent in terms
        if max(exponent) == 1 and sum(exponent) == 2
    }
    constraint_mat = np.vstack(
        (
            _pure_constraints(n_tgts, n_terms, pure_terms),
            _antisymm_double_constraint(n_tgts, n_terms, pure_terms, mixed_terms),
            _antisymm_triple_constraints(n_tgts, n_terms, mixed_terms),
        )
    )

    return np.zeros(len(constraint_mat)), constraint_mat


def _pure_constraints(
    n_tgts: int, n_terms: int, pure_terms: dict[intp, int]
) -> Float2D:
    """Set constraints for coefficients adorning terms like a_i^3 = 0"""
    constraint_mat = np.zeros((n_tgts, n_terms, n_tgts))
    for constr_ind, (tgt_ind, term_ind) in zip(range(n_tgts), pure_terms.items()):
        constraint_mat[constr_ind, term_ind, tgt_ind] = 1.0
    return constraint_mat


def _antisymm_double_constraint(
    n_tgts: int,
    n_terms: int,
    pure_terms: dict[intp, int],
    mixed_terms: dict[frozenset[intp], int],
) -> Float2D:
    """Set constraints for coefficients adorning terms like a_i^2 * a_j=0"""
    constraint_mat_1 = np.zeros((comb(n_tgts, 2), n_terms, n_tgts))  # a_i^2 * a_j
    constraint_mat_2 = np.zeros((comb(n_tgts, 2), n_terms, n_tgts))  # a_i * a_j^2
    for constr_ind, ((tgt_i, tgt_j), mix_term) in zip(
        range(n_tgts), mixed_terms.items()
    ):
        constraint_mat_1[constr_ind, mix_term, tgt_i] = 1.0
        constraint_mat_1[constr_ind, pure_terms[tgt_i], tgt_j] = 1.0
        constraint_mat_2[constr_ind, mix_term, tgt_j] = 1.0
        constraint_mat_2[constr_ind, pure_terms[tgt_j], tgt_i] = 1.0

    return np.concatenate((constraint_mat_1, constraint_mat_2), axis=0)


def _antisymm_triple_constraints(
    n_tgts: int, n_terms: int, mixed_terms: dict[frozenset[intp], int]
) -> Float2D:
    constraint_mat = np.zeros((comb(n_tgts, 3), n_terms, n_tgts))  # a_ia_ja_k

    def find_symm_term(a, b):
        return mixed_terms[frozenset({a, b})]

    for constr_ind, (tgt_i, tgt_j, tgt_k) in enumerate(combo_nr(range(n_tgts), 3)):
        constraint_mat[constr_ind, find_symm_term(tgt_j, tgt_k), tgt_i] = 1
        constraint_mat[constr_ind, find_symm_term(tgt_k, tgt_i), tgt_j] = 1
        constraint_mat[constr_ind, find_symm_term(tgt_i, tgt_j), tgt_k] = 1

    return constraint_mat


def _build_lib_info(
    polyterms: list[tuple[int, Int1D]]
) -> tuple[int, int, dict[int, int], dict[int, int], dict[frozenset[int], int]]:
    """From polynomial, calculate various useful info

    Args:
        polyterms.  The output of PolynomialLibrary.powers_.  Each term is
            a tuple of it's index in the ordering and a 1D array of the
            exponents of each feature.

    Returns:
        the number of targets
        the number of features
        a dictionary from each target to its linear term index
        a dictionary from each target to its quadratic term index
        a dictionary from each pair of targets to its mixed term index
    """
    try:
        n_targets = len(polyterms[0][1])
    except IndexError:
        raise ValueError("Passed a polynomial library with no terms")
    n_features = len(polyterms)
    mixed_terms = {
        frozenset(np.argwhere(exps == 1).flatten()): t_ind
        for t_ind, exps in polyterms
        if max(exps) == 1 and sum(exps) == 2
    }
    pure_terms = {np.argmax(exps): t_ind for t_ind, exps in polyterms if max(exps) == 2}
    linear_terms = {
        np.argmax(exps): t_ind for t_ind, exps in polyterms if sum(exps) == 1
    }
    return n_targets, n_features, linear_terms, pure_terms, mixed_terms
