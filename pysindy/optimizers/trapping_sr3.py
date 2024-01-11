import warnings
from itertools import combinations_with_replacement as combo_wr
from itertools import product
from typing import Tuple
from typing import Union

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
    Generalized trapping variant of sparse relaxed regularized regression.
    This optimizer can be used to identify quadratically nonlinear systems with
    either a-priori globally or locally stable (bounded) solutions.

    This optimizer can be used to minimize five different objective functions:

    .. math::

        0.5\\|y-Xw\\|^2_2 + \\lambda \\times R(w)
        + 0.5\\|Pw-A\\|^2_2/\\eta + \\delta_0(Cw-d)
        + \\delta_{\\Lambda}(A) + \\alpha \\|Qijk\\|
        + \\beta \\|Q_{ijk} + Q_{jik} + Q_{kij}\\|

    where :math:`R(w)` is a regularization function, C is a constraint matrix
    detailing affine constraints on the model coefficients, A is a proxy for
    the quadratic contributions to the energy evolution, and
    :math:`Q_{ijk}` are the quadratic coefficients in the model. For
    provably globally bounded solutions, use :math:`\\alpha >> 1`,
    :math:`\\beta >> 1`, and equality constraints. For maximizing the local
    stability radius of the model one has the choice to do this by
    (1) minimizing the values in :math:`Q_{ijk}`, (2) promoting models
    with skew-symmetrix :math:`Q_{ijk}` coefficients, or
    (3) using inequality constraints for skew-symmetry in :math:`Q_{ijk}`.

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

<<<<<<< HEAD
    eps_solver :
||||||| 5c6e9fd
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
=======
    alpha_m : float, optional (default eta * 0.1)
        Determines the step size in the prox-gradient descent over m.
        For convergence, need alpha_m <= eta / ||w^T * PQ^T * PQ * w||.
        Typically 0.01 * eta <= alpha_m <= 0.1 * eta.

    alpha_A : float, optional (default eta)
        Determines the step size in the prox-gradient descent over A.
        For convergence, need alpha_A <= eta, so typically
        alpha_A = eta is used.

    alpha : float, optional (default 1.0e20)
        Determines the strength of the local stability term ||Qijk||^2 in the
        optimization. The default value is very large so that the
        algorithm default is to ignore this term.

    beta : float, optional (default 1.0e20)
        Determines the strength of the local stability term
        ||Qijk + Qjik + Qkij||^2 in the
        optimization. The default value is very large so that the
        algorithm default is to ignore this term.

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
>>>>>>> origin/trapping_extended
        If threshold != 0, this specifies the error tolerance in the
        CVXPY (OSQP) solve. Default 1.0e-7 (Default is 1.0e-3 in OSQP.)

<<<<<<< HEAD
    relax_optim :
        If relax_optim = True, use the relax-and-split method. If False,
        try a direct minimization on the largest eigenvalue.

||||||| 5c6e9fd
    relax_optim : bool, optional (default True)
        If relax_optim = True, use the relax-and-split method. If False,
        try a direct minimization on the largest eigenvalue.

=======
>>>>>>> origin/trapping_extended
    inequality_constraints : bool, optional (default False)
        If True, CVXPY methods are used.

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

<<<<<<< HEAD
    A0 :
        Initial guess for vector A in the optimization.  Shape (n_targets, n_targets)
        Default None, meaning A is initialized as A = diag(gamma).
||||||| 5c6e9fd
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
=======
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
        If True, prints out the different error terms every iteration.

    verbose_cvxpy : bool, optional (default False)
        Boolean flag which is passed to CVXPY solve function to indicate if
        output should be verbose or not. Only relevant for optimizers that
        use the CVXPY package in some capabity.
>>>>>>> origin/trapping_extended

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

<<<<<<< HEAD
    objective_history_: list
        History of the objective value at each iteration

||||||| 5c6e9fd
=======
    PT_ : np.ndarray, shape (n_targets, n_targets,
                            n_targets, n_targets, n_features)
        Transpose of 1st dimension and 2nd dimension of quadratic coefficient
        part of the P matrix in ||Pw - A||^2

>>>>>>> origin/trapping_extended
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
<<<<<<< HEAD
        *,
        eta: Union[float, None] = None,
        eps_solver: float = 1e-7,
        relax_optim: bool = True,
||||||| 5c6e9fd
        evolve_w=True,
        threshold=0.1,
        eps_solver=1e-7,
        relax_optim=True,
=======
        evolve_w=True,
        threshold=0.1,
        eps_solver=1e-7,
>>>>>>> origin/trapping_extended
        inequality_constraints=False,
<<<<<<< HEAD
        alpha_A: Union[float, None] = None,
        alpha_m: Union[float, None] = None,
        gamma: float = -0.1,
        tol_m: float = 1e-5,
        thresholder: str = "l1",
        accel: bool = False,
        m0: Union[NDArray, None] = None,
        A0: Union[NDArray, None] = None,
        **kwargs,
||||||| 5c6e9fd
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
=======
        eta=None,
        alpha=None,
        beta=None,
        mod_matrix=None,
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
>>>>>>> origin/trapping_extended
    ):
<<<<<<< HEAD
        super().__init__(thresholder=thresholder, **kwargs)
||||||| 5c6e9fd
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
=======
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
        if alpha is None:
            alpha = 1e20
            warnings.warn(
                "alpha was not set, so defaulting to alpha = 1e20 "
                "which is so"
                "large that the ||Qijk|| term in the optimization "
                "will be essentially ignored."
            )
        if beta is None:
            beta = 1e20
            warnings.warn(
                "beta was not set, so defaulting to beta = 1e20 "
                "which is so"
                "large that the ||Qijk + Qjik + Qkij|| "
                "term in the optimization will be essentially ignored."
            )
        if alpha_m < 0 or alpha_m > eta:
            raise ValueError("0 <= alpha_m <= eta")
        if alpha_A < 0 or alpha_A > eta:
            raise ValueError("0 <= alpha_A <= eta")
        if gamma >= 0:
            raise ValueError("gamma must be negative")
        if tol <= 0 or tol_m <= 0 or eps_solver <= 0:
            raise ValueError("tol and tol_m must be positive")

        self.mod_matrix = mod_matrix
        self.evolve_w = evolve_w
>>>>>>> origin/trapping_extended
        self.eps_solver = eps_solver
        self.inequality_constraints = inequality_constraints
        self.m0 = m0
        self.A0 = A0
        self.alpha_A = alpha_A
        self.alpha_m = alpha_m
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tol_m = tol_m
        self.accel = accel
<<<<<<< HEAD
        self.__post_init_guard()
||||||| 5c6e9fd
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
=======
        self.A_history_ = []
        self.m_history_ = []
        self.p_history_ = []
        self.PW_history_ = []
        self.PWeigs_history_ = []
        self.history_ = []
        self.objective_history = objective_history
        self.unbias = False
        self.verbose_cvxpy = verbose_cvxpy
        self.use_constraints = (constraint_lhs is not None) and (
            constraint_rhs is not None
        )
        if inequality_constraints:
            if not evolve_w:
                raise ValueError(
                    "Use of inequality constraints requires solving for xi "
                    " (evolve_w=True)."
                )
            if not self.use_constraints:
                raise ValueError(
                    "Use of inequality constraints requires constraint_rhs "
                    "and constraint_lhs "
                    "variables to be passed to the Optimizer class."
                )
>>>>>>> origin/trapping_extended

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

    def _set_Ptensors(
        self, n_targets: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make the projection tensors used for the algorithm."""
<<<<<<< HEAD
        N = int((n_targets**2 + 3 * n_targets) / 2.0)
||||||| 5c6e9fd
        N = int((r**2 + 3 * r) / 2.0)
=======
        N = self.n_features
        # If bias term is included, need to shift the tensor index
        if N > int((r**2 + 3 * r) / 2.0):
            offset = 1
        else:
            offset = 0

        # If bias term is not included, make it zero
        PC_tensor = np.zeros((r, r, N))
        if offset:
            for i in range(r):
                PC_tensor[i, i, 0] = 1.0
>>>>>>> origin/trapping_extended

        # delta_{il}delta_{jk}
<<<<<<< HEAD
        PL_tensor_unsym = np.zeros((n_targets, n_targets, n_targets, N))
        for i, j in combo_wr(range(n_targets), 2):
            PL_tensor_unsym[i, j, i, j] = 1.0
||||||| 5c6e9fd
        PL_tensor = np.zeros((r, r, r, N))
        PL_tensor_unsym = np.zeros((r, r, r, N))
        for i in range(r):
            for j in range(r):
                for k in range(r):
                    for kk in range(N):
                        if i == k and j == kk:
                            PL_tensor_unsym[i, j, k, kk] = 1.0
=======
        PL_tensor = np.zeros((r, r, r, N))
        PL_tensor_unsym = np.zeros((r, r, r, N))
        for i in range(r):
            for j in range(r):
                for k in range(r):
                    for kk in range(offset, N):
                        if i == k and j == (kk - offset):
                            PL_tensor_unsym[i, j, k, kk] = 1.0
>>>>>>> origin/trapping_extended

        # Now symmetrize PL
<<<<<<< HEAD
        PL_tensor = (PL_tensor_unsym + np.transpose(PL_tensor_unsym, [1, 0, 2, 3])) / 2
||||||| 5c6e9fd
        for i in range(r):
            for j in range(N):
                PL_tensor[:, :, i, j] = 0.5 * (
                    PL_tensor_unsym[:, :, i, j] + PL_tensor_unsym[:, :, i, j].T
                )
=======
        for i in range(r):
            for j in range(offset, N):
                PL_tensor[:, :, i, j] = 0.5 * (
                    PL_tensor_unsym[:, :, i, j] + PL_tensor_unsym[:, :, i, j].T
                )
>>>>>>> origin/trapping_extended

        # if j == k, delta_{il}delta_{N-r+j,n}
<<<<<<< HEAD
        # if j != k, delta_{il}delta_{r+j+k-1,n}
        PQ_tensor = np.zeros((n_targets, n_targets, n_targets, n_targets, N))
        for (i, j, k, kk), n in product(combo_wr(range(n_targets), 4), range(N)):
            if (j == k) and (n == N - n_targets + j) and (i == kk):
                PQ_tensor[i, j, k, kk, n] = 1.0
            if (j != k) and (n == n_targets + j + k - 1) and (i == kk):
                PQ_tensor[i, j, k, kk, n] = 1 / 2
||||||| 5c6e9fd
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
=======
        # if j != k, delta_{il}delta_{r+j*(2*r-j-3)/2+k-1,n} if j<k; swap j & k
        # in the second delta operator if j > k
        # PT projects out the transpose of the 1st dimension and 2nd dimension of Q
        PQ_tensor = np.zeros((r, r, r, r, N))
        PT_tensor = np.zeros((r, r, r, r, N))
        for i in range(r):
            for j in range(r):
                for k in range(r):
                    for kk in range(r):
                        for n in range(N):
                            if (j == k) and (n == N - r + j) and (i == kk):
                                PQ_tensor[i, j, k, kk, n] = 1.0
                                PT_tensor[j, i, k, kk, n] = 1.0
                            if (
                                (j != k)
                                and (
                                    (n - offset)
                                    == r
                                    + np.min([j, k]) * (2 * r - np.min([j, k]) - 3) / 2
                                    + np.max([j, k])
                                    - 1
                                )
                                and (i == kk)
                            ):
                                PQ_tensor[i, j, k, kk, n] = 1 / 2.0
                                PT_tensor[j, i, k, kk, n] = 1 / 2.0
>>>>>>> origin/trapping_extended

        # PM is the sum of PQ and PQ which projects out the sum of Qijk and Qjik
        PM_tensor = PQ_tensor + PT_tensor

        return PC_tensor, PL_tensor_unsym, PL_tensor, PQ_tensor, PT_tensor, PM_tensor

<<<<<<< HEAD
    @staticmethod
    def _check_P_matrix(
        n_tgts: int, n_feat: int, n_feat_expected: int, PL: np.ndarray, PQ: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
||||||| 5c6e9fd
    def _bad_PL(self, PL):
        """Check if PL tensor is properly defined"""
        tol = 1e-10
        return np.any((np.transpose(PL, [1, 0, 2, 3]) - PL) > tol)

    def _bad_PQ(self, PQ):
        """Check if PQ tensor is properly defined"""
        tol = 1e-10
        return np.any((np.transpose(PQ, [0, 2, 1, 3, 4]) - PQ) > tol)

    def _check_P_matrix(self, r, n_features, N):
=======
    def _bad_PL(self, PL):
        """Check if PL tensor is properly defined"""
        tol = 1e-10
        return np.any((np.transpose(PL, [1, 0, 2, 3]) - PL) > tol)

    def _bad_PQ(self, PQ):
        """Check if PQ tensor is properly defined"""
        tol = 1e-10
        return np.any((np.transpose(PQ, [0, 2, 1, 3, 4]) - PQ) > tol)

    def _bad_PT(self, PT):
        """Check if PT tensor is properly defined"""
        tol = 1e-10
        return np.any((np.transpose(PT, [2, 1, 0, 3, 4]) - PT) > tol)

    def _check_P_matrix(self, r, n_features, N):
>>>>>>> origin/trapping_extended
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
<<<<<<< HEAD
        if not np.allclose(
            np.transpose(PL, [1, 0, 2, 3]), PL, atol=1e-10
        ) or not np.allclose(np.transpose(PQ, [0, 2, 1, 3, 4]), PQ, atol=1e-10):
            raise ValueError("PQ/PL tensors were passed but have the wrong symmetry")
        return PL, PQ
||||||| 5c6e9fd
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
=======
        if self.PT_ is None:
            self.PT_ = np.zeros((r, r, r, r, n_features))
            warnings.warn(
                "The PT tensor (a requirement for the stability promotion) was"
                " not set, so setting this tensor to all zeros. "
            )
        elif (self.PT_).shape != (r, r, r, r, n_features) and (self.PT_).shape != (
            r,
            r,
            r,
            r,
            N,
        ):
            self.PT_ = np.zeros((r, r, r, r, n_features))
            warnings.warn(
                "The PT tensor (a requirement for the stability promotion) was"
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
        if self._bad_PT(self.PT_):
            raise ValueError("PT tensor was passed but the symmetries are not correct")

        # If PL/PQ/PT finite and correct, so trapping theorem is being used,
        # then make sure library is quadratic and correct shape
        if (
            np.any(self.PL_ != 0.0)
            or np.any(self.PQ_ != 0.0)
            or np.any(self.PT_ != 0.0)
        ) and n_features != N:
            print(
                "The feature library is the wrong shape or not quadratic, "
                "so please correct this if you are attempting to use the "
                "trapping algorithm with the stability term included. Setting "
                "PL and PQ tensors to zeros for now."
            )
            self.PL_ = np.zeros((r, r, r, n_features))
            self.PQ_ = np.zeros((r, r, r, r, n_features))
            self.PT_ = np.zeros((r, r, r, r, n_features))
>>>>>>> origin/trapping_extended

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

    def _objective(self, x, y, coef_sparse, A, PW, k):
        """Objective function"""
        # Compute the errors
        R2 = (y - np.dot(x, coef_sparse)) ** 2
        A2 = (A - PW) ** 2
        Qijk = np.tensordot(
            self.mod_matrix,
            np.tensordot(self.PQ_, coef_sparse, axes=([4, 3], [0, 1])),
            axes=([1], [0]),
        )
        beta2 = (
            Qijk + np.transpose(Qijk, [1, 2, 0]) + np.transpose(Qijk, [2, 0, 1])
        ) ** 2
        L1 = self.threshold * np.sum(np.abs(coef_sparse.flatten()))
        R2 = 0.5 * np.sum(R2)
        stability_term = 0.5 * np.sum(A2) / self.eta
        alpha_term = 0.5 * np.sum(Qijk**2) / self.alpha
        beta_term = 0.5 * np.sum(beta2) / self.beta

        # convoluted way to print every max_iter / 10 iterations
        if self.verbose and k % max(1, self.max_iter // 10) == 0:
            row = [
                k,
                R2,
                stability_term,
                L1,
                alpha_term,
                beta_term,
                R2 + stability_term + L1 + alpha_term + beta_term,
            ]
            if self.threshold == 0:
                if k % max(int(self.max_iter / 10.0), 1) == 0:
                    print(
                        "{0:5d} ... {1:8.3e} ... {2:8.3e} ... {3:8.2e}"
                        " ... {4:8.2e} ... {5:8.2e} ... {6:8.2e}".format(*row)
                    )
            else:
                print(
                    "{0:5d} ... {1:8.3e} ... {2:8.3e} ... {3:8.2e}"
                    " ... {4:8.2e} ... {5:8.2e} ... {6:8.2e}".format(*row)
                )
        return R2 + stability_term + L1 + alpha_term + beta_term

<<<<<<< HEAD
    def _solve_m_relax_and_split(self, m_prev, m, A, coef_sparse, tk_previous):
||||||| 5c6e9fd
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
=======
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

        # new terms minimizing quadratic piece ||P^Q @ xi||_2^2
        Q = np.reshape(self.PQ_, (r * r * r, N * r), "F")
        cost = cost + cp.sum_squares(Q @ xi) / self.alpha
        Q = np.reshape(self.PQ_, (r, r, r, N * r), "F")
        Q_ep = Q + np.transpose(Q, [1, 2, 0, 3]) + np.transpose(Q, [2, 0, 1, 3])
        Q_ep = np.reshape(Q_ep, (r * r * r, N * r), "F")
        cost = cost + cp.sum_squares(Q_ep @ xi) / self.beta

        # Constraints
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
>>>>>>> origin/trapping_extended
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
            mPM = np.tensordot(self.PM_, m_partial, axes=([2], [0]))
        else:
            mPM = np.tensordot(self.PM_, m, axes=([2], [0]))
        p = np.tensordot(self.mod_matrix, self.PL_ + mPM, axes=([1], [0]))
        PW = np.tensordot(p, coef_sparse, axes=([3, 2], [0, 1]))
        PMW = np.tensordot(self.PM_, coef_sparse, axes=([4, 3], [0, 1]))
        PMW = np.tensordot(self.mod_matrix, PMW, axes=([1], [0]))
        A_b = (A - PW) / self.eta
        PMT_PW = np.tensordot(PMW, A_b, axes=([2, 1], [0, 1]))
        if self.accel:
            m_new = m_partial - self.alpha_m * PMT_PW
        else:
            m_new = m_prev - self.alpha_m * PMT_PW
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

<<<<<<< HEAD
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

||||||| 5c6e9fd
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

=======
>>>>>>> origin/trapping_extended
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
<<<<<<< HEAD
        n_tgts = y.shape[1]
        var_len = n_features * n_tgts
        n_feat_expected = int((n_tgts**2 + 3 * n_tgts) / 2.0)
||||||| 5c6e9fd
        r = y.shape[1]
        N = int((r**2 + 3 * r) / 2.0)
=======
        self.n_features = n_features
        r = y.shape[1]
        N = n_features  # int((r ** 2 + 3 * r) / 2.0)
>>>>>>> origin/trapping_extended

<<<<<<< HEAD
        # Only relevant if the stability term is turned on.
        self.PL_unsym_, self.PL_, self.PQ_ = self._set_Ptensors(n_tgts)
||||||| 5c6e9fd
        # Define PL and PQ tensors, only relevant if the stability term in
        # trapping SINDy is turned on.
        self.PL_unsym_, self.PL_, self.PQ_ = self._set_Ptensors(r)
=======
        if self.mod_matrix is None:
            self.mod_matrix = np.eye(r)

        # Define PL, PQ, PT and PM tensors, only relevant if the stability term in
        # trapping SINDy is turned on.
        (
            self.PC_,
            self.PL_unsym_,
            self.PL_,
            self.PQ_,
            self.PT_,
            self.PM_,
        ) = self._set_Ptensors(r)
>>>>>>> origin/trapping_extended
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
                "Iter",
                "|y-Xw|^2",
                "|Pw-A|^2/eta",
                "|w|_1",
                "|Qijk|/a",
                "|Qijk+...|/b",
                "Total:",
            ]
            print(
<<<<<<< HEAD
                "{: >10} ... {: >10} ... {: >10} ... {: >10} ... {: >10}".format(*row)
||||||| 5c6e9fd
                "{: >10} ... {: >10} ... {: >10} ... {: >10}"
                " ... {: >10}".format(*row)
=======
                "{: >5} ... {: >8} ... {: >10} ... {: >5}"
                " ... {: >8} ... {: >10} ... {: >8}".format(*row)
>>>>>>> origin/trapping_extended
            )

        # initial A
        if self.A0 is not None:
            A = self.A0
<<<<<<< HEAD
        elif np.any(self.PQ_ != 0.0):
            A = np.diag(self.gamma * np.ones(n_tgts))
||||||| 5c6e9fd
        elif np.any(self.PQ_ != 0.0):
            A = np.diag(self.gamma * np.ones(r))
=======
        elif np.any(self.PM_ != 0.0):
            A = np.diag(self.gamma * np.ones(r))
>>>>>>> origin/trapping_extended
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

<<<<<<< HEAD
            # update P tensor from the newest trap center
            mPQ = np.tensordot(trap_ctr, self.PQ_, axes=([0], [0]))
            p = self.PL_ - mPQ
            Pmatrix = p.reshape(n_tgts * n_tgts, n_tgts * n_features)
||||||| 5c6e9fd
            # update P tensor from the newest m
            mPQ = np.tensordot(m, self.PQ_, axes=([0], [0]))
            p = self.PL_ - mPQ
            Pmatrix = p.reshape(r * r, r * n_features)
=======
            # update P tensor from the newest m
            mPM = np.tensordot(self.PM_, m, axes=([2], [0]))
            p = np.tensordot(self.mod_matrix, self.PL_ + mPM, axes=([1], [0]))
            Pmatrix = p.reshape(r * r, r * n_features)
>>>>>>> origin/trapping_extended

            coef_prev = coef_sparse
<<<<<<< HEAD
            if self.relax_optim:
                if self.threshold > 0.0:
                    # sparse relax_and_split
                    coef_sparse = self._update_coef_sparse_rs(
                        var_len, x_expanded, y, Pmatrix, A, coef_prev
||||||| 5c6e9fd
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
=======
            if self.evolve_w:
                if (self.threshold > 0.0) or self.inequality_constraints:
                    coef_sparse = self._solve_sparse_relax_and_split(
                        r, n_features, x_expanded, y, Pmatrix, A, coef_prev
                    )
                else:
                    # if threshold = 0, there is analytic expression
                    # for the solve over the coefficients,
                    # which is coded up here separately
                    pTp = np.dot(Pmatrix.T, Pmatrix)
                    # notice reshaping PQ here requires fortran-ordering
                    PQ = np.tensordot(self.mod_matrix, self.PQ_, axes=([1], [0]))
                    PQ = np.reshape(PQ, (r * r * r, r * n_features), "F")
                    PQTPQ = np.dot(PQ.T, PQ)
                    PQ = np.reshape(self.PQ_, (r, r, r, r * n_features), "F")
                    PQ = np.tensordot(self.mod_matrix, PQ, axes=([1], [0]))
                    PQ_ep = (
                        PQ
                        + np.transpose(PQ, [1, 2, 0, 3])
                        + np.transpose(PQ, [2, 0, 1, 3])
                    )
                    PQ_ep = np.reshape(PQ_ep, (r * r * r, r * n_features), "F")
                    PQTPQ_ep = np.dot(PQ_ep.T, PQ_ep)
                    H = xTx + pTp / self.eta + PQTPQ / self.alpha + PQTPQ_ep / self.beta
                    P_transpose_A = np.dot(Pmatrix.T, A.flatten())
                    coef_sparse = self._solve_nonsparse_relax_and_split(
                        H, xTy, P_transpose_A, coef_prev
>>>>>>> origin/trapping_extended
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

<<<<<<< HEAD
||||||| 5c6e9fd
            if self.relax_optim:
                m_prev, m, A, tk_prev = self._solve_m_relax_and_split(
                    r, n_features, m_prev, m, A, coef_sparse, tk_prev
                )

=======
            # Now solve optimization for m and A
            m_prev, m, A, tk_prev = self._solve_m_relax_and_split(
                r, n_features, m_prev, m, A, coef_sparse, tk_prev
            )

>>>>>>> origin/trapping_extended
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
            mPM = np.tensordot(self.PM_, m, axes=([2], [0]))
            p = np.tensordot(self.mod_matrix, self.PL_ + mPM, axes=([1], [0]))
            self.p_history_.append(p)

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
