import warnings

import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from sklearn.exceptions import ConvergenceWarning
import cvxpy as cp
from ..utils import get_regularization, get_prox
from ..utils import reorder_constraints
from .sr3 import SR3


class clSR3(SR3):
    """
    Sparse relaxed regularized regression.

    Attempts to minimize the objective function

    .. math::

        0.5\\|y-Xw\\|^2_2 + \\lambda \\times R(w)
        + (0.5 / \\eta)\\|Pw-A\\|^2_2 + \\delta_0(Cw-d)
        + \\delta_{}\\Lambda(A)}

    where :math:`R(w)` is a regularization function.
    See the following references for more details:

        Zheng, Peng, et al. "A unified framework for sparse relaxed
        regularized regression: Sr3." IEEE Access 7 (2018): 1404-1423.

        Champion, Kathleen, et al. "A unified sparse optimization framework
        to learn parsimonious physics-informed models from data."
        arXiv preprint arXiv:1906.10612 (2019).

        New paper, Kaptanoglu et al. on trapping SINDy algorithms

    Parameters
    ----------
    threshold : float, optional (default 0.1)
        Determines the strength of the regularization. When the
        regularization function R is the L0 norm, the regularization
        is equivalent to performing hard thresholding, and lambda
        is chosen to threshold at the value given by this parameter.
        This is equivalent to choosing lambda = threshold^2 / (2 * nu).

    tol : float, optional (default 1e-5)
        Tolerance used for determining convergence of the optimization
        algorithm.

    thresholder : string, optional (default 'L0')
        Regularization function to use. Currently implemented options
        are 'L0' (L0 norm), 'L1' (L1 norm), and 'CAD' (clipped
        absolute deviation).

    max_iter : int, optional (default 30)
        Maximum iterations of the optimization algorithm.

    initial_guess : np.ndarray, shape (n_features) or (n_targets, n_features),
            optional (default None)
        Initial guess for coefficients ``coef_``.
        If None, least-squares is used to obtain an initial guess.

    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the L2-norm.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Regularized weight vector(s). This is the v in the objective
        function.

    history_ : list
        History of sparse coefficients. ``history_[k]`` contains the
        sparse coefficients (v in the optimization objective function)
        at iteration k.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import clSR3
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = clSR3(threshold=0.1)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -10.004 1 + 10.004 x0
    x1' = 27.994 1 + -0.993 x0 + -1.000 1 x1
    x2' = -2.662 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        w_evo=True,
        threshold=0.1,
        eta=1.0,
        alpha_A=0.5,
        alpha_m=0.5,
        eigmax=-0.1,
        eigmin=-10,
        tol=1e-5,
        vtol=1e-5,
        thresholder="l0",
        max_iter=30,
        accel=False,
        normalize=False,
        fit_intercept=False,
        copy_X=True,
        initial_guess=None,
        PL=None,
        PQ=None,
        thresholds=None,
        objective_history=None,
        constraint_lhs=None,
        constraint_rhs=None,
        constraint_order="target",
    ):
        super(clSR3, self).__init__(
            max_iter=max_iter,
            initial_guess=initial_guess,
            normalize=normalize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
        )

        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        if eta <= 0:
            raise ValueError("eta must be positive")
        if alpha_m < 0:
            raise ValueError("alpha_m must be positive")
        if alpha_A < 0:
            raise ValueError("alpha_A must be positive")
        if tol <= 0 or vtol <= 0:
            raise ValueError("tol and vtol must be positive")

        self.w_evo = w_evo
        self.threshold = threshold
        self.thresholds = thresholds
        self.alpha_A = alpha_A
        self.alpha_m = alpha_m
        self.eta = eta
        self.A_eigmax = eigmax
        self.A_eigmin = eigmin
        self.PL = PL
        self.PQ = PQ
        self.tol = tol
        self.vtol = vtol
        self.accel = accel
        self.thresholder = thresholder
        self.reg = get_regularization(thresholder)
        self.prox = get_prox(thresholder)
        self.A_history_ = []
        # self.v_history_ = []
        self.m_history_ = []
        self.PW_history_ = []
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

    def _update_coef_constraints(self, H, x_transpose_y, P_transpose_A):
        g = x_transpose_y + P_transpose_A / self.eta
        inv1 = np.linalg.pinv(H,rcond=1e-8)
        inv2 = np.linalg.pinv(
            self.constraint_lhs.dot(inv1).dot(self.constraint_lhs.T),rcond=1e-8
        )

        rhs = g.flatten() + self.constraint_lhs.T.dot(inv2).dot(
            self.constraint_rhs - self.constraint_lhs.dot(inv1).dot(g.flatten())
        )
        rhs = rhs.reshape(g.shape)
        return inv1.dot(rhs)

    def _update_A(self, A_old):
        """Update the symmetrized A matrix"""
        eigvals, eigvecs = np.linalg.eigh(A_old)
        r = A_old.shape[0]
        A = np.diag(eigvals)
        for i in range(r):
            # if eigvals[i] < self.A_eigmin:
            #     A[i, i] = self.A_eigmin
            # elif eigvals[i] > self.A_eigmax:
            if eigvals[i] > self.A_eigmax:
                A[i, i] = self.A_eigmax
        #return np.linalg.inv(eigvecs) @ A @ eigvecs 
        return eigvecs @ A @ np.linalg.inv(eigvecs)

    def _convergence_criterion(self):
        """Calculate the convergence criterion for the optimization"""
        this_coef = self.history_[-1]
        if len(self.history_) > 1:
            last_coef = self.history_[-2]
        else:
            last_coef = np.zeros_like(this_coef)
        err_coef = np.sqrt(np.sum((this_coef - last_coef) ** 2))
        return err_coef

    def _bounded_convergence_criterion(self, PW, A):
        """Calculate the bounded convergence criterion for the optimization"""
        # return np.all(np.diag(self.PW_history_[-1]) < 0)
        err_coef = np.sqrt(np.sum((PW - A) ** 2)) / self.eta
        return err_coef

    def _objective(self, x, y, coef_sparse, A, PW, q):
        """Objective function"""
        # Compute the errors
        R2 = (y - np.dot(x, coef_sparse)) ** 2
        A2 = (A - PW) ** 2
        L1 = self.threshold * np.sum(np.abs(coef_sparse.flatten()))
        if q % 500 == 0:
            print(q, 0.5 * np.sum(R2), 0.5 * np.sum(A2) / self.eta, L1)
        return 0.5 * np.sum(R2) + 0.5 * np.sum(A2) / self.eta + L1

    def _reduce(self, x, y):
        """
        Perform at most ``self.max_iter`` iterations of the clSR3 algorithm.
        Assumes initial guess for coefficients is stored in ``self.coef_``.
        """

        n_samples, n_features = x.shape
        r = self.PL.shape[0]
        Nr = self.PL.shape[-1]

        # Set initial coefficients
        if self.initial_guess is not None:
            self.coef_ = self.initial_guess
        if self.use_constraints and self.constraint_order.lower() == "target":
            self.constraint_lhs = reorder_constraints(self.constraint_lhs,
                                                      n_features)

        coef_sparse = self.coef_.T

        # update objective
        objective_history = []
        objective_history.append(
            self._objective(x, y, coef_sparse, np.zeros((r, r)), np.zeros(r), 0)
        )

        # Precompute some objects for optimization
        PL = self.PL
        PQ = self.PQ
        A = np.diag(self.A_eigmax * np.ones(r))
        delta_jk = np.eye(r)
        delta_il = np.eye(Nr)
        delta_ijkl = np.zeros((Nr, r, r, Nr))
        for i in range(Nr):
            for j in range(r):
                for k in range(r):
                    for z in range(Nr):
                        delta_ijkl[i, j, k, z] = delta_il[i, z] * delta_jk[j, k]

        np.random.seed(1)
        m = np.random.rand(r)-np.ones(r)*0.5  # initial guess for m
        self.m_history_.append(m)
        mPQ = np.zeros(PL.shape)
        for i in range(r):
            for j in range(i + 1, r):
                mPQ[i, j, :, int((i + 1) / 2.0 * (2 * r - i)) + j - 1 - i] = m
        for i in range(r):
            mPQ[i, i, :, Nr - r + i] = m
        for i in range(r):
            for j in range(Nr):
                mPQ[:, :, i, j] = 0.5 * (mPQ[:, :, i, j] + mPQ[:, :, i, j].T)
        p = PL - mPQ
        x_expanded = np.zeros((n_samples, r, n_features, r))
        for i in range(r):
            x_expanded[:, i, :, i] = x
        x_expanded = np.reshape(x_expanded, (n_samples * r, r * n_features))
        xTx = np.dot(x_expanded.T, x_expanded)
        xTy = np.dot(x_expanded.T, y.flatten())
        x_transpose_y = np.dot(x.T, y)
        PW = np.tensordot(p, coef_sparse[:Nr, :], axes=([3, 2], [0, 1]))

        q = 0
        tk_prev = 1
        m_prev = np.zeros(r)
        # Begin optimization loop
        for _ in range(self.max_iter):

            # update W
            Pmatrix = p.reshape(r * r, r * Nr)
            pTp = np.dot(Pmatrix.T, Pmatrix)
            H = xTx + pTp / self.eta
            if self.use_constraints:
                P_transpose_A = np.dot(Pmatrix.T, A.flatten())
                coef_sparse = self._update_coef_constraints(H, xTy, P_transpose_A).reshape(coef_sparse.shape)
            else:
                coef_sparse = cho_solve(cho, x_transpose_y)
            self.history_.append(coef_sparse.T)

            # prox-grad for (A, m)
            # Accelerated prox gradient descent
            if self.accel:
                tk = (1 + np.sqrt(1 + 4 * tk_prev ** 2)) / 2.0
                m_partial = m + (tk_prev - 1.0) / tk * (m - m_prev)
                tk_prev = tk
            else:
                m_partial = m
            # Code incorrect if I comment below line out but not sure why
            mPQ = np.zeros(PL.shape)
            for i in range(r):
                mPQ[i, i, :, Nr - r + i] = m_partial
                for j in range(i + 1, r):
                    ind = int((i + 1) / 2.0 * (2 * r - i)) + j - 1 - i
                    mPQ[i, j, :, ind] = m_partial
            mPQ = 0.5 * (mPQ + np.transpose(mPQ, [1, 0, 2, 3]))
            PW = np.tensordot(PL - mPQ, coef_sparse, axes=([3, 2], [0, 1]))
            PQW = np.tensordot(PQ, coef_sparse, axes=([2], [0]))
            A_b = (A - PW) / self.eta
            PQWT_PW = np.tensordot(PQW.T, A_b, axes=([2, 1], [0, 1]))
            if self.accel:
                m = m_partial - self.alpha_m * PQWT_PW
            else:
                m = m_prev - self.alpha_m * PQWT_PW
            m_prev = m
            # alpha_m = alpha_m*0.999

            # Update A
            # A = self._update_A(PW)
            A = self._update_A(A - self.alpha_A * A_b)

            # (m,A) update finished, append the result
            self.m_history_.append(m)
            self.A_history_.append(A)
            eigvals, eigvecs = np.linalg.eig(PW)
            self.PW_history_.append(np.sort(eigvals))
            # print(np.sort(eigvals))
            
            # update objective
            objective_history.append(
                self._objective(x, y, coef_sparse, A, PW, q)
            )
            q = q + 1

            if (np.sum(np.abs(m - m_prev)) < self.vtol) and self._convergence_criterion() < self.tol:
                # Could not (further) select important features
                # and already found a set of coefficients s.t. As is neg def
                break
        else:
            warnings.warn(
                "clSR3._reduce did not converge after {} iterations.".format(
                    self.max_iter
                ),
                ConvergenceWarning,
            )

        if self.use_constraints and self.constraint_order.lower() == "target":
            self.constraint_lhs = reorder_constraints(
                self.constraint_lhs, n_features, output_order="target"
            )

        self.coef_ = coef_sparse.T
        self.objective_history = objective_history
