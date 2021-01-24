import warnings

import numpy as np
from scipy.linalg import cho_factor
from scipy.linalg import cho_solve
from sklearn.exceptions import ConvergenceWarning

from ..utils import get_regularization
from ..utils import reorder_constraints
from .sr3 import SR3


class wSR3(SR3):
    """
    Sparse relaxed regularized regression.

    Attempts to minimize the objective function

    .. math::

        0.5\\|y-Xw\\|^2_2 + \\lambda \\times R(v)
        + (0.5 / \\nu)\\|w-v\\|^2_2

    where :math:`R(v)` is a regularization function. See the following references
    for more details:

        Zheng, Peng, et al. "A unified framework for sparse relaxed
        regularized regression: Sr3." IEEE Access 7 (2018): 1404-1423.

        Champion, Kathleen, et al. "A unified sparse optimization framework
        to learn parsimonious physics-informed models from data."
        arXiv preprint arXiv:1906.10612 (2019).

    Parameters
    ----------
    threshold : float, optional (default 0.1)
        Determines the strength of the regularization. When the
        regularization function R is the L0 norm, the regularization
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

    thresholder : string, optional (default 'L0')
        Regularization function to use. Currently implemented options
        are 'L0' (L0 norm), 'L1' (L1 norm), and 'CAD' (clipped
        absolute deviation).

    max_iter : int, optional (default 30)
        Maximum iterations of the optimization algorithm.

    initial_guess : np.ndarray, shape (n_features) or (n_targets, n_features), \
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

    coef_full_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s) that are not subjected to the regularization.
        This is the w in the objective function.

    history_ : list
        History of sparse coefficients. ``history_[k]`` contains the
        sparse coefficients (v in the optimization objective function)
        at iteration k.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import wSR3
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = wSR3(threshold=0.1, nu=1)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -10.004 1 + 10.004 x0
    x1' = 27.994 1 + -0.993 x0 + -1.000 1 x1
    x2' = -2.662 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        threshold=0.1,
        nu=1.0,
        eta=1.0,
        alpha=0.5,
        beta=0.5,
        eigmax=-0.1,
        eigmin=-10,
        tol=1e-5,
        thresholder="l0",
        max_iter=30,
        normalize=False,
        fit_intercept=False,
        copy_X=True,
        initial_guess=None,
        Theta=None,
        Xdot=None,
        PL=None,
        PQ=None,
        thresholds=None,
        objective_history=None,
        constraint_lhs=None,
        constraint_rhs=None,
        constraint_order="target",
    ):
        super(wSR3, self).__init__(
            max_iter=max_iter,
            initial_guess=initial_guess,
            normalize=normalize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
        )

        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        if nu <= 0:
            raise ValueError("nu must be positive")
        if eta <= 0:
            raise ValueError("eta must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")

        self.threshold = threshold
        self.thresholds = thresholds
        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.eta = eta
        self.Theta = Theta
        self.Xdot = Xdot
        self.A_eigmax = eigmax
        self.A_eigmin = eigmin
        self.PL = PL
        self.PQ = PQ
        self.tol = tol
        self.thresholder = thresholder
        self.reg = get_regularization(thresholder)
        self.A_history_ = []
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

    def _update_sparse_coef(self, coef_old):
        """Update the regularized weight vector"""
        # if self.thresholds is None:
        #     return super(wSR3, self)._update_sparse_coef(coef_full)
        # else:
        #     coef_sparse = self.prox(coef_full, self.thresholds.T)
        coef_sparse = self.prox(coef_old, self.threshold)
        self.history_.append(coef_sparse.T)
        return coef_sparse

    def _update_A_test(self, A_old):
        """Update the symmetrized A matrix"""
        eigvals, eigvecs = np.linalg.eig(A_old)
        r = A_old.shape[0]
        A = np.diag(eigvals)
        for i in range(r):
            if eigvals[i] < self.A_eigmin:
                A[i, i] = self.A_eigmin
            elif eigvals[i] > self.A_eigmax:
                A[i, i] = self.A_eigmax
            else:
                A[i, i] = eigvals[i]
        A = eigvecs @ A @ np.linalg.pinv(eigvecs)
        self.PW_history_.append(np.diag(eigvals))
        return A

    def _convergence_criterion(self):
        """Calculate the convergence criterion for the optimization"""
        this_coef = self.history_[-1]
        if len(self.history_) > 1:
            last_coef = self.history_[-2]
        else:
            last_coef = np.zeros_like(this_coef)
        err_coef = np.sqrt(np.sum((this_coef - last_coef) ** 2)) / self.nu
        return err_coef

    def _bounded_convergence_criterion(self, PW, A):
        """Calculate the bounded convergence criterion for the optimization"""
        # return np.all(np.diag(self.PW_history_[-1]) < 0)
        err_coef = np.sqrt(np.sum((PW - A) ** 2)) / self.eta
        return err_coef

    def _objective(self, x, y, coef_sparse, A, p):
        """Objective function"""
        Nr = self.PL.shape[-1]
        PW = np.tensordot(p, coef_sparse[:Nr, :], axes=([3, 2], [0, 1]))
        eigvals, eigvecs = np.linalg.eig(PW)
        R2 = (y - np.dot(x, coef_sparse)) ** 2
        A2 = (A - PW) ** 2
        if self.thresholds is None:
            # print(np.sum(A2) / self.eta, np.sum(R2))
            return (
                0.5 * np.sum(R2)
                + self.reg(coef_sparse, 0.5 * self.threshold ** 2 / self.nu)
                + 0.5 * np.sum(A2) / self.eta
            )
        else:
            return (
                0.5 * np.sum(R2)
                + self.reg(coef_sparse, 0.5 * self.thresholds.T ** 2 / self.nu)
                + 0.5 * np.sum(A2) / self.eta
            )

    def _reduce(self, x, y):
        """
        Perform at most ``self.max_iter`` iterations of the wSR3 algorithm.
        Assumes initial guess for coefficients is stored in ``self.coef_``.
        """

        n_samples, n_features = x.shape
        r = self.PL.shape[0]
        Nr = self.PL.shape[-1]

        # Set initial coefficients
        if self.initial_guess is not None:
            self.coef_ = self.initial_guess
        if self.use_constraints and self.constraint_order.lower() == "target":
            self.constraint_lhs = reorder_constraints(self.constraint_lhs, n_features)

        coef_sparse = self.coef_.T
        self.history_.append(coef_sparse.T)

        # Precompute some objects for upcoming least-squares solves.
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
        # H_Xi = np.dot(x.T, x) + np.diag(np.full(x.shape[1], 1.0 / self.nu))
        # cho = cho_factor(H_Xi)
        self.Theta = x
        self.Xdot = y
        x_transpose_y = np.dot(x.T, y)
        x_expanded = np.zeros((n_samples, r, n_features, r))
        for i in range(r):
            x_expanded[:, i, :, i] = x
        x_expanded = np.reshape(x_expanded, (n_samples * r, r * n_features))
        xTx = np.dot(x_expanded.T, x_expanded)
        # Begin optimization loop
        objective_history = []
        for _ in range(self.max_iter):

            # update Xi
            # coef_full = self._update_full_coef(cho, x_transpose_y, coef_sparse)
            # update W -- prox-grad version
            # W_b = (W - coef_full) / self.nu + np.tensordot(
            #     p.T, PW - A, axes=([3, 2], [0, 1])
            # ) / self.eta

            # Update m
            W = coef_sparse[:Nr, :]
            PQ_W = np.tensordot(PQ, W, axes=([2], [0]))
            pqwTpqw = np.tensordot(PQ_W.T, PQ_W, axes=([2, 1], [0, 1]))
            M_inv = np.linalg.pinv(pqwTpqw)
            PL_W = np.tensordot(PL, W, axes=([3, 2], [0, 1]))
            m_b = np.tensordot(PQ_W.T, PL_W - A, axes=([2, 1], [0, 1]))
            m = np.tensordot(M_inv, m_b, axes=([1], [0]))
            self.m_history_.append(m)

            # Switching from coordinate-descent for (Xi, m)
            # to prox-grad for (A, W)
            mPQ = np.zeros(PL.shape)
            for i in range(r):
                for j in range(i + 1, r):
                    mPQ[i, j, :, int((i + 1) / 2.0 * (2 * r - i)) + j - 1 - i] = m
            for i in range(r):
                mPQ[i, i, :, Nr - r + i] = m
            for i in range(r):
                for j in range(Nr):
                    mPQ[:, :, i, j] = 0.5 * (mPQ[:, :, i, j] + mPQ[:, :, i, j].T)
            # Compute 4-index tensor P
            p = PL - mPQ

            # update A
            PW = np.tensordot(p, W, axes=([3, 2], [0, 1]))
            A = self._update_A_test(PW)
            # A_b = -(PW - A) / self.eta
            # A = self._update_A_test(A - self.beta * A_b)
            self.A_history_.append(A)

            # update W
            pTp = np.tensordot(p.T, np.transpose(p, [0, 1, 3, 2]), axes=([3, 2], [0, 1])).reshape(Nr * r, Nr * r)
            H = pTp / self.eta + xTx
            H_inv = np.linalg.pinv(H)
            G = np.tensordot(p.T, A, axes=([3, 2], [0, 1])) / self.eta + x_transpose_y
            if self.use_constraints:
                CHCT_inv = np.linalg.pinv(np.dot(self.constraint_lhs, np.dot(H_inv, self.constraint_lhs.T)))
                CT_CHCTinv_CH = np.dot(self.constraint_lhs.T, np.dot(CHCT_inv, np.dot(self.constraint_lhs, H_inv)))
                H_shifted = np.dot(H_inv, np.eye(r * Nr) - CT_CHCTinv_CH)
                W_shifted = np.dot(H_shifted, G.flatten()).reshape(coef_sparse.shape)
                coef_sparse = self._update_sparse_coef(W_shifted)
            else:
                W_new = np.dot(H_inv, G.flatten()).reshape(coef_sparse.shape)
                coef_sparse = self._update_sparse_coef(W_new)

            # update objective
            objective_history.append(
                self._objective(x, y, coef_sparse, A, p)
            )
            if (
                self._convergence_criterion() < self.tol
                and self._bounded_convergence_criterion(PW, A) < self.tol
            ):
                # Could not (further) select important features
                # and already found a set of coefficients s.t. As is neg def
                break
        else:
            warnings.warn(
                "wSR3._reduce did not converge after {} iterations.".format(
                    self.max_iter
                ),
                ConvergenceWarning,
            )

        if self.use_constraints and self.constraint_order.lower() == "target":
            self.constraint_lhs = reorder_constraints(
                self.constraint_lhs, n_features, output_order="target"
            )

        self.coef_ = np.real(coef_sparse.T)
        # self.coef_full_ = coef_full.T
        self.m_ = m
        self.objective_history = objective_history
