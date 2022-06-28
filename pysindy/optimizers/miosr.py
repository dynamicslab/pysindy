import gurobipy as gp
import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseOptimizer


class MIOSR(BaseOptimizer):
    """Mixed-Integer Optimized Sparse Regression.

    Solves the sparsity constrained regression problem to provable optimality
    .. math::

        0.5\\|y-Xw\\|^2_2 + \\lambda R(u)
        + (0.5 / \\nu)\\|w-u\\|^2_2

    .. math::

        \\text{subject to } \\|w\\|_0 \\leq k

    by using type-1 specially ordered sets (SOS1) to encode the support of
    the coefficients. Can optionally add additional constraints on the
    coefficients or access the gurobi `model` directly for advanced usage.
    See the following reference for additional details:

        Bertsimas, D. and Gurnee, W., 2022. Learning Sparse Nonlinear Dynamics
        via Mixed-Integer Optimization. arXiv preprint arXiv:2206.00176.

    Parameters
    ----------
    target_sparsity : int, optional (default None)
        The maximum number of nonzero coefficients across all dimensions.
        If set, and `group_sparsity` is not set, the model will fit all
        dimensions jointly, potentially reducing statistical efficiency.

    group_sparsity : int tuple, optional (default None)
        Tuple of length n_targets constraining the number of nonzero
        coefficients for each target dimension.

      alpha : float, optional (default 0.01)
        Optional L2 (ridge) regularization on the weight vector.

    regression_timeout : int, optional (default 30)
        The timeout (in seconds) of the gurobi optimizer to solve and prove
        optimality (either per dimension or jointly depending on the
        above sparsity settings).

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

    initial_guess : np.ndarray, shape (n_features) or (n_targets, n_features), \
            optional (default None)
        Initial guess for coefficients ``coef_`` to warmstart the optimizer.

    verbose : bool, optional (default False)
        If True, prints out the Gurobi solver log.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s).
    ind_ : array, shape (n_features,) or (n_targets, n_features)
        Array of 0s and 1s indicating which coefficients of the
        weight vector have not been masked out, i.e. the support of
        ``self.coef_``.
    model : gurobipy.model
        The raw gurobi model being solved.
    """

    def __init__(
        self,
        target_sparsity=5,
        group_sparsity=None,
        alpha=0.01,
        regression_timeout=10,
        fit_intercept=False,
        constraint_lhs=None,
        constraint_rhs=None,
        constraint_order="target",
        normalize_columns=False,
        copy_X=True,
        initial_guess=None,
        verbose=False,
    ):
        super(MIOSR, self).__init__(
            normalize_columns=normalize_columns,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
        )

        if target_sparsity is not None and (
            target_sparsity <= 0 or not isinstance(target_sparsity, int)
        ):
            raise ValueError("target_sparsity must be positive int")
        if constraint_order not in {"target", "feature"}:
            raise ValueError("constraint_order must be one of {'target', 'feature'}")
        if alpha < 0:
            raise ValueError("alpha cannot be negative")

        self.target_sparsity = target_sparsity
        self.group_sparsity = group_sparsity
        self.constraint_lhs = constraint_lhs
        self.constraint_rhs = constraint_rhs
        self.constraint_order = constraint_order
        self.alpha = alpha
        self.initial_guess = initial_guess
        self.regression_timeout = regression_timeout
        self.verbose = verbose

        self.model = None

    def _make_model(self, X, y, k, warm_start=None):
        m = gp.Model()
        n, d = X.shape
        _, r = y.shape

        beta = m.addMVar(
            r * d, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="beta"
        )
        iszero = m.addMVar(r * d, vtype=gp.GRB.BINARY, name="iszero")

        # Sparsity constraint
        for i in range(r * d):
            m.addSOS(gp.GRB.SOS_TYPE1, [beta[i], iszero[i]])
        m.addConstr(iszero.sum() >= (r * d) - k, name="sparsity")

        # Group sparsity constraints
        if self.target_sparsity is not None and self.group_sparsity is not None:
            for i in range(r):
                dimension_sparsity = self.group_sparsity[i]
                print(dimension_sparsity)
                m.addConstr(
                    iszero[i * d : (i + 1) * d].sum() >= d - dimension_sparsity,
                    name=f"group_sparsity{i}",
                )

        # General equality constraints
        if self.constraint_lhs is not None and self.constraint_rhs is not None:
            if self.constraint_order == "feature":
                target_indexing = np.arange(r * d).reshape(r, d, order="F").flatten()
                constraint_lhs = self.constraint_lhs[:, target_indexing]
            else:
                constraint_lhs = self.constraint_lhs
            m.addConstr(
                constraint_lhs @ beta == self.constraint_rhs, name="coefficient_constrs"
            )

        if warm_start is not None:
            warm_start = warm_start.reshape(1, r * d)[0]
            for i in range(d):
                iszero[i].start = abs(warm_start[i]) < 1e-6
                beta[i].start = warm_start[i]

        Quad = np.dot(X.T, X)
        obj = self.alpha * (beta @ beta)
        for i in range(r):
            lin = np.dot(y[:, i].T, X)
            obj += beta[d * i : d * (i + 1)] @ Quad @ beta[d * i : d * (i + 1)]
            obj -= 2 * (lin @ beta[d * i : d * (i + 1)])

        m.setObjective(obj, gp.GRB.MINIMIZE)

        m.params.OutputFlag = 1 if self.verbose else 0
        m.params.timelimit = self.regression_timeout
        m.update()

        self.model = m

        return m, beta

    def _regress(self, X, y, k, warm_start=None):
        """
        Deploy and optimize the MIO formulation of L0-Regression.
        """
        m, beta = self._make_model(X, y, k, warm_start)
        m.optimize()
        return beta.X

    def _reduce(self, x, y):
        """
        Runs MIOSR either per dimension or jointly on all dimensions.

        Assumes an initial guess for coefficients and support are saved in
        ``self.coef_`` and ``self.ind_``.
        """
        regress_jointly = self.target_sparsity is not None

        if self.initial_guess is not None:
            self.coef_ = self.initial_guess

        n, d = x.shape
        n, r = y.shape

        if regress_jointly:
            coefs = self._regress(x, y, self.target_sparsity)
            # Remove nonzero terms due to numerical error
            non_active_ixs = np.argsort(np.abs(coefs))[: -int(self.target_sparsity)]
            coefs[non_active_ixs] = 0
            self.coef_ = coefs.reshape(r, d)
            self.ind_ = (np.abs(self.coef_) > 1e-6).astype(int)
        else:
            for i in range(r):
                k = self.group_sparsity[i]
                warm_start = (
                    None if self.initial_guess is None else self.initial_guess[[i], :]
                )
                coef_i = self._regress(x, y[:, [i]], k, warm_start=warm_start)
                # Remove nonzero terms due to numerical error
                non_active_ixs = np.argsort(np.abs(coef_i))[: -int(k)]
                coef_i[non_active_ixs] = 0
                self.coef_[i, :] = coef_i
            self.ind_ = (np.abs(self.coef_) > 1e-6).astype(int)

    @property
    def complexity(self):
        check_is_fitted(self)
        return np.count_nonzero(self.coef_)
