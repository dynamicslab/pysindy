try:
    import gurobipy as gp
except ImportError:
    raise ImportError(
        "To use MIOSR please install pysindy with pip install pysindy[miosr]"
        "to gain access to a restricted installation of Gurobi."
        "Free unrestricted academic licenses are available at "
        "https://www.gurobi.com/academia/academic-program-and-licenses/"
    )

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .base import BaseOptimizer


class MIOSR(BaseOptimizer):
    """Mixed-Integer Optimized Sparse Regression.

    Solves the sparsity constrained regression problem to provable optimality
    .. math::

        \\|y-Xw\\|^2_2 + \\lambda R(u)

    .. math::

        \\text{subject to } \\|w\\|_0 \\leq k

    by using type-1 specially ordered sets (SOS1) to encode the support of
    the coefficients. Can optionally add additional constraints on the
    coefficients or access the gurobi model directly for advanced usage.
    See the following reference for additional details:

        Bertsimas, D. and Gurnee, W., 2022. Learning Sparse Nonlinear Dynamics
        via Mixed-Integer Optimization. arXiv preprint arXiv:2206.00176.

    Parameters
    ----------
    target_sparsity : int, optional (default 5)
        The maximum number of nonzero coefficients across all dimensions.
        If set, the model will fit all dimensions jointly, potentially reducing
        statistical efficiency.

    group_sparsity : int tuple, optional (default None)
        Tuple of length n_targets constraining the number of nonzero
        coefficients for each target dimension.

    alpha : float, optional (default 0.01)
        Optional L2 (ridge) regularization on the weight vector.

    regression_timeout : int, optional (default 10)
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
        model = gp.Model()
        n_samples, n_features = X.shape
        _, n_targets = y.shape

        coeff_var = model.addMVar(
            n_targets * n_features,
            lb=-gp.GRB.INFINITY,
            vtype=gp.GRB.CONTINUOUS,
            name="coeff_var",
        )
        iszero = model.addMVar(
            n_targets * n_features, vtype=gp.GRB.BINARY, name="iszero"
        )

        # Sparsity constraint
        for i in range(n_targets * n_features):
            model.addSOS(gp.GRB.SOS_TYPE1, [coeff_var[i], iszero[i]])
        model.addConstr(iszero.sum() >= (n_targets * n_features) - k, name="sparsity")

        # Group sparsity constraints
        if self.group_sparsity is not None and n_targets > 1:
            for i in range(n_targets):
                dimension_sparsity = self.group_sparsity[i]
                model.addConstr(
                    iszero[i * n_features : (i + 1) * n_features].sum()
                    >= n_features - dimension_sparsity,
                    name=f"group_sparsity{i}",
                )

        # General equality constraints
        if self.constraint_lhs is not None and self.constraint_rhs is not None:
            if self.constraint_order == "feature":
                target_indexing = (
                    np.arange(n_targets * n_features)
                    .reshape(n_targets, n_features, order="F")
                    .flatten()
                )
                constraint_lhs = self.constraint_lhs[:, target_indexing]
            else:
                constraint_lhs = self.constraint_lhs
            model.addConstr(
                constraint_lhs @ coeff_var == self.constraint_rhs, name="coeff_constrs"
            )

        if warm_start is not None:
            warm_start = warm_start.reshape(1, n_targets * n_features)[0]
            for i in range(n_features):
                iszero[i].start = abs(warm_start[i]) < 1e-6
                coeff_var[i].start = warm_start[i]

        # Equation 15 in paper
        Quad = np.dot(X.T, X)
        obj = self.alpha * (coeff_var @ coeff_var)
        for i in range(n_targets):
            lin = np.dot(y[:, i].T, X)
            obj += (
                coeff_var[n_features * i : n_features * (i + 1)]
                @ Quad
                @ coeff_var[n_features * i : n_features * (i + 1)]
            )
            obj -= 2 * (lin @ coeff_var[n_features * i : n_features * (i + 1)])

        model.setObjective(obj, gp.GRB.MINIMIZE)

        model.params.OutputFlag = 1 if self.verbose else 0
        model.params.timelimit = self.regression_timeout
        model.update()

        self.model = model

        return model, coeff_var

    def _regress(self, X, y, k, warm_start=None):
        """
        Deploy and optimize the MIO formulation of L0-Regression.
        """
        m, coeff_var = self._make_model(X, y, k, warm_start)
        m.optimize()
        return coeff_var.X

    def _reduce(self, x, y):
        """
        Runs MIOSR either per dimension or jointly on all dimensions.

        Assumes an initial guess for coefficients and support are saved in
        ``self.coef_`` and ``self.ind_``.
        """
        if self.initial_guess is not None:
            self.coef_ = self.initial_guess

        n_samples, n_features = x.shape
        _, n_targets = y.shape

        if (
            self.target_sparsity is not None or self.constraint_lhs is not None
        ):  # Regress jointly
            coefs = self._regress(x, y, self.target_sparsity, self.initial_guess)
            # Remove nonzero terms due to numerical error
            non_active_ixs = np.argsort(np.abs(coefs))[: -int(self.target_sparsity)]
            coefs[non_active_ixs] = 0
            self.coef_ = coefs.reshape(n_targets, n_features)
            self.ind_ = (np.abs(self.coef_) > 1e-6).astype(int)
        else:  # Regress dimensionwise
            for i in range(n_targets):
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
