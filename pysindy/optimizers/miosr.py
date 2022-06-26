import time

import gurobipy as gp
import numpy as np
from scipy.integrate import odeint
from sklearn.utils.validation import check_is_fitted

import pysindy as ps
from pysindy.optimizers.base import BaseOptimizer


class MIOSR(BaseOptimizer):
    """
    Parameters
    ----------
    alpha : float, optional (default 0.05)
        Optional L2 (ridge) regularization on the weight vector.
    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.
    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.
    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.
    initial_guess : np.ndarray, shape (n_features) or (n_targets, n_features), \
            optional (default None)
        Initial guess for coefficients ``coef_``.
        If None, least-squares is used to obtain an initial guess.
    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s).
    ind_ : array, shape (n_features,) or (n_targets, n_features)
        Array of 0s and 1s indicating which coefficients of the
        weight vector have not been masked out, i.e. the support of
        ``self.coef_``.
    history_ : list
        History of ``coef_``. ``history_[k]`` contains the values of
        ``coef_`` at iteration k of sequentially thresholded least-squares.
    """

    def __init__(
        self,
        target_sparsity=None,
        group_sparsity=None,
        constraint_lhs=None,
        constraint_rhs=None,
        constraint_order="target",
        alpha=0.05,
        regression_timeout=30,
        normalize_columns=False,
        fit_intercept=False,
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
        self.lambda_2 = alpha
        self.initial_guess = initial_guess
        self.regression_timeout = regression_timeout
        self.verbose = verbose

        # Regression model class variables used for model selection
        self._model_made = False
        self._model = None
        self._beta_vars = None
        self._is_zero_vars = None
        self.solve_times = []
        self.build_times = []

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
                group_cardinality = self.group_sparsity[i]
                print(group_cardinality)
                m.addConstr(
                    iszero[i * d : (i + 1) * d].sum() == d - group_cardinality,
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
        obj = self.lambda_2 * (beta @ beta)
        for i in range(r):
            lin = np.dot(y[:, i].T, X)
            obj += beta[d * i : d * (i + 1)] @ Quad @ beta[d * i : d * (i + 1)]
            obj -= 2 * (lin @ beta[d * i : d * (i + 1)])

        m.setObjective(obj, gp.GRB.MINIMIZE)

        m.params.OutputFlag = 1 if self.verbose else 0
        m.params.timelimit = self.regression_timeout
        m.update()

        self._model_made = True
        self._model = m
        self._beta_vars = beta
        self._is_zero_vars = iszero

    def _change_sparsity(self, new_k):
        sparsity_constr = self._model.getConstrByName("sparsity")
        sparsity_constr.rhs = self._beta_vars.shape[0] - new_k
        self._model.update()
        self.target_sparsity = new_k

    def _change_regularizer(self, new_lambda2):
        coef_change = new_lambda2 - self.lambda_2
        regularizer = gp.quicksum(
            self._beta_vars[i] ** 2 for i in range(self._beta_vars.shape[0])
        )
        new_obj = self._model.getObjective() + coef_change * regularizer
        self._model.setObjective(new_obj)
        self._model.update()
        self.lambda_2 = new_lambda2

    def _regress(self, X, y, k, warm_start=None):
        """
        Deploy and optimize the MIQP formulation of L0-Regression.
        """
        model_construction_start_t = time.time()
        self._make_model(X, y, k, warm_start)
        self.build_times.append(time.time() - model_construction_start_t)
        solve_start_t = time.time()
        self._model.optimize()
        self.solve_times.append(time.time() - solve_start_t)
        return self._beta_vars.X

    def _reduce(self, x, y):
        """Performs at most ``self.max_iter`` iterations of the
        sequentially-thresholded least squares algorithm.
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


if __name__ == "__main__":
    # Generate training data

    def lorenz(x, t):
        return [
            10 * (x[1] - x[0]),
            x[0] * (28 - x[2]) - x[1],
            x[0] * x[1] - 8 / 3 * x[2],
        ]

    dt = 0.001
    t_train = np.arange(0, 10, dt)
    x0_train = [-8, 8, 27]
    x_train = odeint(lorenz, x0_train, t_train)
    x_dot_train_measured = np.array(
        [lorenz(x_train[i], 0) for i in range(t_train.size)]
    )
    # Fit the model
    poly_order = 2
    threshold = 0.05
    model = ps.SINDy(
        optimizer=MIOSR(
            group_sparsity=(2, 3, 2),
            alpha=0.05,
        ),
        feature_library=ps.PolynomialLibrary(degree=poly_order),
    )
    model.fit(x_train, t=dt, x_dot=x_dot_train_measured, unbias=False)
    model.print()
