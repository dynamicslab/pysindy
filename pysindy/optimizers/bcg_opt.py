import numpy as np
from bcg.model_init import Model
from bcg.run_BCG import BCG as BCG_algorithm

from .base import BaseOptimizer


class L1_ball_model(Model):
    """LP model for the L1 ball"""

    alpha = 1.0

    def minimize(self, gradient_at_x=None):
        result = np.zeros(self.dimension)
        if gradient_at_x is None:
            result[0] = self.alpha
        else:
            i = np.argmax(np.abs(gradient_at_x))
            result[i] = -self.alpha if gradient_at_x[i] > 0 else self.alpha
        return result


class L1_ball_with_constraints_model(Model):
    alpha = 0

    def minimize(self, gradient_at_x=None):
        result = np.zeros(self.dimension)
        if gradient_at_x is None:
            result[0] = self.alpha
        else:
            i = np.argmax(np.abs(gradient_at_x))
            result[i] = -self.alpha if gradient_at_x[i] > 0 else self.alpha
        return result


class BCG(BaseOptimizer):
    """
    Blended conditional gradients algorithm.

    Attempts to minimize the objective function

    .. math::

        0.5\\|y-Xw\\|^2_2 + \\lambda \\times R(v)

    where :math:`R(v)` is a regularization function. See the following reference
    for more details:

        Carderera, Alejandro, Sebastian Pokutta, Christof Schütte, and Martin
        Weiser. "CINDy: Conditional gradient-based Identification of Non-linear
        Dynamics--Noise-robust recovery."
        arXiv preprint arXiv:2101.02630 (2021).

    Parameters
    ----------

    tol : float, optional (default 1e-5)
        Tolerance used for determining convergence of the optimization
        algorithm.

    max_iter : int, optional (default 30)
        Maximum iterations of the optimization algorithm.

    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the L2-norm.

    normalize_columns : boolean, optional (default False)
        Normalize the columns of x (the SINDy library terms)

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    constraint_lhs : numpy ndarray, shape (n_constraints, n_features * n_targets), \
            optional (default None)
        The left hand side matrix C of Cw <= d.
        There should be one row per constraint.

    constraint_rhs : numpy ndarray, shape (n_constraints,), optional (default None)
        The right hand side vector d of Cw <= d.

    constraint_order : string, optional (default "target")
        The format in which the constraints ``constraint_lhs`` were passed.
        Must be one of "target" or "feature".
        "target" indicates that the constraints are grouped by target:
        i.e. the first ``n_features`` columns
        correspond to constraint coefficients on the library features for the first
        target (variable), the next ``n_features`` columns to the library features
        for the second target (variable), and so on.
        "feature" indicates that the constraints are grouped by library feature:
        the first ``n_targets`` columns correspond to the first library feature,
        the next ``n_targets`` columns to the second library feature, and so on.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        SINDy model coefficients.

    history_ : list
        History of sparse coefficients. ``history_[k]`` contains the
        sparse coefficients (v in the optimization objective function)
        at iteration k.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import BCG
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = BCG()
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -10.004 1 + 10.004 x0
    x1' = 27.994 1 + -0.993 x0 + -1.000 1 x1
    x2' = -2.662 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        tol=1e-5,
        max_iter=30,
        normalize_columns=False,
        normalize=False,
        fit_intercept=False,
        copy_X=True,
        alpha=None,
        constraint_lhs=None,
        constraint_rhs=None,
        constraint_order="target",
    ):
        super(BCG, self).__init__(
            max_iter=max_iter,
            normalize=normalize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
        )

        if tol <= 0:
            raise ValueError("tol must be positive")

        self.unbias = False
        self.alpha = alpha
        self.tol = tol
        self.normalize_columns = normalize_columns
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

    def _reduce(self, x, y):
        """
        Perform the BCG algorithm.
        """
        n_samples, n_features = x.shape
        n_targets = y.shape[1]

        self.Theta = x
        x_normed = np.copy(x)
        if self.normalize_columns:
            reg = np.zeros(n_features)
            for i in range(n_features):
                reg[i] = 1.0 / np.linalg.norm(x[:, i], 2)
                x_normed[:, i] = reg[i] * x[:, i]
        # you can construct your own configuration dictionary
        config_dictionary = {
            "solution_only": False,
            "verbosity": "quiet",
            "dual_gap_acc": 1e-6,
            "runningTimeLimit": 20,
            "use_LPSep_oracle": False,
            "max_lsFW": 200,
            "strict_dropSteps": True,
            "max_stepsSub": 1000,
            "max_lsSub": 300,
            "LPsolver_timelimit": 500,
            "K": 1,
        }
        if self.alpha is None:
            x_inv = np.linalg.pinv(x_normed)
            print(y.shape, x_inv.shape)
            self.alpha = 2 * np.sum(np.abs(x_inv @ y))
        coef = np.zeros(np.shape(self.coef_))

        # Precompute some objects for optimization
        x_expanded = np.zeros((n_samples, n_targets, n_features, n_targets))
        for i in range(n_targets):
            x_expanded[:, i, :, i] = x
        x_expanded = np.reshape(
            x_expanded, (n_samples * n_targets, n_targets * n_features)
        )

        def oracle(coefs):
            return np.dot(
                (y.flatten() - x_expanded @ coefs), y.flatten() - x_expanded @ coefs
            )

        def oracle_grad(coefs):
            return -2 * x_expanded.T @ (y.flatten() - x_expanded @ coefs)

        L1_ball = L1_ball_model(n_features * n_targets)
        L1_ball.alpha = 1  # self.alpha
        res = BCG_algorithm(
            oracle,
            oracle_grad,
            "/Users/alankaptanoglu/bcg/examples/test.lp",
            config_dictionary,
        )
        print("optimal solution {}".format(res[0]))
        coef = np.reshape(res[0], (n_targets, n_features))
        # for i in range(n_targets):
        #     y0 = y[:, i]
        #
        #     def oracle(coefs):
        #         # defaults to frobenius error
        #         return np.dot((y0 - x_normed @ coefs),
        #                       y0 - x_normed @ coefs)
        #
        #     def oracle_grad(coefs):
        #         return (- 2 * x_normed.T @ (y0 - x_normed @ coefs))
        #
        #     L1_ball = L1_ball_model(n_features)
        #     L1_ball.alpha = self.alpha
        #     res = BCG_algorithm(oracle, oracle_grad, None, config_dictionary)
        #     print('optimal solution {}'.format(res[0]))
        #     coef[i, :] = res[0]
        print(coef)
        print("dual_bound {}".format(res[1]))
        # prediction on train dataset
        mse_train = res[2] / x.shape[0]
        print("mean square error on training dataset {}".format(mse_train))
        if self.normalize_columns:
            self.coef_ = np.multiply(reg, coef.T)
        else:
            self.coef_ = coef
