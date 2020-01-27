from sklearn.linear_model import Lasso

from pysindy.optimizers import BaseOptimizer


class LASSO(BaseOptimizer):
    """
    LASSO algorithm (linear regression with L1 regularization).

    Attempts to minimize the objective function
    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1.

    Parameters
    ----------
    alpha : float, optional (default 1)
        Strength of the L1 regularization

    max_iter : int, optional (default 1000)
        Maximum iterations of the optimization algorithm.

    lasso_kw : dict, optional
        Optional keyword arguments to pass to the LASSO regression
        object.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s)

    iters : int
        Number of iterations performed in the optimization

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import LASSO
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = LASSO(alpha=100)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -0.351 1 x1 + 0.054 x0^2 + 0.329 x0 x1 + 0.009 x1^2
    x1' = -1.357 1^2 + 0.035 1 x1 + 0.710 x0^2 + -0.039 x0 x1 + 0.027 x1
    x2' = 0.369 1^2 + 0.558 1 x0 + 0.150 1 x1 + 0.018 x0^2 + -0.133 x1^2
    """

    def __init__(self, alpha=1.0, lasso_kw=None, max_iter=1000, **kwargs):
        super(LASSO, self).__init__(**kwargs)

        if alpha < 0:
            raise ValueError("alpha cannot be negative")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")

        self.lasso_kw = lasso_kw
        self.alpha = alpha
        self.max_iter = max_iter

    def _reduce(self, x, y):
        """Performs the LASSO regression
        """
        kw = self.lasso_kw or {}
        lasso_model = Lasso(
            alpha=self.alpha, max_iter=self.max_iter, fit_intercept=False, **kw
        )

        lasso_model.fit(x, y)

        self.coef_ = lasso_model.coef_
        self.iters = lasso_model.n_iter_
