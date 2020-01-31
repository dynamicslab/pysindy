from sklearn.linear_model import ElasticNet as SKElasticNet

from pysindy.optimizers import BaseOptimizer


class ElasticNet(BaseOptimizer):
    """
    Linear regression with combined L1 and L2 regularization.

    Minimizes the objective function
    1 / (2 * n_samples) * ||y - Xw||^2_2
    + alpha * l1_ratio * ||w||_1
    + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2.

    Parameters
    ----------
    alpha : float, optional (default 1)
        Strength of the L2 regularization

    l1_ratio : float, optional (default 0.5)
        The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For
        l1_ratio = 0 the penalty is an L2 penalty. For l1_ratio = 1 it is
        an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination
        of L1 and L2.

    max_iter : int, optional (default 1000)
        Maximum iterations of the optimization algorithm.

    elastic_net_kw : dict, optional
        Optional keyword arguments to pass to the ElasticNet regression
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
    >>> from pysindy.optimizers import ElasticNet
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = ElasticNet(alpha=100, l1_ratio=0.8)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -0.328 1 x1 + 0.050 x0^2 + 0.319 x0 x1 + 0.004 x1^2
    x1' = -1.100 1^2 + 0.639 x0^2 + -0.045 x0 x1 + 0.018 x1^2
    x2' = 0.141 1^2 + 0.286 1 x0 + 0.274 1 x1 + 0.125 x0^2 + 0.066 x0 x1 + -0.147 x1^2
    """

    def __init__(
        self, alpha=1.0, l1_ratio=0.5, max_iter=1000, elastic_net_kw={}, **kwargs
    ):
        super(ElasticNet, self).__init__(**kwargs)

        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        if l1_ratio < 0:
            raise ValueError("l1_ratio must be nonnegative")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.elastic_net_kw = elastic_net_kw

    def _reduce(self, x, y):
        """Performs the elastic net regression
        """
        kw = self.elastic_net_kw or {}
        elastic_net_model = SKElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            max_iter=self.max_iter,
            fit_intercept=False,
            **kw
        )

        elastic_net_model.fit(x, y)

        self.coef_ = elastic_net_model.coef_
        self.iters = elastic_net_model.n_iter_
