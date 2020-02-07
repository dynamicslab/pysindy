from sklearn.linear_model import ElasticNet as SKElasticNet

from pysindy.optimizers import BaseOptimizer


class ElasticNet(BaseOptimizer):
    """
    Linear regression with combined L1 and L2 regularization.

    Minimizes the objective function

    .. math::

        (0.5 /  n_{samples}) \\times \\|y - Xw\\|^2_2
        + alpha \\times l1_{ratio} \\times \\|w\\|_1
        + 0.5 \\times alpha \\times (1 - l1_{ratio}) \\times \\|w\\|^2_2.

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

    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    unbias : boolean, optional (default True)
        Whether to perform an extra step of unregularized linear regression to unbias
        the coefficients for the identified support.
        For example, if `STLSQ(alpha=0.1)` is used then the learned coefficients will
        be biased toward 0 due to the L2 regularization.
        Setting `unbias=True` will trigger an additional step wherein the nonzero
        coefficients learned by the `STLSQ` object will be updated using an
        unregularized least-squares fit.

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
        self,
        alpha=1.0,
        l1_ratio=0.5,
        max_iter=1000,
        elastic_net_kw={},
        normalize=False,
        fit_intercept=False,
        copy_X=True,
        unbias=True,
    ):
        super(ElasticNet, self).__init__(
            max_iter=max_iter,
            normalize=normalize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            unbias=unbias,
        )

        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        if l1_ratio < 0:
            raise ValueError("l1_ratio must be nonnegative")

        self.alpha = alpha
        self.l1_ratio = l1_ratio
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
