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
