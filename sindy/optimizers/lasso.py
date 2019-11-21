from sklearn.linear_model import Lasso

from sindy.optimizers import BaseOptimizer


class LASSO(BaseOptimizer):
    def __init__(
        self,
        alpha=1.0,
        lasso_kw=None,
        max_iter=1000,
        **kwargs
    ):
        super(LASSO, self).__init__(**kwargs)

        if alpha < 0:
            raise ValueError('alpha cannot be negative')
        if max_iter <= 0:
            raise ValueError('max_iter must be positive')

        self.lasso_kw = lasso_kw
        self.alpha = alpha
        self.max_iter = max_iter

    def _reduce(self, x, y):
        kw = self.lasso_kw or {}
        lasso_model = Lasso(
            alpha=self.alpha,
            max_iter=self.max_iter,
            fit_intercept=False,
            **kw
        )

        lasso_model.fit(x, y)

        self.coef_ = lasso_model.coef_
        self.iters = lasso_model.n_iter_
