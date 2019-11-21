from sklearn.linear_model import ElasticNet as SKElasticNet

from sindy.optimizers import BaseOptimizer


class ElasticNet(BaseOptimizer):
    def __init__(
        self,
        alpha=1.0,
        l1_ratio=0.5,
        max_iter=1000,
        elastic_net_kw={},
        **kwargs
    ):
        super(ElasticNet, self).__init__(**kwargs)

        if alpha < 0:
            raise ValueError('alpha must be nonnegative')
        if l1_ratio < 0:
            raise ValueError('l1_ratio must be nonnegative')
        if max_iter <= 0:
            raise ValueError('max_iter must be positive')

        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.elastic_net_kw = elastic_net_kw

    def _reduce(self, x, y):
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
