import numpy as np
from sklearn.multioutput import MultiOutputRegressor

from .base import BaseOptimizer

COEF_THRESHOLD = 1e-14


class WrappedOptimizer(BaseOptimizer):
    """Wrapper class for generic optimizers/sparse regression methods

    Enables single target regressors (i.e. those whose predictions are
    1-dimensional) to perform multi target regression (i.e. predictions
    are 2-dimensional).  Also allows unbiasing optimizers that would
    otherwise prohibit it.

    Args:
        optimizer: wrapped optimizer/sparse regression method

    Parameters
    ----------
    optimizer: estimator object
        The optimizer/sparse regressor to be wrapped, implementing ``fit`` and
        ``predict``. ``optimizer`` should also have the attributes ``coef_``,
        ``fit_intercept``, and ``intercept_``. Note that attribute
        ``normalize`` is deprecated as of sklearn versions >= 1.0 and will be
        removed in future versions.

    """

    def __init__(self, optimizer, unbias: bool = True):
        super().__init__(unbias=unbias)
        self.optimizer = MultiOutputRegressor(optimizer)

    def _reduce(self, x, y):
        coef_shape = (y.shape[1], x.shape[1])
        self.coef_ = np.zeros(coef_shape)
        self.ind_ = np.ones(coef_shape)
        self.optimizer.fit(x, y)
        coef_list = [
            np.reshape(est.coef_, (-1, coef_shape[1]))
            for est in self.optimizer.estimators_
        ]
        self.coef_ = np.concatenate(coef_list, axis=0)
        self.ind_ = np.abs(self.coef_) > COEF_THRESHOLD
        if hasattr(self.optimizer, "intercept_"):
            self.intercept_ = self.optimizer.intercept_
        else:
            self.intercept_ = 0.0
        return self

    def predict(self, x):
        prediction = self.optimizer.predict(x)
        if prediction.ndim == 1:
            return prediction[:, np.newaxis]
        else:
            return prediction

    @property
    def complexity(self):
        return np.count_nonzero(self.coef_) + np.count_nonzero(self.intercept_)
