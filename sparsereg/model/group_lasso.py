from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel, _rescale_data
from sklearn.utils.validation import check_X_y


from ..vendor.group_lasso.group_lasso import sparse_group_lasso


class SparseGroupLasso(LinearModel, RegressorMixin):
    def __init__(self, groups, alpha=1.0, rho=0.5, max_iter=1000, tol=1e-4,
                 normalize=False, fit_intercept=True, copy_X=True):
        self.alpha = alpha
        self.rho = rho
        self.groups = groups
        self.max_iter = max_iter
        self.tol = tol
        self.normalize = normalize
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X


    def fit(self, x, y, sample_weight=None):
        x, y = check_X_y(x, y, accept_sparse=[], y_numeric=True, multi_output=False)

        x, y, X_offset, y_offset, X_scale = self._preprocess_data(
            x, y, fit_intercept=self.fit_intercept, normalize=self.normalize,
            copy=self.copy_X, sample_weight=sample_weight)

        if sample_weight is not None:
            x, y = _rescale_data(x, y, sample_weight)

        self.coef_ = sparse_group_lasso(x, y, self.alpha, self.rho, self.groups,
                                        max_iter=self.max_iter, rtol=self.tol)

        self._set_intercept(X_offset, y_offset, X_scale)
        return self
