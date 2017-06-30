import numpy as np

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class RationalFunctionMixin():
    def _transform(self, x, y):
        return np.hstack((x, y.reshape(-1, 1) * x))
    
    def fit(self, x, y):
        x, y = check_X_y(x, y, multi_output=False)
        super().fit(self._transform(x, y), y)
        l = len(self.coef_)//2
        self.coef_nominator_ = self.coef_[:l]
        self.coef_denominator_ = -self.coef_[l:]
        return self
    
    def predict(self, x):
        check_is_fitted(self, "coef_")
        x = check_array(x)
        return (self.intercept_ + x @ self.coef_nominator_) / (1 + x @ self.coef_denominator_)
