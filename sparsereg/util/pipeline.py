from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(TransformerMixin, BaseEstimator):
    def __init__(self, index=slice(None)):
        self.index = index
        self.n_features = None

    def fit(self, x, y=None):
        if len(x.shape) == 2:
            _, self.n_features = x.shape
        else:
            self.n_features = x.shape[0]
        return self

    def transform(self, x, y=None):
        xnew = x[..., self.index]
        if len(xnew.shape) == 2:
            return xnew
        else:
            return xnew.reshape(-1, 1)

    def get_feature_names(self, input_features=None):
        input_features = input_features or ["x_{}".format(i) for i in range(self.n_features)]
        if self.index == slice(None):
            return input_features
        else:
            return [n for i, n in zip(index, input_features) if i]
