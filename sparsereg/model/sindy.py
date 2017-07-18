import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer

from sparsereg.model._base import STRidge, equation


def _derivative(x, dt=1.0):
    dx = np.zeros_like(x)
    dx[1:-1, :] = (x[2:, :] - x[:-2, :]) / (2.*dt)

    dx[0, :] = (x[1, :] - x[0, :]) / dt
    dx[-1, :] = (x[-1, :]-x[-2, :]) / dt

    return dx


class SINDy(BaseEstimator):
    def __init__(self, dt=1.0, alpha=1.0, threshold=0.1, degree=3, n_jobs=1, derivative=None, feature_names=None, kw={}):
        self.alpha = alpha
        self.threshold = threshold
        self.degree = degree
        self.n_jobs = n_jobs
        self.derivative = derivative or FunctionTransformer(func=_derivative, kw_args={"dt": dt})
        self.feature_names = feature_names
        self.kw = kw

    def fit(self, x, y=None):
        xdot = self.derivative.transform(x)

        steps = (("features", PolynomialFeatures(degree=self.degree, include_bias=False)),
                 ("model", STRidge(alpha=self.alpha, threshold=self.threshold, **self.kw)))
        self.model = MultiOutputRegressor(Pipeline(steps), n_jobs=self.n_jobs)
        self.model.fit(x, xdot)
        return self

    def predict(self, x):
        return self.model.predict(x)

    def equations(self):
        names = self.model.estimators_[0].steps[0][1].get_feature_names(input_features=self.feature_names)
        return [equation(est, names) for est in self.model.estimators_]

    def score(self, x, y=None):
        xdot = self.derivative.transform(x)
        return r2_score(self.model.predict(x), xdot)
