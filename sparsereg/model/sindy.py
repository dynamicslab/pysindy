import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer

from sparsereg.preprocessing.symfeat import SymbolicFeatures
from sparsereg.model.base import STRidge, equation


def _derivative(x, dt=1.0):
    if isinstance(dt, (list, np.ndarray)):
        if len(dt) < 3:
            raise ValueError("dt has too few elements")
        dx = np.zeros_like(x)
        dx[1:-1, :] = (x[2:, :] - x[:-2, :]) / (dt[2:] - dt[:-2])

        dx[0, :] = (x[1, :] - x[0, :]) / (dt[1] - dt[0])
        dx[-1, :] = (x[-1, :] - x[-2, :]) / (dt[-1] - dt[-2])
    else:
        dx = np.zeros_like(x)
        dx[1:-1, :] = (x[2:, :] - x[:-2, :]) / (2.*dt)

        dx[0, :] = (x[1, :] - x[0, :]) / dt
        dx[-1, :] = (x[-1, :] - x[-2, :]) / dt

    return dx


class SINDy(BaseEstimator):
    def __init__(self, alpha=1.0, threshold=0.1, degree=3, operators=None, dt=1.0, n_jobs=1, derivative=None, feature_names=None, kw={}):
        self.alpha = alpha
        self.threshold = threshold
        self.degree = degree
        self.operators = operators
        self.n_jobs = n_jobs
        self.derivative = derivative or FunctionTransformer(func=_derivative, kw_args={"dt": dt})
        self.feature_names = feature_names
        self.kw = kw

    def fit(self, x, y=None):
        if y is not None:
            xdot = y
        else:
            xdot = self.derivative.transform(x)

        if self.operators is not None:
            feature_transformer = SymbolicFeatures(exponents=np.linspace(1, self.degree, self.degree), operators=self.operators)
        else:
            feature_transformer = PolynomialFeatures(degree=self.degree, include_bias=False)

        steps = [("features", feature_transformer),
                 ("model", STRidge(alpha=self.alpha, threshold=self.threshold, **self.kw))]
        self.model = MultiOutputRegressor(Pipeline(steps), n_jobs=self.n_jobs)
        self.model.fit(x, xdot)

        self.n_input_features_ = self.model.estimators_[0].steps[0][1].n_input_features_
        self.n_output_features_ = self.model.estimators_[0].steps[0][1].n_output_features_
        return self

    def predict(self, x):
        return self.model.predict(x)

    def equations(self, precision=3):
        names = self.model.estimators_[0].steps[0][1].get_feature_names(input_features=self.feature_names)
        return [equation(est, names, precision=precision) for est in self.model.estimators_]

    def score(self, x, y=None, multioutput="uniform_average"):
        if y is not None:
            xdot = y
        else:
            xdot = self.derivative.transform(x)
        return r2_score(self.model.predict(x), xdot, multioutput=multioutput)

    @property
    def complexity(self):
        return sum(est.steps[1][1].complexity for est in self.model.estimators_)
