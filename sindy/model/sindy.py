
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from sindy.differentiation import differentiation_methods
from sindy.model.optimizers import STLSQ
from sindy.model.base import equation


class SINDy(BaseEstimator):
    def __init__(
            self,
            threshold=0.1,
            feature_library=PolynomialFeatures(),
            differentiation_method=differentiation_methods.centered_difference,
            feature_names=None,
            n_jobs=1
        ):
        self.threshold              = threshold
        self.feature_library        = feature_library
        self.differentiation_method = differentiation_method
        self.feature_names          = feature_names
        self.n_jobs                 = n_jobs

    # For now just write this how we want it to look and then write simplest
    # version of the supporting code possible
    def fit(self, x, t=1, x_dot=None):
        if x_dot is None:
            x_dot = self.differentiation_method(x, t)

        feature_transformer = PolynomialFeatures()

        steps = [
            ("features", feature_transformer),
            ("model", STLSQ(threshold=self.threshold)),
        ]
        self.model = MultiOutputRegressor(Pipeline(steps), n_jobs=self.n_jobs)

        self.model.fit(x, x_dot)

        self.n_input_features_  = self.model.estimators_[0].steps[0][1].n_input_features_
        self.n_output_features_ = self.model.estimators_[0].steps[0][1].n_output_features_

        return self

    def predict(self, x):
        return self.model.predict(x)

    def equations(self, precision=3):
        names = self.model.estimators_[0].steps[0][1].get_feature_names(input_features=self.feature_names)
        return [equation(est, names, precision=precision) for est in self.model.estimators_]

    def score(self, x, y=None, multioutput="uniform_average"):
        if y is not None:
            x_dot = y
        else:
            x_dot = self.derivative.transform(x)
        return r2_score(self.model.predict(x), x_dot, multioutput=multioutput)

    @property
    def complexity(self):
        return sum(est.steps[1][1].complexity for est in self.model.estimators_)

    def differentiate(self, x, t=1):
        return self.differentiation_method(x, t)

    # TODO
    def coefficients(self):
        pass

    # TODO
    def get_feature_names(self):
        pass