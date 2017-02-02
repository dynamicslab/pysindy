from operator import attrgetter

from sklearn.linear_model import enet_path
from sklearn.metrics import mean_squared_error

import symfeat as sf

from .util import pareto_front, cardinality


class FFXModel:
    def __init__(self, coefs, alpha, sym, null=1e-9, metric=mean_squared_error):
        self.null = null
        self.coefs_ = coefs
        self.complexity = cardinality(coefs, null=null)
        self.metric = metric
        self.sym = sym
        self.alpha = alpha

    def predict(self, x):
        features = self.sym.transform(x)
        return features @ self.coefs_

    def score(self, x, y):
        yhat = self.predict(x)
        return self.metric(y, yhat)

    def pprint(self):
        fmt = "{}*{}".format
        expr =  "+".join(fmt(c, n) for c, n in zip(self.coefs_, self.sym.names) if abs(c) >= self.null) or "0"
        return expr


def run_ffx(x, y, exponents, operators, metric=mean_squared_error, **kwargs):
    sym = sf.SymbolicFeatures(exponents=exponents, operators=operators)
    features = sym.fit_transform(x)
    alphas, coefsT, _ = enet_path(features, y, warm_start=False, fit_intercept=True, **kwargs)
    models = [FFXModel(coef, alpha, sym, metric=metric) for coef, alpha in zip(coefsT.T, alphas)]
    for model in models:
        model.score_ = model.score(x, y)
    return pareto_front(models, "complexity", "score_")
