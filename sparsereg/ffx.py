from collections import defaultdict

from sklearn.linear_model import enet_path
from sklearn.metrics import mean_squared_error

import symfeat as sf


def pareto_front_2d(models, attr1, attr2):
    d = defaultdict(lambda: defaultdict(list))
    for model in models:
        d[getattr(model, attr1)][getattr(model, attr2)] = model
    return [d[a1][min(d[a1])] for a1 in d]


def cardinality(x):
    return sum(map(bool, x))


class FFXModel:
    def __init__(self, coefs, alpha, sym, metric=mean_squared_error):
        self.coefs_ = coefs
        self.complexity = cardinality(coefs)
        self.metric = metric
        self.sym = sym

    def predict(self, x):
        features = self.sym.transform(x)
        return features @ self.coefs_

    def score(self, x, y):
        yhat = self.predict(x)
        return self.metric(y, yhat)

    def pprint(self):
        fmt = "{}*{}".format
        expr =  "+".join(fmt(c, n) for c, n in zip(self.coefs_, self.sym.names) if c != 0) or "0"
        return expr


def run_ffx(x, y, exponents, operators, metric=mean_squared_error, **kwargs):
    sym = sf.SymbolicFeatures(exponents=exponents, operators=operators)
    features = sym.fit_transform(x)
    alphas, coefsT, _ = enet_path(features, y, warm_start=True, fit_intercept=False, **kwargs)
    models = [FFXModel(coef, alpha, sym, metric=metric) for coef, alpha in zip(coefsT.T, alphas)]
    for model in models:
        model.score_ = model.score(x, y)
    return pareto_front_2d(models, "complexity", "score_")
