from operator import attrgetter
from itertools import product

from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNet
from joblib import Parallel, delayed
import numpy as np

import symfeat as sf

from .util import pareto_front, cardinality, nrmse


class FFXModel(BaseEstimator):
    def __init__(self, coefs, alpha, l1_ratio, sym, precision=-6, metric=nrmse):
        self.null = 10**precision
        self.coefs_ = np.array([round(c, -precision) if abs(c) >= self.null else 0 for c in coefs])
        self.complexity = cardinality(coefs, null=self.null)
        self.metric = metric
        self.sym = sym
        self.alpha = alpha
        self.l1_ratio = l1_ratio

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

    def __eq__(self, other):
        return np.allclose(self.coefs_, other.coefs_)

    def __hash__(self):
        return hash(tuple(self.coefs_))


def _fit(alpha, l1_ratio, x, y, params):
    net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, **params).fit(x, y)
    return alpha, l1_ratio, net.coef_.copy()


def enet(x, y, alphas, l1_ratios, n_jobs, **params):
    with Parallel(n_jobs=n_jobs) as parallel:
        for alpha in alphas:
            yield from parallel(delayed(_fit)(alpha, l1_ratio, x, y, params) for l1_ratio in l1_ratios)


def _get_alphas(alpha_max, num_alphas, eps):
    st, fin = np.log10(alpha_max*eps), np.log10(alpha_max)
    alphas1 = np.logspace(st, fin, num=num_alphas*10)[::-1][:int(num_alphas/4)]
    alphas2 = np.logspace(st, fin, num=num_alphas)
    return sorted(set(alphas1).union(alphas2), reverse=True)


def run_ffx(x, y, exponents, operators, num_alphas=300, metric=nrmse, l1_ratios=(0.1, 0.3, 0.5, 0.7, 0.9, 0.95),
            eps=1e-70, max_complexity=100, target_score=0.01, min_models=40, alpha_max=1000, n_jobs=1, **params):

    sym = sf.SymbolicFeatures(exponents=exponents, operators=operators)

    features = sym.fit_transform(x)
    alphas = _get_alphas(alpha_max, num_alphas, eps)

    models = (FFXModel(coef, alpha, l1_ratio, sym, metric=metric)
              for alpha, l1_ratio, coef in enet(features, y, alphas, l1_ratios, n_jobs=n_jobs, **params))

    considered = []
    for model in models:
        model.score_ = model.score(x, y)
        considered.append(model)
        if (model.score_ <= target_score) or (model.complexity >= max_complexity):
            break

    return sorted(pareto_front(considered, "complexity", "score_"), key=attrgetter("complexity"))
