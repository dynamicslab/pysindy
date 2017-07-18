from operator import attrgetter
from itertools import product
from collections import namedtuple
from copy import deepcopy

from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
from joblib import Parallel, delayed
import numpy as np

from sparsereg.preprocessing.symfeat import SymbolicFeatures
from sparsereg.model._base import RationalFunctionMixin
from sparsereg.util import pareto_front, rmse


class MyPipeline(Pipeline):
    def __hash__(self):
        return hash(joblib.hash((self._final_estimator.coef_, self._final_estimator.intercept_)))

    def __eq__(self, other):
        if self._final_estimator.coef_.shape != other._final_estimator.coef_.shape:
            return False

        return np.allclose(self._final_estimator.coef_, other._final_estimator.coef_) and \
               np.allclose(self._final_estimator.intercept_, other._final_estimator.intercept_)

class FFXElasticNet(ElasticNet):
    def score(self, x, y):
        return rmse(self.predict(x) - y)

class FFXRationalElasticNet(RationalFunctionMixin, FFXElasticNet):
    pass


Strategy = namedtuple("Strategy", "exponents operators consider_products base")

def build_strategies(exponents, operators):
    linear = Strategy(exponents=[1], operators={}, consider_products=False, base=FFXElasticNet)
    rational = Strategy(exponents=[1], operators={}, consider_products=False, base=FFXRationalElasticNet)
    full_exponents = Strategy(exponents=exponents, operators={}, consider_products=True, base=FFXElasticNet)
    full_exponents_rational = Strategy(exponents=exponents, operators={}, consider_products=True, base=FFXRationalElasticNet)
    full_operators = Strategy(exponents=exponents, operators=operators, consider_products=True, base=FFXElasticNet)
    #full_operators_rational = Strategy(exponents=exponents, operators=operators, consider_products=True, base=FFXRationalElasticNet)
    return [linear, rational, full_exponents, full_exponents_rational, full_operators]

def _get_alphas(alpha_max, num_alphas, eps):
    st, fin = np.log10(alpha_max*eps), np.log10(alpha_max)
    alphas1 = np.logspace(st, fin, num=num_alphas*10)[::-1][:int(num_alphas/4)]
    alphas2 = np.logspace(st, fin, num=num_alphas)
    return sorted(set(alphas1).union(alphas2), reverse=True)


def run_strategy(strategy, x_train, x_test, y_train, y_test, alphas, l1_ratios, target_score, **kw):
    est = MyPipeline((("features", SymbolicFeatures(exponents=strategy.exponents,
                                                    operators=strategy.operators,
                                                    consider_products=strategy.consider_products)),
                      ("regression", strategy.base(warm_start=True, **kw))))

    models = []
    for alpha, l1_ratio in product(alphas, l1_ratios):
        est.set_params(regression__l1_ratio=l1_ratio, regression__alpha=alpha)

        est = est.fit(x_train, y_train)
        models.append(deepcopy(est))
        models[-1].train_score_ = est.score(x_train, y_train)
        models[-1].test_score_ = est.score(x_train, y_train)
        models[-1].complexity_ = np.count_nonzero(est.steps[1][1].coef_)
        if models[-1].train_score_ <= target_score:
            break

    return models


def run_ffx(x, y, exponents, operators, num_alphas=100, l1_ratios=(0.1, 0.3, 0.5, 0.7, 0.9, 0.95),
            eps=1e-30, target_score=0.01, alpha_max=100, n_jobs=1, random_state=None, strategies=None, **kw):

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state)


    strategies = strategies or build_strategies(exponents, operators)
    alphas = _get_alphas(alpha_max, num_alphas, eps)

    non_dominated_models = []

    for strategy in strategies:
        print(strategy)
        models = run_strategy(strategy, x_train, x_test, y_train, y_test, alphas, l1_ratios, target_score, **kw)
        front = pareto_front(models, "complexity_", "test_score_")
        non_dominated_models.extend(front)
        if any(model.test_score_ <= target_score for model in front):
            break

    return sorted(pareto_front(non_dominated_models, "complexity_", "test_score_"), key=attrgetter("complexity_"))
