from operator import attrgetter
from itertools import product
from collections import namedtuple
from copy import deepcopy
import warnings

from sklearn.base import TransformerMixin, BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y, check_random_state
import joblib
import numpy as np

from sparsereg.preprocessing.symfeat import SymbolicFeatures
from sparsereg.model._base import RationalFunctionMixin, PrintMixin
from sparsereg.util import pareto_front, rmse, aic


class ColumnSelector(TransformerMixin, BaseEstimator):
    def __init__(self, index=slice(None)):
        self.index = index
        self.n_features = None

    def fit(self, x, y=None):
        _, self.n_features = x.shape
        return self

    def transform(self, x, y=None):
        return x[..., self.index]

    def get_feature_names(self, input_features=None):
        input_features = input_features or ["x_{}".format(i) for i in range(self.n_features)]
        if self.index == slice(None):
            return input_features
        else:
            return [n for i, n in zip(index, input_features) if i]


class FFXModel(Pipeline):
    def __init__(self, strategy, **kw):
        self.strategy = strategy
        self.kw = kw
        super().__init__(steps=(
               ("selection", ColumnSelector(index=self.strategy.index)),
               ("features", SymbolicFeatures(exponents=self.strategy.exponents,
                                             operators=self.strategy.operators,
                                             consider_products=self.strategy.consider_products)),
               ("regression", strategy.base(warm_start=False, **self.kw))))

    def __hash__(self):
        return hash(joblib.hash((self._final_estimator.coef_, self._final_estimator.intercept_)))

    def __eq__(self, other):
        if self._final_estimator.coef_.shape != other._final_estimator.coef_.shape:
            return False

        return np.allclose(self._final_estimator.coef_, other._final_estimator.coef_) and \
               np.allclose(self._final_estimator.intercept_, other._final_estimator.intercept_)

    def print_model(self, input_features=None):
        for step in self.steps[:-1]:
            input_features = step[1].get_feature_names(input_features)
        return self._final_estimator.print_model(input_features)


class FFXElasticNet(PrintMixin, ElasticNet):
    def score(self, x, y):
        return rmse(self.predict(x) - y)


class FFXRationalElasticNet(RationalFunctionMixin, FFXElasticNet):
    pass


Strategy = namedtuple("Strategy", "exponents operators consider_products index base")


def build_strategies(exponents, operators):
    strategies = []
    linear = Strategy(exponents=[1], operators={}, consider_products=False, index=slice(None), base=FFXElasticNet)
    rational = Strategy(exponents=[1], operators={}, consider_products=False, index=slice(None), base=FFXRationalElasticNet)
    strategies.append(linear)
    strategies.append(rational)
    if sorted(exponents) != [1]:
        full_exponents = Strategy(exponents=exponents, operators={}, consider_products=True, index=slice(None), base=FFXElasticNet)
        full_exponents_rational = Strategy(exponents=exponents, operators={}, consider_products=True, index=slice(None), base=FFXRationalElasticNet)
        strategies.append(full_exponents)
        strategies.append(full_exponents_rational)
    if operators:
        full_operators = Strategy(exponents=exponents, operators=operators, consider_products=True, index=slice(None), base=FFXElasticNet)
        strategies.append(full_operators)
    #full_operators_rational = Strategy(exponents=exponents, operators=operators, consider_products=True, base=FFXRationalElasticNet)
    def strategy_generator(front):
        yield from strategies
    return strategy_generator


def _get_alphas(alpha_max, num_alphas, eps):
    st, fin = np.log10(alpha_max*eps), np.log10(alpha_max)
    alphas1 = np.logspace(st, fin, num=num_alphas*10)[::-1][:int(num_alphas/4)]
    alphas2 = np.logspace(st, fin, num=num_alphas)
    return sorted(set(alphas1).union(alphas2), reverse=True)


def enet_path(est, x_train, x_test, y_train, y_test, alphas, l1_ratio, target_score):
    models = []
    for alpha in alphas:
        est.set_params(regression__l1_ratio=l1_ratio, regression__alpha=alpha)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            est = est.fit(x_train, y_train)
        if not est._final_estimator.n_iter_ == est._final_estimator.max_iter:
            models.append(deepcopy(est))
            models[-1].train_score_ = est.score(x_train, y_train)
            models[-1].test_score_ = est.score(x_train, y_train)
            models[-1].complexity_ = np.count_nonzero(est._final_estimator.coef_)
            if models[-1].train_score_ <= target_score:
                break
        else: # refresh estimator
            est = clone(est)
    return models


def run_strategy(strategy, x_train, x_test, y_train, y_test, alphas, l1_ratios, target_score, n_jobs, **kw):

    est = FFXModel(strategy, **kw)
    with joblib.Parallel(n_jobs=n_jobs) as parallel:
        paths = parallel(joblib.delayed(enet_path)(est, x_train, x_test, y_train, y_test, alphas, l1_ratio, target_score)
                                                   for l1_ratio in l1_ratios)
    
    return [model for path in paths for model in path]


def run_ffx(x_train, x_test, y_train, y_test, exponents, operators, num_alphas=100, l1_ratios=(0.1, 0.3, 0.5, 0.7, 0.9, 0.95),
            eps=1e-30, target_score=0.01, alpha_max=100, random_state=None, strategies=None, n_jobs=1, **kw):

    strategies = strategies or build_strategies(exponents, operators)
    alphas = _get_alphas(alpha_max, num_alphas, eps)

    non_dominated_models = []

    for strategy in strategies(non_dominated_models):
        models = run_strategy(strategy, x_train, x_test, y_train, y_test, alphas, l1_ratios, target_score, n_jobs, **kw)
        front = pareto_front(models, "complexity_", "test_score_")
        non_dominated_models.extend(front)
        if any(model.test_score_ <= target_score for model in front):
            break

    return sorted(pareto_front(non_dominated_models, "complexity_", "test_score_"), key=attrgetter("complexity_"))


class WeightedEnsembleEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, estimators, weights):
        self.estimators = estimators
        self.weights = weights

    def fit(self, x, y=None):
        return self

    def predict(self, x):
        return np.sum([w * est.predict(x) for w, est in zip(self.weights, self.estimators)], axis=0)

    def print_model(self, input_features=None):
        return "+".join(["{}*({})".format(w, est.print_model(input_features))
                         for w, est in zip(self.weights, self.estimators)])


class FFX(BaseEstimator, RegressorMixin):
    def __init__(self, l1_ratios=(0.4, 0.8, 0.95), num_alphas=100, alpha_max=100,
                 eps=1e-10, random_state=None, strategies=None, target_score=0.01, decision="min",
                 exponents=[1, 2], operators={}, kw={}, n_jobs=1):

        self.l1_ratios = l1_ratios
        self.num_alphas = num_alphas
        self.alpha_max = alpha_max
        self.eps = eps
        self.random_state = check_random_state(random_state)
        self.strategies = strategies
        self.target_score = target_score
        self.exponents = exponents
        self.operators = operators
        self.kw = kw
        self.decision = decision
        self.n_jobs = n_jobs

    def fit(self, x, y=None):
        x, y = check_X_y(x, y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=self.random_state)
        self.front = run_ffx(x_train, x_test, y_train, y_test,
                             self.exponents, self.operators, self.num_alphas, self.l1_ratios,
                             self.eps, self.target_score, self.alpha_max, self.random_state,
                             self.strategies, n_jobs=self.n_jobs, **self.kw)
        self.make_model(x_test, y_test)
        return self

    def predict(self, x):
        return self._model.predict(x)

    def make_model(self, x_test, y_test):
        residuals = [y_test - est.predict(x_test) for est in self.front]
        complexities = [est.complexity_ for est in self.front]
        # scores = [est.score_ for est in self.front]
        aic_scores = np.array([aic(res, c) for res, c in zip(residuals, complexities)])
        aic_scores -= np.min(aic_scores)

        if self.decision == "weight":
            weights = np.exp(-aic_scores/2  )
            weights /= np.sum(weights)
            self._model = WeightedEnsembleEstimator(self.front, weights)
        else:
            self._model = self.front[np.argmin(aic_scores)]

    def print_model(self, input_features=None):
        return self._model.print_model(input_features)
