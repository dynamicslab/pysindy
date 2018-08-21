from operator import attrgetter
from itertools import product
from collections import namedtuple
from copy import deepcopy
import warnings

from sklearn.base import TransformerMixin, BaseEstimator, RegressorMixin, clone
from sklearn.linear_model.coordinate_descent import ElasticNet, _pre_fit
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y, check_random_state
import joblib
import numpy as np

from sparsereg.preprocessing.symfeat import SymbolicFeatures
from sparsereg.model.base import RationalFunctionMixin, PrintMixin
from sparsereg.util import pareto_front, nrmse, aic
from sparsereg.util.pipeline import ColumnSelector


class FFXModel(Pipeline):
    def __init__(self, strategy, **kw):
        self.strategy = strategy
        self.kw = kw
        super().__init__(steps=[
               ("selection", ColumnSelector(index=self.strategy.index)),
               ("features", SymbolicFeatures(exponents=self.strategy.exponents,
                                             operators=self.strategy.operators,
                                             consider_products=self.strategy.consider_products)),
               ("regression", strategy.base(warm_start=True, **self.kw))])

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

    def pre_compute(self, x, y):
        pass


class FFXElasticNet(PrintMixin, ElasticNet):
    """Mixin, implements only the `score` method."""
    def score(self, x, y):
        """Score using the nrmse."""
        return nrmse(self.predict(x), y)


class FFXRationalElasticNet(RationalFunctionMixin, FFXElasticNet):
    pass


Strategy = namedtuple("Strategy", "exponents operators consider_products index base")

Strategy.__doc__ = """
Class for holding strategy data.

:param iterable exponents: exponents to consider
:param dict operators: mapping str (name) to callable (operator)
:param bool consider_products: Whether to consider products of features as features
:param slice index: Which columns of the features provided to `fit` will be used.
:param type base: type of FFX variant to use.
"""

def build_strategies(exponents, operators, rational=True):
    strategies = []
    linear = Strategy(exponents=[1], operators={}, consider_products=False, index=slice(None), base=FFXElasticNet)
    strategies.append(linear)
    if rational:
        rational = Strategy(exponents=[1], operators={}, consider_products=False, index=slice(None), base=FFXRationalElasticNet)
        strategies.append(rational)
    if sorted(exponents) != [1]:
       full_exponents = Strategy(exponents=exponents, operators={}, consider_products=True, index=slice(None), base=FFXElasticNet)
       strategies.insert(1, full_exponents)
       if rational:
           full_exponents_rational = Strategy(exponents=exponents, operators={}, consider_products=True, index=slice(None), base=FFXRationalElasticNet)
           strategies.append(full_exponents_rational)

    if operators:
        if exponents != [1]:
            simple_operators = Strategy(exponents=[1], operators=operators, consider_products=True, index=slice(None), base=FFXElasticNet)
            strategies.append(simple_operators)
        full_operators = Strategy(exponents=exponents, operators=operators, consider_products=True, index=slice(None), base=FFXElasticNet)
        strategies.append(full_operators)

    def strategy_generator(front):
        yield from strategies
    return strategy_generator


def _get_alphas(alpha_max, num_alphas, eps):
    """Return list of alphas between `alpha_max` and `alpha_max*eps` in descending order.

    The returned alphas will be logarithmically spaced mostly.
    Close to zero `approx. 1/4 * num_alphas` will be inserted with a 10x finer logarithmic spacing.
    """
    st, fin = np.log10(alpha_max*eps), np.log10(alpha_max)
    alphas1 = np.logspace(st, fin, num=num_alphas*10)[::-1][:int(num_alphas/4)]
    alphas2 = np.logspace(st, fin, num=num_alphas)
    return sorted(set(alphas1).union(alphas2), reverse=True)


def _path_is_saturated(models, n_tail=15, digits=4):
    if len(models) <= n_tail:
        return False
    else:
        return round(models[-1].train_score_, digits) == round(models[-n_tail].train_score_, digits)


def enet_path(est, x_train, x_test, y_train, y_test, num_alphas, eps,
              l1_ratio, target_score, n_tail, max_complexity):
    models = []

    trafo = Pipeline(steps=est.steps[:-1])
    final = est._final_estimator
    fit_intercept = final.fit_intercept
    normalize = final.normalize
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        features = trafo.fit_transform(x_train)

    if isinstance(final, RationalFunctionMixin):
        features = est._final_estimator._transform(features, y_train)

    X, y, X_offset, y_offset, X_scale, precompute, Xy = _pre_fit(features, y_train, None, True, normalize=normalize,
                                                                fit_intercept=fit_intercept, copy=True)


    n_samples = X.shape[0]
    alpha_max = np.abs(np.nanmax(X.T @ y) / (n_samples * l1_ratio))

    est.set_params(regression__precompute=precompute, regression__fit_intercept=False,
                   regression__normalize=False, regression__warm_start=True)

    est_ = FFXElasticNet()
    est_.set_params(**final.get_params())

    for alpha in _get_alphas(alpha_max, num_alphas, eps):
        est_.set_params(l1_ratio=l1_ratio, alpha=alpha)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            est_.fit(X, y, check_input=False)

        model = deepcopy(est)
        model.set_params(regression__fit_intercept=fit_intercept, regression__normalize=normalize,
                         regression__l1_ratio=l1_ratio, regression__alpha=alpha)

        for attr in ["coef_", "intercept_", "n_iter_"]:
            setattr(model._final_estimator, attr, getattr(est_, attr))

        model._final_estimator._set_intercept(X_offset, y_offset, X_scale)
        if isinstance(model._final_estimator, RationalFunctionMixin):
            model._final_estimator._arrange_coef()

        model.train_score_ = model.score(x_train, y_train)
        model.test_score_ = model.score(x_test, y_test)
        model.complexity_ = np.count_nonzero(model._final_estimator.coef_)
        models.append(model)

        if model.train_score_ <= target_score:
            # print("Reached target score")
            break
        elif model.complexity_ >= max_complexity:
            # print("Reached target complexity")
            break
        elif _path_is_saturated(models, n_tail=n_tail):
            # print("Stagnation in train score")
            break
    return models


def run_strategy(strategy, x_train, x_test, y_train, y_test, num_alphas, eps, l1_ratios,
                 target_score, n_tail, max_complexity, n_jobs, **kw):

    est = FFXModel(strategy, **kw)
    with joblib.Parallel(n_jobs=n_jobs) as parallel:
        paths = parallel(joblib.delayed(enet_path)(est, x_train, x_test, y_train, y_test,
                                                   num_alphas, eps, l1_ratio, target_score, n_tail,
                                                   max_complexity)
                                                   for l1_ratio in l1_ratios)
    return [model for path in paths for model in path]


def run_ffx(x_train, x_test, y_train, y_test, exponents, operators, num_alphas=100,
            l1_ratios=(0.1, 0.3, 0.5, 0.7, 0.9, 0.95), eps=1e-30, target_score=0.01,
            max_complexity=50, n_tail=15, random_state=None, strategies=None, n_jobs=1,
            rational=True, **kw):

    strategies = strategies or build_strategies(exponents, operators, rational=rational)

    non_dominated_models = []

    for strategy in strategies(non_dominated_models):
        models = run_strategy(strategy, x_train, x_test, y_train, y_test, num_alphas, eps,
                              l1_ratios, target_score, n_tail, max_complexity, n_jobs, **kw)
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
    def __init__(self, l1_ratios=(0.4, 0.8, 0.95), num_alphas=30,
                 eps=1e-5, random_state=None, strategies=None, target_score=0.01,
                 n_tail=5, decision="min", max_complexity=50,
                 exponents=[1, 2], operators={}, n_jobs=1, rational=True, **kw):
        """Fast Function eXtraction model.

        :param iterable l1_ratios: Determines ratio of l1 to l2 penalty term
        :param int num_alphas: Determines numbers of different ratios of cost function to penalty term. `0<= l1_ratio <= 1`.
        :param float eps: ratio of smallest to largest alpha considered. (`0 < eps < 1`)
        :param int random_state:
        :param iterable strategies: `Strategy` s to consider
        :param float target_score: break condition on cost function for innermost loop
        :param int n_tail: length of path (in alpha) to check into past for saturation
        :param str decision: one of ``'weight'`` or ``'min'``
        :param float max_complexity: break condition on model complexity for innermost loop
        :param iterable exponents: can contain float and negative values
        :param dict operators: mapping operator name to callable (of one variable)
        :param int n_jobs:
        :param bool rational: Whether to consider general rational functions as well
        :param kw:

        The implemented algorithm is found in `http://dx.doi.org/10.1007/978-1-4614-1770-5_13`.

        A `Strategy` is determined by a set of nonlinear functions from which an extended set of features
        will be generated by evaluating these functions on all given features.
        You can either supply the strategies directly via the `strategies` parameter or let the strategies be generated.
        Generation of strategies is configured by the parameters `exponents`, `operators` and `rational`.
        When `strategies` is given, `exponents`, `operators` and `rational` have no effect.

        Strategy generation takes place in the following manner:


        `exponents`:
            Orders of the monomials to consider for each single feature. (No products between features here).
            `exponents` is an iterable of numbers (floats and negative values are possible, 1 will always automatically be included.)
            The first step in strategy generation is calculating all monomials.
        `operators`:
            mapping of str to callable taking one parameter.
            All callables in `operators` will be evaluated on all monomials from the first step

        products
            Not configurable. Always consider all products of each operator feature from the second step with each monomial feature from the first.
            And all products of monomial features with all monomial features based on a different feature
            (thus generating mixed products up to order `2*max(exponents)`).
        `rational`
            If true, do not only consider generalized linear models from all basis functions
            but consider also rational functions using the rational function trick described `here <`http://dx.doi.org/10.1007/978-1-4614-1770-5_13>`_

        For each `Strategy`, an elastic net optimizer will be run with many combinations of l1_ratio and alpha.
        A `l1_ratio` of 0 corresponds to ridge regression (only l2 penalty), a `l1_ratio` of 1 corresponds to
        LASSO regression (only l1 penalty).
        `alpha` determines the amount of regularization, where `alpha=0` would mean now regularization and
        `alpha -> infty` would mean only regularization.
        For details on the used elastic net algorithm see :class:`sklearn.linear_model.ElasticNet`.

        The number of alphas is loosely determined by `num_alpha` (the actual number is close and never smaller).
        The maximum value of the considered `alpha` is determined dynamically based on Tibshirani's "Strong Rules", see
        `https://doi.org/10.1111/j.1467-9868.2011.01004.x`_
        The rule gives an `alpha` for which the fitted model will (in most relevant cases)
        have a complexity of 0 (no nonzero terms).
        This maximum alpha also depends on the l1_ratio, therefore the iteration over alpha takes place
        in the innermost loop.

        The innermost loop would iterate from the maximum alpha to `eps` times the maximum alpha.
        With increasing `alpha`, the complexity (number of non-zero terms) is expected to increase, whereas the
        cost (nrmse evaluated on the training set) is expected to decrease.

        The innermost loop has three break conditions:

        1. train_score
           If the cost is less or equal to `target_score`
        2. complexity
           If the complexity  is greater or equal to `max_complexity`
        3. saturation
           No significant improvement in the cost during the last `n_tail` iterations.
           (Significant -> last 4 decimal digits)

        To obtain a single model from the Pareto front of models, the Akaike information criterion (AIC)
        is used (see `https://en.wikipedia.org/wiki/Akaike_information_criterion`).
        How it is used is determined by the `decision` parameter.
        If `decision == 'min'`, the model with the smallest AIC is taken,
        if `decision == 'weighted'`, the resulting model will be a linear combination of all models
        the front consists of, weighted by `exp((min(AIC)-AIC)/2)`.

        """
        self.l1_ratios = l1_ratios
        self.num_alphas = num_alphas
        self.eps = eps
        self.random_state = check_random_state(random_state)
        self.strategies = strategies
        self.target_score = target_score
        self.n_tail = n_tail
        self.exponents = exponents
        self.operators = operators
        self.kw = kw
        self.decision = decision
        self.max_complexity = max_complexity
        self.n_jobs = n_jobs
        self.rational = rational

    def fit(self, x, y=None):
        x, y = check_X_y(x, y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=self.random_state)
        self.front = run_ffx(x_train, x_test, y_train, y_test,
                             self.exponents, self.operators, num_alphas=self.num_alphas, l1_ratios=self.l1_ratios,
                             target_score=self.target_score, n_tail=self.n_tail, random_state=self.random_state,
                             strategies=self.strategies, n_jobs=self.n_jobs, max_complexity=self.max_complexity,
                             rational=self.rational, eps=self.eps, **self.kw)
        self.make_model(x_test, y_test)
        return self

    def predict(self, x):
        return self._model.predict(x)

    def score(self, x, y):
        return self._model.score(x, y)

    def make_model(self, x_test, y_test):
        residuals = [y_test - est.predict(x_test) for est in self.front]
        complexities = [est.complexity_ for est in self.front]
        aic_scores = np.array([aic(res, c) for res, c in zip(residuals, complexities)])
        aic_scores -= np.min(aic_scores)

        if self.decision == "weight":
            weights = np.exp(-aic_scores / 2)
            weights /= np.sum(weights)
            self._model = WeightedEnsembleEstimator(self.front, weights)
        else:
            self._model = self.front[np.argmin(aic_scores)]

    def print_model(self, input_features=None):
        return self._model.print_model(input_features)
