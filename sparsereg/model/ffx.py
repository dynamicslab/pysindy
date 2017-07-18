from operator import attrgetter
from itertools import product

from sklearn.linear_model import ElasticNet
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib
from joblib import Parallel, delayed
import numpy as np

from sparsereg.preprocessing.symfeat import SymbolicFeatures
from sparsereg.util import pareto_front, nrmse


class MyPipeline(Pipeline):
    def __hash__(self):
        return hash(joblib.hash((self._final_estimator.coef_, self._final_estimator.intercept_)))

    def __eq__(self, other):
        return np.allclose(self._final_estimator.coef_, other._final_estimator.coef_) and \
               np.allclose(self._final_estimator.intercept_, other._final_estimator.intercept_)

def _get_alphas(alpha_max, num_alphas, eps):
    st, fin = np.log10(alpha_max*eps), np.log10(alpha_max)
    alphas1 = np.logspace(st, fin, num=num_alphas*10)[::-1][:int(num_alphas/4)]
    alphas2 = np.logspace(st, fin, num=num_alphas)
    return sorted(set(alphas1).union(alphas2), reverse=True)


def fit_(strategy, x_train, x_test, y_train, y_test, **kw):
    exponents, operators, base = strategy
    model = MyPipeline((("features", SymbolicFeatures(exponents=exponents, operators=operators)),
                        ("regression", base(**kw))))
    model = model.fit(x_train, y_train)
    model.test_score_ = model.score(x_test, y_test)
    model.complexity_ = np.count_nonzero(model.steps[1][1].coef_)
    return model


def run_ffx(x, y, exponents, operators, num_alphas=100, metric=nrmse, l1_ratios=(0.1, 0.3, 0.5, 0.7, 0.9, 0.95),
            eps=1e-70, target_score=0.01, min_models=40, alpha_max=1000, n_jobs=1, random_state=None, **kw):

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=random_state)

    base = ElasticNet
    base.score = lambda self, x, y: nrmse(y, self.predict(x))

    strategies = [(exponents, operators, base)]
    alphas = _get_alphas(alpha_max, num_alphas, eps)

    non_dominated_models = []

    for strategy in strategies:
        with Parallel(n_jobs=-1) as parallel:
            models = parallel(delayed(fit_)(strategy, x_train, x_test, y_train, y_test, alpha=alpha, l1_ratio=l1_ratio, **kw)
                            for alpha, l1_ratio in product(alphas, l1_ratios))
        front = pareto_front(models, "complexity_", "test_score_")
        non_dominated_models.extend(front)
        if any(model.test_score_ <= target_score for model in front):
            break

    return sorted(pareto_front(non_dominated_models, "complexity_", "test_score_"), key=attrgetter("complexity_"))
