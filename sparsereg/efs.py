import re
import random
import warnings

import numpy as np

from sklearn.base import RegressorMixin, TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoLarsCV, Lasso

from sparsereg.net import net

operators = {
    "add": np.add,
    "subtract": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "square": np.square,
    "sin": np.sin,
    "cos": np.cos
}

def size(name):
    pattern = r"[\(,]"
    return len(re.findall(pattern, name)) + 1


def mutate(names, importance, toursize, operators, rng=random):
    f = rng.choice(list(operators))
    arity = getattr(operators[f], "nin", None) or operators[f].__code__.co_argcount
    parents = []
    size = min(toursize, len(names))
    for _ in range(arity):
        candidates = rng.sample(names, size)
        parent = sorted(candidates, key=lambda i: importance[names.index(i)])[0]
        parents.append(parent)
 
    args = ",".join(parents)
    name = f + "(" + args + ")"
    return operators[f], name, [names.index(p) for p in parents]


def get_importance(coefs, scores):
    return np.array([[score if c else 0 for c in coef] for coef, score in zip(coefs, scores)]).sum(axis=0)


def _check_rng(state):
    if isinstance(state, random.Random):
        return state
    elif isinstance(state, int):
        rng = random.Random()
        rng.seed(state)
        return rng
    else:
        return random.Random()

def _transform(x, names, operators):
    args = ",".join("x_{}".format(i) for i in range(x.shape[1]))
    funcs = [eval("lambda {}: {}".format(args, code), {**operators}) for code in names]
    data = np.array([f(*x.T) for f in funcs]).T
    return data


class LibTrafo(BaseEstimator, TransformerMixin):
    def __init__(self, names, operators):
        self.names = names[:]
        self.operators = operators
    
    def fit(self, x, y=None):
        return self

    def transform(self, x, y=None):
        return _transform(x, self.names, self.operators)


def _fit_model(x, y, names, operators, **kw):
    steps = ("trafo", LibTrafo(names, operators)), ("lasso", LassoLarsCV(**kw))
    model = Pipeline(steps).fit(x, y)
    return model, model.score(x, y)


class EFS(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, q=1, mu=1, max_size=5, t=0.95, toursize=5, max_stall_iter=20, max_iter=2000, random_state=None, operators=operators, max_coarsity=2, n_jobs=1):
        self.q = q
        self.mu = mu
        self.max_size = max_size
        self.t = t
        self.toursize = toursize
        self.max_iter = max_iter
        self.max_stall_iter = max_stall_iter
        self.max_coarsity = max_coarsity
        self.operators = operators
        self.n_jobs = n_jobs
        self.rng = _check_rng(random_state)

    def fit(self, x, y):
        n_samples, p = x.shape

        linear_names = ["x_{}".format(i) for i in range(p)]
        names = linear_names[:]
        data = [x[:, i] for i in range(p)]

        models = net(Lasso, x, y, max_coarsity=self.max_coarsity).values()
        scores = [model.score(x, y) for model in models]
        coefs = [model.coef_ for model in models]
        
        importance = get_importance(coefs, scores)

        stall_iter = 0

        best_names = linear_names[:]
        best_model, best_score = _fit_model(x, y, best_names, self.operators, n_jobs=self.n_jobs)
        for _ in range(self.max_iter):
            old_names = sorted(names[:])
            stall_iter += 1
            new_names = []
            new_data = []

            while len(new_names + names) < p*(self.mu + 1 + self.q):
                f, new_name, parents = mutate(names, importance, self.toursize, self.operators, self.rng)
                if size(new_name) <= self.max_size and new_name not in new_names and new_name not in names:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        feature = f(*[data[i] for i in parents])
                        if np.all(np.isfinite(feature)) and all(abs(np.corrcoef(feature, data[i]))[1, 0] <= self.t for i in parents):
                            new_names.append(new_name)
                            new_data.append(feature)

            names.extend(new_names)
            data.extend(new_data)
            models = net(Lasso, np.array(data).T, y, max_coarsity=self.max_coarsity).values()
            scores = [model.score(np.array(data).T, y) for model in models]
            coefs = [model.coef_ for model in models]
            importance = list(get_importance(coefs, scores))
            names_to_discard = [n for n in sorted(names, key=lambda x: importance[names.index(x)], reverse=True) if n not in linear_names][-self.mu*p:]
            for n in names_to_discard:
                i = names.index(n)
                names.pop(i)
                data.pop(i)
                importance.pop(i)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model, score = _fit_model(x, y, names, self.operators, n_jobs=self.n_jobs)
            
            if score > best_score:
                best_model = model
                best_score = score
                stall_iter = 0

            elif stall_iter >= self.max_stall_iter:
                break

        self.model = best_model
        return self
    
    def predict(self, x):
        return self.model.predict(x)

    def transform(self, x, y=None):
        return self.model.steps[0][-1].transform(x)