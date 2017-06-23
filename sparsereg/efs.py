import re
import random
import warnings

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso
from sklearn.preprocessing import FunctionTransformer

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
    "cube": lambda x: np.power(x, 3),
    "sin": np.sin,
    "cos": np.cos
}

def size(name):
    pattern = r"[\(,]"
    return len(re.findall(pattern, name)) + 1


def mutate(names, importance, toursize, rng=random):
    f = rng.choice(list(operators))
    arity = getattr(operators[f], "nin", None) or operators[f].__code__.co_argcount
    parents = []
    for _ in range(arity):
        candidates = rng.sample(names, toursize)
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
        return random
    
class EFS(BaseEstimator, RegressorMixin):
    def __init__(self, q=3, mu=4, max_size=5, t=0.95, toursize=2, gen=200, random_state=None):
        self.q = q
        self.mu = mu
        self.max_size = max_size
        self.t = t
        self.toursize = toursize
        self.gen = gen
        self.rng = _check_rng(random_state)

    def fit(self, x, y):
        n_samples, p = x.shape

        linear_names = ["x_{}".format(i) for i in range(p)]
        names = ["x_{}".format(i) for i in range(p)]
        data = [x[:, i] for i in range(p)]

        models = net(Lasso, x, y).values()
        scores = [model.score(x, y) for model in models]
        coefs = [model.coef_ for model in models]
        
        importance = get_importance(coefs, scores)

        for i in range(self.gen):
            new_names = []
            new_data = []
            while len(new_names + names) < self.mu + p + self.q:
                f, new_name, parents = mutate(names, importance, self.toursize, self.rng)
                if size(new_name) <= self.max_size and new_name not in new_names and new_name not in names:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        feature = f(*[data[i] for i in parents])
                    if np.all(np.isfinite(feature)):
                        new_names.append(new_name)
                        new_data.append(feature)
            
            names.extend(new_names)
            data.extend(new_data)
            models = net(Lasso, np.array(data).T, y).values()
            scores = [model.score(np.array(data).T, y) for model in models]
            coefs = [model.coef_ for model in models]

            importance = list(get_importance(coefs, scores))

            names_to_discard = [n for n in sorted(names, key=lambda x: importance[names.index(x)]) if n not in linear_names][self.mu:]
            for n in names_to_discard:
                i = names.index(n)
                names.pop(i)
                data.pop(i)
                importance.pop(i)

        self.names = names
        return self