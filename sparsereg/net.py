from collections import defaultdict
import warnings

import numpy as np
from sklearn.exceptions import FitFailedWarning


def complexity(estimator):
    return np.count_nonzero(estimator.coef_)


def net(estimator, x, y, attr="alpha", max_coarsity=2, filter=True, r_max=1e3, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return _net(estimator, x, y, attr=attr, max_coarsity=max_coarsity, filter=filter, r_max=r_max, **kw)


def _net(estimator, x, y, attr="alpha", max_coarsity=2, filter=True, r_max=1e3, **kw):
    n_features = x.shape[1]

    memory = defaultdict(list)   # just a convenience list; this information is redundant
    models = defaultdict(list)

    def fit_in_memory(r):
        if not any(r in memory[k] for k in memory):
            est = estimator(**{**kw, **{attr: r}}).fit(x, y)
            c = complexity(est)
            memory[c].append(r)
            models[c].append(est)
            return c


    fit_in_memory(0)
    while True:
        try:
            c = fit_in_memory(r_max)
            if c == 0:
                r_max *= 0.8
            else:
                break
        except FitFailedWarning:
            r_max *= 0.8


    # greedy forward
    def greed_forward(c_lower, c_upper, coarsity):
        upper = min(memory[c_lower])
        lower = max(memory[c_upper])
        for r in np.linspace(lower, upper, 2**coarsity)[::-1]:
            fit_in_memory(r)

    # greedy search for transitions

    coarsity = 1
    all_expected = list(range(min(memory), max(memory) + 1))

    while True:
        it = ((d, e) for d, e in zip(sorted(memory.keys()), all_expected) if d != e)
        try:
            discovered, expected = next(it)
        except StopIteration:
            break
        n_keys = len(memory)
        greed_forward(max([k for k in memory if k < expected]), discovered, coarsity)
        if n_keys < len(memory):
            coarsity = 0
        else:
            coarsity += 1
            if coarsity >= max_coarsity:
                if discovered == max(all_expected):
                    break
                else:
                    all_expected.pop(all_expected.index(expected))
                    coarsity = 0
    if filter:
        return {k: min(v, key=lambda x: getattr(x, attr)) for k, v in models.items()}
    else:
        return models
