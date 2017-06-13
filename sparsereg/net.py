from collections import defaultdict

import numpy as np
from sklearn.exceptions import FitFailedWarning


def complexity(estimator):
    return np.count_nonzero(estimator.coef_)


def net(estimator, x, y, attr="alpha", max_coarsity=2, filter=True, r_max=1e3, **kw):
    n_features = x.shape[1]

    memory = defaultdict(list)   # just a convenience list; this information is redundant
    models = defaultdict(list)

    def fit_in_memory(r):
        est = estimator(**{**kw, **{attr: r}}).fit(x, y)
        c = complexity(est)
        memory[c].append(r)
        models[c].append(est)
        return est
    
    fit_in_memory(0)
    while True:
        try:
            fit_in_memory(r_max)
            break
        except FitFailedWarning:
            r_max *= 0.8

    # greedy forward
    def greed_forward(c_lower, c_upper, coarsity):
        upper = min(memory[c_lower])
        lower = max(memory[c_upper])
        delta = 1./(coarsity*(upper - lower)*(c_upper - c_lower))
        for r in np.linspace(lower + delta, upper, 1./delta)[::-1]:
            fit_in_memory(r)
    

    # greedy search for transitions

    coarsity = 1
    all_expected = list(range(min(memory), max(memory) + 1))

    while True:
        it = zip(sorted(memory.keys()), all_expected)
        for discovered, expected in it:
            if discovered != expected:
                n_keys = len(memory)
                greed_forward(max([k for k in memory if k < expected]), discovered, coarsity)
                if n_keys < len(memory):
                    coarsity = 1
                else:
                    coarsity += 1
                    if coarsity >= max_coarsity:
                        all_expected.pop(all_expected.index(expected))
                        coarsity = 1
                break
        else:
            break
            
    if filter:
        return {k: min(v, key=lambda x: getattr(x, attr)) for k, v in models.items()}
    else:
        return models