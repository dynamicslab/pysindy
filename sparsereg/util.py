import numpy as np


def dominates(a, b):
    return all(ai <= bi for ai, bi in zip(a, b)) and not a == b

def pareto_front(models, *attrs):
    """Simple cull.
    """
    get_fit = lambda m: [getattr(m, attr) for attr in attrs]
    front = set()
    for m in models:
        dominated = set()
        for f in front:
            fitf = get_fit(f)
            fitm = get_fit(m)
            if dominates(fitm, fitf):
                dominated.add(f)
            elif dominates(fitf, fitm):
                break
        else:
            front.add(m)
        front -= dominated
    return front

def cardinality(x, null=1e-9):
    return sum(map(lambda x: abs(x) >= null, x))

def rmse(x):
    return np.sqrt(np.mean(x**2))

def nrmse(x, y):
    return rmse(x-y)/(max(x) - min(x))
