from itertools import chain
from operator import attrgetter

import numpy as np


def dominates(a, b):
    return all(ai <= bi for ai, bi in zip(a, b)) and not a == b


def _pareto_front(models, *attrs, id=False):
    """Helper function. Performs simple cull algorithm"""
    if id:
        get_fit = lambda x: x
    else:
        get_fit = attrgetter(*attrs)

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
    
    return sorted(front, key=get_fit)


def pareto_front(models, *attrs, all=False, id=False):
    """
    Simple cull. Can recursively determine all fronts.
    """
    if not all:
        return _pareto_front(models, *attrs, id=id)
    
    else:
        fronts = []
        models = set(models)
        while models:
            fronts.append(_pareto_front(models, *attrs, id=id))
            models -= set(fronts[-1])
        return fronts


def crowding_distance(models, *attrs, id=False):
    """
    Assumes models in lexicographical sorted.
    """
    if id:
        get_fit = lambda x: x
    else:
        get_fit = lambda m: [getattr(m, attr) for attr in attrs]

    f = np.array([get_fit(m) for m in sorted(models)])
    scale = np.max(f, axis=0) - np.min(f, axis=0)

    with np.errstate(invalid="ignore"):
        dist = np.sum(abs(np.roll(f, 1, axis=0) - np.roll(f, -1, axis=0) ) / scale, axis=1)
    dist[0] = np.infty
    dist[-1] = np.infty
    return dist


def sort_non_dominated(models, *attrs, id=False):
    """
    NSGA2 based sorting
    """
    fronts = pareto_front(models[:], *attrs, all=True, id=id)

    distances = [crowding_distance(front, *attrs, id=id) for front in fronts] # if len(front) > 2 else list(np.zeros_like(front))

    # fd = (list of models, list of distances)
    # convert that into (model, distance) tuples
    # sort in descending order by distance
    # resolve the nested chain
    ranked = chain.from_iterable(sorted(zip(*fd), key=lambda x: x[1]) for fd in zip(fronts, distances))

    return [m for (m, d) in ranked] # discard the distance


def normalize(x, order=2):
    m = 1.0 / np.linalg.norm(x, ord=order, axis=0)
    return m * x, m


def cardinality(x, null=1e-9):
    return sum(map(lambda x: abs(x) >= null, x))


def rmse(x):
    return np.sqrt(np.mean(x**2))


def nrmse(x, y):
    return rmse(x-y)/(max(x) - min(x))
