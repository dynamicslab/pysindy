import itertools
import collections
import random

import pytest

from sparsereg.util import *


try:
    random.choices   # only in 3.6+
except AttributeError:

    def choices(l, k):
        for _ in range(k):
            yield random.choice(l)

    random.choices = choices

dominates_cases = [
    ((1.0, 1.0), (1.0, 1.0), False, False),
    ((1.0, 0.8), (1.0, 1.0), True, False),
    ((1.0, 1.0), (0.8, 1.0), False, True),
    ((1.0, 0.8), (0.8, 1.0), False, False),
    ((1.0, 0.8, 1.2), (0.8, 1.0, 1.1), False, False),
]


@pytest.mark.parametrize("case", dominates_cases)
def test_dominates(case):
    a, b, result_ab, result_ba = case
    assert dominates(a, b) == result_ab
    assert dominates(b, a) == result_ba


class Model:
    def __init__(self, a, b):
        self.a = a
        self.b = b


def test_pareto_front_attrs():
    amax = 1.0
    amin = 0.7
    bmax = 1.3
    bmin = 1.0

    front_fitness = [(1, 1), (0.9, 1.1), (0.8, 1.2), (0.7, 1.3)]

    models = [Model(a, b) for a, b in front_fitness]
    models.extend([Model(a + 0.01 * abs(random.random()) + 0.01, b + 0.01 *
                         abs(random.random()) + 0.01) for a, b in random.choices(front_fitness, k=50)])
    models = list(filter(lambda m: m.b <= bmax, models))

    front = pareto_front(models, "a", "b")

    for m in front:
        assert amin <= m.a <= amax
        assert bmin <= m.b <= bmax


@pytest.fixture(scope="function")
def models():

    m = [
        (0, 0),
        (0, 1),
        (1, 0),
        (0.5, 0.5),
        (1, 1),
    ]
    return m


def test_pareto_front_tpl(models):

    fronts = pareto_front(models, all=True)

    assert len(fronts) == 3
    assert models[0] in fronts[0]
    assert models[-1] in fronts[-1]

@pytest.mark.xfail()
def test_pareto_front_duplicates():
    base = collections.namedtuple("base", "a b c")

    models = ("a", 0, 1), ("b", 0, 1), ("c", 1, 0), ("d", 0.5, 0.5)
    models = [base(*m) for m in models]
    front = pareto_front(models, "b", "c")
    print(front)
    assert False


cd_cases = (
    (models()[1:-1], ()),
    ([Model(a, b) for (a, b) in sorted(models()[1:-1])], ("a", "b")),
)

@pytest.mark.parametrize("cd_case", cd_cases)
def test_crowding_distance(cd_case):
    m, attr = cd_case
    dist = crowding_distance(m, *attr)

    assert dist[0] == dist[-1] == np.infty
    assert dist[1] == 2


def test_non_dominated_sorting(models):
    ranked = sort_non_dominated(models)

    index = [0, 3, 1, 2, 4]

    assert ranked[0] == models[0]
    assert ranked[1] == models[-2]
    assert ranked[-1] == models[-1]

    assert set(ranked[2:4]) == set(models[1:3])

    assert index == sort_non_dominated(models, index=True)


@pytest.mark.parametrize("exp_null", range(6))
def test_cardinality(exp_null):
    coef = [10**(-i) for i in range(7)]
    assert cardinality(coef, null=10.0**(-exp_null)) == exp_null + 1


def test_normalize():
    n = 10
    order = 2
    x = np.ones(shape=(n, 1))

    x, m = normalize(x, order=order)
    assert m[0] == 1 / n**(1.0 / order)
