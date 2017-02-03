import itertools
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


def test_pareto_front():
    amax = 1.0
    amin = 0.7
    bmax = 1.3
    bmin = 1.0

    front_fitness = [(1, 1), (0.9, 1.1), (0.8, 1.2), (0.7, 1.3)]

    class Model:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    models = [Model(a, b) for a, b in front_fitness]
    models.extend([Model(a + 0.01*abs(random.random()) + 0.01, b + 0.01*abs(random.random()) + 0.01 ) for a, b in random.choices(front_fitness, k=50)])
    models = filter(lambda m: m.b <= bmax, models)

    front = pareto_front(models, "a", "b")

    for m in front:
        assert amin <= m.a <= amax
        assert bmin <= m.b <= bmax


@pytest.mark.parametrize("exp_null", range(6))
def test_cardinality(exp_null):
    coef =  [10**(-i) for i in range(7)]
    assert cardinality(coef, null=10.0**(-exp_null)) == exp_null + 1
