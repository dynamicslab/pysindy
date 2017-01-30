import itertools

import pytest

import sparsereg


def test_pareto_front_2d():

    class Model:
        def __init__(self, a, b):
            self.a = a
            self.b = b

    models = [Model(a, b) for a, b in itertools.combinations(range(10), 2)]

    front = sparsereg.ffx.pareto_front_2d(models, "a", "b")

    assert len(front) == 9
    assert all([m.a == m.b - 1 for m in front])
