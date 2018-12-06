from collections import namedtuple

import numpy as np

from sparsereg.model.ffx import *
from sparsereg.model.ffx import _path_is_saturated  # , _path_is_overfit


def test_build_strategies():
    strats = list(build_strategies([1, 2], {})([]))
    assert len(strats) == 4

    strats = list(build_strategies([1, 2], {}, rational=False)([]))
    assert len(strats) == 2

    strats = list(build_strategies([1], {}, rational=False)([]))
    assert len(strats) == 1

    strats = list(build_strategies([1], {}, rational=False)([]))
    assert len(strats) == 1

    strats = list(build_strategies([1], {"sin": np.sin}, rational=False)([]))
    assert len(strats) == 2

    strats = list(build_strategies([1, 2], {"sin": np.sin}, rational=False)([]))
    assert len(strats) == 4


def test__path_is_saturated():
    model = namedtuple("Model", ["train_score_"])
    models = [model(1) for _ in range(10)]

    assert not _path_is_saturated(models, n_tail=len(models) + 1)
    assert _path_is_saturated(models, n_tail=1)
    assert _path_is_saturated(models, n_tail=5)


# def test__path_is_overfit():
#     model = namedtuple("Model", ["test_score_"])
#     models = [model(-i) for i in range(10)]
#
#     assert not _path_is_overfit(models)
#     assert not _path_is_overfit([models[0]])
