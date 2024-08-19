import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pysindy.utils import AxesArray
from pysindy.utils import get_prox
from pysindy.utils import get_regularization
from pysindy.utils import reorder_constraints
from pysindy.utils import validate_control_variables


def test_reorder_constraints_1D():
    n_feats = 3
    n_tgts = 2
    target_order = np.array(
        [f"t{i}f{j}" for i in range(n_tgts) for j in range(n_feats)]
    )
    feature_order = np.array(
        [f"t{i}f{j}" for j in range(n_feats) for i in range(n_tgts)]
    )

    result = reorder_constraints(target_order, n_feats, output_order="feature")
    np.testing.assert_array_equal(result.flatten(), feature_order)

    result = reorder_constraints(feature_order, n_feats, output_order="target")
    np.testing.assert_array_equal(result.flatten(), target_order)


def test_reorder_constraints_2D():
    n_feats = 3
    n_tgts = 2
    n_const = 2
    target_order = np.array(
        [
            [f"c{k}t{i}f{j}" for i in range(n_tgts) for j in range(n_feats)]
            for k in range(n_const)
        ]
    )
    feature_order = np.array(
        [
            [f"c{k}t{i}f{j}" for j in range(n_feats) for i in range(n_tgts)]
            for k in range(n_const)
        ]
    )

    result = reorder_constraints(target_order, n_feats, output_order="feature")
    np.testing.assert_array_equal(result, feature_order)

    result = reorder_constraints(feature_order, n_feats, output_order="target")
    np.testing.assert_array_equal(result, target_order)


def test_validate_controls():
    with pytest.raises(ValueError):
        validate_control_variables(1, [])
    with pytest.raises(ValueError):
        validate_control_variables([], 1)
    with pytest.raises(ValueError):
        validate_control_variables([], [1])
    arr = AxesArray(np.ones(4).reshape((2, 2)), axes={"ax_time": 0, "ax_coord": 1})
    with pytest.raises(ValueError):
        validate_control_variables([arr], [arr[:1]])
    u_mod = validate_control_variables([arr], [arr], trim_last_point=True)
    assert u_mod[0].n_time == 1


@pytest.mark.parametrize(
    ["regularization", "expected"], [("l0", 4), ("l1", 16), ("l2", 68)]
)
def test_get_regularization_1d(regularization, expected):
    data = np.array([[0, 3, 5]]).T
    lam = np.array([[2]])

    reg = get_regularization(regularization)
    result = reg(data, lam)
    assert result == expected


@pytest.mark.parametrize(
    ["regularization", "expected"], [("l0", 8), ("l1", 52), ("l2", 408)]
)
def test_get_regularization_2d(regularization, expected):
    data = np.array([[0, 3, 5], [7, 11, 0]]).T
    lam = np.array([[2]])

    reg = get_regularization(regularization)
    result = reg(data, lam)
    assert result == expected


@pytest.mark.parametrize(
    ["regularization", "expected"],
    [("weighted_l0", 2.5), ("weighted_l1", 8.5), ("weighted_l2", 30.5)],
)
def test_get_weighted_regularization_1d(regularization, expected):
    data = np.array([[0, 3, 5]]).T
    lam = np.array([[3, 2, 0.5]]).T

    reg = get_regularization(regularization)
    result = reg(data, lam)
    assert result == expected


@pytest.mark.parametrize(
    ["regularization", "expected"],
    [("weighted_l0", 16.5), ("weighted_l1", 158.5), ("weighted_l2", 1652.5)],
)
def test_get_weighted_regularization_2d(regularization, expected):
    data = np.array([[0, 3, 5], [7, 11, 0]]).T
    lam = np.array([[3, 2, 0.5], [1, 13, 17]]).T

    reg = get_regularization(regularization)
    result = reg(data, lam)
    assert result == expected


@pytest.mark.parametrize(
    ["regularization", "expected"],
    [
        ("l0", np.array([[0, 3, 5]]).T),
        ("l1", np.array([[0, 0, 2]]).T),
        ("l2", np.array([[-2 / 7, 3 / 7, 5 / 7]]).T),
    ],
)
def test_get_prox_1d(regularization, expected):
    data = np.array([[-2, 3, 5]]).T
    lam = np.array([[3]])

    prox = get_prox(regularization)
    result = prox(data, lam)
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ["regularization", "expected"],
    [
        ("l0", np.array([[0, 3, 5], [-7, 11, 0]]).T),
        ("l1", np.array([[0, 0, 2], [-4, 8, 0]]).T),
        ("l2", np.array([[-2 / 7, 3 / 7, 5 / 7], [-7 / 7, 11 / 7, 0 / 7]]).T),
    ],
)
def test_get_prox_2d(regularization, expected):
    data = np.array([[-2, 3, 5], [-7, 11, 0]]).T
    lam = np.array([[3]])

    prox = get_prox(regularization)
    result = prox(data, lam)
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ["regularization", "expected"],
    [
        ("l0", np.array([[0, 3, 5]]).T),
        ("l1", np.array([[0, 1, 4.5]]).T),
        ("l2", np.array([[-2 / 7, 3 / 5, 5 / 2]]).T),
    ],
)
def test_get_weighted_prox_1d(regularization, expected):
    data = np.array([[-2, 3, 5]]).T
    lam = np.array([[3, 2, 0.5]]).T

    prox = get_prox(regularization)
    result = prox(data, lam)
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ["regularization", "expected"],
    [
        ("l0", np.array([[0, 3, 5], [-7, 11, 0]]).T),
        ("l1", np.array([[0, 1, 4.5], [-6, 0, 0]]).T),
        ("l2", np.array([[-2 / 7, 3 / 5, 5 / 2], [-7 / 3, 11 / 27, 0 / 35]]).T),
    ],
)
def test_get_weighted_prox_2d(regularization, expected):
    data = np.array([[-2, 3, 5], [-7, 11, 0]]).T
    lam = np.array([[3, 2, 0.5], [1, 13, 17]]).T

    prox = get_prox(regularization)
    result = prox(data, lam)
    assert_array_equal(result, expected)
