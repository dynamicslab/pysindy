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
    ["regularization", "lam", "expected"],
    [
        ("l0", 2, 4),
        ("l1", 2, 14),
        ("l2", 2, 58),
        ("weighted_l0", np.array([[3, 2]]).T, 5),
        ("weighted_l1", np.array([[3, 2]]).T, 16),
        ("weighted_l2", np.array([[3, 2]]).T, 62),
    ],
)
def test_get_regularization(regularization, lam, expected):
    data = np.array([[-2, 5]]).T

    reg = get_regularization(regularization)
    result = reg(data, lam)
    assert result == expected


@pytest.mark.parametrize("regularization", ["l0", "l1", "l2"])
@pytest.mark.parametrize(
    "lam",
    [
        np.array([[1, 2]]),
        np.array([[1]]),
    ],
)
def test_get_prox_and_regularization_bad_shape(regularization, lam):
    data = np.array([[-2, 5]]).T
    reg = get_regularization(regularization)
    with pytest.raises(ValueError):
        reg(data, lam)
    prox = get_prox(regularization)
    with pytest.raises(ValueError):
        prox(data, lam)


@pytest.mark.parametrize(
    "regularization", ["weighted_l0", "weighted_l1", "weighted_l2"]
)
@pytest.mark.parametrize(
    "lam",
    [
        np.array([[1, 2]]),
        1,
    ],
)
def test_get_weighted_prox_and_regularization_bad_shape(regularization, lam):
    data = np.array([[-2, 5]]).T
    reg = get_regularization(regularization)
    with pytest.raises(ValueError):
        reg(data, lam)
    prox = get_prox(regularization)
    with pytest.raises(ValueError):
        prox(data, lam)


@pytest.mark.parametrize(
    ["regularization", "lam", "expected"],
    [
        ("l0", 1, np.array([[2]])),
        ("l1", 0.5, np.array([[1.5]])),
        ("l2", 0.5, np.array([[1]])),
        ("weighted_l0", np.array([[1]]), np.array([[2]])),
        ("weighted_l1", np.array([[0.5]]), np.array([[1.5]])),
        ("weighted_l2", np.array([[0.5]]), np.array([[1]])),
    ],
)
def test_get_prox(regularization, lam, expected):
    data = np.array([[2]])

    prox = get_prox(regularization)
    result = prox(data, lam)
    assert_array_equal(result, expected)
