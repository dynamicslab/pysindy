import numpy as np
import pytest

from pysindy.utils import AxesArray
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
