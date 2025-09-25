import numpy as np
from numpy.testing import assert_array_equal

from pysindy._weak import _get_spatial_endpoints

def test_get_spatial_endpoints():
    expected = ((-1, 3, 0), (4, 10, 1.5))
    x = np.linspace(expected[0][0], expected[1][0])
    y = np.linspace(expected[0][1], expected[1][1])
    z = np.linspace(expected[0][2], expected[1][2])
    st_grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
    result = _get_spatial_endpoints(st_grid)
    assert_array_equal(result[0], expected[0])
    assert_array_equal(result[1], expected[1])