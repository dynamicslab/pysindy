"""Numerical contracts for Fourier features with non-floating measurements."""
import cmath
import math

import numpy as np
import pytest
from sklearn.pipeline import make_pipeline

from pysindy import FourierLibrary
from pysindy import STLSQ


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.uint8, bool])
@pytest.mark.parametrize(
    "include_sin, include_cos", [(True, True), (True, False), (False, True)]
)
@pytest.mark.parametrize("n_frequencies", [1, 3])
def test_fourier_integer_measurements(dtype, include_sin, include_cos, n_frequencies):
    # Higher harmonics of 100 must not overflow narrow integer measurements.
    x = np.array([[0], [1], [100]], dtype=dtype)
    original = x.copy()
    functions = ([math.sin] if include_sin else []) + (
        [math.cos] if include_cos else []
    )
    expected = np.array(
        [
            [f(k * int(value)) for k in range(1, n_frequencies + 1) for f in functions]
            for value in x[:, 0]
        ]
    )
    library = FourierLibrary(n_frequencies, include_sin, include_cos)
    actual = library.fit_transform(x)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(x, original)
    assert np.issubdtype(actual.dtype, np.floating)
    assert actual.shape[1] == len(library.get_feature_names())


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
def test_fourier_preserves_inexact_precision(dtype):
    x = np.array([[-0.5], [0.25], [1]], dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):
        x += 0.25j
    expected = np.array(
        [[cmath.sin(complex(value)), cmath.cos(complex(value))] for value in x[:, 0]]
    )
    actual = FourierLibrary().fit_transform(x)

    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_fourier_integer_trajectories_with_spatial_axes():
    x = np.arange(-6, 6, dtype=np.int16).reshape(2, 3, 2)
    trajectories = [x, x[:1]]
    library = FourierLibrary(n_frequencies=2).fit(trajectories)
    actual = library.transform(trajectories)

    for trajectory, features in zip(trajectories, actual):
        expected = np.array(
            [
                [
                    math.sin(a),
                    math.cos(a),
                    math.sin(b),
                    math.cos(b),
                    math.sin(2 * a),
                    math.cos(2 * a),
                    math.sin(2 * b),
                    math.cos(2 * b),
                ]
                for a, b in trajectory.reshape(-1, 2).tolist()
            ]
        ).reshape(*trajectory.shape[:-1], 8)
        np.testing.assert_allclose(features, expected, atol=1e-14)
        assert features.ax_coord == 2
        assert features.shape == expected.shape


def test_fourier_pipeline_identifies_periodic_drift_from_integer_angles():
    # d(theta)/dt = 2 sin(theta) - 0.5 cos(theta), measured at integer angles.
    x = np.arange(-4, 5, dtype=np.int64).reshape(-1, 1)
    x_dot = np.array([[2 * math.sin(int(t)) - 0.5 * math.cos(int(t))] for t in x[:, 0]])
    model = make_pipeline(FourierLibrary(), STLSQ(threshold=0, alpha=0))
    model.fit(x, x_dot)

    np.testing.assert_allclose(model[-1].coef_, [[2, -0.5]], atol=1e-12)
    held_out = np.array([[-5], [5]], dtype=np.int64)
    expected = [[2 * math.sin(int(t)) - 0.5 * math.cos(int(t))] for t in held_out[:, 0]]
    np.testing.assert_allclose(model.predict(held_out), expected, atol=1e-12)
