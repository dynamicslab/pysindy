"""
Shared pytest fixtures for unit tests.
"""

import pytest
import numpy as np
from scipy.integrate import odeint

from pysindy.feature_library import CustomLibrary


@pytest.fixture
def data_1d():
    t = np.linspace(0, 5, 100)
    x = 2 * t.reshape(-1, 1)
    return x, t


@pytest.fixture
def data_1d_bad_shape():
    t = np.linspace(0, 5, 100)
    x = 2 * t
    return x, t


@pytest.fixture
def data_lorenz():
    def lorenz(z, t):
        return [
            10 * (z[1] - z[0]),
            z[0] * (28 - z[2]) - z[1],
            z[0] * z[1] - 8 / 3 * z[2],
        ]

    t = np.linspace(0, 5, 100)
    x0 = [8, 27, -7]
    x = odeint(lorenz, x0, t)

    return x, t


@pytest.fixture
def data_multiple_trajctories():
    def lorenz(z, t):
        return [
            10 * (z[1] - z[0]),
            z[0] * (28 - z[2]) - z[1],
            z[0] * z[1] - 8 / 3 * z[2],
        ]

    n_points = [10, 50, 100]
    initial_conditions = [[8, 27, -7], [9, 28, -8], [-1, 10, 1]]

    x_list = []
    t_list = []
    for n, x0 in zip(n_points, initial_conditions):
        t = np.linspace(0, 5, n)
        t_list.append(t)
        x_list.append(odeint(lorenz, x0, t))

    return x_list, t_list


@pytest.fixture
def data_discrete_time():
    def logistic_map(x, mu):
        return mu * x * (1 - x)

    n_steps = 100
    mu = 3.6
    x = np.zeros((n_steps))
    x[0] = 0.5
    for i in range(1, n_steps):
        x[i] = logistic_map(x[i - 1], mu)

    return x


@pytest.fixture
def data_discrete_time_multiple_trajectories():
    def logistic_map(x, mu):
        return mu * x * (1 - x)

    n_steps = 100
    mus = [1, 2.3, 3.6]
    x = [np.zeros((n_steps)) for mu in mus]
    for i, mu in enumerate(mus):
        x[i][0] = 0.5
        for k in range(1, n_steps):
            x[i][k] = logistic_map(x[i][k - 1], mu)

    return x


@pytest.fixture
def data_derivative_1d():
    x = 2 * np.linspace(1, 100, 100)
    x_dot = 2 * np.ones(100).reshape(-1, 1)
    return x, x_dot


@pytest.fixture
def data_derivative_2d():
    x = np.zeros((100, 2))
    x[:, 0] = 2 * np.linspace(1, 100, 100)
    x[:, 1] = -10 * np.linspace(1, 100, 100)

    x_dot = np.ones((100, 2))
    x_dot[:, 0] *= 2
    x_dot[:, 1] *= -10
    return x, x_dot


@pytest.fixture
def data_custom_library():
    library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
    function_names = [lambda s: str(s), lambda s: str(s) + "^2", lambda s: "0"]

    return CustomLibrary(
        library_functions=library_functions, function_names=function_names
    )
