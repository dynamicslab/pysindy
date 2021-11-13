"""
Shared pytest fixtures for unit tests.
"""
import numpy as np
import pytest
from scipy.integrate import odeint

from pysindy.differentiation import FiniteDifference
from pysindy.feature_library import CustomLibrary
from pysindy.feature_library import FourierLibrary
from pysindy.feature_library import GeneralizedLibrary
from pysindy.feature_library import PDELibrary
from pysindy.feature_library import PolynomialLibrary
from pysindy.feature_library import SINDyPILibrary
from pysindy.utils.odes import lorenz
from pysindy.utils.odes import lorenz_control


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

    t = np.linspace(0, 5, 500)
    x0 = [8, 27, -7]
    x = odeint(lorenz, x0, t)

    return x, t


@pytest.fixture
def data_lorenz_control():

    t = np.linspace(0, 5, 500)
    x0 = [8, 27, -7]
    x = odeint(lorenz_control, x0, t)

    return x, t


@pytest.fixture
def data_multiple_trajctories():
    def lorenz(z, t):
        return [
            10 * (z[1] - z[0]),
            z[0] * (28 - z[2]) - z[1],
            z[0] * z[1] - 8 / 3 * z[2],
        ]

    n_points = [100, 200, 500]
    initial_conditions = [
        [8, 27, -7],
        [-10.9595724, 21.7346758, 24.5722540],
        [-3.95406365, -9.21825124, 12.07459147],
    ]

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
def data_1d_random_pde():
    nx = 8
    t = np.linspace(0, 10, nx)
    dt = t[1] - t[0]
    x = np.linspace(0, 10, nx)
    u = np.random.randn(nx, nx, 1)
    u_dot = np.zeros_like(u)
    for i in range(len(x)):
        u_dot[i, :, :] = FiniteDifference()._differentiate(u[i, :, :], t=dt)
    u_flattened = np.reshape(u, (nx * nx, 1))
    u_dot_flattened = np.reshape(u_dot, (nx * nx, 1))

    return x, u_flattened, u_dot_flattened


@pytest.fixture
def data_2d_random_pde():
    nx = 8
    ny = 8
    t = np.linspace(0, 10, nx)
    dt = t[1] - t[0]
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 10, ny)
    X, Y = np.meshgrid(x, y)
    spatial_grid = np.asarray([X, Y]).T
    u = np.random.randn(nx, ny, nx, 2)
    u_dot = np.zeros(u.shape)
    for i in range(nx):
        for j in range(ny):
            u_dot[i, j, :, :] = FiniteDifference()._differentiate(u[i, j, :, :], t=dt)
    u_flattened = np.reshape(u, (nx * ny * nx, 2))
    u_dot_flattened = np.reshape(u_dot, (nx * ny * nx, 2))

    return spatial_grid, u_flattened, u_dot_flattened


@pytest.fixture
def data_3d_random_pde():
    nx = 8
    t = np.linspace(0, 10, nx)
    dt = t[1] - t[0]
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 10, nx)
    z = np.linspace(0, 10, nx)
    (
        X,
        Y,
        Z,
    ) = np.meshgrid(x, y, z, indexing="ij")
    spatial_grid = np.asarray([X, Y, Z])
    spatial_grid = np.transpose(spatial_grid, axes=[1, 2, 3, 0])
    n = len(x)
    u = np.random.randn(n, n, n, n, 2)
    u_dot = np.zeros(u.shape)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                u_dot[i, j, k, :, :] = FiniteDifference()._differentiate(
                    u[i, j, k, :, :], t=dt
                )
    u_flattened = np.reshape(u, (n ** 4, 2))
    u_dot_flattened = np.reshape(u_dot, (n ** 4, 2))

    return spatial_grid, u_flattened, u_dot_flattened


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
def data_2dspatial():
    u = np.zeros((100, 50, 2))
    x = np.linspace(1, 100, 100)
    y = np.linspace(1, 50, 50)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u[:, :, 0] = np.cos(X) * np.sin(Y)
    u[:, :, 1] = -np.sin(X) * np.cos(Y) ** 2
    return u


@pytest.fixture
def data_custom_library():
    library_functions = [
        lambda x: x,
        lambda x: x ** 2,
        lambda x: 0 * x,
        lambda x, y: x * y,
    ]
    function_names = [
        lambda s: str(s),
        lambda s: str(s) + "^2",
        lambda s: "0",
        lambda s, t: str(s) + " " + str(t),
    ]

    return CustomLibrary(
        library_functions=library_functions, function_names=function_names
    )


@pytest.fixture
def data_generalized_library():
    tensor_array = [[1, 1]]
    inputs_temp = np.tile([0, 1, 2], 2)
    inputs_per_library = np.reshape(inputs_temp, (2, 3))
    return GeneralizedLibrary(
        [PolynomialLibrary(), FourierLibrary()],
        tensor_array=tensor_array,
        inputs_per_library=inputs_per_library,
    )


@pytest.fixture
def data_sindypi_library():
    library_functions = [
        lambda x: x,
        lambda x: x ** 2,
        lambda x, y: x * y,
    ]
    x_dot_library_functions = [lambda x: x]
    function_names = [
        lambda s: str(s),
        lambda s: str(s) + "^2",
        lambda s, t: str(s) + " " + str(t),
        lambda s: str(s),
    ]
    t = np.linspace(0, 5, 500)

    return SINDyPILibrary(
        library_functions=library_functions,
        x_dot_library_functions=x_dot_library_functions,
        function_names=function_names,
        t=t,
    )


@pytest.fixture
def data_ode_library():
    library_functions = [
        lambda x: x,
        lambda x: x ** 2,
        lambda x, y: x * y,
    ]
    function_names = [
        lambda s: str(s),
        lambda s: str(s) + "^2",
        lambda s, t: str(s) + " " + str(t),
    ]

    return PDELibrary(
        library_functions=library_functions,
        function_names=function_names,
    )


@pytest.fixture
def data_pde_library():
    spatial_grid = np.linspace(0, 10)
    library_functions = [
        lambda x: x,
        lambda x: x ** 2,
        lambda x, y: x * y,
    ]
    function_names = [
        lambda s: str(s),
        lambda s: str(s) + "^2",
        lambda s, t: str(s) + " " + str(t),
    ]

    return PDELibrary(
        library_functions=library_functions,
        function_names=function_names,
        spatial_grid=spatial_grid,
        derivative_order=4,
    )


@pytest.fixture
def data_linear_oscillator_corrupted():
    t = np.linspace(0, 1, 100)
    x = 3 * np.exp(-2 * t)
    y = 0.5 * np.exp(t)
    np.random.seed(1)
    corrupt_idxs = np.random.choice(np.arange(1, t.size - 1), t.size // 20)
    x[corrupt_idxs] = 0
    X = np.stack((x, y), axis=-1)
    X_dot = FiniteDifference(order=2)(X, t)

    # build an array of the indices of samples that should be trimmed
    trimmed_idxs = np.concatenate((corrupt_idxs - 1, corrupt_idxs, corrupt_idxs + 1))
    trimming_array = np.ones(X.shape[0])
    trimming_array[trimmed_idxs] = 0.0

    return X, X_dot, trimming_array


@pytest.fixture
def data_linear_combination():
    t = np.linspace(0, 5, 100)
    x = np.stack((np.exp(t), np.sin(t), np.cos(t)), axis=-1)
    y = np.stack((x[:, 0] + x[:, 1], x[:, 1] + x[:, 2]), axis=-1)

    return x, y


# Datasets with control inputs


@pytest.fixture
def data_lorenz_c_1d():
    def u_fun(t):
        return np.sin(2 * t)

    def lorenz(z, t):
        return [
            10 * (z[1] - z[0]) + u_fun(t) ** 2,
            z[0] * (28 - z[2]) - z[1],
            z[0] * z[1] - 8 / 3 * z[2],
        ]

    t = np.linspace(0, 5, 500)
    x0 = [8, 27, -7]
    x = odeint(lorenz, x0, t)
    u = u_fun(t)

    return x, t, u, u_fun


@pytest.fixture
def data_lorenz_c_2d():
    def u_fun(t):
        return np.column_stack([np.sin(2 * t), t ** 2])

    def lorenz(z, t):
        u = u_fun(t)
        return [
            10 * (z[1] - z[0]) + u[0, 0] ** 2,
            z[0] * (28 - z[2]) - z[1],
            z[0] * z[1] - 8 / 3 * z[2] - u[0, 1],
        ]

    t = np.linspace(0, 5, 500)
    x0 = [8, 27, -7]
    x = odeint(lorenz, x0, t)
    u = u_fun(t)

    return x, t, u, u_fun


@pytest.fixture
def data_discrete_time_c():
    def logistic_map(x, mu, ui):
        return mu * x * (1 - x) + ui

    n_steps = 100
    mu = 3.6

    u = 0.01 * np.random.randn(n_steps)
    x = np.zeros((n_steps))
    x[0] = 0.5
    for i in range(1, n_steps):
        x[i] = logistic_map(x[i - 1], mu, u[i - 1])

    return x, u


@pytest.fixture
def data_discrete_time_c_multivariable():
    def logistic_map(x, mu, u):
        return mu * x * (1 - x) + u[0] * u[1]

    n_steps = 100
    mu = 3.6

    u1 = 0.1 * np.random.randn(n_steps)
    u2 = 0.1 * np.random.randn(n_steps)
    u = np.column_stack((u1, u2))
    x = np.zeros((n_steps))
    x[0] = 0.5
    for i in range(1, n_steps):
        x[i] = logistic_map(x[i - 1], mu, u[i - 1])

    return x, u


@pytest.fixture
def data_discrete_time_multiple_trajectories_c():
    def logistic_map(x, mu, ui):
        return mu * x * (1 - x) + ui

    n_steps = 100
    mus = [1, 2.3, 3.6]
    u = [0.001 * np.random.randn(n_steps) for mu in mus]
    x = [np.zeros((n_steps)) for mu in mus]
    for i, mu in enumerate(mus):
        x[i][0] = 0.5
        for k in range(1, n_steps):
            x[i][k] = logistic_map(x[i][k - 1], mu, u[i][k - 1])

    return x, u
