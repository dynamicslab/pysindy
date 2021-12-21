"""
Shared pytest fixtures for unit tests.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pysindy.differentiation import FiniteDifference
from pysindy.feature_library import CustomLibrary
from pysindy.feature_library import FourierLibrary
from pysindy.feature_library import GeneralizedLibrary
from pysindy.feature_library import PDELibrary
from pysindy.feature_library import PolynomialLibrary
from pysindy.feature_library import SINDyPILibrary
from pysindy.utils.odes import logistic_map
from pysindy.utils.odes import logistic_map_control
from pysindy.utils.odes import logistic_map_multicontrol
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
    x = solve_ivp(lorenz, (t[0], t[-1]), x0, t_eval=t).y.T

    return x, t


@pytest.fixture
def data_multiple_trajctories():

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
        x_list.append(solve_ivp(lorenz, (t[0], t[-1]), x0, t_eval=t).y.T)

    return x_list, t_list


@pytest.fixture
def data_discrete_time():

    n_steps = 100
    mu = 3.6
    x = np.zeros((n_steps))
    x[0] = 0.5
    for i in range(1, n_steps):
        x[i] = logistic_map(x[i - 1], mu)

    return x


@pytest.fixture
def data_discrete_time_multiple_trajectories():

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
    u_dot = FiniteDifference(axis=1)._differentiate(u, t=dt)
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
    u_dot = FiniteDifference(axis=2)._differentiate(u, t=dt)
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
    u_dot = FiniteDifference(axis=3)._differentiate(u, t=dt)
    u_flattened = np.reshape(u, (n ** 4, 2))
    u_dot_flattened = np.reshape(u_dot, (n ** 4, 2))

    return spatial_grid, u_flattened, u_dot_flattened


@pytest.fixture
def data_2d_resolved_pde():
    nx = 8
    ny = 8
    nt = 1000
    t = np.linspace(0, 10, nt)
    dt = t[1] - t[0]
    x = np.linspace(0, 10, nx)
    y = np.linspace(0, 10, ny)
    X, Y = np.meshgrid(x, y)
    spatial_grid = np.asarray([X, Y]).T
    u = np.random.randn(nx, ny, nt, 2)
    u_dot = FiniteDifference(axis=-2)._differentiate(u, t=dt)
    u_flattened = np.reshape(u, (nx * ny * nt, 2))
    u_dot_flattened = np.reshape(u_dot, (nx * ny * nt, 2))
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
def data_custom_library_bias():
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
        library_functions=library_functions,
        function_names=function_names,
        include_bias=True,
    )


@pytest.fixture
def data_quadratic_library():
    library_functions = [
        lambda x: x,
        lambda x, y: x * y,
        lambda x: x ** 2,
    ]
    function_names = [
        lambda x: str(x),
        lambda x, y: "{} * {}".format(x, y),
        lambda x: "{}^2".format(x),
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
        if len(np.shape(t)) == 0:
            return np.column_stack([np.sin(2 * t), 0])
        else:
            return np.column_stack([np.sin(2 * t), np.zeros(len(t))])

    t = np.linspace(0, 5, 500)
    x0 = [8, 27, -7]
    x = solve_ivp(lorenz_control, (t[0], t[-1]), x0, t_eval=t, args=(u_fun,)).y.T
    u = u_fun(t)

    return x, t, u, u_fun


@pytest.fixture
def data_lorenz_c_2d():
    def u_fun(t):
        return np.column_stack([np.sin(2 * t), t ** 2])

    t = np.linspace(0, 5, 500)
    x0 = [8, 27, -7]
    x = solve_ivp(lorenz_control, (t[0], t[-1]), x0, t_eval=t, args=(u_fun,)).y.T
    u = u_fun(t)

    return x, t, u, u_fun


@pytest.fixture
def data_discrete_time_c():

    n_steps = 100
    mu = 3.6

    u = 0.01 * np.random.randn(n_steps)
    x = np.zeros((n_steps))
    x[0] = 0.5

    for i in range(1, n_steps):
        x[i] = logistic_map_control(x[i - 1], mu, u[i - 1])

    return x, u


@pytest.fixture
def data_discrete_time_c_multivariable():

    n_steps = 100
    mu = 3.6

    u1 = 0.1 * np.random.randn(n_steps)
    u2 = 0.1 * np.random.randn(n_steps)
    u = np.column_stack((u1, u2))
    x = np.zeros((n_steps))
    x[0] = 0.5
    for i in range(1, n_steps):
        x[i] = logistic_map_multicontrol(x[i - 1], mu, u[i - 1])

    return x, u


@pytest.fixture
def data_discrete_time_multiple_trajectories_c():

    n_steps = 100
    mus = [1, 2.3, 3.6]
    u = [0.001 * np.random.randn(n_steps) for mu in mus]
    x = [np.zeros((n_steps)) for mu in mus]
    for i, mu in enumerate(mus):
        x[i][0] = 0.5
        for k in range(1, n_steps):
            x[i][k] = logistic_map_control(x[i][k - 1], mu, u[i][k - 1])

    return x, u
