"""
Shared pytest fixtures for unit tests.
"""
from pathlib import Path
from tempfile import gettempdir

import jax
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pysindy.differentiation import FiniteDifference
from pysindy.differentiation import SpectralDerivative
from pysindy.feature_library import CustomLibrary
from pysindy.feature_library import FourierLibrary
from pysindy.feature_library import GeneralizedLibrary
from pysindy.feature_library import PDELibrary
from pysindy.feature_library import PolynomialLibrary
from pysindy.utils.odes import logistic_map
from pysindy.utils.odes import logistic_map_control
from pysindy.utils.odes import logistic_map_multicontrol
from pysindy.utils.odes import lorenz
from pysindy.utils.odes import lorenz_control

jax.config.update("jax_platform_name", "cpu")


def pytest_addoption(parser):
    parser.addoption(
        "--external-notebook",
        action="append",
        default=[],
        help=(
            "name of notebook to test.  Only valid if running"
            " test_notebooks.test_external"
        ),
    )
    parser.addoption(
        "--jax-pcc",
        action="store_true",
        help=(
            "Whether to cache @jax.jit compilations to disk."
            "It speeds up the slowest tests appx 2x, writing files to /tmp/jax_cache"
        ),
    )


def pytest_generate_tests(metafunc):
    if "external_notebook" in metafunc.fixturenames:
        metafunc.parametrize(
            "external_notebook",
            [
                Path(f.lstrip('"').rstrip('"'))
                for f in metafunc.config.getoption("external_notebook")
            ],
        )


@pytest.fixture(scope="session")
def set_jax_pcc_env(request):
    if request.config.getoption("--jax-pcc"):
        cache_dir = Path(gettempdir()) / "jax_cache/"
        cache_dir.mkdir(exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(cache_dir))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", -1)
        jax.config.update("jax_persistent_cache_enable_xla_caches", "all")


@pytest.fixture(scope="session")
def data_1d():
    t = np.linspace(0, 1, 10)
    x = 2 * t.reshape(-1, 1)
    return x, t


@pytest.fixture(scope="session")
def data_lorenz():
    t = np.linspace(0, 1, 12)
    x0 = [8, 27, -7]
    x = solve_ivp(lorenz, (t[0], t[-1]), x0, t_eval=t).y.T

    return x, t


@pytest.fixture
def data_multiple_trajectories():
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


@pytest.fixture(scope="session")
def diffuse_multiple_trajectories():
    def diffuse(t, u, dx, nx):
        u = np.reshape(u, (nx, 1))
        du = SpectralDerivative(d=2, axis=0)(u, dx)
        return np.reshape(u * du, nx)

    # Required for accurate solve_ivp results
    integrator_keywords = {}
    integrator_keywords["rtol"] = 1e-8
    integrator_keywords["method"] = "LSODA"
    integrator_keywords["atol"] = 1e-8
    N = 25
    h0 = 1.0
    L = 5
    T = 1
    t = np.linspace(0, T, N)
    x = np.arange(0, N) * L / N
    ep = 0.5 * h0
    y0 = np.reshape(
        h0 + ep * np.cos(4 * np.pi / L * x) + ep * np.cos(2 * np.pi / L * x), N
    )
    dx = x[1] - x[0]
    sol1 = solve_ivp(
        diffuse, (t[0], t[-1]), y0=y0, t_eval=t, args=(dx, N), **integrator_keywords
    )
    u = [np.reshape(sol1.y, (N, N, 1))]
    return [t], x, u


@pytest.fixture(scope="session")
def data_discrete_time():
    n_steps = 100
    mu = 3.6
    x = np.zeros((n_steps))
    x[0] = 0.5
    for i in range(1, n_steps):
        x[i] = logistic_map(x[i - 1], mu)

    return x.reshape((-1, 1))


@pytest.fixture(scope="session")
def data_discrete_time_multiple_trajectories():
    n_steps = 100
    mus = [1, 2.3, 3.6]
    x = [np.zeros((n_steps)) for mu in mus]
    for i, mu in enumerate(mus):
        x[i][0] = 0.5
        for k in range(1, n_steps):
            x[i][k] = logistic_map(x[i][k - 1], mu)

    return [xi.reshape((-1, 1)) for xi in x]


@pytest.fixture(scope="session")
def data_1d_random_pde():
    n = 100
    t = np.linspace(0, 10, n)
    dt = t[1] - t[0]
    x = np.linspace(0, 10, n)
    u = np.random.randn(n, n, 1)
    u_dot = FiniteDifference(axis=1)(u, t=dt)
    return t, x, u, u_dot


@pytest.fixture(scope="session")
def data_2d_random_pde():
    n = 4
    t = np.linspace(0, 10, n)
    dt = t[1] - t[0]
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 10, n)
    X, Y = np.meshgrid(x, y)
    spatial_grid = np.asarray([X, Y]).T
    u = np.random.randn(n, n, n, 2)
    u_dot = FiniteDifference(axis=2)(u, t=dt)
    return spatial_grid, u, u_dot


@pytest.fixture(scope="session")
def data_3d_random_pde():
    n = 4
    t = np.linspace(0, 10, n)
    dt = t[1] - t[0]
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 10, n)
    z = np.linspace(0, 10, n)
    (
        X,
        Y,
        Z,
    ) = np.meshgrid(x, y, z, indexing="ij")
    spatial_grid = np.asarray([X, Y, Z])
    spatial_grid = np.transpose(spatial_grid, axes=[1, 2, 3, 0])
    u = np.random.randn(n, n, n, n, 2)
    u_dot = FiniteDifference(axis=3)(u, t=dt)
    return spatial_grid, u, u_dot


@pytest.fixture(scope="session")
def data_5d_random_pde():
    n = 4
    t = np.linspace(0, n, n)
    dt = t[1] - t[0]
    v = np.linspace(0, 10, n)
    w = np.linspace(0, 10, n)
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 10, n)
    z = np.linspace(0, 10, n)
    V, W, X, Y, Z = np.meshgrid(v, w, x, y, z, indexing="ij")
    spatial_grid = np.asarray([V, W, X, Y, Z])
    spatial_grid = np.transpose(spatial_grid, axes=[1, 2, 3, 4, 5, 0])
    u = np.random.randn(n, n, n, n, n, n, 2)
    u_dot = FiniteDifference(axis=5)(u, t=dt)
    return spatial_grid, u, u_dot


@pytest.fixture(scope="session")
def data_2d_resolved_pde():
    n = 8
    nt = 1000
    t = np.linspace(0, 10, nt)
    dt = t[1] - t[0]
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 10, n)
    X, Y = np.meshgrid(x, y)
    spatial_grid = np.asarray([X, Y]).T
    u = np.random.randn(n, n, nt, 2)
    u_dot = FiniteDifference(axis=-2)(u, t=dt)
    return spatial_grid, u, u_dot


@pytest.fixture(scope="session")
def data_derivative_1d():
    x = 2 * np.linspace(1, 100, 100).reshape((-1, 1))
    x_dot = 2 * np.ones(100).reshape((-1, 1))
    return x, x_dot


@pytest.fixture(scope="session")
def data_derivative_quasiperiodic_1d():
    t = np.arange(1000) * 2 * np.pi / 1000
    x = 2 * np.sin(t)
    x_dot = 2 * np.cos(t)
    return t, x, x_dot


@pytest.fixture(scope="session")
def data_derivative_2d():
    x = np.zeros((100, 2))
    x[:, 0] = 2 * np.linspace(1, 100, 100)
    x[:, 1] = -10 * np.linspace(1, 100, 100)

    x_dot = np.ones((100, 2))
    x_dot[:, 0] *= 2
    x_dot[:, 1] *= -10
    return x, x_dot


@pytest.fixture(scope="session")
def data_derivative_quasiperiodic_2d():
    t = np.arange(1000) * 2 * np.pi / 1000
    x = np.zeros((1000, 2))
    x[:, 0] = 2 * np.sin(t)
    x[:, 1] = 2 * np.cos(2 * t)
    x_dot = np.zeros((1000, 2))
    x_dot[:, 0] = 2 * np.cos(t)
    x_dot[:, 1] = -4 * np.sin(2 * t)
    return t, x, x_dot


@pytest.fixture(scope="session")
def data_2dspatial():
    u = np.zeros((100, 50, 2))
    x = np.linspace(1, 100, 100)
    y = np.linspace(1, 50, 50)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u[:, :, 0] = np.cos(X) * np.sin(Y)
    u[:, :, 1] = -np.sin(X) * np.cos(Y) ** 2
    return x, y, u


@pytest.fixture
def custom_library():
    library_functions = [
        lambda x: x,
        lambda x: x**2,
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
def custom_library_bias():
    library_functions = [
        lambda x: x,
        lambda x: x**2,
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
def generalized_library():
    tensor_array = [[1, 1]]
    return GeneralizedLibrary(
        [PolynomialLibrary(), FourierLibrary()],
        tensor_array=tensor_array,
    )


@pytest.fixture
def sindypi_library(data_lorenz):
    function_library = PolynomialLibrary(degree=2, include_bias=False)
    _, t = data_lorenz

    return PDELibrary(
        function_library=function_library,
        temporal_grid=t,
        implicit_terms=True,
        derivative_order=1,
    )


@pytest.fixture
def ode_library():
    function_library = PolynomialLibrary(degree=2, include_bias=False)

    return PDELibrary(
        function_library=function_library,
    )


@pytest.fixture
def pde_library(data_lorenz):
    _, spatial_grid = data_lorenz
    function_library = PolynomialLibrary(degree=2, include_bias=False)

    return PDELibrary(
        function_library=function_library,
        spatial_grid=spatial_grid,
        derivative_order=4,
    )


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def data_linear_combination():
    t = np.linspace(0, 3, 100).reshape((-1, 1))
    lib = PolynomialLibrary(2, include_bias=False)
    x = lib.fit_transform(np.hstack([t, -2 * t, 3 * t]))
    y = np.stack((x[:, 0] + x[:, 1], x[:, 1] + x[:, 2], x[:, 0] + x[:, 2]), axis=-1)

    return x, y


# Datasets with control inputs


@pytest.fixture(scope="session")
def data_lorenz_c_1d():
    def u_fun(t):
        if len(np.shape(t)) == 0:
            return np.column_stack([np.sin(2 * t), 0])
        else:
            return np.column_stack([np.sin(2 * t), np.zeros(len(t))])

    t = np.linspace(0, 1, 100)
    x0 = [8, 27, -7]
    x = solve_ivp(lorenz_control, (t[0], t[-1]), x0, t_eval=t, args=(u_fun,)).y.T
    u = u_fun(t)

    return x, t, u, u_fun


@pytest.fixture(scope="session")
def data_lorenz_c_2d():
    def u_fun(t):
        return np.column_stack([np.sin(2 * t), t**2])

    t = np.linspace(0, 1, 100)
    x0 = [8, 27, -7]
    x = solve_ivp(lorenz_control, (t[0], t[-1]), x0, t_eval=t, args=(u_fun,)).y.T
    u = u_fun(t)

    return x, t, u, u_fun


@pytest.fixture(scope="session")
def data_discrete_time_c():
    n_steps = 100
    mu = 3.6

    u = 0.01 * np.random.randn(n_steps)
    x = np.zeros((n_steps))
    x[0] = 0.5

    for i in range(1, n_steps):
        x[i] = logistic_map_control(x[i - 1], mu, u[i - 1])

    return x.reshape((-1, 1)), u.reshape((-1, 1))


@pytest.fixture(scope="session")
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

    return x.reshape((-1, 1)), u


@pytest.fixture(scope="session")
def data_discrete_time_multiple_trajectories_c():
    n_steps = 100
    mus = [1, 2.3, 3.6]
    u = [0.001 * np.random.randn(n_steps) for mu in mus]
    x = [np.zeros((n_steps, 1)) for mu in mus]
    for i, mu in enumerate(mus):
        x[i][0] = 0.5
        for k in range(1, n_steps):
            x[i][k] = logistic_map_control(x[i][k - 1], mu, u[i][k - 1])

    return [xi.reshape((-1, 1)) for xi in x], [ui.reshape((-1, 1)) for ui in u]
