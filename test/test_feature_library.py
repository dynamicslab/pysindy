"""
Unit tests for feature libraries.
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pysindy import SINDy
from pysindy.differentiation import FiniteDifference
from pysindy.feature_library import ConcatLibrary
from pysindy.feature_library import CustomLibrary
from pysindy.feature_library import FourierLibrary
from pysindy.feature_library import GeneralizedLibrary
from pysindy.feature_library import IdentityLibrary
from pysindy.feature_library import ParameterizedLibrary
from pysindy.feature_library import PDELibrary
from pysindy.feature_library import PolynomialLibrary
from pysindy.feature_library import SINDyPILibrary
from pysindy.feature_library import TensoredLibrary
from pysindy.feature_library import WeakPDELibrary
from pysindy.feature_library.base import BaseFeatureLibrary
from pysindy.optimizers import SINDyPI
from pysindy.optimizers import STLSQ


def test_form_custom_library():
    library_functions = [lambda x: x, lambda x: x**2, lambda x: 0 * x]
    function_names = [
        lambda s: str(s),
        lambda s: "{}^2".format(s),
        lambda s: "0",
    ]

    # Test with user-supplied function names
    CustomLibrary(library_functions=library_functions, function_names=function_names)

    # Test without user-supplied function names
    CustomLibrary(library_functions=library_functions, function_names=None)


def test_form_pde_library():
    library_functions = [lambda x: x, lambda x: x**2, lambda x: 0 * x]
    function_names = [
        lambda s: str(s),
        lambda s: "{}^2".format(s),
        lambda s: "0",
    ]

    # Test with user-supplied function names
    PDELibrary(library_functions=library_functions, function_names=function_names)

    # Test without user-supplied function names
    PDELibrary(library_functions=library_functions, function_names=None)


def test_form_sindy_pi_library():
    library_functions = [lambda x: x, lambda x: x**2, lambda x: 0 * x]
    function_names = [
        lambda s: str(s),
        lambda s: "{}^2".format(s),
        lambda s: "0",
    ]
    # Test with user-supplied function names
    SINDyPILibrary(library_functions=library_functions, function_names=function_names)

    # Test without user-supplied function names
    SINDyPILibrary(library_functions=library_functions, function_names=None)


def test_bad_parameters():
    with pytest.raises(ValueError):
        PolynomialLibrary(degree=-1).fit(np.array([]))
    with pytest.raises(ValueError):
        PolynomialLibrary(degree=1.5).fit(np.array([]))
    with pytest.raises(ValueError):
        PolynomialLibrary(include_interaction=False, interaction_only=True).fit(
            np.array([])
        )
    with pytest.raises(ValueError):
        FourierLibrary(n_frequencies=-1)
    with pytest.raises(ValueError):
        FourierLibrary(n_frequencies=-1)
    with pytest.raises(ValueError):
        FourierLibrary(n_frequencies=2.2)
    with pytest.raises(ValueError):
        FourierLibrary(include_sin=False, include_cos=False)
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x**2, lambda x: 0 * x]
        function_names = [lambda s: str(s), lambda s: "{}^2".format(s)]
        CustomLibrary(
            library_functions=library_functions, function_names=function_names
        )


@pytest.mark.parametrize(
    "params",
    [
        dict(function_names=[lambda s: str(s), lambda s: "{}^2".format(s)]),
        dict(derivative_order=1),
        dict(derivative_order=3),
        dict(spatial_grid=range(10)),
        dict(spatial_grid=range(10), derivative_order=-1),
        dict(spatial_grid=np.zeros((10, 10))),
        dict(spatial_grid=np.zeros((10, 10, 10, 10, 10))),
    ],
)
def test_pde_library_bad_parameters(params):
    params["library_functions"] = [lambda x: x, lambda x: x**2, lambda x: 0 * x]
    with pytest.raises(ValueError):
        PDELibrary(**params)


@pytest.mark.parametrize(
    "params",
    [
        dict(spatiotemporal_grid=range(10), p=-1),
        dict(spatiotemporal_grid=range(10), H_xt=-1),
        dict(spatiotemporal_grid=range(10), H_xt=11),
        dict(spatiotemporal_grid=range(10), K=-1),
        dict(),
        dict(
            spatiotemporal_grid=np.asarray(np.meshgrid(range(10), range(10))).T,
            H_xt=-1,
        ),
        dict(
            spatiotemporal_grid=np.transpose(
                np.asarray(np.meshgrid(range(10), range(10), range(10), indexing="ij")),
                axes=[1, 2, 3, 0],
            ),
            H_xt=-1,
        ),
        dict(
            spatiotemporal_grid=np.transpose(
                np.asarray(np.meshgrid(range(10), range(10), range(10), indexing="ij")),
                axes=[1, 2, 3, 0],
            ),
            H_xt=11,
        ),
    ],
)
def test_weak_pde_library_bad_parameters(params):
    params["library_functions"] = [lambda x: x, lambda x: x**2, lambda x: 0 * x]
    with pytest.raises(ValueError):
        WeakPDELibrary(**params)


@pytest.mark.parametrize(
    "params",
    [
        dict(libraries=[]),
        dict(
            libraries=[PolynomialLibrary(), PolynomialLibrary()], tensor_array=[[0, 0]]
        ),
        dict(
            libraries=[PolynomialLibrary(), PolynomialLibrary()], tensor_array=[[0, 1]]
        ),
        dict(
            libraries=[PolynomialLibrary(), PolynomialLibrary()], tensor_array=[[1, -1]]
        ),
        dict(
            libraries=[PolynomialLibrary(), PolynomialLibrary()], tensor_array=[[2, 1]]
        ),
        dict(libraries=[PolynomialLibrary(), PolynomialLibrary()], tensor_array=[1, 1]),
        dict(
            libraries=[PolynomialLibrary(), PolynomialLibrary()],
            tensor_array=[[1, 1, 1]],
        ),
        dict(
            libraries=[PolynomialLibrary(), PolynomialLibrary()],
            inputs_per_library=[[0, 1], [0, 100]],
        ),
        dict(
            libraries=[PolynomialLibrary(), PolynomialLibrary()],
            inputs_per_library=[[0, 1]],
        ),
        dict(
            libraries=[PolynomialLibrary(), PolynomialLibrary()],
            inputs_per_library=[[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        ),
        dict(
            libraries=[PolynomialLibrary(), PolynomialLibrary()],
            inputs_per_library=[[0, 1, 2], [0, 1, -1]],
        ),
    ],
)
def test_generalized_library_bad_parameters(data_lorenz, params):
    with pytest.raises(ValueError):
        lib = GeneralizedLibrary(**params)
        x, t = data_lorenz
        lib.fit(x)


@pytest.mark.parametrize(
    "params",
    [
        dict(num_parameters=0, num_features=1),
        dict(num_parameters=1, num_features=0),
        dict(feature_library=None),
        dict(parameter_library=None),
    ],
)
def test_parameterized_library_bad_parameters(data_lorenz, params):
    with pytest.raises(ValueError):
        lib = ParameterizedLibrary(**params)
        x, t = data_lorenz
        lib.fit(x)


@pytest.mark.parametrize(
    "params",
    [
        dict(
            library_functions=[lambda x: x, lambda x: x**2, lambda x: 0 * x],
            function_names=[lambda s: str(s), lambda s: "{}^2".format(s)],
        ),
        dict(
            x_dot_library_functions=[lambda x: x, lambda x: x**2, lambda x: 0 * x],
            function_names=[lambda s: str(s), lambda s: "{}^2".format(s)],
        ),
        dict(x_dot_library_functions=[lambda x: x, lambda x: x**2, lambda x: 0 * x]),
        dict(),
        dict(
            library_functions=[lambda x: x, lambda x: x**2],
            x_dot_library_functions=[lambda x: x, lambda x: x**2],
            function_names=[lambda s: s, lambda s: s + s],
        ),
    ],
)
def test_sindypi_library_bad_params(params):
    with pytest.raises(ValueError):
        SINDyPILibrary(**params)


@pytest.mark.parametrize(
    "library",
    [
        IdentityLibrary(),
        PolynomialLibrary(),
        PolynomialLibrary(include_bias=False),
        FourierLibrary(),
        IdentityLibrary() + PolynomialLibrary(),
        pytest.lazy_fixture("custom_library"),
        pytest.lazy_fixture("custom_library_bias"),
        pytest.lazy_fixture("generalized_library"),
        pytest.lazy_fixture("ode_library"),
        pytest.lazy_fixture("sindypi_library"),
    ],
)
def test_fit_transform(data_lorenz, library):
    x, t = data_lorenz
    library.fit_transform(x)
    check_is_fitted(library)


@pytest.mark.parametrize(
    "library",
    [
        IdentityLibrary(),
        PolynomialLibrary(),
        PolynomialLibrary(include_bias=False),
        FourierLibrary(),
        IdentityLibrary() + PolynomialLibrary(),
        pytest.lazy_fixture("custom_library"),
        pytest.lazy_fixture("custom_library_bias"),
        pytest.lazy_fixture("generalized_library"),
        pytest.lazy_fixture("ode_library"),
        pytest.lazy_fixture("pde_library"),
        pytest.lazy_fixture("sindypi_library"),
    ],
)
def test_change_in_data_shape(data_lorenz, library):
    x, t = data_lorenz
    library.fit(x)
    with pytest.raises(ValueError):
        library.transform(x[:, 1:])


@pytest.mark.parametrize(
    "library, shape",
    [
        (IdentityLibrary(), 3),
        (PolynomialLibrary(include_bias=False), 9),
        (PolynomialLibrary(), 10),
        (IdentityLibrary() + PolynomialLibrary(), 13),
        (FourierLibrary(), 6),
        (pytest.lazy_fixture("custom_library_bias"), 13),
        (pytest.lazy_fixture("custom_library"), 12),
        (pytest.lazy_fixture("generalized_library"), 76),
        (pytest.lazy_fixture("ode_library"), 9),
        (pytest.lazy_fixture("sindypi_library"), 39),
    ],
)
def test_output_shape(data_lorenz, library, shape):
    x, t = data_lorenz
    y = library.fit_transform(x)
    expected_shape = (x.shape[0], shape)
    assert y.shape == expected_shape
    assert library.size > 0


@pytest.mark.parametrize(
    "library",
    [
        IdentityLibrary(),
        PolynomialLibrary(),
        PolynomialLibrary(include_bias=False),
        FourierLibrary(),
        PolynomialLibrary() + FourierLibrary(),
        pytest.lazy_fixture("custom_library"),
        pytest.lazy_fixture("custom_library_bias"),
        pytest.lazy_fixture("generalized_library"),
        pytest.lazy_fixture("ode_library"),
        pytest.lazy_fixture("sindypi_library"),
    ],
)
def test_get_feature_names(data_lorenz, library):
    with pytest.raises(NotFittedError):
        library.get_feature_names()

    x, t = data_lorenz
    library.fit_transform(x)
    feature_names = library.get_feature_names()
    assert isinstance(feature_names, list)
    assert isinstance(feature_names[0], str)

    input_features = ["a"] * x.shape[1]
    library.get_feature_names(input_features=input_features)
    assert isinstance(feature_names, list)
    assert isinstance(feature_names[0], str)


# Catch-all for various combinations of options and
# inputs for polynomial features
@pytest.mark.parametrize(
    "kwargs, sparse_format",
    [
        ({"degree": 4}, csr_matrix),
        ({"include_bias": True}, csr_matrix),
        ({"include_bias": False}, csr_matrix),
        ({"include_bias": False, "interaction_only": True}, csr_matrix),
        (
            {
                "include_bias": False,
                "interaction_only": False,
                "include_interaction": False,
            },
            csr_matrix,
        ),
        (
            {
                "include_bias": False,
                "interaction_only": False,
                "include_interaction": True,
            },
            csr_matrix,
        ),
        ({"include_interaction": False}, lambda x: x),
        ({"include_interaction": False, "include_bias": True}, lambda x: x),
    ],
)
def test_polynomial_options(data_lorenz, kwargs, sparse_format):
    x, t = data_lorenz
    library = PolynomialLibrary(**kwargs)
    out = library.fit_transform(sparse_format(x))
    check_is_fitted(library)
    expected = len(library.powers_)
    result = out.shape[1]
    assert result == expected


# Catch-all for various combinations of options and
# inputs for Fourier features
def test_fourier_options(data_lorenz):
    x, t = data_lorenz

    library = FourierLibrary(include_cos=False)
    library.fit_transform(x)
    check_is_fitted(library)


def test_not_implemented(data_lorenz):
    x, t = data_lorenz
    library = BaseFeatureLibrary()

    with pytest.raises(NotImplementedError):
        library.fit(x)

    with pytest.raises(NotImplementedError):
        library.transform(x)

    with pytest.raises(NotImplementedError):
        library.get_feature_names(x)


def test_concat(data_lorenz):
    x, t = data_lorenz
    ident_lib = IdentityLibrary()
    poly_lib = PolynomialLibrary()
    concat_lib = ident_lib + poly_lib
    assert isinstance(concat_lib, ConcatLibrary)
    concat_lib.fit(x)
    check_is_fitted(concat_lib)
    concat_lib.fit_transform(x)
    check_is_fitted(concat_lib)


def test_tensored(data_lorenz):
    x, t = data_lorenz
    ident_lib = IdentityLibrary()
    poly_lib = PolynomialLibrary()
    tensored_lib = ident_lib * poly_lib
    assert isinstance(tensored_lib, TensoredLibrary)
    tensored_lib.fit(x)
    check_is_fitted(tensored_lib)
    tensored_lib.fit_transform(x)
    check_is_fitted(tensored_lib)


@pytest.mark.parametrize(
    "library",
    [
        IdentityLibrary(),
        PolynomialLibrary(),
        FourierLibrary(),
        PolynomialLibrary() + FourierLibrary(),
        pytest.lazy_fixture("custom_library"),
        pytest.lazy_fixture("generalized_library"),
        pytest.lazy_fixture("ode_library"),
        pytest.lazy_fixture("pde_library"),
        pytest.lazy_fixture("sindypi_library"),
    ],
)
def test_not_fitted(data_lorenz, library):
    x, t = data_lorenz

    with pytest.raises(NotFittedError):
        library.transform(x)


def test_generalized_library(data_lorenz):
    x, t = data_lorenz
    poly_library = PolynomialLibrary(include_bias=False)
    fourier_library = FourierLibrary()
    library_functions = [
        lambda x: np.exp(x),
        lambda x: 1.0 / x,
        lambda x, y: np.sin(x + y),
    ]
    custom_library = CustomLibrary(
        library_functions=library_functions,
    )

    tensor_array = [[0, 1, 1], [1, 0, 1]]

    inputs_per_library = [[1, 2], [0, 2], [0]]

    # First try without tensor libraries and subset of the input variables
    sindy_library = GeneralizedLibrary(
        [poly_library, fourier_library, custom_library],
    )
    sindy_opt = STLSQ(threshold=0.25)
    model = SINDy(
        optimizer=sindy_opt,
        feature_library=sindy_library,
    )
    model.fit(x, t=t)
    model.print()
    model.get_feature_names()
    assert len(model.get_feature_names()) == 24

    # Repeat with feature names
    feature_names = ["x", "y", "z"]
    model = SINDy(
        optimizer=sindy_opt, feature_library=sindy_library, feature_names=feature_names
    )
    model.fit(x, t=t)
    model.print()
    model.get_feature_names()
    assert len(model.get_feature_names()) == 24

    # Next try with tensor libraries but still all the input variables
    sindy_library = GeneralizedLibrary(
        [poly_library, fourier_library, custom_library], tensor_array=tensor_array
    )
    model = SINDy(
        optimizer=sindy_opt,
        feature_library=sindy_library,
    )
    model.fit(x, t=t)
    model.print()
    # 24 + (9 * 6) = 54 + (9 * 9) = 81
    assert len(model.get_feature_names()) == 159

    # Repeat with feature_names
    sindy_library = GeneralizedLibrary(
        [poly_library, fourier_library, custom_library], tensor_array=tensor_array
    )
    model = SINDy(
        optimizer=sindy_opt, feature_library=sindy_library, feature_names=feature_names
    )
    model.fit(x, t=t)
    model.print()
    # 24 + (9 * 6) = 54 + (9 * 9) = 81
    assert len(model.get_feature_names()) == 159

    sindy_library = GeneralizedLibrary(
        [poly_library, fourier_library, custom_library],
        tensor_array=tensor_array,
        inputs_per_library=inputs_per_library,
    )
    model = SINDy(
        optimizer=sindy_opt,
        feature_library=sindy_library,
    )
    model.fit(x, t=t)
    assert len(model.get_feature_names()) == 29

    # Repeat with feature names
    sindy_library = GeneralizedLibrary(
        [poly_library, fourier_library, custom_library],
        tensor_array=tensor_array,
        inputs_per_library=inputs_per_library,
    )
    model = SINDy(
        optimizer=sindy_opt, feature_library=sindy_library, feature_names=feature_names
    )
    model.fit(x, t=t)
    assert len(model.get_feature_names()) == 29


def test_generalized_library_pde(data_1d_random_pde):
    t, x, u, u_dot = data_1d_random_pde
    poly_library = PolynomialLibrary(include_bias=False)
    fourier_library = FourierLibrary()
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_library = PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=2,
        spatial_grid=x,
        include_bias=True,
    )

    # First try without tensor libraries and subset of the input variables
    sindy_library = GeneralizedLibrary(
        [poly_library, fourier_library, pde_library],
    )
    sindy_opt = STLSQ(threshold=0.25)
    model = SINDy(
        optimizer=sindy_opt,
        feature_library=sindy_library,
    )
    model.fit(u, t=t)
    model.print()
    model.get_feature_names()
    assert len(model.get_feature_names()) == 13


def test_generalized_library_weak_pde(data_1d_random_pde):
    t, x, u, u_dot = data_1d_random_pde
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    X, T = np.meshgrid(x, t)
    XT = np.array([X, T]).T
    weak_library1 = WeakPDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=2,
        spatiotemporal_grid=XT,
        include_bias=True,
    )
    library_functions = [lambda x: x * x * x]
    library_function_names = [lambda x: x + x + x]
    weak_library2 = WeakPDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=0,
        spatiotemporal_grid=XT,
    )

    # First try without tensor libraries and subset of the input variables
    sindy_library = GeneralizedLibrary(
        [weak_library1, weak_library2],
    )
    sindy_opt = STLSQ(threshold=0.25)
    model = SINDy(
        optimizer=sindy_opt,
        feature_library=sindy_library,
    )
    model.fit(u, t=t)
    model.print()
    model.get_feature_names()
    assert len(model.get_feature_names()) == 10


def test_parameterized_library(diffuse_multiple_trajectories):
    t, spatial_grid, xs = diffuse_multiple_trajectories
    us = []
    for i in range(len(xs)):
        u = np.zeros(xs[0].shape)
        us = us + [u]

    library_functions = [lambda x: x]
    library_function_names = [lambda x: x]

    feature_lib = PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=2,
        spatial_grid=spatial_grid,
    )

    parameter_lib = PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=0,
        include_bias=True,
    )

    pde_lib = ParameterizedLibrary(
        feature_library=feature_lib,
        parameter_library=parameter_lib,
        num_features=1,
        num_parameters=1,
    )

    X, T = np.meshgrid(spatial_grid, t, indexing="ij")
    XT = np.transpose([X, T], [1, 2, 0])

    np.random.seed(100)
    weak_feature_lib = WeakPDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=2,
        spatiotemporal_grid=XT,
        K=100,
    )
    np.random.seed(100)
    weak_parameter_lib = WeakPDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=0,
        spatiotemporal_grid=XT,
        K=100,
        include_bias=True,
    )

    weak_lib = ParameterizedLibrary(
        feature_library=weak_feature_lib,
        parameter_library=weak_parameter_lib,
        num_features=1,
        num_parameters=1,
    )

    optimizer = STLSQ(threshold=0.5, alpha=1e-8, normalize_columns=False)
    model = SINDy(
        feature_library=pde_lib, optimizer=optimizer, feature_names=["u", "c"]
    )
    model.fit(xs, u=us, t=t)
    assert abs(model.coefficients()[0, 4] - 1) < 1e-1
    assert np.all(model.coefficients()[0, :4] == 0)
    assert np.all(model.coefficients()[0, 5:] == 0)

    optimizer = STLSQ(threshold=0.25, alpha=1e-8, normalize_columns=False)
    model = SINDy(
        feature_library=weak_lib, optimizer=optimizer, feature_names=["u", "c"]
    )
    model.fit(xs, u=us, t=t)
    assert abs(model.coefficients()[0, 4] - 1) < 1e-1
    assert np.all(model.coefficients()[0, :4] == 0)
    assert np.all(model.coefficients()[0, 5:] == 0)


# Helper function for testing PDE libraries
def pde_library_helper(library, u):
    base_opt = STLSQ(normalize_columns=True, alpha=1e-10, threshold=0)
    model = SINDy(optimizer=base_opt, feature_library=library)
    model.fit(u)
    assert np.any(base_opt.coef_ != 0.0)


def test_1D_pdes(data_1d_random_pde):
    _, spatial_grid, u, _ = data_1d_random_pde
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=4,
        spatial_grid=spatial_grid,
        include_bias=True,
    )
    pde_library_helper(pde_lib, u)


def test_2D_pdes(data_2d_random_pde):
    spatial_grid, u, _ = data_2d_random_pde
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=2,
        spatial_grid=spatial_grid,
        include_bias=True,
    )
    pde_library_helper(pde_lib, u)


def test_3D_pdes(data_3d_random_pde):
    spatial_grid, u, _ = data_3d_random_pde
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=2,
        spatial_grid=spatial_grid,
        include_bias=True,
    )
    pde_library_helper(pde_lib, u)


def test_5D_pdes(data_5d_random_pde):
    spatial_grid, u, _ = data_5d_random_pde
    pde_lib = PDELibrary(
        derivative_order=1,
        spatial_grid=spatial_grid,
        include_bias=True,
    )
    pde_library_helper(pde_lib, u)


def test_1D_weak_pdes():
    n = 10
    t = np.linspace(0, 10, n)
    x = np.linspace(0, 10, n)
    u = np.random.randn(n, n, 1)
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    X, T = np.meshgrid(x, t, indexing="ij")
    spatiotemporal_grid = np.asarray([X, T])
    spatiotemporal_grid = np.transpose(spatiotemporal_grid, axes=[1, 2, 0])
    pde_lib = WeakPDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=4,
        spatiotemporal_grid=spatiotemporal_grid,
        H_xt=2,
        include_bias=True,
    )
    pde_library_helper(pde_lib, u)


def test_2D_weak_pdes():
    n = 5
    t = np.linspace(0, 10, n)
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 10, n)
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    spatiotemporal_grid = np.asarray([X, Y, T])
    spatiotemporal_grid = np.transpose(spatiotemporal_grid, axes=[1, 2, 3, 0])
    u = np.random.randn(n, n, n, 1)
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = WeakPDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=2,
        spatiotemporal_grid=spatiotemporal_grid,
        H_xt=4,
        K=10,
        include_bias=True,
    )
    pde_library_helper(pde_lib, u)


def test_3D_weak_pdes():
    n = 5
    t = np.linspace(0, 10, n)
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 10, n)
    z = np.linspace(0, 10, n)
    X, Y, Z, T = np.meshgrid(x, y, z, t, indexing="ij")
    spatiotemporal_grid = np.asarray([X, Y, Z, T])
    spatiotemporal_grid = np.transpose(spatiotemporal_grid, axes=[1, 2, 3, 4, 0])
    u = np.random.randn(n, n, n, n, 2)
    pde_lib = WeakPDELibrary(
        derivative_order=2,
        spatiotemporal_grid=spatiotemporal_grid,
        H_xt=4,
        K=10,
        include_bias=True,
    )
    pde_library_helper(pde_lib, u)


def test_5D_weak_pdes():
    n = 5
    t = np.linspace(0, 10, n)
    v = np.linspace(0, 10, n)
    w = np.linspace(0, 10, n)
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 10, n)
    z = np.linspace(0, 10, n)
    V, W, X, Y, Z, T = np.meshgrid(v, w, x, y, z, t, indexing="ij")
    spatiotemporal_grid = np.asarray([V, W, X, Y, Z, T])
    spatiotemporal_grid = np.transpose(spatiotemporal_grid, axes=[1, 2, 3, 4, 5, 6, 0])
    u = np.random.randn(n, n, n, n, n, n, 2)
    pde_lib = WeakPDELibrary(
        derivative_order=2,
        spatiotemporal_grid=spatiotemporal_grid,
        H_xt=4,
        K=10,
        include_bias=True,
    )
    pde_library_helper(pde_lib, u)


def test_sindypi_library(data_lorenz):
    x, t = data_lorenz
    x_library_functions = [
        lambda x: x,
        lambda x, y: x * y,
        lambda x: x**2,
    ]
    x_dot_library_functions = [lambda x: x]

    library_function_names = [
        lambda x: x,
        lambda x, y: x + y,
        lambda x: x + x,
        lambda x: x,
    ]
    sindy_library = SINDyPILibrary(
        library_functions=x_library_functions,
        x_dot_library_functions=x_dot_library_functions,
        t=t,
        function_names=library_function_names,
        include_bias=True,
    )
    sindy_opt = SINDyPI(threshold=0.1, thresholder="l1")
    model = SINDy(
        optimizer=sindy_opt,
        feature_library=sindy_library,
        differentiation_method=FiniteDifference(),
    )
    model.fit(x, t=t)
    assert np.shape(sindy_opt.coef_) == (40, 40)

    sindy_opt = SINDyPI(threshold=1, thresholder="l1", model_subset=[3])
    model = SINDy(
        optimizer=sindy_opt,
        feature_library=sindy_library,
        differentiation_method=FiniteDifference(),
    )
    model.fit(x, t=t)
    assert np.sum(sindy_opt.coef_ == 0.0) == 40.0 * 39.0 and np.any(
        sindy_opt.coef_[3, :] != 0.0
    )


@pytest.mark.parametrize(
    ("include_interaction", "interaction_only", "bias", "expected"),
    [
        (True, True, True, ((), (0,), (0, 1), (1,))),
        (False, False, False, ((0,), (0, 0), (1,), (1, 1))),
    ],
)
def test_polynomial_combinations(include_interaction, interaction_only, bias, expected):
    combos = PolynomialLibrary._combinations(
        n_features=2,
        degree=2,
        include_interaction=include_interaction,
        interaction_only=interaction_only,
        include_bias=bias,
    )
    result = tuple(sorted(list(combos)))
    assert result == expected
