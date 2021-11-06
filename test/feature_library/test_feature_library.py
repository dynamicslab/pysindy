"""
Unit tests for feature libraries.
"""
import numpy as np
import pytest
from scipy.integrate import trapezoid
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from pysindy import SINDy
from pysindy.differentiation import FiniteDifference
from pysindy.feature_library import ConcatLibrary
from pysindy.feature_library import CustomLibrary
from pysindy.feature_library import FourierLibrary
from pysindy.feature_library import IdentityLibrary
from pysindy.feature_library import PDELibrary
from pysindy.feature_library import PolynomialLibrary
from pysindy.feature_library import SINDyPILibrary
from pysindy.feature_library import SpatiotemporalLibrary
from pysindy.feature_library.base import BaseFeatureLibrary
from pysindy.optimizers import SINDyPI
from pysindy.optimizers import STLSQ


def test_form_custom_library():
    library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
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
    library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
    function_names = [
        lambda s: str(s),
        lambda s: "{}^2".format(s),
        lambda s: "0",
    ]

    # Test with user-supplied function names
    PDELibrary(library_functions=library_functions, function_names=function_names)

    # Test without user-supplied function names
    PDELibrary(library_functions=library_functions, function_names=None)


def test_form_spatiotemporal_library():
    library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
    function_names = [
        lambda s: str(s),
        lambda s: "{}^2".format(s),
        lambda s: "0",
    ]
    x = np.linspace(0, 10, 10)
    y = np.linspace(0, 10, 20)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Test with user-supplied function names
    SpatiotemporalLibrary(
        library_functions=library_functions,
        function_names=function_names,
        spatiotemporal_variables=[X, Y],
    )

    # Test without user-supplied function names
    SpatiotemporalLibrary(
        library_functions=library_functions,
        function_names=None,
        spatiotemporal_variables=[X, Y],
    )


def test_form_sindy_pi_library():
    library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
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
        PolynomialLibrary(degree=-1)
    with pytest.raises(ValueError):
        PolynomialLibrary(degree=1.5)
    with pytest.raises(ValueError):
        PolynomialLibrary(include_interaction=False, interaction_only=True)
    with pytest.raises(ValueError):
        FourierLibrary(n_frequencies=-1)
    with pytest.raises(ValueError):
        FourierLibrary(n_frequencies=-1)
    with pytest.raises(ValueError):
        FourierLibrary(n_frequencies=2.2)
    with pytest.raises(ValueError):
        FourierLibrary(include_sin=False, include_cos=False)
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        function_names = [lambda s: str(s), lambda s: "{}^2".format(s)]
        CustomLibrary(
            library_functions=library_functions, function_names=function_names
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        function_names = [lambda s: str(s), lambda s: "{}^2".format(s)]
        x = np.linspace(0, 10, 10)
        y = np.linspace(0, 10, 20)
        X, Y = np.meshgrid(x, y, indexing="ij")
        SpatiotemporalLibrary(
            library_functions=library_functions,
            function_names=function_names,
            spatiotemporal_variables=[X, Y],
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2]
        SpatiotemporalLibrary(
            library_functions=library_functions,
            spatiotemporal_variables=np.linspace(0, 10, 10),
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2]
        X = np.linspace(0, 10, 10)
        Y = np.linspace(0, 10, 10)
        SpatiotemporalLibrary(
            library_functions=library_functions, spatiotemporal_variables=[X, Y]
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        function_names = [lambda s: str(s), lambda s: "{}^2".format(s)]
        PDELibrary(library_functions=library_functions, function_names=function_names)
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        PDELibrary(library_functions=library_functions, derivative_order=-1)
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        PDELibrary(library_functions=library_functions, spatial_grid=range(10))
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        PDELibrary(library_functions=library_functions, derivative_order=3)
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        PDELibrary(
            library_functions=library_functions,
            spatial_grid=range(10),
            derivative_order=-1,
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        PDELibrary(
            library_functions=library_functions,
            spatial_grid=np.zeros((10, 10)),
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        PDELibrary(
            library_functions=library_functions,
            spatial_grid=np.zeros((10, 10, 10, 10, 10)),
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        PDELibrary(
            library_functions=library_functions,
            spatial_grid=range(10),
            temporal_grid=range(10),
            weak_form=True,
            p=-1,
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        PDELibrary(
            library_functions=library_functions,
            weak_form=True,
            spatial_grid=range(10),
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        PDELibrary(
            library_functions=library_functions,
            spatial_grid=range(10),
            temporal_grid=range(10),
            weak_form=True,
            Hx=-1,
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        PDELibrary(
            library_functions=library_functions,
            spatial_grid=range(10),
            temporal_grid=range(10),
            weak_form=True,
            Hx=11,
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        x = range(0, 10)
        y = range(0, 10)
        X, Y = np.meshgrid(x, y)
        spatial_grid = np.asarray([X, Y]).T
        PDELibrary(
            library_functions=library_functions,
            spatial_grid=spatial_grid,
            temporal_grid=range(10),
            weak_form=True,
            Hy=-1,
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        x = range(0, 10)
        y = range(0, 10)
        X, Y = np.meshgrid(x, y)
        spatial_grid = np.asarray([X, Y]).T
        PDELibrary(
            library_functions=library_functions,
            spatial_grid=range(10),
            temporal_grid=range(10),
            weak_form=True,
            Hy=11,
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        x = range(0, 10)
        y = range(0, 10)
        z = range(0, 10)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        spatial_grid = np.asarray([X, Y, Z])
        spatial_grid = np.transpose(spatial_grid, axes=[1, 2, 3, 0])
        PDELibrary(
            library_functions=library_functions,
            spatial_grid=spatial_grid,
            temporal_grid=range(10),
            weak_form=True,
            Hz=-1,
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        x = range(0, 10)
        y = range(0, 10)
        z = range(0, 10)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        spatial_grid = np.asarray([X, Y, Z])
        spatial_grid = np.transpose(spatial_grid, axes=[1, 2, 3, 0])
        PDELibrary(
            library_functions=library_functions,
            spatial_grid=spatial_grid,
            temporal_grid=range(10),
            weak_form=True,
            Hz=11,
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        PDELibrary(
            library_functions=library_functions,
            spatial_grid=range(10),
            temporal_grid=range(10),
            weak_form=True,
            K=-1,
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        PDELibrary(
            library_functions=library_functions,
            spatial_grid=range(10),
            temporal_grid=range(10),
            weak_form=True,
            Ht=11,
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        PDELibrary(
            library_functions=library_functions,
            spatial_grid=range(10),
            temporal_grid=range(10),
            weak_form=True,
            Ht=-1,
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        PDELibrary(
            library_functions=library_functions,
            spatial_grid=range(10),
            temporal_grid=np.zeros((10, 3)),
            weak_form=True,
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        function_names = [lambda s: str(s), lambda s: "{}^2".format(s)]
        SINDyPILibrary(
            library_functions=library_functions, function_names=function_names
        )
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        function_names = [lambda s: str(s), lambda s: "{}^2".format(s)]
        SINDyPILibrary(
            x_dot_library_functions=library_functions, function_names=function_names
        )
    with pytest.raises(ValueError):
        SINDyPILibrary()
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
        SINDyPILibrary(x_dot_library_functions=library_functions)
    with pytest.raises(ValueError):
        library_functions = [lambda x: x, lambda x: x ** 2]
        function_names = [lambda s: s, lambda s: s + s]
        SINDyPILibrary(
            library_functions=library_functions,
            x_dot_library_functions=library_functions,
            function_names=function_names,
        )


@pytest.mark.parametrize(
    "library",
    [
        IdentityLibrary(),
        PolynomialLibrary(),
        FourierLibrary(),
        IdentityLibrary() + PolynomialLibrary(),
        pytest.lazy_fixture("data_custom_library"),
        pytest.lazy_fixture("data_spatiotemporal_library"),
        pytest.lazy_fixture("data_ode_library"),
        pytest.lazy_fixture("data_pde_library"),
        pytest.lazy_fixture("data_sindypi_library"),
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
        FourierLibrary(),
        IdentityLibrary() + PolynomialLibrary(),
        pytest.lazy_fixture("data_custom_library"),
        pytest.lazy_fixture("data_ode_library"),
        pytest.lazy_fixture("data_pde_library"),
        pytest.lazy_fixture("data_sindypi_library"),
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
        (PolynomialLibrary(), 10),
        (IdentityLibrary() + PolynomialLibrary(), 13),
        (FourierLibrary(), 6),
        (pytest.lazy_fixture("data_custom_library"), 12),
        (pytest.lazy_fixture("data_spatiotemporal_library"), 7),
        (pytest.lazy_fixture("data_ode_library"), 9),
        (pytest.lazy_fixture("data_pde_library"), 129),
        (pytest.lazy_fixture("data_sindypi_library"), 39),
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
        FourierLibrary(),
        PolynomialLibrary() + FourierLibrary(),
        pytest.lazy_fixture("data_custom_library"),
        pytest.lazy_fixture("data_spatiotemporal_library"),
        pytest.lazy_fixture("data_ode_library"),
        pytest.lazy_fixture("data_pde_library"),
        pytest.lazy_fixture("data_sindypi_library"),
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


@pytest.mark.parametrize("sparse_format", [csc_matrix, csr_matrix, coo_matrix])
def test_polynomial_sparse_inputs(data_lorenz, sparse_format):
    x, t = data_lorenz
    library = PolynomialLibrary()
    library.fit_transform(sparse_format(x))
    check_is_fitted(library)


# Catch-all for various combinations of options and
# inputs for polynomial features
@pytest.mark.parametrize(
    "kwargs, sparse_format",
    [
        ({"degree": 4}, csr_matrix),
        ({"include_bias": True}, csr_matrix),
        ({"include_interaction": False}, lambda x: x),
        ({"include_interaction": False, "include_bias": True}, lambda x: x),
    ],
)
def test_polynomial_options(data_lorenz, kwargs, sparse_format):
    x, t = data_lorenz
    library = PolynomialLibrary(**kwargs)
    library.fit_transform(sparse_format(x))
    check_is_fitted(library)


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


def test_concat():
    ident_lib = IdentityLibrary()
    poly_lib = PolynomialLibrary()
    concat_lib = ident_lib + poly_lib
    assert isinstance(concat_lib, ConcatLibrary)


@pytest.mark.parametrize(
    "library",
    [
        IdentityLibrary(),
        PolynomialLibrary(),
        FourierLibrary(),
        PolynomialLibrary() + FourierLibrary(),
        pytest.lazy_fixture("data_custom_library"),
        pytest.lazy_fixture("data_spatiotemporal_library"),
        pytest.lazy_fixture("data_ode_library"),
        pytest.lazy_fixture("data_pde_library"),
        pytest.lazy_fixture("data_sindypi_library"),
    ],
)
def test_not_fitted(data_lorenz, library):
    x, t = data_lorenz

    with pytest.raises(NotFittedError):
        library.transform(x)


@pytest.mark.parametrize(
    "library",
    [
        IdentityLibrary(),
        PolynomialLibrary(),
        FourierLibrary(),
        PolynomialLibrary() + FourierLibrary(),
    ],
)
def test_library_ensemble(data_lorenz, library):
    x, t = data_lorenz
    library.fit(x)
    N = library.n_output_features_
    library.library_ensemble = True
    xp = library.transform(x)
    assert N == xp.shape[1] + 1
    library.ensemble_indices = [0, 1]
    xp = library.transform(x)
    assert N == xp.shape[1] + 2


@pytest.mark.parametrize(
    "library",
    [
        IdentityLibrary,
        PolynomialLibrary,
        FourierLibrary,
    ],
)
def test_bad_library_ensemble(data_lorenz, library):
    x, t = data_lorenz
    with pytest.raises(ValueError):
        library = library(ensemble_indices=-1)


# Try various size spatiotemporal inputs for spatiotemporal library
def test_spatiotemporal_shapes(data_lorenz):
    data, t = data_lorenz
    x = np.linspace(0, 10, 10)
    y = np.linspace(0, 10, 50)
    X, Y = np.meshgrid(x, y, indexing="ij")
    library_functions = [lambda x: x, lambda x: x ** 2]
    library = SpatiotemporalLibrary(
        library_functions=library_functions, spatiotemporal_variables=[X, Y]
    )
    library.fit_transform(data)
    check_is_fitted(library)
    x = np.linspace(0, 10, 5)
    y = np.linspace(0, 10, 10)
    z = np.linspace(0, 10, 10)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    library = SpatiotemporalLibrary(
        library_functions=library_functions, spatiotemporal_variables=[X, Y, Z]
    )
    library.fit_transform(data)
    check_is_fitted(library)
    x = np.linspace(0, 10, 5)
    y = np.linspace(0, 10, 5)
    z = np.linspace(0, 10, 5)
    t = np.linspace(0, 10, 4)
    X, Y, Z, T = np.meshgrid(x, y, z, t, indexing="ij")
    library = SpatiotemporalLibrary(
        library_functions=library_functions, spatiotemporal_variables=[X, Y, Z, T]
    )
    library.fit_transform(data)
    check_is_fitted(library)


def test_1D_pdes():
    t = np.linspace(0, 10, 8)
    dt = t[1] - t[0]
    x = np.linspace(0, 10, 8)
    nx = len(x)
    u = np.random.randn(nx, len(t), 1)
    u_dot = np.zeros(u.shape)
    for i in range(len(x)):
        u_dot[i, :, :] = FiniteDifference()._differentiate(u[i, :, :], t=dt)
    u_flattened = np.reshape(u, (nx * len(t), 1))
    u_dot_flattened = np.reshape(u_dot, (nx * len(t), 1))

    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=4,
        spatial_grid=x,
        include_bias=True,
        is_uniform=True,
    )
    opt = STLSQ(normalize_columns=True, alpha=1e-10, threshold=0)
    model = SINDy(optimizer=opt, feature_library=pde_lib)
    model.fit(u_flattened, x_dot=u_dot_flattened)
    assert np.any(opt.coef_ != 0.0)
    n_features = len(model.get_feature_names())
    model.fit(u_flattened, x_dot=u_dot_flattened, ensemble=True, n_models=10)
    assert np.shape(model.coef_list) == (10, 1, n_features)
    model.fit(u_flattened, x_dot=u_dot_flattened, library_ensemble=True, n_models=10)
    assert np.shape(model.coef_list) == (10, 1, n_features)


def test_1D_pdelibrary_plus_spatiotemporal_library():
    t = np.linspace(0, 10, 8)
    dt = t[1] - t[0]
    x = np.linspace(0, 10, 8)
    nx = len(x)
    u = np.random.randn(nx, len(t), 1)
    u_dot = np.zeros(u.shape)
    for i in range(len(x)):
        u_dot[i, :, :] = FiniteDifference()._differentiate(u[i, :, :], t=dt)
    u_flattened = np.reshape(u, (nx * len(t), 1))
    u_dot_flattened = np.reshape(u_dot, (nx * len(t), 1))

    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=4,
        spatial_grid=x,
        include_bias=True,
        is_uniform=True,
    )
    spatiotemporal_lib = SpatiotemporalLibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        spatiotemporal_variables=[x],
        include_bias=False,
    )
    sindy_lib = pde_lib + spatiotemporal_lib
    opt = STLSQ(normalize_columns=True, alpha=1e-10, threshold=0)
    model = SINDy(optimizer=opt, feature_library=sindy_lib)
    model.fit(u_flattened, x_dot=u_dot_flattened)
    assert np.any(opt.coef_ != 0.0)
    n_features = len(model.get_feature_names())
    model.fit(u_flattened, x_dot=u_dot_flattened, ensemble=True, n_models=10)
    assert np.shape(model.coef_list) == (10, 1, n_features)
    model.fit(u_flattened, x_dot=u_dot_flattened, library_ensemble=True, n_models=10)
    assert np.shape(model.coef_list) == (10, 1, n_features)


def test_2D_pdes():
    t = np.linspace(0, 10, 8)
    dt = t[1] - t[0]
    x = np.linspace(0, 10, 8)
    y = np.linspace(0, 10, 8)
    X, Y = np.meshgrid(x, y)
    spatial_grid = np.asarray([X, Y]).T
    nx = len(x)
    ny = len(y)
    u = np.random.randn(nx, ny, len(t), 2)
    u_dot = np.zeros(u.shape)
    for i in range(nx):
        for j in range(ny):
            u_dot[i, j, :, :] = FiniteDifference()._differentiate(u[i, j, :, :], t=dt)
    u_flattened = np.reshape(u, (nx * ny * len(t), 2))
    u_dot_flattened = np.reshape(u_dot, (nx * ny * len(t), 2))

    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=2,
        spatial_grid=spatial_grid,
        include_bias=True,
        is_uniform=True,
    )
    opt = STLSQ(normalize_columns=True, alpha=1e-10, threshold=0)
    model = SINDy(optimizer=opt, feature_library=pde_lib)
    model.fit(u_flattened, x_dot=u_dot_flattened)
    assert np.any(opt.coef_ != 0.0)
    n_features = len(model.get_feature_names())
    model.fit(
        u_flattened, x_dot=u_dot_flattened, ensemble=True, n_models=10, n_subset=20
    )
    assert np.shape(model.coef_list) == (10, 2, n_features)
    model.fit(u_flattened, x_dot=u_dot_flattened, library_ensemble=True, n_models=10)
    assert np.shape(model.coef_list) == (10, 2, n_features)


def test_3D_pdes():
    t = np.linspace(0, 10, 8)
    dt = t[1] - t[0]
    x = np.linspace(0, 10, 8)
    y = np.linspace(0, 10, 8)
    z = np.linspace(0, 10, 8)
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

    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=2,
        spatial_grid=spatial_grid,
        include_bias=True,
        is_uniform=True,
    )
    opt = STLSQ(normalize_columns=True, alpha=1e-10, threshold=0)
    model = SINDy(optimizer=opt, feature_library=pde_lib)
    model.fit(u_flattened, x_dot=u_dot_flattened)
    assert np.any(opt.coef_ != 0.0)
    n_features = len(model.get_feature_names())
    model.fit(
        u_flattened, x_dot=u_dot_flattened, ensemble=True, n_models=10, n_subset=20
    )
    assert np.shape(model.coef_list) == (10, 2, n_features)
    model.fit(u_flattened, x_dot=u_dot_flattened, library_ensemble=True, n_models=10)
    assert np.shape(model.coef_list) == (10, 2, n_features)


def test_1D_weak_pdes():
    t = np.linspace(0, 10, 20)
    x = np.linspace(0, 10, 20)
    nx = len(x)
    u = np.random.randn(nx, len(t), 1)
    u_flattened = np.reshape(u, (nx * len(t), 1))
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=4,
        spatial_grid=x,
        Hx=0.1,
        Ht=0.1,
        temporal_grid=t,
        include_bias=True,
        K=5,
        is_uniform=False,
        weak_form=True,
        num_pts_per_domain=20,
    )

    K = pde_lib.K
    num_pts_per_domain = pde_lib.num_pts_per_domain
    u_dot_integral = np.zeros((K, 1))
    u_shaped = np.reshape(u, (len(x), len(t)))
    u_interp = RectBivariateSpline(x, t, u_shaped)
    for k in range(K):
        X = np.ravel(pde_lib.X[k, :, :])
        t = np.ravel(pde_lib.t[k, :, :])
        u_new = u_interp.ev(X, t)
        u_new = np.reshape(u_new, (num_pts_per_domain, num_pts_per_domain, 1))
        w_diff = pde_lib._smooth_ppoly(
            np.reshape(pde_lib.xgrid_k[k, :], (num_pts_per_domain, 1)),
            pde_lib.tgrid_k[k, :],
            k,
            0,
            0,
            0,
            1,
        )
        u_dot_integral[k] = (-1) * (
            trapezoid(
                trapezoid(u_new * w_diff, x=pde_lib.xgrid_k[k, :], axis=0),
                x=pde_lib.tgrid_k[k, :],
                axis=0,
            )
        )

    opt = STLSQ(normalize_columns=True, alpha=1e-10, threshold=0)
    model = SINDy(optimizer=opt, feature_library=pde_lib)
    model.fit(u_flattened, x_dot=u_dot_integral)
    assert np.any(opt.coef_ != 0.0)
    n_features = len(model.get_feature_names())
    model.fit(u_flattened, x_dot=u_dot_integral, ensemble=True, n_models=10)
    assert np.shape(model.coef_list) == (10, 1, n_features)
    model.fit(u_flattened, x_dot=u_dot_integral, library_ensemble=True, n_models=10)
    assert np.shape(model.coef_list) == (10, 1, n_features)


def test_2D_weak_pdes():
    t = np.linspace(0, 10, 8)
    x = np.linspace(0, 10, 8)
    y = np.linspace(0, 10, 8)
    X, Y = np.meshgrid(x, y)
    spatial_grid = np.asarray([X, Y]).T
    nx = len(x)
    ny = len(y)
    u = np.random.randn(nx, ny, len(t), 1)
    u_flattened = np.reshape(u, (nx * ny * len(t), 1))
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=4,
        spatial_grid=spatial_grid,
        Hx=0.1,
        Hy=0.1,
        Ht=0.1,
        K=2,
        temporal_grid=t,
        include_bias=True,
        is_uniform=False,
        weak_form=True,
        num_pts_per_domain=10,
    )

    K = pde_lib.K
    num_pts_per_domain = pde_lib.num_pts_per_domain
    u_dot_integral = np.zeros((K, 1))
    u_shaped = np.reshape(u, (nx, ny, len(t), 1))
    u_interp = RegularGridInterpolator((x, y, t), u_shaped[:, :, :, 0])
    for k in range(K):
        X = np.ravel(pde_lib.X[k, :, :, :])
        Y = np.ravel(pde_lib.Y[k, :, :, :])
        t = np.ravel(pde_lib.t[k, :, :, :])
        XYt = np.array((X, Y, t)).T
        u_new = u_interp(XYt)
        u_new = np.reshape(
            u_new, (num_pts_per_domain, num_pts_per_domain, num_pts_per_domain, 1)
        )
        w_diff = pde_lib._smooth_ppoly(
            np.transpose((pde_lib.xgrid_k[k, :], pde_lib.ygrid_k[k, :])),
            pde_lib.tgrid_k[k, :],
            k,
            0,
            0,
            0,
            1,
        )
        u_dot_integral[k, :] = (-1) * (
            trapezoid(
                trapezoid(
                    trapezoid(u_new * w_diff, x=pde_lib.xgrid_k[k, :], axis=0),
                    x=pde_lib.ygrid_k[k, :],
                    axis=0,
                ),
                x=pde_lib.tgrid_k[k, :],
                axis=0,
            )
        )

    opt = STLSQ(normalize_columns=True, alpha=1e-10, threshold=0)
    model = SINDy(optimizer=opt, feature_library=pde_lib)
    model.fit(u_flattened, x_dot=u_dot_integral)
    assert np.any(opt.coef_ != 0.0)
    n_features = len(model.get_feature_names())
    model.fit(u_flattened, x_dot=u_dot_integral, ensemble=True, n_models=10)
    assert np.shape(model.coef_list) == (10, 1, n_features)
    model.fit(u_flattened, x_dot=u_dot_integral, library_ensemble=True, n_models=10)
    assert np.shape(model.coef_list) == (10, 1, n_features)


def test_3D_weak_pdes():
    t = np.linspace(0, 10, 7)
    x = np.linspace(0, 10, 7)
    y = np.linspace(0, 10, 7)
    z = np.linspace(0, 10, 7)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    spatial_grid = np.asarray([X, Y, Z])
    spatial_grid = np.transpose(spatial_grid, axes=[1, 2, 3, 0])
    n = len(x)
    u = np.random.randn(n, n, n, n, 2)
    u_flattened = np.reshape(u, (n ** 4, 2))
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=4,
        spatial_grid=spatial_grid,
        Hx=0.1,
        Hy=0.1,
        Hz=0.1,
        Ht=0.1,
        K=2,
        temporal_grid=t,
        include_bias=True,
        is_uniform=False,
        weak_form=True,
        num_pts_per_domain=4,
    )

    K = pde_lib.K
    num_pts_per_domain = pde_lib.num_pts_per_domain
    u_dot_integral = np.zeros((K, 2))
    for kk in range(2):
        u_interp = RegularGridInterpolator(
            (
                pde_lib.spatial_grid[:, 0, 0, 0],
                pde_lib.spatial_grid[0, :, 0, 1],
                pde_lib.spatial_grid[0, 0, :, 2],
                t,
            ),
            u[:, :, :, :, kk],
        )
        for k in range(K):
            XYZt = np.array(
                (
                    np.ravel(pde_lib.X[k, :, :, :, :]),
                    np.ravel(pde_lib.Y[k, :, :, :, :]),
                    np.ravel(pde_lib.Z[k, :, :, :, :]),
                    np.ravel(pde_lib.t[k, :, :, :, :]),
                )
            ).T
            u_new = u_interp(XYZt)
            u_new = np.reshape(
                u_new,
                (
                    num_pts_per_domain,
                    num_pts_per_domain,
                    num_pts_per_domain,
                    num_pts_per_domain,
                    1,
                ),
            )
            w_diff = pde_lib._smooth_ppoly(
                np.transpose(
                    (
                        pde_lib.xgrid_k[k, :],
                        pde_lib.ygrid_k[k, :],
                        pde_lib.zgrid_k[k, :],
                    )
                ),
                pde_lib.tgrid_k[k, :],
                k,
                0,
                0,
                0,
                1,
            )
            u_dot_integral[k, :] = (-1) * (
                trapezoid(
                    trapezoid(
                        trapezoid(
                            trapezoid(u_new * w_diff, x=pde_lib.xgrid_k[k, :], axis=0),
                            x=pde_lib.ygrid_k[k, :],
                            axis=0,
                        ),
                        x=pde_lib.zgrid_k[k, :],
                        axis=0,
                    ),
                    x=pde_lib.tgrid_k[k, :],
                    axis=0,
                )
            )

    opt = STLSQ(normalize_columns=True, alpha=1e-10, threshold=0)
    model = SINDy(optimizer=opt, feature_library=pde_lib)
    model.fit(u_flattened, x_dot=u_dot_integral)
    assert np.any(opt.coef_ != 0.0)
    n_features = len(model.get_feature_names())
    model.fit(u_flattened, x_dot=u_dot_integral, ensemble=True, n_models=10)
    assert np.shape(model.coef_list) == (10, 2, n_features)
    model.fit(u_flattened, x_dot=u_dot_integral, library_ensemble=True, n_models=10)
    assert np.shape(model.coef_list) == (10, 2, n_features)


def test_sindypi_library(data_lorenz):
    x, t = data_lorenz
    x_library_functions = [
        lambda x: x,
        lambda x, y: x * y,
        lambda x: x ** 2,
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
        t=t[1:-1],
        function_names=library_function_names,
        include_bias=True,
    )
    sindy_opt = SINDyPI(threshold=0.1, thresholder="l1")
    model = SINDy(
        optimizer=sindy_opt,
        feature_library=sindy_library,
        differentiation_method=FiniteDifference(drop_endpoints=True),
    )
    model.fit(x, t=t)
    assert np.shape(sindy_opt.coef_) == (40, 40)
    sindy_opt = SINDyPI(threshold=0.1, thresholder="l1", model_subset=[3])
    model = SINDy(
        optimizer=sindy_opt,
        feature_library=sindy_library,
        differentiation_method=FiniteDifference(drop_endpoints=True),
    )
    model.fit(x, t=t)
    assert np.sum(sindy_opt.coef_ == 0.0) == 40.0 * 39.0 and np.any(
        sindy_opt.coef_[3, :] != 0.0
    )

    sindy_library.get_feature_names()
