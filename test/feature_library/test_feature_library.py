"""
Unit tests for feature libraries.
"""
import numpy as np
import pytest
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
from pysindy.feature_library import GeneralizedLibrary
from pysindy.feature_library import IdentityLibrary
from pysindy.feature_library import PDELibrary
from pysindy.feature_library import PolynomialLibrary
from pysindy.feature_library import SINDyPILibrary
from pysindy.feature_library import TensoredLibrary
from pysindy.feature_library import WeakPDELibrary
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
    params["library_functions"] = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
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
        dict(spatiotemporal_grid=range(10), H_xt=11),
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
    params["library_functions"] = [lambda x: x, lambda x: x ** 2, lambda x: 0 * x]
    with pytest.raises(ValueError):
        WeakPDELibrary(**params)


@pytest.mark.parametrize(
    "params",
    [
        dict(libraries=[]),
        dict(libraries=[PolynomialLibrary, WeakPDELibrary]),
        dict(libraries=[PolynomialLibrary, PolynomialLibrary], tensor_array=[[0, 0]]),
        dict(libraries=[PolynomialLibrary, PolynomialLibrary], tensor_array=[[0, 1]]),
        dict(libraries=[PolynomialLibrary, PolynomialLibrary], tensor_array=[[1, -1]]),
        dict(libraries=[PolynomialLibrary, PolynomialLibrary], tensor_array=[[2, 1]]),
        dict(libraries=[PolynomialLibrary, PolynomialLibrary], tensor_array=[1, 1]),
        dict(
            libraries=[PolynomialLibrary, PolynomialLibrary], tensor_array=[[1, 1, 1]]
        ),
        dict(
            libraries=[PolynomialLibrary, PolynomialLibrary],
            inputs_per_library=np.array([[0, 1], [0, 100]]),
        ),
        dict(
            libraries=[PolynomialLibrary, PolynomialLibrary],
            inputs_per_library=np.array([0, 0]),
        ),
        dict(
            libraries=[PolynomialLibrary, PolynomialLibrary],
            inputs_per_library=np.array([[0, 1]]),
        ),
        dict(
            libraries=[PolynomialLibrary, PolynomialLibrary],
            inputs_per_library=np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
        ),
        dict(
            libraries=[PolynomialLibrary, PolynomialLibrary],
            inputs_per_library=np.array([[0, 1, 2], [0, 1, -1]]),
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
        dict(
            library_functions=[lambda x: x, lambda x: x ** 2, lambda x: 0 * x],
            function_names=[lambda s: str(s), lambda s: "{}^2".format(s)],
        ),
        dict(
            x_dot_library_functions=[lambda x: x, lambda x: x ** 2, lambda x: 0 * x],
            function_names=[lambda s: str(s), lambda s: "{}^2".format(s)],
        ),
        dict(x_dot_library_functions=[lambda x: x, lambda x: x ** 2, lambda x: 0 * x]),
        dict(),
        dict(
            library_functions=[lambda x: x, lambda x: x ** 2],
            x_dot_library_functions=[lambda x: x, lambda x: x ** 2],
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
        pytest.lazy_fixture("data_custom_library"),
        pytest.lazy_fixture("data_custom_library_bias"),
        pytest.lazy_fixture("data_generalized_library"),
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
        PolynomialLibrary(include_bias=False),
        FourierLibrary(),
        IdentityLibrary() + PolynomialLibrary(),
        pytest.lazy_fixture("data_custom_library"),
        pytest.lazy_fixture("data_custom_library_bias"),
        pytest.lazy_fixture("data_generalized_library"),
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
        (PolynomialLibrary(include_bias=False), 9),
        (PolynomialLibrary(), 10),
        (IdentityLibrary() + PolynomialLibrary(), 13),
        (FourierLibrary(), 6),
        (pytest.lazy_fixture("data_custom_library_bias"), 13),
        (pytest.lazy_fixture("data_custom_library"), 12),
        (pytest.lazy_fixture("data_generalized_library"), 76),
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
        PolynomialLibrary(include_bias=False),
        FourierLibrary(),
        PolynomialLibrary() + FourierLibrary(),
        pytest.lazy_fixture("data_custom_library"),
        pytest.lazy_fixture("data_custom_library_bias"),
        pytest.lazy_fixture("data_generalized_library"),
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
        ({"include_bias": False}, csr_matrix),
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
        pytest.lazy_fixture("data_custom_library"),
        pytest.lazy_fixture("data_generalized_library"),
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
        pytest.lazy_fixture("data_custom_library"),
        pytest.lazy_fixture("data_generalized_library"),
        pytest.lazy_fixture("data_ode_library"),
        pytest.lazy_fixture("data_pde_library"),
        pytest.lazy_fixture("data_sindypi_library"),
    ],
)
def test_library_ensemble(data_lorenz, library):
    x, t = data_lorenz
    library.fit(x)
    n_output_features = library.n_output_features_
    library.library_ensemble = True
    xp = library.transform(x)
    assert n_output_features == xp.shape[1] + 1
    library.ensemble_indices = [0, 1]
    xp = library.transform(x)
    assert n_output_features == xp.shape[1] + 2
    library.ensemble_indices = np.zeros(1000, dtype=int).tolist()
    with pytest.raises(ValueError):
        xp = library.transform(x)


@pytest.mark.parametrize(
    "library",
    [
        IdentityLibrary,
        PolynomialLibrary,
        FourierLibrary,
    ],
)
def test_bad_library_ensemble(library):
    with pytest.raises(ValueError):
        library = library(ensemble_indices=-1)


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

    inputs_per_library = np.tile([0, 1, 2], 3)
    inputs_per_library = np.reshape(inputs_per_library, (3, 3))
    inputs_per_library[0, 0] = 1
    inputs_per_library[1, 1] = 0
    inputs_per_library[2, 1] = 0
    inputs_per_library[2, 2] = 0

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
        is_uniform=True,
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
        is_uniform=True,
    )
    library_functions = [lambda x: x * x * x]
    library_function_names = [lambda x: x + x + x]
    weak_library2 = WeakPDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=0,
        spatiotemporal_grid=XT,
        is_uniform=True,
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


# Helper function for testing PDE libraries
def pde_library_helper(library, u, coef_first_dim):
    opt = STLSQ(normalize_columns=True, alpha=1e-10, threshold=0)
    model = SINDy(optimizer=opt, feature_library=library)
    model.fit(u)
    assert np.any(opt.coef_ != 0.0)

    n_features = len(model.get_feature_names())
    model.fit(u, ensemble=True, n_subset=50, n_models=10)
    assert np.shape(model.coef_list) == (10, coef_first_dim, n_features)

    model.fit(u, library_ensemble=True, n_models=10)
    assert np.shape(model.coef_list) == (10, coef_first_dim, n_features)


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
        is_uniform=True,
    )
    pde_library_helper(pde_lib, u, 1)


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
        is_uniform=True,
    )
    pde_library_helper(pde_lib, u, 2)


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
        is_uniform=True,
    )
    pde_library_helper(pde_lib, u, 2)


def test_5D_pdes(data_5d_random_pde):
    spatial_grid, u, _ = data_5d_random_pde
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
    pde_library_helper(pde_lib, u, 2)


def test_1D_weak_pdes():
    n = 4
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
        H_xt=0.1,
        include_bias=True,
        K=5,
        is_uniform=False,
        num_pts_per_domain=20,
    )
    pde_library_helper(pde_lib, u, 1)


def test_2D_weak_pdes():
    n = 4
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
        derivative_order=4,
        spatiotemporal_grid=spatiotemporal_grid,
        H_xt=0.1,
        K=2,
        include_bias=True,
        is_uniform=False,
        num_pts_per_domain=10,
    )
    pde_library_helper(pde_lib, u, 1)


def test_3D_weak_pdes():
    n = 4
    t = np.linspace(0, 10, n)
    x = np.linspace(0, 10, n)
    y = np.linspace(0, 10, n)
    z = np.linspace(0, 10, n)
    X, Y, Z, T = np.meshgrid(x, y, z, t, indexing="ij")
    spatiotemporal_grid = np.asarray([X, Y, Z, T])
    spatiotemporal_grid = np.transpose(spatiotemporal_grid, axes=[1, 2, 3, 4, 0])
    u = np.random.randn(n, n, n, n, 2)
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = WeakPDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=4,
        spatiotemporal_grid=spatiotemporal_grid,
        H_xt=0.1,
        K=2,
        include_bias=True,
        is_uniform=False,
        num_pts_per_domain=4,
    )
    pde_library_helper(pde_lib, u, 2)


def test_5D_weak_pdes():
    n = 4
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
    library_functions = [lambda x: x, lambda x: x * x]
    library_function_names = [lambda x: x, lambda x: x + x]
    pde_lib = WeakPDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=2,
        spatiotemporal_grid=spatiotemporal_grid,
        K=2,
        include_bias=True,
        is_uniform=False,
        num_pts_per_domain=4,
    )
    pde_library_helper(pde_lib, u, 2)


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

    sindy_opt = SINDyPI(threshold=1, thresholder="l1", model_subset=[3])
    model = SINDy(
        optimizer=sindy_opt,
        feature_library=sindy_library,
        differentiation_method=FiniteDifference(drop_endpoints=True),
    )
    model.fit(x, t=t)
    assert np.sum(sindy_opt.coef_ == 0.0) == 40.0 * 39.0 and np.any(
        sindy_opt.coef_[3, :] != 0.0
    )
