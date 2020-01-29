"""
Unit tests for feature libraries.
"""
import pytest
from sklearn.exceptions import NotFittedError
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix

from pysindy.feature_library.feature_library import BaseFeatureLibrary
from pysindy.feature_library import CustomLibrary
from pysindy.feature_library import FourierLibrary
from pysindy.feature_library import PolynomialLibrary


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
    "library",
    [
        PolynomialLibrary(),
        FourierLibrary(),
        pytest.lazy_fixture("data_custom_library"),
    ],
)
def test_fit_transform(data_lorenz, library):
    x, t = data_lorenz
    library.fit_transform(x)


@pytest.mark.parametrize(
    "library",
    [
        PolynomialLibrary(),
        FourierLibrary(),
        pytest.lazy_fixture("data_custom_library"),
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
        (PolynomialLibrary(), 10),
        (FourierLibrary(), 6),
        (pytest.lazy_fixture("data_custom_library"), 9),
    ],
)
def test_output_shape(data_lorenz, library, shape):
    x, t = data_lorenz
    y = library.fit_transform(x)
    expected_shape = (x.shape[0], shape)
    assert y.shape == expected_shape

    library.size


@pytest.mark.parametrize(
    "library",
    [
        PolynomialLibrary(),
        FourierLibrary(),
        pytest.lazy_fixture("data_custom_library"),
    ],
)
def test_get_feature_names(data_lorenz, library):
    with pytest.raises(NotFittedError):
        library.get_feature_names()

    x, t = data_lorenz
    library.fit_transform(x)
    library.get_feature_names()

    input_features = ['a'] * x.shape[1]
    library.get_feature_names(input_features=input_features)


# Catch-all for various combinations of options and
# inputs for polynomial features
def test_polynomial_options(data_lorenz):
    x, t = data_lorenz

    # Check sparse inputs
    library = PolynomialLibrary()
    library.fit_transform(csc_matrix(x))
    library.fit_transform(csr_matrix(x))
    library.fit_transform(coo_matrix(x))

    library = PolynomialLibrary(degree=4)
    library.fit_transform(csr_matrix(x))

    library = PolynomialLibrary(include_bias=True)
    library.fit_transform(csr_matrix(x))

    library = PolynomialLibrary(include_interaction=False)
    library.fit_transform(x)

    library = PolynomialLibrary(
        include_interaction=False, include_bias=True
    )


# Catch-all for various combinations of options and
# inputs for Fourier features
def test_fourier_options(data_lorenz):
    x, t = data_lorenz

    library = FourierLibrary(include_cos=False)
    library.fit_transform(x)


def test_not_implemented(data_lorenz):
    x, t = data_lorenz
    library = BaseFeatureLibrary()

    with pytest.raises(NotImplementedError):
        library.fit(x)

    with pytest.raises(NotImplementedError):
        library.transform(x)

    with pytest.raises(NotImplementedError):
        library.get_feature_names(x)
