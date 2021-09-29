import pytest
from numpy.testing import assert_allclose
from sklearn import __version__
from sklearn.exceptions import NotFittedError

from pysindy import FourierLibrary
from pysindy import SINDy
from pysindy import STLSQ
from pysindy.deeptime import SINDyEstimator
from pysindy.deeptime import SINDyModel


def test_estimator_has_model(data_lorenz):
    x, t = data_lorenz

    estimator = SINDyEstimator()
    assert not estimator.has_model

    estimator.fit(x, t=t)
    assert estimator.has_model


def test_estimator_fetch_model(data_lorenz):
    x, t = data_lorenz

    estimator = SINDyEstimator()
    assert estimator.fetch_model() is None

    estimator.fit(x, t=t)
    assert isinstance(estimator.fetch_model(), SINDyModel)


def test_model_sindy_equivalence(data_lorenz_c_1d):
    x, t, u, _ = data_lorenz_c_1d

    model = SINDyEstimator().fit(x, t=t, u=u).fetch_model()
    sindy_model = SINDy().fit(x, t=t, u=u)

    assert_allclose(model.coefficients(), sindy_model.coefficients())
    print(sindy_model.n_features_in_)
    if float(__version__[:3]) >= 1.0:
        assert model.n_features_in_ == sindy_model.n_features_in_
    else:
        assert model.n_input_features_ == sindy_model.n_input_features_
    assert model.n_output_features_ == sindy_model.n_output_features_
    assert model.n_control_features_ == sindy_model.n_control_features_


def test_model_has_sindy_methods(data_lorenz):
    x, t = data_lorenz
    model = SINDyEstimator().fit(x, t=t).fetch_model()

    assert hasattr(model, "predict")
    assert hasattr(model, "simulate")
    assert hasattr(model, "score")
    assert hasattr(model, "print")
    assert hasattr(model, "equations")


def test_model_unfitted_library(data_derivative_2d):
    x, x_dot = data_derivative_2d
    optimizer = STLSQ().fit(x, x_dot)
    library = FourierLibrary()

    with pytest.raises(NotFittedError):
        SINDyModel(optimizer, library)


def test_model_unfitted_optimizer(data_lorenz):
    x, t = data_lorenz
    optimizer = STLSQ()
    library = FourierLibrary().fit(x)

    with pytest.raises(NotFittedError):
        SINDyModel(optimizer, library)


def test_model_copy(data_lorenz):
    x, t = data_lorenz
    model = SINDyEstimator().fit(x, t=t).fetch_model()
    model_copy = model.copy()

    assert model is not model_copy
