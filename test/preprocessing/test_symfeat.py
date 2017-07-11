import pytest
import numpy as np
import pickle

import symfeat as sf

@pytest.fixture
def data():
    np.random.seed(42)
    d = np.random.normal(size=(10, 2))
    d[0, 0] = 0
    return d


def test_ConstantFeature(data):
    const = sf.ConstantFeature()
    assert const.name == "1"
    np.testing.assert_allclose(data[:, 0]**0, const.transform(data))


@pytest.mark.parametrize("index", range(2))
@pytest.mark.parametrize("exponent", [1, 2, -1, -2])
def test_SimpleFeature(exponent, data, index):
    simple = sf.SimpleFeature(exponent, index=index)
    np.testing.assert_allclose(data[:, index]**exponent, simple.transform(data))


def test_SimpleFeature_raise():
    with pytest.raises(ValueError):
        sf.SimpleFeature(0)


def test_SimpleFeature_name():
    simple = sf.SimpleFeature(1)
    assert simple.name == "x_0"

    simple = sf.SimpleFeature(2, index=2)
    assert simple.name == "x_2**2"


def test_OperatorFeature(data):
    simple = sf.SimpleFeature(1)

    operator = np.exp
    operator_name = "exp"

    op = sf.OperatorFeature(simple, operator, operator_name=operator_name)

    assert op.name == "{}(x_0)".format(operator_name)
    np.testing.assert_allclose(operator(data[:, simple.index]), op.transform(data))


def test_ProductFeature(data):
    simple10 = sf.SimpleFeature(1, index=0)
    simple11 = sf.SimpleFeature(1, index=1)

    prod = sf.ProductFeature(simple10, simple11)

    assert prod.name == "x_0*x_1"

    np.testing.assert_allclose(data[:, 0]*data[:, 1], prod.transform(data))


def test_SymbolicFeatures(data):
    operators = {"log": np.log}
    exponents = [2]

    sym = sf.SymbolicFeatures(exponents, operators)
    features = sym.fit_transform(data)

    names = sym.names

    assert len(names) == features.shape[1]
    assert features.shape[0] == data.shape[0]
    np.testing.assert_allclose(features[:, 0], np.ones_like(data[:,0]))


def test_SymbolicFeatures_remove_id(data):
    """ProductFeature x_i * x_i**2 == x_i**3
    """
    operators = {}
    exponents = [1, 2, 3]
    sym = sf.SymbolicFeatures(exponents, operators).fit(data)
    # const + simple * 2 + products - excluded
    assert len(sym.names) == 1 + 2*3 + 15 - 2


def test_SymbolicFeatures_redundant_data():
    data = np.ones(shape=(10, 10))
    exponents = [1]
    operators = {}
    sym = sf.SymbolicFeatures(exponents, operators).fit(data)
    assert len(sym.names) == 1


def test_SymbolicFeatures_no_const():
    data = np.ones(shape=(10, 10))
    exponents = [1]
    operators = {}
    sym = sf.SymbolicFeatures(exponents, operators, const=False).fit(data)
    assert sym.names[0] == "x_0"


def test_SymbolicFeatures_pickle(data):
    exponents = [1]
    operators = {}
    sym = sf.SymbolicFeatures(exponents, operators)
    assert pickle.loads(pickle.dumps(sym)).__dict__ == sym.__dict__
    sym.fit(data)
    assert pickle.loads(pickle.dumps(sym)).names == sym.names
