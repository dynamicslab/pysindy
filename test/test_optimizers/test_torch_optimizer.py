import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pysindy.optimizers import TorchOptimizer
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary


def make_synthetic(n_samples=200, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, 3))
    # True W
    W = np.array([[1.0, 0.0, -0.5], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    Y = X @ W.T + noise * rng.normal(size=(n_samples, 3))
    return X, Y


def test_basic_fit_shapes():
    X, Y = make_synthetic()
    opt = TorchOptimizer(max_iter=50, threshold=1e-2, alpha_l1=1e-3, seed=1)
    opt.fit(X, Y)
    assert opt.coef_.shape == (Y.shape[1], X.shape[1])
    assert opt.ind_.shape == (Y.shape[1], X.shape[1])
    assert len(opt.history_) >= 1


def test_unbias_and_sparsity():
    X, Y = make_synthetic(noise=0.01)
    opt = TorchOptimizer(max_iter=100, threshold=0.05, alpha_l1=1e-3, seed=2, unbias=True)
    opt.fit(X, Y)
    # Check some sparsity present
    assert np.count_nonzero(opt.coef_) < opt.coef_.size


def test_multi_target_with_sindylib():
    # Integration with SINDy and a small library
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 200)
    x = np.stack([
        np.sin(2*np.pi*t),
        np.cos(2*np.pi*t),
        0.5*np.sin(4*np.pi*t)
    ], axis=1)
    lib = PolynomialLibrary(degree=2)
    opt = TorchOptimizer(max_iter=50, threshold=1e-2, alpha_l1=1e-3, seed=0)
    model = SINDy(optimizer=opt, feature_library=lib)
    model.fit(x, t=t[1]-t[0])
    score = model.score(x, t=t[1]-t[0])
    assert score > 0.8

