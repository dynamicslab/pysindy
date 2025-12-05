import numpy as np
import pytest

from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import TorchOptimizer

torch = pytest.importorskip("torch")


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
    opt = TorchOptimizer(
        max_iter=100, threshold=0.05, alpha_l1=1e-3, seed=2, unbias=True
    )
    opt.fit(X, Y)
    # Check some sparsity present
    assert np.count_nonzero(opt.coef_) < opt.coef_.size


def test_multi_target_with_sindylib():
    # Integration with SINDy and a small library
    t = np.linspace(0, 1, 200)
    x = np.stack(
        [np.sin(2 * np.pi * t), np.cos(2 * np.pi * t), 0.5 * np.sin(4 * np.pi * t)],
        axis=1,
    )
    lib = PolynomialLibrary(degree=2)
    opt = TorchOptimizer(max_iter=50, threshold=1e-2, alpha_l1=1e-3, seed=0)
    model = SINDy(optimizer=opt, feature_library=lib)
    model.fit(x, t=t[1] - t[0])
    score = model.score(x, t=t[1] - t[0])
    assert score > 0.8


# New tests for broader coverage


def test_sparse_ind_hard_thresholding_effect():
    # With extremely large threshold and one iteration, only columns listed in
    # sparse_ind should be hard-thresholded to zero; others should retain small
    # non-zero values from the single gradient step.
    X, Y = make_synthetic(n_samples=120, noise=0.0, seed=42)
    thr = 1e9
    opt = TorchOptimizer(
        max_iter=1,
        step_size=1e-3,
        threshold=thr,
        alpha_l1=0.0,
        seed=0,
        sparse_ind=[0],  # only first feature is forced to zero by hard threshold
    )
    opt.fit(X, Y)
    # Column 0 should be exactly zero (hard-thresholded)
    assert np.allclose(opt.coef_[:, 0], 0.0)
    # At least one coefficient outside column 0 should remain non-zero
    assert np.any(np.abs(opt.coef_[:, 1:]) > 0.0)


@pytest.mark.parametrize("opt_name", ["sgd", "adam", "adamw", "cadamw"])
def test_optimizer_variants_run(opt_name):
    X, Y = make_synthetic(n_samples=100, noise=0.01, seed=3)
    opt = TorchOptimizer(
        optimizer=opt_name, max_iter=30, threshold=1e-3, alpha_l1=1e-4, seed=1
    )
    opt.fit(X, Y)
    assert opt.coef_.shape == (Y.shape[1], X.shape[1])


def test_early_stopping_via_patience_and_min_delta():
    # Use a huge min_delta so improvements never count; should stop after
    # early_stopping_patience iterations instead of max_iter.
    X, Y = make_synthetic(n_samples=150, noise=0.01, seed=4)
    patience = 2
    opt = TorchOptimizer(
        max_iter=200,
        threshold=0.0,
        alpha_l1=0.0,
        seed=0,
        early_stopping_patience=patience,
        min_delta=1e9,
    )
    opt.fit(X, Y)
    # history_ includes the initial entry (from BaseOptimizer) + per-iteration appends
    # Ensure we stopped well before max_iter due to patience
    assert len(opt.history_) <= 1 + patience + 1  # a few iterations at most


def test_cuda_device_selection_warning_or_success():
    X, Y = make_synthetic(n_samples=50, noise=0.0, seed=5)
    if not torch.cuda.is_available():
        with pytest.warns(UserWarning, match="CUDA not available; falling back to CPU"):
            opt = TorchOptimizer(device="cuda", max_iter=5, threshold=0.0, seed=0)
            opt.fit(X, Y)
            assert opt.coef_.shape == (Y.shape[1], X.shape[1])
    else:
        opt = TorchOptimizer(device="cuda", max_iter=10, threshold=1e-4, seed=0)
        opt.fit(X, Y)
        assert opt.coef_.shape == (Y.shape[1], X.shape[1])


def test_complexity_property_matches_manual():
    X, Y = make_synthetic(n_samples=120, noise=0.02, seed=6)
    opt = TorchOptimizer(max_iter=60, threshold=5e-2, alpha_l1=1e-3, seed=6)
    opt.fit(X, Y)
    manual = np.count_nonzero(opt.coef_) + np.count_nonzero(opt.intercept_)
    assert opt.complexity == manual


def test_verbose_prints_progress(capsys):
    X, Y = make_synthetic(n_samples=60, noise=0.0, seed=7)
    opt = TorchOptimizer(max_iter=5, threshold=1e-4, alpha_l1=0.0, verbose=True, seed=0)
    opt.fit(X, Y)
    captured = capsys.readouterr().out
    assert "[TorchSINDy]" in captured and "iter=" in captured


def test_history_tracking_shapes_and_length():
    X, Y = make_synthetic(n_samples=80, noise=0.0, seed=8)
    # Provide a zero initial guess so the first history entry (initial) certainly
    # differs from later iterates after optimization updates.
    init = np.zeros((Y.shape[1], X.shape[1]))
    opt = TorchOptimizer(
        max_iter=8, threshold=1e-4, alpha_l1=1e-4, seed=0, initial_guess=init
    )
    opt.fit(X, Y)
    # History should contain the initial coefficients + at least one update
    assert isinstance(opt.history_, list)
    assert len(opt.history_) >= 2
    shapes_ok = all(h.shape == opt.coef_.shape for h in opt.history_)
    assert shapes_ok
    # The first history entry (initial guess) should differ from a later iterate
    assert not np.allclose(opt.history_[0], opt.history_[-1])
