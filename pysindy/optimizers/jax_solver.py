"""
JAX-based SINDy optimizer using proximal gradient + iterative hard-thresholding.

This module provides a high-performance optimizer implemented with JAX to solve
Sparse Identification of Nonlinear Dynamics (SINDy) regression problems. It
minimizes a smooth data-fit term and applies proximal/thresholding operations to
promote sparsity in the discovered dynamical system.

Key features
- Batched multi-target regression on CPU or GPU/TPU (if available).
- Proximal L1 shrinkage and hard thresholding to encourage sparse models.
- Optimizers: SGD, Adam, AdamW with minimal implementations in JAX.
-   Note: The CAdamW variant is currently not available in this implementation.
- Best-solution tracking across iterations and early-stopping support.
- Compatible with PySINDy BaseOptimizer interface and ensembling.

Optional dependencies
- JAX is optional at import time; an ImportError will be raised during fit if
  jax is not available. Code paths and annotations avoid import-time failures.

Notes
-----
- Thresholding and proximal operations operate on coefficient magnitudes. A small
  numerical threshold (1e-14) is used to derive support masks for `ind_`.
- When `sparse_ind` is provided, thresholding affects only the specified columns.
- Early stopping halts iterations when the objective fails to improve by at least
  `min_delta` for `early_stopping_patience` consecutive steps.
- The optimizer tracks and restores the best solution observed across iterations.
"""
import warnings
from typing import List
from typing import Optional

import numpy as np

from .base import BaseOptimizer

try:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    jax = None  # type: ignore
    jnp = None  # type: ignore


def _soft_threshold(t: jnp.ndarray, lam: float):
    if lam <= 0:
        return t
    return jnp.sign(t) * jnp.maximum(jnp.abs(t) - lam, 0.0)


def _hard_threshold(t: jnp.ndarray, thr: float):
    if thr <= 0:
        return t
    return t * (jnp.abs(t) >= thr)


class JaxOptimizer(BaseOptimizer):
    """JAX-powered optimizer for sparse SINDy regression.

    Objective
        J(W) = (1/N) * ||Y - X W^T||_F^2 + alpha_l1 * ||W||_1

    Parameters
    ----------
    threshold : float, default 1e-1
        Minimum magnitude for a coefficient. Values with |coef| < threshold
        are set to zero after each iteration (hard-threshold).
    alpha_l1 : float, default 0.0
        L1 penalty weight. If > 0, soft-thresholding is applied after the
        gradient step to shrink coefficients.
    step_size : float, default 1e-1
        Learning rate for the chosen Torch optimizer.
    max_iter : int, default 1000
        Maximum number of iterations.
    optimizer : {"sgd", "adam", "adamw", "cadamw"}, default "adam"
        Which optimizer to use for the smooth part of the objective.
    normalize_columns : bool, default False
        See BaseOptimizer; if True, columns of X are normalized before fitting.
    copy_X : bool, default True
        See BaseOptimizer; controls whether X is copied or may be overwritten.
    initial_guess : np.ndarray or None, default None
        Warm-start coefficients; shape (n_targets, n_features).
    verbose : bool, default False
        If True, prints periodic progress including loss and sparsity.
    device : {"cpu", "cuda"} or None, default None
        Torch device to use. If None, uses CPU; if "cuda" is requested but not
        available, falls back to CPU with a warning.
    seed : int or None, default None
        Random seed for reproducibility of Torch and NumPy.
    sparse_ind : list[int] or None, default None
        If provided, thresholding only applies to these feature indices. Other
        indices remain unaffected by hard-thresholding.
    unbias : bool, default True
        See BaseOptimizer; when True, performs an unbiased refit on the selected
        support after optimization.
    early_stopping_patience : int, default 0
        If > 0, stop early when the objective has not improved by at least
        `min_delta` for this many consecutive iterations.
    min_delta : float, default 0.0
        Minimum improvement to reset patience; small positive values help
        prevent stopping on floating-point noise.
    Attributes
    ----------
    coef_ : np.ndarray, shape (n_targets, n_features)
        Optimized SINDy coefficients.
    history_ : list of np.ndarray
        Coefficient history; `history_[k]` is the coefficient matrix after
        iteration k.
    ind_ : np.ndarray, shape (n_targets, n_features)
        Boolean mask of nonzero coefficients (|coef| > 1e-14).
    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import JaxOptimizer
    >>> lorenz = lambda z,t : [10 * (z[1] - z[0]),
    >>>                        z[0] * (28 - z[2]) - z[1],
    >>>                        z[0] * z[1] - 8 / 3 * z[2]]
    >>> t = np.arange(0, 2, .002)
    >>> x = odeint(lorenz, [-8, 8, 27], t)
    >>> opt = JaxOptimizer(threshold=.1, alpha_l1=.01, max_iter=1000)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1] - t[0])
    >>> model.print()

    x0' = -9.973 x0 + 9.973 x1
    x1' = -0.129 1 + 27.739 x0 + -0.949 x1 + -0.993 x0 x2
    x2' = -2.656 x2 + 0.996 x0 x1
    """

    def __init__(
        self,
        threshold: float = 1e-1,
        alpha_l1: float = 0.0,
        step_size: float = 1e-1,
        max_iter: int = 1000,
        optimizer: str = "adam",
        normalize_columns: bool = False,
        copy_X: bool = True,
        initial_guess: Optional[np.ndarray] = None,
        verbose: bool = False,
        seed: Optional[int] = None,
        sparse_ind: Optional[List[int]] = None,
        unbias: bool = True,
        early_stopping_patience: int = 100,
        min_delta: float = 1e-10,
    ):
        super().__init__(
            max_iter=max_iter,
            normalize_columns=normalize_columns,
            initial_guess=initial_guess,
            copy_X=copy_X,
            unbias=unbias,
        )
        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        if alpha_l1 < 0:
            raise ValueError("alpha_l1 cannot be negative")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        if optimizer not in ("sgd", "adam", "adamw"):
            raise ValueError("optimizer must be 'sgd', 'adam', or 'adamw'")
        if early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")
        if min_delta < 0:
            raise ValueError("min_delta cannot be negative")
        self.threshold = float(threshold)
        self.alpha_l1 = float(alpha_l1)
        self.step_size = float(step_size)
        self.verbose = bool(verbose)
        self.seed = seed
        self.opt_name = optimizer
        self.sparse_ind = sparse_ind
        self.early_stopping_patience = int(early_stopping_patience)
        self.min_delta = float(min_delta)
        self.stability_eps = 1e-14

        if jax is None:
            warnings.warn(
                "JAX is not installed; "
                "JaxOptimizer will not run until jax is available."
            )

    def _reduce(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        if jax is None:
            raise ImportError("JAX is required for JaxOptimizer. Please install jax.")

        # Seed control
        if self.seed is not None:
            # jax uses PRNG keys.
            # We just seed numpy for determinism in thresholds/history
            np.random.seed(self.seed)

        X = jnp.asarray(x, dtype=jnp.float64)
        Y = jnp.asarray(y, dtype=jnp.float64)
        n_samples, n_features = X.shape
        n_targets = Y.shape[1]

        if self.coef_ is None:
            W = jnp.zeros((n_targets, n_features), dtype=jnp.float64)
        else:
            W = jnp.asarray(self.coef_, dtype=jnp.float64)

        # sparse mask
        sparse_mask = None
        if self.sparse_ind is not None:
            sparse_mask = jnp.zeros((n_targets, n_features), dtype=bool)
            sparse_mask = sparse_mask.at[:, self.sparse_ind].set(True)

        def loss_fn(W_):
            Y_pred = X @ W_.T
            residual = Y_pred - Y
            mse = jnp.sum(residual**2) / n_samples
            l1 = self.alpha_l1 * jnp.sum(jnp.abs(W_)) if self.alpha_l1 > 0 else 0.0
            return mse + l1

        grad_fn = jax.grad(lambda W_: loss_fn(W_))

        # Optimizer states for Adam/AdamW
        m = jnp.zeros_like(W)
        v = jnp.zeros_like(W)
        beta1, beta2, eps, wd = 0.9, 0.999, 1e-8, 0.0
        if self.opt_name == "adamw":
            wd = 1e-4  # small default weight decay

        last_mask = None
        best_obj = None
        best_W = None
        patience_counter = 0

        def step_adam(W, m, v, g, t):
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g * g)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            W = W - self.step_size * m_hat / (jnp.sqrt(v_hat) + eps)
            if wd > 0:
                W = W - self.step_size * wd * W
            return W, m, v

        # Loop
        for it in range(self.max_iter):
            g = grad_fn(W)
            if self.opt_name == "sgd":
                W = W - self.step_size * g
            else:
                W, m, v = step_adam(W, m, v, g, it + 1)

            # proximal L1
            if self.alpha_l1 > 0:
                W = _soft_threshold(W, self.alpha_l1 * self.step_size)
            # hard threshold
            if self.threshold > 0:
                kept = _hard_threshold(W, self.threshold)
                if sparse_mask is None:
                    W = kept
                else:
                    W = jnp.where(sparse_mask, kept, W)

            # evaluate objective
            obj = float(loss_fn(W))
            if best_obj is None or obj < (best_obj - self.min_delta):
                best_obj = obj
                best_W = W
                patience_counter = 0
            else:
                patience_counter += 1

            # track history and early stop
            coef_np = np.array(W)
            self.history_.append(coef_np)
            mask = np.abs(coef_np) >= max(self.threshold, self.stability_eps)
            if last_mask is not None and np.array_equal(mask, last_mask):
                break
            last_mask = mask
            if 0 < self.early_stopping_patience <= patience_counter:
                break

            if self.verbose and (
                it % max(1, self.max_iter // 10) == 0 or it == self.max_iter - 1
            ):
                mse_val = float(np.sum((np.array(X @ W.T - Y)) ** 2) / n_samples)
                l0 = int(np.sum(np.abs(coef_np) >= self.threshold))
                print(f"[JaxSINDy] iter={it} mse={mse_val:.4e} L0={l0} obj={obj:.4e}")

        final_W = np.array(best_W if best_W is not None else W)
        self.coef_ = final_W
        self.ind_ = np.abs(self.coef_) > self.stability_eps

    @property
    def complexity(self):
        return np.count_nonzero(self.coef_) + np.count_nonzero(self.intercept_)
