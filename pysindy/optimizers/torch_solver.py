"""
PyTorch-based SINDy optimizer using proximal gradient + iterative hard-thresholding.

This module provides a high-performance optimizer implemented with PyTorch to solve
Sparse Identification of Nonlinear Dynamics (SINDy) regression problems. It
minimizes a smooth data-fit term and applies proximal/thresholding operations to
promote sparsity in the discovered dynamical system.

Key features
- Batched multi-target regression on CPU or GPU (if available).
- Proximal L1 shrinkage and hard thresholding to encourage sparse models.
- Optional optimizers: SGD, Adam, AdamW, and custom CAdamW.
- Best-solution tracking across iterations and early-stopping support.
- Compatible with PySINDy BaseOptimizer interface and ensembling.

Optional dependencies
- PyTorch is optional at import time; a RuntimeError will be raised during fit if
  torch is not available. Code paths and annotations avoid import-time failures.
- CAdamW is a custom optimizer shipped with the project and used when requested.

Usage
-----
Example: Fit a SINDy model with the Torch optimizer.

    >>> import numpy as np
    >>> from pysindy import SINDy
    >>> from pysindy.feature_library import PolynomialLibrary
    >>> from pysindy.optimizers.torch_solver import TorchOptimizer
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((500, 3))
    >>> Y = X @ np.array([[1.0, 0.0, -0.5], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]).T
    >>> opt = TorchOptimizer(threshold=0.05, alpha_l1=1e-3, step_size=1e-2, max_iter=500, early_stopping_patience=50)
    >>> lib = PolynomialLibrary(degree=2)
    >>> model = SINDy(optimizer=opt, feature_library=lib)
    >>> model.fit(X, t=0.01)
    >>> model.equations()  # doctest: +ELLIPSIS
    [...]

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
from typing import Optional, TYPE_CHECKING

import numpy as np

from .base import BaseOptimizer

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore

if TYPE_CHECKING:  # only for type checkers
    import torch as torch_types  # noqa: F401


import math
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

# Taken from https://github.com/kyleliang919/C-Optim/blob/main/c_adamw.py
class CAdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self.init_lr = lr

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # apply weight decay
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                # compute norm gradient
                mask = (exp_avg * grad > 0).to(grad.dtype)
                # mask = mask * (mask.numel() / (mask.sum() + 1)) ## original implementation, leaving it here for record
                mask.div_(
                    mask.mean().clamp_(min=1e-3)
                )  # https://huggingface.co/rwightman/timm-optim-caution found this implementation is more favourable in many cases
                norm_grad = (exp_avg * mask) / denom
                p.add_(norm_grad, alpha=-step_size)
        return loss


def _soft_threshold(t, lam: float):
    """Soft-thresholding proximal operator.

    Applies element-wise soft-shrinkage: sign(t) * max(|t| - lam, 0).

    Parameters
    - t: torch.Tensor with coefficients.
    - lam: float threshold (lambda) controlling the shrinkage amount.

    Returns
    - torch.Tensor of same shape as `t` after soft-thresholding.
    """
    if lam <= 0:
        return t
    return torch.sign(t) * torch.clamp(torch.abs(t) - lam, min=0.0)


def _hard_threshold(t, thr: float):
    """Hard-thresholding operator.

    Zeros out elements whose absolute value is below the given threshold.

    Parameters
    - t: torch.Tensor with coefficients.
    - thr: float magnitude threshold.

    Returns
    - torch.Tensor of same shape as `t` with small entries set to zero.
    """
    if thr <= 0:
        return t
    return t * (torch.abs(t) >= thr)


class TorchOptimizer(BaseOptimizer):
    """Torch-powered optimizer for sparse SINDy regression.

    This optimizer minimizes the objective

        J(W) = (1/N) * ||Y - X W^T||_F^2 + alpha_l1 * ||W||_1

    using gradient-based updates (SGD/Adam/AdamW/CAdamW), followed by proximal
    soft-thresholding (L1) and hard-thresholding to encourage sparsity. It supports
    multi-target regression (rows correspond to targets) and optional GPU acceleration.

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
        Final coefficients (best observed if early stopping is enabled).
    ind_ : np.ndarray[bool], shape (n_targets, n_features)
        Support mask where True indicates non-zero (above tiny threshold).
    history_ : list[np.ndarray]
        Sequence of coefficient matrices recorded at each iteration.
    intercept_ : float or np.ndarray
        Intercept term (handled by BaseOptimizer; not fit here).

    Notes
    -----
    - Best-solution tracking: the optimizer records the coefficient state with
      the lowest objective value and restores it at the end.
    - Early-stopping: controlled by `early_stopping_patience` and `min_delta`.
    - The objective combines mean-squared error and optional L1 penalty.
    - For reproducible results, set `seed` and avoid non-deterministic CUDA ops.
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
        device: Optional[str] = None,
        seed: Optional[int] = None,
        sparse_ind: Optional[list[int]] = None,
        unbias: bool = True,
        early_stopping_patience: int = 0,
        min_delta: float = 0.0,
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
        if optimizer not in ("sgd", "adam", "adamw", "cadamw"):
            raise ValueError("optimizer must be 'sgd', 'adam', 'adamw', or 'cadamw'")
        if early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")
        if min_delta < 0:
            raise ValueError("min_delta cannot be negative")
        self.threshold = float(threshold)
        self.alpha_l1 = float(alpha_l1)
        self.step_size = float(step_size)
        self.verbose = bool(verbose)
        self.torch_device = device or "cpu"
        self.seed = seed
        self.opt_name = optimizer
        self.sparse_ind = sparse_ind
        self.early_stopping_patience = int(early_stopping_patience)
        self.min_delta = float(min_delta)
        if torch is None:
            # Delay hard failure to fit-time to allow import of module without torch
            warnings.warn(
                "PyTorch is not installed; TorchSINDyOptimizer will not run until torch is available.")

    def _reduce(self, x: np.ndarray, y: np.ndarray) -> None:
        """Core optimization loop.

        This method performs up to `max_iter` iterations of gradient-based updates
        on the smooth objective, followed by proximal soft-thresholding and hard
        thresholding to enforce sparsity. It maintains a history of coefficients,
        tracks the best objective value, and optionally stops early.

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_features)
            Feature matrix (SINDy library output). Normalization may be applied
            by the BaseOptimizer depending on constructor options.
        y : np.ndarray, shape (n_samples, n_targets)
            Target matrix (derivatives).

        Side effects
        ------------
        - Updates `self.coef_` with the best observed coefficients.
        - Updates `self.ind_` as a boolean support mask.
        - Appends snapshots to `self.history_` each iteration.

        Raises
        ------
        ImportError
            If PyTorch is not installed at run time.
        """
        if torch is None:
            raise ImportError("PyTorch is required for TorchSINDyOptimizer. Please install torch.")
        # Select device
        if self.torch_device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA not available; falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(self.torch_device)
        # Seed control
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        # Data to torch
        dtype = torch.float64  # match numpy default precision
        X = torch.as_tensor(x, dtype=dtype, device=device)
        Y = torch.as_tensor(y, dtype=dtype, device=device)
        n_samples, n_features = X.shape
        n_targets = Y.shape[1]

        # Parameter tensor W: shape (n_targets, n_features)
        if self.coef_ is None:
            W = torch.zeros((n_targets, n_features), dtype=dtype, device=device)
        else:
            W = torch.as_tensor(self.coef_, dtype=dtype, device=device)
        W.requires_grad_(True)

        # Optimizer for smooth loss
        if self.opt_name == "adam":
            opt = torch.optim.Adam([W], lr=self.step_size)
        elif self.opt_name == "adamw":
            opt = torch.optim.AdamW([W], lr=self.step_size)
        elif self.opt_name == "cadamw":
            opt = CAdamW([W], lr=self.step_size)
        else:
            opt = torch.optim.SGD([W], lr=self.step_size, momentum=0.9)

        # Support mask helper: restrict thresholding to specified indices
        sparse_mask = None
        if self.sparse_ind is not None:
            sparse_mask = torch.zeros((n_targets, n_features), dtype=torch.bool, device=device)
            sparse_mask[:, self.sparse_ind] = True

        def loss_fn(W_):
            """Compute objective: MSE/N + alpha_l1 * ||W||_1."""
            Y_pred = X @ W_.T  # (n_samples, n_targets)
            residual = Y_pred - Y
            mse = (residual.pow(2)).sum() / n_samples
            if self.alpha_l1 > 0:
                l1 = self.alpha_l1 * torch.abs(W_).sum()
            else:
                l1 = torch.zeros((), dtype=dtype, device=device)
            return mse + l1

        last_mask = None
        best_obj = None
        best_W = None
        patience_counter = 0

        # Simple gradient stepping just as you would do in learning a model on pytorch
        for it in range(self.max_iter):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(W)
            loss.backward()
            # Gradient step via optimizer
            opt.step()
            # Prox for L1
            if self.alpha_l1 > 0:
                with torch.no_grad():
                    W[:] = _soft_threshold(W, self.alpha_l1 * self.step_size)
            # Hard-threshold
            with torch.no_grad():
                if self.threshold > 0:
                    if sparse_mask is None:
                        W[:] = _hard_threshold(W, self.threshold)
                    else:
                        kept = _hard_threshold(W, self.threshold)
                        W[:] = torch.where(sparse_mask, kept, W)
            # Evaluate objective and update best
            with torch.no_grad():
                obj = float(loss_fn(W).cpu().numpy())
                if best_obj is None or (best_obj - obj) > self.min_delta:
                    best_obj = obj
                    best_W = W.detach().clone()
                    patience_counter = 0
                else:
                    patience_counter += 1
            # Track history and early stop if support unchanged or patience exceeded
            with torch.no_grad():
                coef_np = W.detach().cpu().numpy()
                self.history_.append(coef_np)
                mask = np.abs(coef_np) >= max(self.threshold, 1e-14)
                if last_mask is not None and np.array_equal(mask, last_mask):
                    break
                last_mask = mask
                if 0 < self.early_stopping_patience <= patience_counter:
                    break
            if self.verbose and (it % max(1, self.max_iter // 10) == 0 or it == self.max_iter - 1):
                mse_val = float(((X @ W.T - Y).pow(2)).sum().cpu().numpy()) / n_samples
                l0 = int((torch.abs(W) >= self.threshold).sum().item())
                print(f"[TorchSINDy] iter={it} mse={mse_val:.4e} L0={l0} obj={obj:.4e}")

        # Final coefficients back to numpy: use best if available
        final_W = (best_W if best_W is not None else W).detach().cpu().numpy()
        self.coef_ = final_W
        # ind_ based on tiny threshold
        self.ind_ = np.abs(self.coef_) > 1e-14

    @property
    def complexity(self):
        """Model complexity measure.

        Returns the number of non-zero coefficients plus the number of non-zero
        intercepts (if any). This should allow comparing sparsity across optimizers.

        Returns
        -------
        int
            Complexity score: count_nonzero(coef_) + count_nonzero(intercept_).
        """
        return np.count_nonzero(self.coef_) + np.count_nonzero(self.intercept_)
