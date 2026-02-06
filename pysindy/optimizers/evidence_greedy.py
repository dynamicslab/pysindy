"""
EvidenceGreedy optimizer: greedy Bayesian evidence-based sparse regression.

See :class:`pysindy.optimizers.EvidenceGreedy` for full documentation.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ridge_regression

from .base import BaseOptimizer
from .base import _normalize_features


class EvidenceGreedy(BaseOptimizer):
    """
    Backward evidence-based sparse regression for SINDy.

    This optimizer performs backward feature elimination driven by the
    Bayesian log evidence for a linear Gaussian model with an isotropic
    Gaussian prior on the coefficients. For each target dimension y_j,
    we assume

    .. math::

        w &\\sim \\mathcal{N}(0, \\alpha^{-1} I), \\\\
        y_j \\mid w &\\sim \\mathcal{N}(\\Theta w, \\sigma^2 I),

    where ``alpha`` is the prior precision on the coefficients
    (sigma_p^{-2}) and ``_sigma2`` is the observation noise variance
    (sigma^2).

    #TODO: make it more ml style than statistics look at STLSQ

    The algorithm:

      1. Starts from the full support (all library terms active).
      2. At each step, temporarily removes each active term in turn.
      3. For each candidate support, computes the Bayesian log evidence
         log p(y_j | alpha, _sigma2, support) using precomputed
         statistics G = Theta^T Theta and b_j = Theta^T y_j.
      4. Accepts the removal that yields the largest increase in evidence.
      5. Stops when no single removal increases the evidence.

    Parameters
    ----------
    alpha : float, default=1.0
        Prior precision on the coefficients (sigma_p^{-2}). Must be positive.
        The prior is defined in the feature space actually used by the
        optimizer. In particular, when ``normalize_columns=True``, ``alpha``
        controls an isotropic Gaussian prior on the coefficients in the
        normalized library. Changing ``normalize_columns`` without retuning
        ``alpha`` will generally change the effective strength of the
        regularization.

    _sigma2 : float, default= (float precision**2)
        Observation noise variance (sigma^2). Must be positive.

    max_iter : int or None
        Maximum number of elimination steps. If None, at most M - 1
        removals are allowed.



    normalize_columns : bool, default=True
        Passed to :class:`~pysindy.optimizers.base.BaseOptimizer`. If True,
        BOTH the columns of the library matrix AND the target variables are normalized before regression.
        The Bayesian prior and ridge penalty are then applied in this
        normalized space. The learned coefficients are mapped back
        to the original scale when stored in ``coef_``.
        Note that when normalized_columns is True, the ``alpha`` is typically of order 1.0.

    copy_X : bool, default=True
        Passed to :class:`~pysindy.optimizers.base.BaseOptimizer`. If True,
        input data are copied.

    initial_guess : array-like of shape (n_targets, n_features) or None, \
            default=None
        Currently ignored by the greedy algorithm; present for API compatibility
        with :class:`~pysindy.optimizers.base.BaseOptimizer`.

    unbias : bool, default=False
        Whether to perform an additional unregularized refit after support
        selection. For a Bayesian evidence interpretation the regularized
        posterior mean is natural, so the default is False.

    verbose : bool, default=False
        If True, prints a short trace of evidence values during backward
        elimination for each target dimension.

    Attributes
    ----------
    coef_ : ndarray of shape (n_targets, n_features)
        Final coefficient matrix Xi. Row i contains the coefficients for
        the i-th target variable, with zeros outside the selected support.

    ind_ : ndarray of bool of shape (n_targets, n_features)
        Boolean support mask corresponding to ``coef_``. ``ind_[i, j]`` is
        True if the j-th library function is active in the equation for the
        i-th target.

    history_ : list of ndarray
        Minimal coefficient history kept for compatibility with other
        optimizers. By convention ``history_[-1]`` is the final coefficient
        matrix ``coef_``.

    evidence_history_ : list of list of dict
        Per-target evidence traces. ``evidence_history_[i]`` is a list of
        dictionaries recording the support size and log evidence at each
        backward-elimination step for the i-th target, e.g.::

            {"step": k,
             "removed": j,
             "support_size": K,
             "log_evidence": value}

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import EvidenceGreedy
    >>>
    >>> # Lorenz system
    >>> lorenz = lambda z, t: [
    ...     10 * (z[1] - z[0]),
    ...     z[0] * (28 - z[2]) - z[1],
    ...     z[0] * z[1] - 8 / 3 * z[2],
    ... ]
    >>> t = np.arange(0, 10, 0.01)
    >>> x = odeint(lorenz, [-8, 8, 27], t)
    >>>
    >>> # Add noise to the measurements
    >>> sigma_x = 1e-2
    >>> x = x + sigma_x * np.random.normal(size=x.shape)
    >>>
    >>> opt = EvidenceGreedy(alpha=1e-6, max_iter=20, normalize_columns=False)
    >>> model = BINDy(optimizer=opt)
    >>> model.fit(x, t=t[1] - t[0])
    >>> model.print()

    (x0)' = -9.979 x0 + 9.980 x1\
    (x1)' = 27.807 x0 + -0.963 x1 + -0.995 x0 x2\
    (x2)' = -2.658 x2 + 0.997 x0 x1 
    """

    def __init__(
        self,
        alpha: float = 1.0,
        _sigma2: float = (np.finfo(float).eps) ** 2,
        max_iter: int | None = None,
        normalize_columns: bool = True,
        copy_X: bool = True,
        initial_guess: np.ndarray | None = None,
        unbias: bool = False,
        verbose: bool = False,
    ):

        if max_iter is not None and max_iter <= 0:
            raise ValueError("max_iter must be positive or None.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        if _sigma2 <= 0:
            raise ValueError("_sigma2 (noise variance) must be positive.")

        # Treat max_iter=None as no limit, but BaseOptimizer requires a positive int
        if max_iter is None:
            max_iter = sys.maxsize
        elif max_iter <= 0:
            raise ValueError("max_iter must be a positive integer or None")

        self.alpha = float(alpha)
        self._sigma2 = float(_sigma2)
        self.verbose = bool(verbose)
        self.max_iter = max_iter

        super().__init__(
            max_iter=max_iter,
            normalize_columns=normalize_columns,
            initial_guess=initial_guess,  # TODO: Documentation
            copy_X=copy_X,  # TODO: just leave
            unbias=unbias,
        )

    @staticmethod
    def TemporalNoisePropagation(
        differentiator,
        t,
        sigma_x: float,
    ) -> float:
        """
        Estimate the derivative noise variance _sigma2 induced by a
        finite-difference differentiator.

        This treats ``differentiator._differentiate`` as a linear operator
        x -> L_dt x. By sending an identity matrix of size T through the
        operator, we reconstruct the finite-difference matrix L_dt and use

            Var[eta_k] = sigma_x**2 * sum_j L_dt[k, j]**2

        for the induced derivative noise at row k. The returned _sigma2 is
        the average of this variance over all rows that contain only
        finite values.
        # TODO: support pointwise noise variance in future versions.

        Parameters
        ----------
        differentiator : object
            Differentiation object with a ``_differentiate(X, t)`` method
            and an ``axis`` attribute, e.g.
            :class:`pysindy.differentiation.FiniteDifference`.
        t : array_like of shape (n_samples,)
            Time grid passed to the differentiator.
        sigma_x : float
            Standard deviation of the additive measurement noise on the
            state x(t). Must be non-negative.

        Returns
        -------
        _sigma2 : float
            Estimated variance of the induced noise on the differentiated
            signal.
        """
        t = np.asarray(t)
        if t.ndim != 1:
            raise ValueError("t must be a 1D time grid.")
        if sigma_x < 0:
            raise ValueError("sigma_x must be non-negative.")

        n_samples = t.shape[0]
        X_probe = np.eye(n_samples, dtype=float)

        # Reconstruct L_dt as the image of the identity under the operator.
        L_dt = differentiator._differentiate(X_probe, t)

        if L_dt.shape != (n_samples, n_samples):
            raise RuntimeError(
                "Unexpected shape from differentiator._differentiate; "
                f"expected ({n_samples}, {n_samples}), got {L_dt.shape}."
            )

        # Some boundary rows may be NaN depending on drop_endpoints / periodic.
        # Here, we remove those rows from the estimate.
        finite_row_mask = np.all(np.isfinite(L_dt), axis=1)
        if not np.any(finite_row_mask):
            raise RuntimeError(
                "Could not find any rows of the finite-difference operator "
                "without NaNs; check differentiator settings."
            )

        # In this version, we use the mean variance over all valid time points as the noise estimate after propagation.
        # TODO: Support pointwise noise variance in future versions.
        row_norm_sq = np.sum(L_dt[finite_row_mask] ** 2, axis=1)
        factor = float(np.mean(row_norm_sq))

        return float(sigma_x**2 * factor)

    def _unbias(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Optional unregularized refit on the selected support.

        This mirrors the STLSQ behaviour in the simple case:
        keep the support ``ind_`` found by the evidence-greedy search,
        but recompute the coefficients on that support by ordinary
        least squares (no ridge penalty).

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Library matrix Theta(X) after preprocessing / normalization
            by :class:`BaseOptimizer`.
        y : ndarray of shape (n_samples, n_targets)
            Target derivatives after preprocessing.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_features = x.shape
        _, n_targets = y.shape

        if self.coef_.shape != (n_targets, n_features):
            raise RuntimeError(
                "EvidenceGreedy._unbias: unexpected coef_ shape "
                f"{self.coef_.shape}, expected {(n_targets, n_features)}."
            )

        # For each target dimension, refit LS on its active columns.
        for i in range(n_targets):
            active_mask = self.ind_[i]
            if not np.any(active_mask):
                # No active terms for this target; nothing to refit.
                continue

            X_active = x[:, active_mask]
            y_i = y[:, i]

            # STLSQ style: use LinearRegression(fit_intercept=False)
            optvar = LinearRegression(fit_intercept=False).fit(X_active, y_i).coef_

            # Overwrite only active coefficients; inactive remain zero.
            self.coef_[i, active_mask] = optvar

    def _reduce(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Run backward evidence selection for each target dimension.

        Parameters
        ----------
        x : ndarray of shape (n_samples, n_features)
            Library matrix Theta(X). This has already been preprocessed
            by BaseOptimizer (and may be normalized).

        y : ndarray of shape (n_samples, n_targets)
            Target derivatives.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        n_samples, n_features = x.shape  # T, M
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_targets = y.shape[1]  # N

        # BaseOptimizer only normalise the library, but for the Bayesian framework, we also need to normalize the targets. Normalising this help make sure the parameter is also rescaled to unit order.
        if self.normalize_columns:
            y_norm, y = _normalize_features(y)

        # Shared Gram matrix and RHS for all outputs:
        G = x.T @ x  # (M, M) = Theta^T Theta
        B = x.T @ y  # (M, N) = Theta^T Y
        yTy_all = np.sum(y**2, axis=0)  # (N,) = [y_j^T y_j]

        coef = np.zeros((n_targets, n_features), dtype=float)
        ind = np.zeros((n_targets, n_features), dtype=bool)
        all_histories: list[list[dict[str, float]]] = []

        for j in range(n_targets):
            b = B[:, j]  # (M,)
            yTy = float(yTy_all[j])  # scalar
        
            eps = float(np.finfo(float).eps)
            if (not np.isfinite(yTy)) or (yTy <= eps):
                coef[j, :] = 0.0
                ind[j, :] = False

                # Log evidence of the empty model (K=0). m_N is ignored for K=0.
                log_ev = _log_evidence_from_G(
                    G_active=np.zeros((0, 0), dtype=float),
                    b_active=np.zeros((0,), dtype=float),
                    yTy=yTy,
                    n_samples=n_samples,
                    alpha=self.alpha,
                    _sigma2=float(self._sigma2),
                    m_N=None,
                )
                history_j = [
                    {
                        "step": 0,
                        "support_size": 0,
                        "log_evidence": float(log_ev),
                    }
                ]
                all_histories.append(history_j)

                # Keep history_ format consistent: one snapshot for this target
                history_tmp = np.full((n_targets, n_features), np.nan, dtype=float)
                history_tmp[j, :] = 0.0
                self.history_.append(history_tmp)
                continue

            # Since the target (Y) is also possibly normalized, we need to rescale _sigma2 accordingly.
            if self.normalize_columns:
                yn = float(y_norm[j])
                # Prevent divide-by-zero / inf inflation
                denom = max(yn * yn, eps)
                _sigma2_ = float(self._sigma2) / denom
            else:
                _sigma2_ = float(self._sigma2)

            coef_j, ind_j, history_j, coef_hist = _backward_evidence_greedy_single(
                x=x,
                y_col=y[:, j],
                G=G,
                b=b,
                yTy=yTy,
                n_samples=n_samples,
                alpha=self.alpha,
                _sigma2=_sigma2_,
                max_iter=self.max_iter,
                verbose=self.verbose,
            )

            coef[j, :] = coef_j
            ind[j, :] = ind_j
            all_histories.append(history_j)

            ## For history, we need to reshape to match the format of other optimizers.
            for i in range(np.shape(coef_hist)[1]):
                history_tmp = np.full((n_targets, n_features), np.nan, dtype=float)
                history_tmp[j,:] = coef_hist[:,i]
                self.history_.append(history_tmp)

        self.coef_ = coef
        self.ind_ = ind

        # Map coefficients back to original scale if normalized.
        if self.normalize_columns:
            self.coef_ = self.coef_ * y_norm.reshape(-1, 1)

        # Expose full evidence traces if required.
        self.evidence_history_ = all_histories


def _ridge_map(
    X_active: np.ndarray,
    y_active: np.ndarray,
    alpha_prior: float,
    _sigma2: float,
    ridge_kw: dict | None = None,
) -> np.ndarray:
    """
    Compute the MAP coefficients for a given active set using ridge regression.

    This solves the ridge problem

        argmin_w ||y - X w||^2 + lambda ||w||^2,

    where lambda = alpha_prior * _sigma2, corresponding to a Gaussian
    prior w ~ N(0, alpha_prior^{-1} I) and noise variance _sigma2.

    Any LinAlgWarning raised by the underlying solver is converted into a
    RuntimeWarning, but the returned coefficients are still used.
    """
    X_active = np.asarray(X_active)
    y_active = np.asarray(y_active).ravel()

    lam = alpha_prior * _sigma2
    kw = ridge_kw or {}

    # Follow the STLSQ pattern: use ridge_regression and handle LinAlgWarning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings("always", category=LinAlgWarning)
        coef = ridge_regression(X_active, y_active, lam, **kw)

    # If any LinAlgWarning occurred, surface a warning to the user but continue.
    for w in caught:
        if issubclass(w.category, LinAlgWarning):
            warnings.warn(
                "EvidenceGreedy: linear algebra warning encountered while "
                "computing MAP coefficients; results may be unreliable.",
                RuntimeWarning,
            )
            break

    return coef


def _log_evidence_from_G(
    G_active: np.ndarray,
    b_active: np.ndarray,
    yTy: float,
    n_samples: int,
    alpha: float,
    _sigma2: float,
    m_N: np.ndarray | None,
) -> float:
    """
    Compute the Bayesian log evidence for a given active set and posterior mean.

    Notation:

      - y in R^T, Theta in R^{T x M}
      - alpha = sigma_p^{-2}
      - beta = sigma^{-2}
      - G = Theta^T Theta,  b = Theta^T y,  yTy = y^T y
      - Lambda = alpha I_M + beta G_active
      - m_N is the posterior mean on the active set

    Evidence approximation:

        log p(y) =
            -1/2 [ T log(2 pi) + T log sigma^2
                   + log|Lambda| - M log alpha
                   + beta ||y - Theta m_N||^2
                   + alpha ||m_N||^2 ]

    where

        ||y - Theta m_N||^2
            = yTy - 2 m_N^T b_active + m_N^T G_active m_N

    Parameters
    ----------
    G_active : ndarray, shape (K, K)
        Gram matrix for active features.

    b_active : ndarray, shape (K,)
        Theta^T y restricted to active features.

    yTy : float
        y^T y (scalar).

    n_samples : int
        T, number of time samples.

    alpha : float
        Prior precision on weights.

    _sigma2 : float
        Observation noise variance.

    m_N : ndarray of shape (K,) or None
        Posterior mean coefficients for the active set. For the empty
        model (K == 0), this is ignored and may be None.

    Returns
    -------
    log_ev : float
        Bayesian log evidence.
    """
    G_active = np.asarray(G_active)
    b_active = np.asarray(b_active)

    K = G_active.shape[0]

    # Degenerate empty model: p(y) = N(0, _sigma2 I)
    if K == 0:
        term1 = n_samples * np.log(2.0 * np.pi)
        term2 = n_samples * np.log(_sigma2)
        term3 = (1.0 / _sigma2) * yTy
        log_ev = -0.5 * (term1 + term2 + term3)
        return float(log_ev)

    if m_N is None:
        raise ValueError("m_N must be provided for a non-empty active set.")

    m_N = np.asarray(m_N).reshape(-1)
    if m_N.shape[0] != K:
        raise ValueError("m_N has incompatible shape for the active set.")

    beta = 1.0 / _sigma2

    # Residual norm using precomputed stats:
    #   ||y - Theta m_N||^2 = yTy - 2 m_N^T b_active + m_N^T G_active m_N
    residual_sq = yTy - 2.0 * float(m_N.T @ b_active) + float(m_N.T @ (G_active @ m_N))

    # log|Lambda|
    Lambda = alpha * np.eye(K) + beta * G_active
    sign, logdet_Lambda = np.linalg.slogdet(Lambda)
    if sign <= 0:
        # Numerically bad model; treat as very low evidence.
        return float(-np.inf)

    term1 = n_samples * np.log(2.0 * np.pi)
    term2 = n_samples * np.log(_sigma2)
    term3 = logdet_Lambda - K * np.log(alpha)
    term4 = (1.0 / _sigma2) * residual_sq
    term5 = alpha * float(m_N.T @ m_N)

    log_ev = -0.5 * (term1 + term2 + term3 + term4 + term5)
    return float(log_ev)


def _backward_evidence_greedy_single(
    x: np.ndarray,
    y_col: np.ndarray,
    G: np.ndarray,
    b: np.ndarray,
    yTy: float,
    n_samples: int,
    alpha: float,
    _sigma2: float,
    max_iter: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """
    Backward greedy evidence maximization for a single output dimension.

    Parameters
    ----------
    x : ndarray, shape (n_samples, M)
        Library matrix Theta(X) for this regression problem.

    y_col : ndarray, shape (n_samples,)
        Single target column y_j.

    G : ndarray, shape (M, M)
        Full Gram matrix Theta^T Theta.

    b : ndarray, shape (M,)
        Full vector Theta^T y_j.

    yTy : float
        Scalar y_j^T y_j.

    n_samples : int
        Number of time samples T.

    alpha : float
        Prior precision on weights.

    _sigma2 : float
        Observation noise variance.

    max_iter : int
        Maximum number of elimination steps. At most M - 1 steps are needed.

    verbose : bool, optional (default False)
        If True, prints a short trace of evidence values.

    Returns
    -------
    coef_full : ndarray, shape (M,)
        Final coefficient vector (zeros outside the selected support).

    active_mask : ndarray, shape (M,), dtype bool
        Boolean mask for active features.

    history : list of dict
        Diagnostics for each step:
        [{"step": ..., "support_size": ..., "log_evidence": ...}, ...]
    """
    x = np.asarray(x)
    y_col = np.asarray(y_col).ravel()
    G = np.asarray(G)
    b = np.asarray(b)

    n_samples_x, M = x.shape
    if n_samples_x != n_samples:
        raise ValueError("Mismatch between n_samples and x.shape[0].")
    if G.shape != (M, M):
        raise ValueError("G must have shape (M, M).")
    if b.shape[0] != M:
        raise ValueError("Dimensions of G and b are inconsistent.")

    # Start with full support
    active = np.ones(M, dtype=bool)
    history: list[dict[str, float]] = []
    

    # Initial MAP estimate on the full support
    J_full = np.where(active)[0]
    m_full = _ridge_map(x[:, J_full], y_col, alpha_prior=alpha, _sigma2=_sigma2)

    log_ev = _log_evidence_from_G(
        G_active=G,
        b_active=b,
        yTy=yTy,
        n_samples=n_samples,
        alpha=alpha,
        _sigma2=_sigma2,
        m_N=m_full,
    )

    best_log_ev = log_ev
    best_m = np.zeros(M, dtype=float)
    best_m[J_full] = m_full
    best_active = active.copy()

    if verbose:
        print(
            f"[EvidenceGreedy] start: "
            f"support={np.count_nonzero(active)}, "
            f"log_evidence={best_log_ev:.3f}"
        )

    history.append(
        {
            "step": 0,
            "support_size": int(np.count_nonzero(active)),
            "log_evidence": float(best_log_ev),
        }
    )

    # At most M - 1 removals are possible.
    # If max_iter is None, perform a full backward elimination
    # with at most M - 1 removals.
    if max_iter is None:
        n_steps_max = max(M - 1, 0)
    else:
        n_steps_max = min(max_iter, max(M - 1, 0))

    m_hist = np.zeros((M,n_steps_max+1),dtype=float)

    for step in range(1, n_steps_max + 1):
        active_indices = np.where(active)[0]
        if active_indices.size <= 1:
            break

        best_step_log_ev = -np.inf
        best_step_idx: int | None = None
        best_step_m_full: np.ndarray | None = None

        # Try removing each currently active feature
        for idx in active_indices:
            mask_candidate = active.copy()
            mask_candidate[idx] = False
            J = np.where(mask_candidate)[0]

            if J.size == 0:
                # Evaluate the empty model analytically
                log_ev_J = _log_evidence_from_G(
                    G_active=G[np.ix_(J, J)],
                    b_active=b[J],
                    yTy=yTy,
                    n_samples=n_samples,
                    alpha=alpha,
                    _sigma2=_sigma2,
                    m_N=None,
                )
                m_full_candidate = np.zeros(M, dtype=float)
            else:
                G_J = G[np.ix_(J, J)]
                b_J = b[J]
                m_J = _ridge_map(x[:, J], y_col, alpha_prior=alpha, _sigma2=_sigma2)
                log_ev_J = _log_evidence_from_G(
                    G_active=G_J,
                    b_active=b_J,
                    yTy=yTy,
                    n_samples=n_samples,
                    alpha=alpha,
                    _sigma2=_sigma2,
                    m_N=m_J,
                )
                m_full_candidate = np.zeros(M, dtype=float)
                m_full_candidate[J] = m_J

            if log_ev_J > best_step_log_ev:
                best_step_log_ev = log_ev_J
                best_step_idx = int(idx)
                best_step_m_full = m_full_candidate
                m_hist[:, step] = m_full_candidate

        # If no candidate improves evidence, stop
        if best_step_log_ev <= best_log_ev or best_step_idx is None:
            if verbose:
                print(
                    f"[EvidenceGreedy] stop at step {step}: "
                    f"no evidence improvement "
                    f"(current={best_log_ev:.3f}, "
                    f"best_candidate={best_step_log_ev:.3f})"
                )
            break

        # Accept the best removal
        active[best_step_idx] = False
        best_log_ev = best_step_log_ev
        best_m = best_step_m_full  # already full-length (M,)
        best_active = active.copy()

        if verbose:
            print(
                f"[EvidenceGreedy] step {step}: removed term {best_step_idx}, "
                f"support={np.count_nonzero(active)}, "
                f"log_evidence={best_log_ev:.3f}"
            )

        history.append(
            {
                "step": step,
                "removed": int(best_step_idx),
                "support_size": int(np.count_nonzero(active)),
                "log_evidence": float(best_log_ev),
            }
        )

    return best_m, best_active, history, m_hist
