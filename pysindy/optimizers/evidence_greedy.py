"""EvidenceGreedy optimizer: greedy Bayesian evidence-based sparse regression."""
import sys
import warnings

import numpy as np
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import ridge_regression

from .base import _normalize_features
from .base import BaseOptimizer


class EvidenceGreedy(BaseOptimizer):
    r"""
    Sparse Regression by maximizing Bayesian evidence
    through greedy elimination of features

    This optimizer performs backward model selection
    (i.e.feature elimination) driven by the
    Bayesian log evidence for a linear Gaussian model with an isotropic
    Gaussian prior on the coefficients. For each target dimension y_{tgt},
    we assume

    .. math::

        w &\sim \mathcal{N}\!\left(0,\ \alpha^{-1} I\right), \\
        y_{tgt} \mid w &\sim \mathcal{N}\!\left(\Theta w,\ \sigma^2 I\right),

    where ``alpha`` is the prior precision on the coefficients
    (sigma_p^{-2}) and ``_sigma2`` is the observation noise variance
    (sigma^2).

    The algorithm:

    #. Start from the full support (all library terms active).
    #. At each step, temporarily remove each active term in turn.
    #. For each candidate support, compute the Bayesian log evidence
       :math:`\log p(y_{tgt} \mid \alpha, \sigma^2, \mathrm{support})` using the
       precomputed statistics :math:`G=\Theta^\top\Theta` and
       :math:`b_{tgt}=\Theta^\top y_{tgt}`.
    #. Accept the removal that yields the largest increase in evidence.
    #. Stop when no single removal increases the evidence.



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
        Maximum number of elimination steps. If None, at most n_features - 1
        removals are allowed.

    normalize_columns : bool, default=True
        Passed to :class:`~pysindy.optimizers.base.BaseOptimizer`.
        If True, BOTH the columns of the library matrix and the target
        variables are normalized before regression. The Bayesian prior
        and ridge penalty are then applied in this normalized space. The
        learned coefficients are mapped back to the original scale when
        stored in ``coef_``.

        Note that when ``normalize_columns=True``, ``alpha`` is typically of
        order 1.0.

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
        Boolean support mask corresponding to ``coef_``. ``ind_[i, tgt]`` is
        True if the tgt-th library function is active in the equation for the
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
             "removed": tgt,
             "support_size": (number of active features after removal),
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

    Example output::

        (x0)' = -9.979 x0 + 9.980 x1
        (x1)' = 27.807 x0 - 0.963 x1 - 0.995 x0 x2
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
            initial_guess=initial_guess,
            copy_X=copy_X,
            unbias=unbias,
        )

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

        # Getting dimensions
        n_samples, n_features = x.shape
        n_targets = y.shape[1]

        # Numerical precision threshold for treating norms as zero.
        # Also used to prevent division by zero in when
        # normalizing sigma2 to sigma2_scaled.
        eps_precision = float(np.finfo(float).eps)

        # BaseOptimizer only normalise the library, but for the Bayesian
        # framework, we also need to normalize the targets.
        # Normalising this help make sure the parameter
        # is also rescaled to unit order.
        y_norm, y_normalised = _normalize_features(y)
        if self.normalize_columns:
            y = y_normalised
            # Since y is normalized, y^T y = n_samples
            yTy_all = np.ones(n_features, dtype=float)
        else:
            yTy_all = y_norm**2.0

        # Shared Gram matrix and RHS for all outputs:
        G = x.T @ x  # (n_features, n_features) = Theta^T Theta
        B = x.T @ y  # (n_features, N) = Theta^T Y

        coef = np.zeros((n_targets, n_features), dtype=float)
        ind = np.zeros((n_targets, n_features), dtype=bool)
        self.evidence_history_: list[list[dict[str, float]]] = []

        for tgt in range(n_targets):
            b = B[:, tgt]  # (n_features,)
            yTy = float(yTy_all[tgt])  # scalar

            # In case y is a zero vector or close to it,
            # output as an empty model.
            if y_norm[tgt] ** 2.0 <= eps_precision:
                coef[tgt, :] = 0.0
                ind[tgt, :] = False

                # Log evidence of the empty model (n_features=0).
                # m_N is ignored for n_features=0.
                log_ev = _log_evidence_laplace_appx(
                    G=np.zeros((0, 0), dtype=float),
                    b=np.zeros((0,), dtype=float),
                    yTy=yTy,
                    n_samples=n_samples,
                    alpha=self.alpha,
                    _sigma2=float(self._sigma2),
                    m_N=None,
                )
                history_tgt = [
                    {
                        "step": 0,
                        "support_size": 0,
                        "log_evidence": float(log_ev),
                    }
                ]
                self.evidence_history_.append(history_tgt)

                # Consistent history_ format
                history_tmp = np.full((n_targets, n_features), np.nan, dtype=float)
                history_tmp[tgt, :] = 0.0
                self.history_.append(history_tmp)
                continue

            # Since the target (Y) is also possibly normalized,
            # we need to rescale _sigma2 accordingly.

            if self.normalize_columns:
                yn = float(y_norm[tgt])
                # Prevent division by zero / inf
                denom = max(yn * yn, eps_precision)
                sigma2_scaled = float(self._sigma2) / denom
            else:
                sigma2_scaled = float(self._sigma2)

            (
                coef_tgt,
                ind_tgt,
                history_tgt,
                coef_hist,
            ) = _backward_evidence_greedy_single(
                x=x,
                y_col=y[:, tgt],
                G=G,
                b=b,
                yTy=yTy,
                n_samples=n_samples,
                alpha=self.alpha,
                _sigma2=sigma2_scaled,
                max_iter=self.max_iter,
                verbose=self.verbose,
            )

            coef[tgt, :] = coef_tgt
            ind[tgt, :] = ind_tgt
            self.evidence_history_.append(history_tgt)

            # For history, we need to reshape to match the format of other optimizers.
            for i in range(np.shape(coef_hist)[1]):
                history_tmp = np.full((n_targets, n_features), np.nan, dtype=float)
                history_tmp[tgt, :] = coef_hist[:, i]
                self.history_.append(history_tmp)

        self.coef_ = coef
        self.ind_ = ind

        # Map coefficients back to original scale if normalized.
        if self.normalize_columns:
            self.coef_ = self.coef_ * y_norm.reshape(-1, 1)


def _ridge_map(
    X: np.ndarray,
    y: np.ndarray,
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

    lam = alpha_prior * _sigma2
    kw = ridge_kw or {}

    # Follow the STLSQ pattern: use ridge_regression and handle LinAlgWarning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings("always", category=LinAlgWarning)
        coef = ridge_regression(X, y, lam, **kw)

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


def _log_evidence_laplace_appx(
    G: np.ndarray,
    b: np.ndarray,
    yTy: float,
    n_samples: int,
    alpha: float,
    _sigma2: float,
    m_N: np.ndarray | None,
) -> float:
    r"""
    Compute the Bayesian log evidence for a given active set and posterior mean.

    Evidence approximation:

        log p(y) =
            -1/2 [ n_samples log(2 pi) + n_samples log _sigma2
                   + log|Lambda| - n_features log alpha
                   + ||y - Theta m_N||^2 / _sigma2
                   + alpha ||m_N||^2 ]

    where

        ||y - Theta m_N||^2
            = yTy - 2 m_N^T b + m_N^T G m_N

    and

        Lambda = alpha I_(n_features) + G / _sigma2

    Parameters
    ----------
    G : ndarray, shape (n_features, n_features)
        Gram matrix for active features
        (i.e. Theta^T Theta restricted to active features).

    b : ndarray, shape (n_features,)
        Theta^T y restricted to active features.

    yTy : float
        y^T y (scalar).

    n_samples : int
        T, number of time samples.

    alpha : float
        Prior precision on weights.

    _sigma2 : float
        Observation noise variance.

    m_N : ndarray of shape (n_features,) or None
        Posterior mean coefficients for the active set. For the empty
        model (n_features == 0), this is ignored and may be None.

    Returns
    -------
    log_ev : float
        Bayesian log evidence.

    """

    n_features = G.shape[0]

    # Degenerate empty model: p(y) = N(0, _sigma2 I)
    if n_features == 0:
        term1 = n_samples * np.log(2.0 * np.pi)
        term2 = n_samples * np.log(_sigma2)
        term4 = (1.0 / _sigma2) * yTy
        log_ev = -0.5 * (term1 + term2 + term4)
        return float(log_ev)

    if m_N is None:
        raise ValueError("m_N must be provided for a non-empty active set.")

    # m_N = np.asarray(m_N).reshape(-1)
    if m_N.shape[0] != n_features:
        raise ValueError("m_N has incompatible shape for the active set.")

    beta = 1.0 / _sigma2

    # Residual norm using precomputed stats:
    #   ||y - Theta m_N||^2 = yTy - 2 m_N^T b + m_N^T G m_N
    residual_sq = yTy - 2.0 * float(m_N.T @ b) + float(m_N.T @ (G @ m_N))

    # log|Lambda|
    Lambda = alpha * np.eye(n_features) + beta * G
    sign, logdet_Lambda = np.linalg.slogdet(Lambda)
    if sign <= 0:
        # Numerically bad model; treat as very low evidence.
        return float(-np.inf)

    term1 = n_samples * np.log(2.0 * np.pi)
    term2 = n_samples * np.log(_sigma2)
    term3 = logdet_Lambda - n_features * np.log(alpha)
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
    x : ndarray, shape (n_samples, n_features)
        Library matrix Theta(X) for this regression problem.

    y_col : ndarray, shape (n_samples,)
        Single target column y_{tgt}.

    G : ndarray, shape (n_features, n_features)
        Full Gram matrix Theta^T Theta.

    b : ndarray, shape (n_features,)
        Full vector Theta^T y_{tgt}.

    yTy : float
        Scalar y_{tgt}^T y_{tgt}.

    n_samples : int
        Number of time samples T.

    alpha : float
        Prior precision on weights.

    _sigma2 : float
        Observation noise variance.

    max_iter : int
        Maximum number of elimination steps. At most n_features - 1 steps are needed.

    verbose : bool, optional (default False)
        If True, prints a short trace of evidence values.

    Returns
    -------
    coef_full : ndarray, shape (n_features,)
        Final coefficient vector (zeros outside the selected support).

    active_mask : ndarray, shape (n_features,), dtype bool
        Boolean mask for active features.

    history : list of dict
        Diagnostics for each step:
        [{"step": ..., "support_size": ..., "log_evidence": ...}, ...]

    """

    n_samples_x, n_features = x.shape

    # Start with full support
    active = np.ones(n_features, dtype=bool)
    history: list[dict[str, float]] = []

    # Initial MAP estimate on the full support
    m_full = _ridge_map(x[:, active], y_col, alpha_prior=alpha, _sigma2=_sigma2)

    log_ev = _log_evidence_laplace_appx(
        G=G,
        b=b,
        yTy=yTy,
        n_samples=n_samples,
        alpha=alpha,
        _sigma2=_sigma2,
        m_N=m_full,
    )

    best_log_ev = log_ev
    best_m = np.zeros(n_features, dtype=float)
    best_m[active] = m_full
    best_active = active.copy()

    if verbose:
        print(
            f"[EvidenceGreedy] start: "
            f"support={np.count_nonzero(active)}, "
            f"log_evidence={best_log_ev: .3f}"
        )

    history.append(
        {
            "step": 0,
            "support_size": int(np.count_nonzero(active)),
            "log_evidence": float(best_log_ev),
        }
    )

    # At most n_features - 1 removals are possible.
    # If max_iter is None, perform a full backward elimination
    # with at most n_features - 1 removals.
    if max_iter is None:
        n_steps_max = max(n_features - 1, 0)
    else:
        n_steps_max = min(max_iter, max(n_features - 1, 0))

    m_hist = np.zeros((n_features, n_steps_max + 1), dtype=float)

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

            if mask_candidate.sum() == 0:
                m_N = None
            else:
                m_N = _ridge_map(
                    x[:, mask_candidate], y_col, alpha_prior=alpha, _sigma2=_sigma2
                )
                # Evaluate the empty model analytically
            log_ev_mask = _log_evidence_laplace_appx(
                G=G[np.ix_(mask_candidate, mask_candidate)],
                b=b[mask_candidate],
                yTy=yTy,
                n_samples=n_samples,
                alpha=alpha,
                _sigma2=_sigma2,
                m_N=m_N,
            )
            m_full_candidate = np.zeros(n_features, dtype=float)
            if mask_candidate.sum() != 0:
                m_full_candidate[mask_candidate] = m_N

            if log_ev_mask > best_step_log_ev:
                best_step_log_ev = log_ev_mask
                best_step_idx = int(idx)
                best_step_m_full = m_full_candidate
                m_hist[:, step] = m_full_candidate

        # If no candidate improves evidence, stop
        if best_step_log_ev <= best_log_ev or best_step_idx is None:
            if verbose:
                print(
                    f"[EvidenceGreedy] stop at step {step}: "
                    f"no evidence improvement "
                    f"(current={best_log_ev: .3f}, "
                    f"best_candidate={best_step_log_ev: .3f})"
                )
            break

        # Accept the best removal
        active[best_step_idx] = False
        best_log_ev = best_step_log_ev
        best_m = best_step_m_full  # already full-length (n_features,)
        best_active = active.copy()

        if verbose:
            print(
                f"[EvidenceGreedy] step {step}: removed term {best_step_idx}, "
                f"support={np.count_nonzero(active)}, "
                f"log_evidence={best_log_ev: .3f}"
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
