import numpy as np

from .._typing import Float1D
from .._typing import Float2D
from ..differentiation import FiniteDifference


def TemporalNoisePropagation(
    differentiator: FiniteDifference,
    t: Float1D | Float2D,
    sigma_x: float,
) -> float:
    """
    Estimate the derivative noise variance ``_sigma2`` induced by a
    finite-difference differentiator.

    This treats ``differentiator._differentiate`` as a linear operator
    mapping ``x`` to ``L x``. By sending an identity matrix of size ``T``
    through the operator, we reconstruct the finite-difference matrix
    ``L`` and use

    .. math::

        \\mathrm{Var}[\\eta_k] = \\sigma_x^2 \\sum_{tgt} L_{k tgt}^2

    for the induced derivative noise at row ``k``. The returned
    ``_sigma2`` is the mean of this variance over rows that contain only
    finite values.

    Notes
    -----
    This implementation currently returns a single averaged noise variance.
    Pointwise noise variance may be supported in future versions.

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
        state ``x(t)``. Must be non-negative.

    Returns
    -------
    _sigma2 : float
        Estimated variance of the induced noise on the differentiated
        signal.

    """

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

    # In this version, we use the mean variance over all
    # valid time points as the noise estimate after propagation.

    # TODO: Support pointwise noise variance in future versions.
    row_norm_sq = np.sum(L_dt[finite_row_mask] ** 2, axis=1)
    factor = float(np.mean(row_norm_sq))

    return float(sigma_x**2 * factor)
