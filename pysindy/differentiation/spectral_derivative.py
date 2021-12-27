import numpy as np

from .base import BaseDifferentiation


class SpectralDerivative(BaseDifferentiation):
    """Spectral derivatives.
    Assumes uniform grid, and utilizes FFT to approximate a derivative.
    Works well for derivatives in periodic dimensions.
    Equivalent to a maximal-order finite difference, but runs in O(NlogN).

    Parameters
    ----------

    d : int
        The order of derivative to take
    axis: int, optional (default 0)
        The axis to differentiate along

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.differentiation import SpectralDerivative
    >>> t = np.arange(0,1,0.1)
    >>> X = np.vstack((np.sin(t), np.cos(t))).T
    >>> sd = SpectralDerivative()
    >>> sd._differentiate(X, t)
    array([[ 6.28318531e+00,  2.69942771e-16],
       [ 5.08320369e+00, -3.69316366e+00],
       [ 1.94161104e+00, -5.97566433e+00],
       [-1.94161104e+00, -5.97566433e+00],
       [-5.08320369e+00, -3.69316366e+00],
       [-6.28318531e+00,  7.10542736e-16],
       [-5.08320369e+00,  3.69316366e+00],
       [-1.94161104e+00,  5.97566433e+00],
       [ 1.94161104e+00,  5.97566433e+00],
       [ 5.08320369e+00,  3.69316366e+00]])
    """

    def __init__(self, d=1, axis=0):
        self.d = d
        self.axis = axis

    def _differentiate(self, x, t):
        """
        Calculate a spectral derivative.
        """

        if not np.isscalar(t):
            t = t[1] - t[0]

        q = np.fft.fft(x, axis=self.axis)
        n = x.shape[self.axis]
        dims = np.ones(x.ndim, dtype=int)
        dims[self.axis] = n
        freqs = np.zeros(n, dtype=np.complex128)
        positives = np.arange(n // 2 + 1)
        negatives = np.setdiff1d(np.arange(n), positives)
        freqs[: n // 2 + 1] = positives * 2 * np.pi / (n * t)
        freqs[n // 2 + 1 :] = (negatives - n) * 2 * np.pi / (n * t)

        if x.dtype is complex:
            return np.fft.ifft(
                np.reshape(1j * freqs, dims) ** self.d * q, axis=self.axis
            )
        else:
            return np.fft.ifft(
                np.reshape(1j * freqs, dims) ** self.d * q, axis=self.axis
            ).real.astype(x.dtype)
