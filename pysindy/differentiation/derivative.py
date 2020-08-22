"""
Wrapper classes for differentiation methods from the `derivative` package.

Some default values used here may differ from those used in `derivative`.
"""
from derivative.dglobal import Spectral
from derivative.dglobal import Spline
from derivative.dglobal import TrendFiltered
from derivative.differentiation import Derivative
from derivative.dlocal import FiniteDifference
from derivative.dlocal import SavitzkyGolay
from derivative import dxdt
from numpy import arange
from numpy import vectorize
from sklearn.base import BaseEstimator

from pysindy.utils.base import validate_input


class DifferentiationMixin:
    """
    Mixin class adapting objects from the derivative package to the
    method calls used in PySINDy
    """

    def _differentiate(self, x, t=1):
        if isinstance(t, (int, float)):
            if t < 0:
                raise ValueError("t must be a positive constant or an array")
            t = arange(x.shape[0]) * t
        return self.d(x, t, axis=0)

    def __call__(self, x, t=1):
        x = validate_input(x)
        return self._differentiate(x, t)


class SINDyDerivative(BaseEstimator):
    """
    Wrapper class for differentiation classes from the ``derivative`` package.

    Imbues the class with ``_differentiate`` and ``__call__`` methods which are
    used by PySINDy.

    Parameters
    ----------
    derivative_kws: dictionary, optional
        Keyword arguments to be passed to the ``derivative.dxdt`` method.

    Notes
    -----
    See the `derivative documentation <https://derivative.readthedocs.io/en/latest/>`_
    for acceptable keywords.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        Modification of the pysindy method to allow unknown kwargs. This allows using
        the full range of derivative parameters that are not defined as member variables
        in sklearn grid search.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        else:
            self.kwargs.update(params)

        return self

    def get_params(self, deep=True):
        """Get parameters."""
        params = super().get_params(deep)

        if isinstance(self.kwargs, dict):
            params.update(self.kwargs)

        return params

    def _differentiate(self, x, t=1):
        if isinstance(t, (int, float)):
            if t < 0:
                raise ValueError("t must be a positive constant or an array")
            t = arange(x.shape[0]) * t

        return dxdt(x, t, axis=0, **self.kwargs)

    def __call__(self, x, t=1):
        x = validate_input(x, t=t)
        return self._differentiate(x, t)


class SpectralDifferentiator(Spectral, BaseEstimator, DifferentiationMixin):
    """Wrapper for spectral derivatives."""

    def __init__(self, filter=vectorize(lambda f: 1)):
        super().__init__(filter=filter)


class SplineDifferentiator(Spline, BaseEstimator, DifferentiationMixin):
    """Wrapper for spline-based derivatives."""

    def __init__(self, s, order=3, periodic=False):
        super().__init__(s=s, order=order, periodic=periodic)


class TrendFilteredDifferentiator(TrendFiltered, BaseEstimator, DifferentiationMixin):
    """Wrapper for derivatives based on Total Squared Variations."""

    def __init__(self, order, alpha, **kwargs):
        super().__init__(order=order, alpha=alpha, **kwargs)


class FiniteDifferenceDifferentiator(
    FiniteDifference, BaseEstimator, DifferentiationMixin
):
    """Wrapper for finite difference derivatives."""

    def __init__(self, k, periodic=False):
        super().__init__(k=k, periodic=periodic)


class SavitzkyGolayDifferentiator(SavitzkyGolay, BaseEstimator, DifferentiationMixin):
    """Wrapper for Savitky Golay derivatives."""

    def __init__(self, order, left, right, use_iwindow=False):
        super().__init__(order=order, left=left, right=right, use_iwindow=use_iwindow)
