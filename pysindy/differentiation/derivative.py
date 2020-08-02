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
    derivative_class: subclass of ``derivative.Derivative``
        The derivative class to be wrapped.
        Must be inherit from the ``Derivative`` class of the ``derivative`` package.
    """

    def __init__(self, derivative_class):
        if isinstance(derivative_class, Derivative):
            self.derivative_class = derivative_class

    def _differentiate(self, x, t=1):
        if isinstance(t, (int, float)):
            if t < 0:
                raise ValueError("t must be a positive constant or an array")
            t = arange(x.shape[0]) * t
        return self.derivative_class.d(x, t, axis=0)

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
