"""
Wrapper classes for differentiation methods from the :doc:`derivative:index` package.

Some default values used here may differ from those used in :doc:`derivative:index`.
"""
from derivative import dxdt
from numpy import arange

from .base import BaseDifferentiation


class SINDyDerivative(BaseDifferentiation):
    """
    Wrapper class for differentiation classes from the :doc:`derivative:index` package.
    This class is meant to provide all the same functionality as the
    `dxdt <https://derivative.readthedocs.io/en/latest/api.html\
        #derivative.differentiation.dxdt>`_ method.

    This class also has ``_differentiate`` and ``__call__`` methods which are
    used by PySINDy.

    Parameters
    ----------
    derivative_kws: dictionary, optional
        Keyword arguments to be passed to the
        `dxdt <https://derivative.readthedocs.io/en/latest/api.html\
        #derivative.differentiation.dxdt>`_
        method.

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
            self.kwargs.update(params["kwargs"])

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
