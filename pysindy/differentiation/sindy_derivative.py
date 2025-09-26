"""
Wrapper classes for differentiation methods from the :doc:`derivative:index` package.

Some default values used here may differ from those used in :doc:`derivative:index`.
"""
import numpy as np
from derivative import methods

from ..utils import AxesArray
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

    def __init__(self, axis=0, save_smooth=True, **kwargs):
        self.axis = axis
        self.kwargs = kwargs
        self.save_smooth = save_smooth

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
            self.save_smooth = params.get("save_smooth", self.save_smooth)

        return self

    def get_params(self, deep=True):
        """Get parameters."""
        params = super().get_params(deep)

        if isinstance(self.kwargs, dict):
            params.update(self.kwargs)
        params["save_smooth"] = self.save_smooth

        return params

    def _differentiate(self, x, t=1):
        differentiator = methods[self.kwargs["kind"]](
            **{k: v for k, v in self.kwargs.items() if k != "kind"}
        )
        t = AxesArray(np.take(t, indices=0, axis=-1), {"ax_time": 0})
        x_dot = differentiator.d(x, t, axis=self.axis)
        if self.save_smooth:
            self.smoothed_x_ = differentiator.x(x, t, axis=self.axis)
        else:
            self.smoothed_x_ = x
        return x_dot
