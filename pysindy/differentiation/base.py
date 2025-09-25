"""
Base class for numerical differentiation methods
"""
import abc
from typing import cast
from typing import TypeVar

import numpy as np
from sklearn.base import BaseEstimator

from .._typing import FloatND
from ..utils import AxesArray
from ..utils import comprehend_axes

FloatArray = TypeVar("FloatArray", bound=FloatND)


class BaseDifferentiation(BaseEstimator):
    """
    Base class for differentiation methods.

    Simply forces differentiation methods to implement a
    ``_differentiate`` function.

    Attributes:
        smoothed_x_: Methods that smooth x before differentiating save
            that value here.  Methods that do not simply save x here.
    """

    def __init__(self):
        pass

    # Force subclasses to implement this
    @abc.abstractmethod
    def _differentiate(self, x, t: AxesArray):
        raise NotImplementedError

    def __call__(self, x: FloatArray, t: FloatND | float = 1.0) -> FloatArray:
        """
        Numerically differentiate data.

        Parameters
        ----------
        x: array-like, shape (*n_spatial, n_samples, n_input_features)
            Data to be differentiated. Rows of x should correspond to the same
            point in time.

        t: float or numpy array of shape (n_samples,) or (n_samples, 1)
            If t is a float, it is interpreted as the timestep between
            samples in x.
            If t is a numpy array, it specifies the times corresponding
            to the rows of x. That is, t[i] should be the time at which
            the measurements x[i, :] were taken.
            The points in t are assumed to be increasing.

        Returns
        -------
        x_dot: array-like
            Numerical time derivative of x. Entries where derivatives were
            not computed will have the value np.nan.
        """
        if isinstance(t, np.ScalarType):
            if t < 0:
                raise ValueError(
                    "if t is passed as a scalar to represent dt, "
                    f"it must be >0.  Received {t}"
                )
            nt = x.shape[-2]
            t = np.arange(0, t * nt, t).reshape((-1, 1))
        t = cast(FloatND, t)
        if t.ndim == 1:
            t = t[:, np.newaxis]
        t = AxesArray(t, comprehend_axes(t))
        return self._differentiate(x, t)
