"""
Base class for numerical differentiation methods
"""

import abc

from pysindy.utils.base import validate_input


class BaseDifferentiation:
    """
    Base class for differentiation methods.

    Simply forces differentiation methods to implement a
    _differentiate function.
    """

    def __init__(self):
        pass

    # Force subclasses to implement this
    @abc.abstractmethod
    def _differentiate(self, x, t):
        """
        Numerically differentiate data.

        Parameters
        ----------
        x: array-like, shape (n_samples, n_input_features)
            Data to be differentiated. Rows of x should correspond to the same
            point in time.

        t: float or numpy array of shape [n_samples]
            If t is a float, it is interpreted as the timestep between
            samples in x.
            If t is a numpy array, it specifies the times corresponding
            to the rows of x. That is, t[i] should be the time at which
            the measurements x[i, :] were taken.
            The points in t are assumed to be increasing.

        Returns
        -------
        x_dot: array-like, shape (n_samples, n_input_features)
            Numerical time derivative of x. Entries where derivatives were
            not computed will have the value np.nan.
        """
        raise NotImplementedError

    def __call__(self, x, t=1):
        x = validate_input(x)
        return self._differentiate(x, t)
