import numpy as np

from pysindy.differentiation import BaseDifferentiation


class FiniteDifference(BaseDifferentiation):
    """
    Finite difference derivatives.

    For now only first and second order finite difference methods have been
    implemented.

    Parameters
    ----------
    order: int, 1 or 2, optional (default 2)
        The order of the finite difference method to be used.
        If 1, first order forward difference will be used.
        If 2, second order centered difference will be used.

    drop_endpoints: boolean, optional (default False)
        Whether or not derivatives are computed for endpoints.
        If False, endpoints will be set to np.nan.
        Note that which points are endpoints depends on the method
        being used.

    Returns
    -------
    self: returns an instance of self
    """

    def __init__(self, order=2, drop_endpoints=False):
        if order <= 0 or not isinstance(order, int):
            raise ValueError("order must be a positive int")
        elif order > 2:
            raise NotImplementedError

        self.order = order
        self.drop_endpoints = drop_endpoints

    def _differentiate(self, x, t):
        """
        Apply finite difference method.
        """
        if self.order == 1:
            return self._forward_difference(x, t)
        else:
            return self._centered_difference(x, t)

    def _forward_difference(self, x, t=1):
        """
        First order forward difference
        (and 2nd order backward difference for final point).

        Note that in order to maintain compatibility with sklearn the,
        array returned, x_dot, always satisfies np.ndim(x_dot) == 2.
        """

        x_dot = np.full_like(x, fill_value=np.nan)

        # Uniform timestep (assume t contains dt)
        if np.isscalar(t):
            x_dot[:-1, :] = (x[1:, :] - x[:-1, :]) / t
            if not self.drop_endpoints:
                x_dot[-1, :] = (
                    3 * x[-1, :] / 2 - 2 * x[-2, :] + x[-3, :] / 2
                ) / t

        # Variable timestep
        else:
            t_diff = t[1:] - t[:-1]
            x_dot[:-1, :] = (x[1:, :] - x[:-1, :]) / t_diff[:, None]
            if not self.drop_endpoints:
                x_dot[-1, :] = (
                    3 * x[-1, :] / 2 - 2 * x[-2, :] + x[-3, :] / 2
                ) / t_diff[-1]

        return x_dot

    def _centered_difference(self, x, t=1):
        """
        Second order centered difference
        with third order forward/backward difference at endpoints.

        Warning: Sometimes has trouble with nonuniform grid spacing
        near boundaries

        Note that in order to maintain compatibility with sklearn the,
        array returned, x_dot, always satisfies np.ndim(x_dot) == 2.
        """
        x_dot = np.full_like(x, fill_value=np.nan)

        # Uniform timestep (assume t contains dt)
        if np.isscalar(t):
            x_dot[1:-1, :] = (x[2:, :] - x[:-2, :]) / (2 * t)
            if not self.drop_endpoints:
                x_dot[0, :] = (
                    -11 / 6 * x[0, :]
                    + 3 * x[1, :]
                    - 3 / 2 * x[2, :]
                    + x[3, :] / 3
                ) / t
                x_dot[-1, :] = (
                    11 / 6 * x[-1, :]
                    - 3 * x[-2, :]
                    + 3 / 2 * x[-3, :]
                    - x[-4, :] / 3
                ) / t

        # Variable timestep
        else:
            t_diff = t[2:] - t[:-2]
            x_dot[1:-1, :] = (x[2:, :] - x[:-2, :]) / t_diff[:, None]
            if not self.drop_endpoints:
                x_dot[0, :] = (
                    -11 / 6 * x[0, :]
                    + 3 * x[1, :]
                    - 3 / 2 * x[2, :]
                    + x[3, :] / 3
                ) / (t_diff[0] / 2)
                x_dot[-1, :] = (
                    11 / 6 * x[-1, :]
                    - 3 * x[-2, :]
                    + 3 / 2 * x[-3, :]
                    - x[-4, :] / 3
                ) / (t_diff[-1] / 2)

        return x_dot
