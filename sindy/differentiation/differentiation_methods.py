import abc
import numpy as np
from scipy.signal import savgol_filter

from sindy.utils.base import validate_input


class BaseDifferentiation:
    def __init__(self):
        pass

    # Force subclasses to implement this
    @abc.abstractmethod
    def _differentiate(self, x, t):
        """Differentiate the data"""
        raise NotImplementedError


class FiniteDifference(BaseDifferentiation):

    def __init__(self, order=2, drop_endpoints=False):
        if order <= 0 or not isinstance(order, int):
            raise ValueError("order must be a positive int")
        elif order > 2:
            raise NotImplementedError()

        self.order = order
        self.drop_endpoints = drop_endpoints

    def _differentiate(self, x, t):
        if self.order == 1:
            return self._forward_difference(x, t)
        else:
            return self._centered_difference(x, t)

    def __call__(self, x, t=1):
        x = validate_input(x)
        return self._differentiate(x, t)

    # First order forward difference
    # (and 2nd order backward difference for final point)
    def _forward_difference(self, x, t=1):
        x_dot = np.full_like(x, fill_value=np.nan)

        # Uniform timestep (assume t contains dt)
        if np.isscalar(t):
            x_dot[:-1, :] = (x[1:, :] - x[:-1, :]) / t
            if not self.drop_endpoints:
                x_dot[-1, :] = (
                    (
                        3 * x[-1, :] / 2
                        - 2 * x[-2, :]
                        + x[-3, :] / 2
                    )
                    / t
                )
            return x_dot

        # Variable timestep
        else:
            t_diff = t[1:] - t[:-1]
            x_dot[:-1, :] = (x[1:, :] - x[:-1, :]) / t_diff[
                :, None
            ]
            if not self.drop_endpoints:
                x_dot[-1, :] = (
                    (
                        3 * x[-1, :] / 2
                        - 2 * x[-2, :]
                        + x[-3, :] / 2
                    )
                    / t_diff[-1]
                )
            return x_dot

    # Second order centered difference
    # with third order forward/backward difference at endpoints.
    # Warning: Sometimes has trouble with nonuniform grid spacing
    # near boundaries
    def _centered_difference(self, x, t=1):
        x_dot = np.full_like(x, fill_value=np.nan)

        # Uniform timestep (assume t contains dt)
        if np.isscalar(t):
            x_dot[1:-1, :] = (x[2:, :] - x[:-2, :]) / (2 * t)
            if not self.drop_endpoints:
                x_dot[0, :] = (
                    (
                        -11 / 6 * x[0, :]
                        + 3 * x[1, :]
                        - 3 / 2 * x[2, :]
                        + x[3, :] / 3
                    )
                    / t
                )
                x_dot[-1, :] = (
                    (
                        11 / 6 * x[-1, :]
                        - 3 * x[-2, :]
                        + 3 / 2 * x[-3, :]
                        - x[-4, :] / 3
                    )
                    / t
                )
            return x_dot

        # Variable timestep
        else:
            t_diff = t[2:] - t[:-2]
            x_dot[1:-1, :] = (x[2:, :] - x[:-2, :]) / t_diff[
                :, None
            ]
            if not self.drop_endpoints:
                x_dot[0, :] = (
                    (
                        -11 / 6 * x[0, :]
                        + 3 * x[1, :]
                        - 3 / 2 * x[2, :]
                        + x[3, :] / 3
                    )
                    / (t_diff[0] / 2)
                )
                x_dot[-1, :] = (
                    (
                        11 / 6 * x[-1, :]
                        - 3 * x[-2, :]
                        + 3 / 2 * x[-3, :]
                        - x[-4, :] / 3
                    )
                    / (t_diff[-1] / 2)
                )
            return x_dot


class SmoothedFiniteDifference(FiniteDifference):
    def __init__(
        self,
        smoother=savgol_filter,
        smoother_kws={},
        **kwargs
    ):
        super(SmoothedFiniteDifference, self).__init__(**kwargs)
        self.smoother = smoother
        self.smoother_kws = smoother_kws

        if smoother is savgol_filter:
            if 'window_length' not in smoother_kws:
                self.smoother_kws['window_length'] = 11
            if 'polyorder' not in smoother_kws:
                self.smoother_kws['polyorder'] = 3
            self.smoother_kws['axis'] = 0

    def _differentiate(self, x, t):
        x = self.smoother(x, **self.smoother_kws)
        return super(SmoothedFiniteDifference, self)._differentiate(x, t)
