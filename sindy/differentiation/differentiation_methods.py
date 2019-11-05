import numpy as np
from sklearn.utils.validation import check_array


# TODO: add smoothing
class FiniteDifference:

    def __init__(self, order=0, drop_endpoints=False):
        if order < 0 or not isinstance(order, int):
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
        # Validate input
        if np.ndim(x) == 1:
            x = x.reshape(-1, 1)
        check_array(x)

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