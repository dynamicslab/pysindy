import warnings

import numpy as np

from .base import BaseDifferentiation


class FiniteDifference(BaseDifferentiation):
    """Finite difference derivatives.

    For now only first and second order finite difference methods have been
    implemented.

    Parameters
    ----------
    order: int, optional (default 2)
        The order of the finite difference method to be used.
        If 1, first order forward difference will be used.
        If 2, second order centered difference will be used.

    d : int, 1, 2, 3, 4 optional (default 1)
        The order of derivative to take (d > 3 inaccurate).

    axis: int, optional (default 0)
        The axis to differentiate along

    is_uniform : boolean, optional (default False)
        Parameter to tell the differentiation that, although a N-dim
        grid is passed, it is uniform so can use dx instead of the full
        grid array.

    drop_endpoints: boolean, optional (default False)
        Whether or not derivatives are computed for endpoints.
        If False, endpoints will be set to np.nan.
        Note that which points are endpoints depends on the method
        being used.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.differentiation import FiniteDifference
    >>> t = np.linspace(0,1,5)
    >>> X = np.vstack((np.sin(t),np.cos(t))).T
    >>> fd = FiniteDifference()
    >>> fd._differentiate(X, t)
    array([[ 1.00114596,  0.00370551],
           [ 0.95885108, -0.24483488],
           [ 0.8684696 , -0.47444711],
           [ 0.72409089, -0.67456051],
           [ 0.53780339, -0.84443737]])
    """

    def __init__(self, order=2, d=1, axis=0, is_uniform=False, drop_endpoints=False):
        if order <= 0 or not isinstance(order, int):
            raise ValueError("order must be a positive int")
        if d < 1:
            raise ValueError("differentiation order must be a positive int")
        elif order > 2:
            raise NotImplementedError

        if d >= 4:
            warnings.warn(
                "Finite differences of arbitrary order are permitted"
                " but please note that d >= 4 finite differences"
                " will dramatically amplify any numerical noise."
            )

        if d > 1 and order != 2:
            raise ValueError(
                "For second or third order derivatives, order must equal 2 "
                "because only centered differences are implemented"
            )

        self.d = d
        self.order = order
        self.drop_endpoints = drop_endpoints
        self.is_uniform = is_uniform
        self.axis = axis

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
        if self.is_uniform and not np.isscalar(t):
            t = t[1] - t[0]

        x_dot = np.full_like(x, fill_value=np.nan)
        s0 = [slice(dim) for dim in x.shape]
        s0[self.axis] = slice(0, -1, None)
        sm1 = [slice(dim) for dim in x.shape]
        sm1[self.axis] = slice(-1, None, None)
        sm2 = [slice(dim) for dim in x.shape]
        sm2[self.axis] = slice(-2, -1, None)
        sm3 = [slice(dim) for dim in x.shape]
        sm3[self.axis] = slice(-3, -2, None)

        # Uniform timestep (assume t contains dt)
        if np.isscalar(t):
            x_dot[tuple(s0)] = (np.roll(x, -1, axis=self.axis) - x)[tuple(s0)] / t
            if not self.drop_endpoints:
                x_dot[tuple(sm1)] = (
                    3 * x[tuple(sm1)] / 2 - 2 * x[tuple(sm2)] + x[tuple(sm3)] / 2
                ) / t

        # Variable timestep
        else:
            dims = np.ones(x.ndim, dtype=int)
            dims[self.axis] = x.shape[self.axis] - 1
            t_diff = np.reshape(t[1:] - t[:-1], dims)

            x_dot[tuple(s0)] = (np.roll(x, -1, axis=self.axis) - x)[tuple(s0)] / t_diff
            if not self.drop_endpoints:
                x_dot[tuple(sm1)] = (
                    3 * x[tuple(sm1)] / 2 - 2 * x[tuple(sm2)] + x[tuple(sm3)] / 2
                ) / t_diff[-1]

        return x_dot

    def _centered_difference(self, x, t=1, d=None):
        """
        Second order centered difference
        with third order forward/backward difference at endpoints.

        Warning: Sometimes has trouble with nonuniform grid spacing
        near boundaries

        Note that in order to maintain compatibility with sklearn the,
        array returned, x_dot, always satisfies np.ndim(x_dot) == 2.
        """
        if self.is_uniform and not np.isscalar(t):
            t = t[1] - t[0]

        if d is None:
            d = self.d

        x_dot = np.full_like(x, fill_value=np.nan)

        s0 = [slice(dim) for dim in x.shape]
        s0[self.axis] = slice(1, -1, None)
        s1 = [slice(dim) for dim in x.shape]
        s1[self.axis] = slice(2, None, None)
        s2 = [slice(dim) for dim in x.shape]
        s2[self.axis] = slice(None, -2, None)
        s3 = [slice(dim) for dim in x.shape]
        s3[self.axis] = slice(2, -2, None)
        s4 = [slice(dim) for dim in x.shape]
        s4[self.axis] = slice(4, None, None)
        s5 = [slice(dim) for dim in x.shape]
        s5[self.axis] = slice(3, -1, None)
        s6 = [slice(dim) for dim in x.shape]
        s6[self.axis] = slice(1, -3, None)
        s7 = [slice(dim) for dim in x.shape]
        s7[self.axis] = slice(None, -4, None)

        sp0 = [slice(dim) for dim in x.shape]
        sp0[self.axis] = slice(0, 1, None)
        sp1 = [slice(dim) for dim in x.shape]
        sp1[self.axis] = slice(1, 2, None)
        sp2 = [slice(dim) for dim in x.shape]
        sp2[self.axis] = slice(2, 3, None)
        sp3 = [slice(dim) for dim in x.shape]
        sp3[self.axis] = slice(3, 4, None)
        sp4 = [slice(dim) for dim in x.shape]
        sp4[self.axis] = slice(4, 5, None)
        sp5 = [slice(dim) for dim in x.shape]
        sp5[self.axis] = slice(5, 6, None)

        sm1 = [slice(dim) for dim in x.shape]
        sm1[self.axis] = slice(-1, None, None)
        sm2 = [slice(dim) for dim in x.shape]
        sm2[self.axis] = slice(-2, -1, None)
        sm3 = [slice(dim) for dim in x.shape]
        sm3[self.axis] = slice(-3, -2, None)
        sm4 = [slice(dim) for dim in x.shape]
        sm4[self.axis] = slice(-4, -3, None)
        sm5 = [slice(dim) for dim in x.shape]
        sm5[self.axis] = slice(-5, -4, None)
        sm6 = [slice(dim) for dim in x.shape]
        sm6[self.axis] = slice(-6, -5, None)

        if d == 1:
            # Uniform timestep (assume t contains dt)
            if np.isscalar(t):
                x_dot[tuple(s0)] = (x[tuple(s1)] - x[tuple(s2)]) / (2 * t)
                if not self.drop_endpoints:
                    x_dot[tuple(sp0)] = (
                        -11 / 6 * x[tuple(sp0)]
                        + 3 * x[tuple(sp1)]
                        - 3 / 2 * x[tuple(sp2)]
                        + x[tuple(sp3)] / 3
                    ) / t
                    x_dot[tuple(sm1)] = (
                        11 / 6 * x[tuple(sm1)]
                        - 3 * x[tuple(sm2)]
                        + 3 / 2 * x[tuple(sm3)]
                        - x[tuple(sm4)] / 3
                    ) / t

            # Variable timestep
            else:
                dims = np.ones(x.ndim, dtype=int)
                dims[self.axis] = x.shape[self.axis] - 2
                t_diff = np.reshape(t[2:] - t[:-2], dims)
                x_dot[tuple(s0)] = (x[tuple(s1)] - x[tuple(s2)]) / t_diff
                if not self.drop_endpoints:
                    x_dot[tuple(sp0)] = (
                        -11 / 6 * x[tuple(sp0)]
                        + 3 * x[tuple(sp1)]
                        - 3 / 2 * x[tuple(sp2)]
                        + x[tuple(sp3)] / 3
                    ) / (t_diff[tuple(sp0)] / 2)
                    x_dot[tuple(sm1)] = (
                        11 / 6 * x[tuple(sm1)]
                        - 3 * x[tuple(sm2)]
                        + 3 / 2 * x[tuple(sm3)]
                        - x[tuple(sm4)] / 3
                    ) / (t_diff[tuple(sm1)] / 2)

        if d == 2:
            # Uniform timestep (assume t contains dt)
            if np.isscalar(t):
                x_dot[tuple(s0)] = (x[tuple(s1)] - 2 * x[tuple(s0)] + x[tuple(s2)]) / (
                    t ** 2
                )
                if not self.drop_endpoints:
                    x_dot[tuple(sp0)] = (
                        2 * x[tuple(sp0)]
                        - 5 * x[tuple(sp1)]
                        + 4 * x[tuple(sp2)]
                        - x[tuple(sp3)]
                    ) / (t ** 2)
                    x_dot[tuple(sm1)] = (
                        2 * x[tuple(sm1)]
                        - 5 * x[tuple(sm2)]
                        + 4 * x[tuple(sm3)]
                        - x[tuple(sm4)]
                    ) / (t ** 2)

            # Variable timestep
            else:
                dims = np.ones(x.ndim, dtype=int)
                dims[self.axis] = x.shape[self.axis] - 2
                t_diff = np.reshape(t[2:] - t[:-2], dims)

                x_dot[tuple(s0)] = (x[tuple(s1)] - 2 * x[tuple(s0)] + x[tuple(s2)]) / (
                    (t_diff / 2.0) ** 2
                )
                if not self.drop_endpoints:
                    x_dot[tuple(sp0)] = (
                        2 * x[tuple(sp0)]
                        - 5 * x[tuple(sp1)]
                        + 4 * x[tuple(sp2)]
                        - x[tuple(sp3)]
                    ) / ((t_diff[tuple(sp0)] / 2.0) ** 2)
                    x_dot[tuple(sm1)] = (
                        2 * x[tuple(sm1)]
                        - 5 * x[tuple(sm2)]
                        + 4 * x[tuple(sm3)]
                        - x[tuple(sm4)]
                    ) / ((t_diff[tuple(sm1)] / 2.0) ** 2)

        if d == 3:
            # Uniform timestep (assume t contains dt)
            if np.isscalar(t):
                x_dot[tuple(s3)] = (
                    x[tuple(s4)] / 2.0
                    - x[tuple(s5)]
                    + x[tuple(s6)]
                    - x[tuple(s7)] / 2.0
                ) / (t ** 3)
                if not self.drop_endpoints:
                    x_dot[tuple(sp0)] = (
                        -2.5 * x[tuple(sp0)]
                        + 9 * x[tuple(sp1)]
                        - 12 * x[tuple(sp2)]
                        + 7 * x[tuple(sp3)]
                        - 1.5 * x[tuple(sp4)]
                    ) / (t ** 3)
                    x_dot[tuple(sp1)] = (
                        -2.5 * x[tuple(sp1)]
                        + 9 * x[tuple(sp2)]
                        - 12 * x[tuple(sp3)]
                        + 7 * x[tuple(sp4)]
                        - 1.5 * x[tuple(sp5)]
                    ) / (t ** 3)
                    x_dot[tuple(sm1)] = (
                        2.5 * x[tuple(sm1)]
                        - 9 * x[tuple(sm2)]
                        + 12 * x[tuple(sm3)]
                        - 7 * x[tuple(sm4)]
                        + 1.5 * x[tuple(sm5)]
                    ) / (t ** 3)
                    x_dot[tuple(sm2)] = (
                        2.5 * x[tuple(sm2)]
                        - 9 * x[tuple(sm3)]
                        + 12 * x[tuple(sm4)]
                        - 7 * x[tuple(sm5)]
                        + 1.5 * x[tuple(sm6)]
                    ) / (t ** 3)

            # Variable timestep
            else:
                dims = np.ones(x.ndim, dtype=int)
                dims[self.axis] = x.shape[self.axis] - 4
                t_diff = np.reshape(t[4:] - t[:-4], dims)
                x_dot[tuple(s3)] = (
                    x[tuple(s4)] / 2.0
                    - x[tuple(s5)]
                    + x[tuple(s6)]
                    - x[tuple(s7)] / 2.0
                ) / ((t_diff / 4.0) ** 3)
                if not self.drop_endpoints:
                    x_dot[tuple(sp0)] = (
                        -2.5 * x[tuple(sp0)]
                        + 9 * x[tuple(sp1)]
                        - 12 * x[tuple(sp2)]
                        + 7 * x[tuple(sp3)]
                        - 1.5 * x[tuple(sp4)]
                    ) / ((t_diff[tuple(sp0)] / 4.0) ** 3)
                    x_dot[tuple(sp1)] = (
                        -2.5 * x[tuple(sp1)]
                        + 9 * x[tuple(sp2)]
                        - 12 * x[tuple(sp3)]
                        + 7 * x[tuple(sp4)]
                        - 1.5 * x[tuple(sp5)]
                    ) / ((t_diff[tuple(sp1)] / 4.0) ** 3)
                    x_dot[tuple(sm1)] = (
                        2.5 * x[tuple(sm1)]
                        - 9 * x[tuple(sm2)]
                        + 12 * x[tuple(sm3)]
                        - 7 * x[tuple(sm4)]
                        + 1.5 * x[tuple(sm5)]
                    ) / ((t_diff[tuple(sm1)] / 4.0) ** 3)
                    x_dot[tuple(sm2)] = (
                        2.5 * x[tuple(sm2)]
                        - 9 * x[tuple(sm3)]
                        + 12 * x[tuple(sm4)]
                        - 7 * x[tuple(sm5)]
                        + 1.5 * x[tuple(sm6)]
                    ) / ((t_diff[tuple(sm2)] / 4.0) ** 3)

        if d > 3:
            return self._centered_difference(
                self._centered_difference(x, t, d=3), t, d=self.d - 3
            )

        return x_dot
