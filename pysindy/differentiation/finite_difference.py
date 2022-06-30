import numpy as np

from .base import BaseDifferentiation


class FiniteDifference(BaseDifferentiation):
    """Finite difference derivatives.

    Parameters
    ----------
    order: int, optional (default 2)
        The order of the finite difference method to be used.
        Currently only centered differences are implemented, for even order
        and left-off-centered differences for odd order.

    d : int, optional (default 1)
        The order of derivative to take.  Must be positive integer.

    axis: int, optional (default 0)
        The axis to differentiate along.

    is_uniform : boolean, optional (default False)
        Parameter to tell the differentiation that, although a N-dim
        grid is passed, it is uniform so can use dx instead of the full
        grid array.

    drop_endpoints: boolean, optional (default False)
        Whether or not derivatives are computed for endpoints.
        If False, endpoints will be set to np.nan.
        Note that which points are endpoints depends on the method
        being used.

    periodic: boolean, optional (default False)
        Whether to use periodic boundary conditions for endpoints.
        Use forward differences for periodic=False and periodic boundaries
        with centered differences for periodic=True on the boundaries.
        No effect if drop_endpoints=True

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.differentiation import FiniteDifference
    >>> t = np.linspace(0, 1, 5)
    >>> X = np.vstack((np.sin(t), np.cos(t))).T
    >>> fd = FiniteDifference()
    >>> fd._differentiate(X, t)
    array([[ 1.00114596,  0.00370551],
           [ 0.95885108, -0.24483488],
           [ 0.8684696 , -0.47444711],
           [ 0.72409089, -0.67456051],
           [ 0.53780339, -0.84443737]])
    """

    def __init__(
        self,
        order=2,
        d=1,
        axis=0,
        is_uniform=False,
        drop_endpoints=False,
        periodic=False,
    ):

        if order <= 0 or not isinstance(order, int):
            raise ValueError("order must be a positive int")
        if d <= 0:
            raise ValueError("differentiation order must be a positive int")

        self.d = int(d)
        self.order = int(order)
        self.is_uniform = is_uniform
        self.axis = axis
        self.drop_endpoints = drop_endpoints
        self.periodic = periodic
        self.n_stencil = int(2 * ((self.d + 1) // 2) - 1 + self.order)
        self.n_stencil_forward = self.d + self.order

        if self.d >= self.n_stencil:
            raise ValueError(
                "This combination of d and order is not implemented. "
                "It is required that d >= stencil_size, where "
                "stencil_size = 2 * (d + 1) // 2 - 1 + order. "
            )

    def _coefficients(self, t):
        nt = len(t)
        self.stencil_inds = np.array(
            [np.arange(i, nt - self.n_stencil + i + 1) for i in range(self.n_stencil)]
        )
        self.stencil = np.transpose(t[self.stencil_inds])

        pows = np.arange(self.n_stencil)[np.newaxis, :, np.newaxis]
        matrices = (
            self.stencil
            - t[
                (self.n_stencil - 1) // 2 : -(self.n_stencil - 1) // 2,
                np.newaxis,
            ]
        )[:, np.newaxis, :] ** pows
        b = np.zeros(self.n_stencil)
        b[self.d] = np.math.factorial(self.d)
        return np.linalg.solve(matrices, [b])

    def _coefficients_boundary_forward(self, t):
        # use the same stencil for each boundary point,
        # but change the evaluation point
        left = np.arange(self.n_stencil_forward)[:, np.newaxis] * np.ones(
            (self.n_stencil - 1) // 2, dtype=int
        )
        if self.order % 2 == 0:
            right_len = (self.n_stencil - 1) // 2
        else:
            right_len = 1 + (self.n_stencil - 1) // 2
        right = (-1 - np.arange(self.n_stencil_forward))[:, np.newaxis] * np.ones(
            right_len, dtype=int
        )
        tinds = np.concatenate(
            [
                np.arange((self.n_stencil - 1) // 2, dtype=int),
                np.flip(-1 - np.arange(right_len, dtype=int)),
            ]
        )
        self.stencil_inds = np.concatenate([left, right], axis=1)

        pows = np.arange(self.n_stencil_forward)[np.newaxis, :, np.newaxis]

        if np.isscalar(t):
            matrices = np.transpose(
                (t * (self.stencil_inds - tinds)[:, np.newaxis, :]) ** pows
            )
        else:
            matrices = np.transpose(
                ((t[self.stencil_inds] - t[tinds])[:, np.newaxis, :]) ** pows
            )

        b = np.zeros(self.stencil_inds.shape).T
        b[:, self.d] = np.math.factorial(self.d)
        return np.linalg.solve(matrices, b)

    def _coefficients_boundary_periodic(self, t):
        # use centered periodic stencils
        left = (np.arange(self.n_stencil) - (self.n_stencil - 1) // 2)[
            :, np.newaxis
        ] + np.arange((self.n_stencil - 1) // 2, dtype=int)
        right = np.flip(
            (-1 - np.arange(self.n_stencil) + (self.n_stencil - 1) // 2)[:, np.newaxis]
            - np.arange((self.n_stencil - 1) // 2, dtype=int),
            axis=1,
        )
        self.stencil_inds = np.concatenate([left, right], axis=1)
        tinds = np.concatenate(
            [
                np.arange((self.n_stencil - 1) // 2, dtype=int),
                np.flip(-1 - np.arange((self.n_stencil - 1) // 2, dtype=int)),
            ]
        )
        pows = np.arange(self.n_stencil)[np.newaxis, :, np.newaxis]

        if np.isscalar(t):
            matrices = (
                np.transpose(
                    t
                    * (
                        np.concatenate(
                            [
                                np.ones((self.n_stencil - 1) // 2),
                                -np.ones((self.n_stencil - 1) // 2),
                            ]
                        )
                        * (np.arange(self.n_stencil) - (self.n_stencil - 1) // 2)[
                            :, np.newaxis
                        ]
                    )[:, np.newaxis, :]
                )
                ** pows
            )
        else:
            period = t[-1] - t[0] + (t[1] - t[0])
            matrices = np.transpose(
                (
                    (
                        np.mod(t[self.stencil_inds] - t[tinds] + period / 2, period)
                        - period / 2
                    )[:, np.newaxis, :]
                )
                ** pows
            )

        b = np.zeros(self.stencil_inds.shape).T
        b[:, self.d] = np.math.factorial(self.d)
        return np.linalg.solve(matrices, b)

    def _constant_coefficients(self, dt):
        pows = np.arange(self.n_stencil)[:, np.newaxis]
        matrices = (dt * (np.arange(self.n_stencil) - (self.n_stencil - 1) // 2))[
            np.newaxis, :
        ] ** pows
        b = np.zeros(self.n_stencil)
        b[self.d] = np.math.factorial(self.d)
        return np.linalg.solve(matrices, b)

    def _accumulate(self, coeffs, x):
        # slice to select the stencil indices
        s = [slice(None)] * len(x.shape)
        s[self.axis] = self.stencil_inds

        # a new axis is introduced after self.axis for the stencil indices
        # To contract with the coefficients, roll by -self.axis to put it first
        # Then roll back by self.axis to return the order
        trans = np.roll(np.arange(len(x.shape) + 1), -self.axis)
        return np.transpose(
            np.einsum(
                "ij...,ij->j...",
                np.transpose(x[tuple(s)], axes=trans),
                np.transpose(coeffs),
            ),
            np.roll(np.arange(len(x.shape)), self.axis),
        )

    def _differentiate(self, x, t):
        """
        Apply finite difference method.
        """
        x_dot = np.full_like(x, fill_value=np.nan)
        s = [slice(None)] * len(x.shape)

        if self.axis < 0:
            # Need to do this for _accumulate function to work properly?
            self.axis = len(x.shape) + self.axis

        # Central differences in interior of domain
        if np.isscalar(t) or self.is_uniform:
            dt = t
            if not np.isscalar(t):
                dt = t[1] - t[0]

            coeffs = self._constant_coefficients(dt)
            dims = np.array(x.shape)
            dims[self.axis] = x.shape[self.axis] - (self.n_stencil - 1)
            interior = np.zeros(dims)
            # Slightly faster version of self._accumulate for uniform grid
            for i in range(self.n_stencil):
                if abs(coeffs[i]) > 0:
                    start = i
                    stop = -(self.n_stencil - start - 1)
                    if stop >= 0:
                        stop = None
                    s[self.axis] = slice(start, stop)
                    interior = interior + x[tuple(s)] * coeffs[i]
        else:
            coeffs = self._coefficients(t)
            interior = self._accumulate(coeffs, x)
        s[self.axis] = slice((self.n_stencil - 1) // 2, -(self.n_stencil - 1) // 2)
        x_dot[tuple(s)] = interior

        # Boundaries
        if not self.drop_endpoints:
            # Forward differences on boundary
            if not self.periodic:
                coeffs = self._coefficients_boundary_forward(t)
                boundary = self._accumulate(coeffs, x)

                if self.order % 2 == 0:
                    right_len = (self.n_stencil - 1) // 2
                else:
                    right_len = 1 + (self.n_stencil - 1) // 2
                s[self.axis] = np.concatenate(
                    [
                        np.arange((self.n_stencil - 1) // 2, dtype=int),
                        np.flip(-1 - np.arange(right_len, dtype=int)),
                    ]
                )
            # Central differences on boundary with periodic bcs
            else:
                coeffs = self._coefficients_boundary_periodic(t)
                boundary = self._accumulate(coeffs, x)
                s[self.axis] = np.concatenate(
                    [
                        np.arange(0, (self.n_stencil - 1) // 2),
                        -np.flip(1 + np.arange(1, (self.n_stencil - 1) // 2)),
                        np.array([-1]),
                    ]
                )
            x_dot[tuple(s)] = boundary
        return x_dot
