from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky
from jax.scipy.linalg import solve_triangular
from typing_extensions import Self

from .._typing import KernelFunc
from .base import LSQInterpolant
from .kernels import Kernel
from .kerneltools import diagpart
from .kerneltools import eval_k
from .kerneltools import get_kernel_block_ops
from .kerneltools import nth_derivative_operator_1d
from .utils import l2reg_lstsq


class RKHSInterpolant(LSQInterpolant):
    """
    RKHS function in from R1 to Rd, modeling a d-dimensional trajectory as a function
    of time. Uses a fixed basis, requires the time points that objective depends on
    upon instantiation to build basis based on representer theorem.
    """

    kernel: KernelFunc | Kernel
    derivative_orders: tuple[int, ...]
    nugget: float

    def __init__(
        self,
        kernel: KernelFunc | Kernel,
        derivative_orders: tuple[int, ...] = (0, 1),
        nugget=1e-5,
    ) -> None:
        """
        dimension: Dimension of the system
        time_points: time points that we include from basis from canonical feature map
        derivative_orders: Orders of derivatives that we wish to model and include in
        the basis.
        """
        self.kernel = kernel
        self.derivative_orders = derivative_orders
        self.nugget = nugget
        self.basis_operators = tuple(
            nth_derivative_operator_1d(n) for n in self.derivative_orders
        )

    def fit_time(self, dimension: int, time_points: jax.Array) -> Self:
        self.dimension = dimension
        self.time_points = time_points
        self.num_params = len(self.derivative_orders) * len(time_points) * dimension

        self.evaluation_kmat = get_kernel_block_ops(
            self.kernel, (eval_k,), self.basis_operators, output_dim=self.dimension
        )
        RKHS_mat = get_kernel_block_ops(
            self.kernel, self.basis_operators, self.basis_operators, self.dimension
        )(self.time_points, self.time_points)
        self.gram_mat = RKHS_mat + self.nugget * diagpart(RKHS_mat)

        self.cholT = cholesky(self.gram_mat, lower=False)

        return self

    def fit_obs(self, t: jax.Array, obs: jax.Array, noise_var: float) -> jax.Array:
        """Only works for fitting observations of the system, not derivatives."""
        if not hasattr(self, "gram_mat"):
            raise ValueError(
                "You must call fit_time before calling fit_obs. "
                "fit_obs requires the gram matrix to be set up first."
            )
        K_obs = get_kernel_block_ops(
            k=self.kernel,
            ops_left=(eval_k,),
            ops_right=self.basis_operators,
            output_dim=self.dimension,
        )(t, self.time_points)
        if noise_var == 0.0 and (
            jnp.any(t != self.time_points)
            or len(self.basis_operators) != 1
            or self.basis_operators[0].func != nth_derivative_operator_1d(0).func  # type: ignore # noqa: E501
            or self.basis_operators[0].args != nth_derivative_operator_1d(0).args  # type: ignore # noqa: E501
        ):
            raise ValueError(
                "Cannot exactly interpolate unless if observation times"
                "match basis times and no derivative operators are present."
            )
        M = solve_triangular(self.cholT.T, K_obs.T, lower=True).T
        params_chol_basis = l2reg_lstsq(M, obs.flatten(), reg=noise_var)
        return solve_triangular(self.cholT, params_chol_basis, lower=False)

    def interpolate(
        self,
        x: jax.Array,
        t: jax.Array,
        t_colloc: jax.Array,
        diff_order: int = 0,
        noise_var: float = 0,
    ) -> jax.Array:
        """Fit a copy of this interpolant to observations x at time t

        This does not modify the original interpolant.

        Arguments:
            x: Observations of the system at time t
            t: Time points of the observations
            t_colloc: Points at which to interpolate
            diff_order: Order of the derivative to evaluate
            noise_var: The variance of measurement noise error

        Returns:
            A smooth nth-order derivative that interpolates the data
        """
        proxy_interpolant = RKHSInterpolant(self.kernel, (0,), self.nugget)
        proxy_interpolant.fit_time(x.shape[-1], t)
        params = proxy_interpolant.fit_obs(t, x, noise_var=noise_var)
        if diff_order == 0:
            return proxy_interpolant(t_colloc, params)
        return proxy_interpolant.derivative(t_colloc, params, diff_order)

    def _evaluate_operator(self, t, params, operator):
        evaluation_matrix = get_kernel_block_ops(
            k=self.kernel,
            ops_left=(operator,),
            ops_right=self.basis_operators,
            output_dim=self.dimension,
        )(t, self.time_points)
        return evaluation_matrix @ params

    def __call__(self, t, params) -> Any:
        return self._evaluate_operator(t, params, eval_k).reshape(
            t.shape[0], self.dimension
        )

    def derivative(self, t, params, diff_order=1) -> Any:
        return self._evaluate_operator(
            t, params, nth_derivative_operator_1d(diff_order)
        ).reshape(t.shape[0], self.dimension)
