from abc import ABC
from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky
from jax.scipy.linalg import solve_triangular
from jsindy.kernels import ConstantKernel
from jsindy.kernels import fit_kernel
from jsindy.kernels import fit_kernel_partialobs
from jsindy.kernels import Kernel
from jsindy.kernels import ScalarMaternKernel
from jsindy.kernels import softplus_inverse
from jsindy.kerneltools import diagpart
from jsindy.kerneltools import eval_k
from jsindy.kerneltools import get_kernel_block_ops
from jsindy.kerneltools import nth_derivative_operator_1d
from jsindy.util import l2reg_lstsq
from jsindy.util import row_block_diag


class TrajectoryModel(ABC):
    system_dim: int

    def __call__(self, t, z):
        pass

    def initalize_fit(self, t, x):
        pass

    def derivative(self, t, z, diff_order=1):
        pass


class RKHSInterpolant(TrajectoryModel):
    """
    Args:
        dimension: Dimension of the system
        time_points: time points that we include from basis from canonical feature map
        derivative_orders: Orders of derivatives that we wish to model and include in
        the basis.
    """

    kernel: Kernel
    # dimension: int
    # time_points: jax.Array
    # derivative_orders: tuple[int, ...]
    # num_params: int

    def __init__(
        self,
        kernel=None,
        derivative_orders: tuple[int, ...] = (0, 1),
        nugget=1e-5,
    ) -> None:
        if kernel is None:
            kernel = ConstantKernel(variance=5.0) + ScalarMaternKernel(
                p=5, variance=10.0
            )
        self.kernel = kernel
        self.is_attached = False
        self.derivative_orders = derivative_orders
        self.nugget = nugget

    def __str__(self):
        return f"""
        RKHS Trajectory Model
        kernel: {self.kernel.__str__()}
        derivative_orders: {self.derivative_orders}
        nugget: {self.nugget}
        """

    def initialize(
        self,
        t,
        x,
        t_colloc,
        params,
        sigma2_est=None,
    ):
        params["sigma2_est"] = sigma2_est
        self.attach(t_obs=t, x_obs=x, basis_time_points=t_colloc)
        self.system_dim = x.shape[1]
        self.num_basis = len(self.derivative_orders) * len(t_colloc)
        self.tot_params = self.system_dim * self.num_basis
        return params

    def initialize_partialobs(
        self,
        t,
        y,
        v,
        t_colloc,
        params,
        sigma2_est=None,
    ):
        params["sigma2_est"] = sigma2_est
        self.attach_partialobs(t_obs=t, y=y, v=v, basis_time_points=t_colloc)
        self.system_dim = v.shape[1]
        self.num_basis = len(self.derivative_orders) * len(t_colloc)
        self.tot_params = self.system_dim * self.num_basis
        return params

    def attach(self, t_obs, x_obs, basis_time_points):
        self.dimension = x_obs.shape[1]
        self.time_points = basis_time_points
        self.basis_operators = tuple(
            nth_derivative_operator_1d(n) for n in self.derivative_orders
        )
        self.num_params = (
            len(self.derivative_orders) * len(basis_time_points) * self.dimension
        )
        self.evaluation_kmat = get_kernel_block_ops(
            self.kernel, (eval_k,), self.basis_operators, output_dim=self.dimension
        )
        self.RKHS_mat = get_kernel_block_ops(
            self.kernel, self.basis_operators, self.basis_operators, self.dimension
        )(self.time_points, self.time_points)
        self.RKHS_mat = self.RKHS_mat + self.nugget * diagpart(self.RKHS_mat)
        self.regmat = self.RKHS_mat
        self.is_attached = True

    def attach_partialobs(self, t_obs, y, v, basis_time_points):
        self.dimension = v.shape[1]
        self.time_points = basis_time_points
        self.basis_operators = tuple(
            nth_derivative_operator_1d(n) for n in self.derivative_orders
        )
        self.num_params = (
            len(self.derivative_orders) * len(basis_time_points) * self.dimension
        )
        self.evaluation_kmat = get_kernel_block_ops(
            self.kernel, (eval_k,), self.basis_operators, output_dim=self.dimension
        )
        self.RKHS_mat = get_kernel_block_ops(
            self.kernel, self.basis_operators, self.basis_operators, self.dimension
        )(self.time_points, self.time_points)
        self.RKHS_mat = self.RKHS_mat + self.nugget * diagpart(self.RKHS_mat)
        self.regmat = self.RKHS_mat
        self.is_attached = True

    def _evaluate_operator(self, t, z, operator):
        evaluation_matrix = get_kernel_block_ops(
            k=self.kernel,
            ops_left=(operator,),
            ops_right=self.basis_operators,
            output_dim=self.dimension,
        )(t, self.time_points)
        return evaluation_matrix @ z

    def __call__(self, t, z) -> Any:
        return self.predict(t, z)

    def predict(self, t, z):
        return self._evaluate_operator(t, z, eval_k).reshape(t.shape[0], self.dimension)

    def derivative(self, t, z, diff_order=1) -> Any:
        return self._evaluate_operator(
            t, z, nth_derivative_operator_1d(diff_order)
        ).reshape(t.shape[0], self.dimension)

    def get_fitted_params(self, t, obs, lam=1e-4):
        A = get_kernel_block_ops(
            k=self.kernel,
            ops_left=(eval_k,),
            ops_right=self.basis_operators,
            output_dim=self.dimension,
        )(t, self.time_points)

        K = self.regmat

        M = A.T @ A + lam * K
        M = M + 1e-7 * jnp.diag(M)
        return jnp.linalg.solve(M, A.T @ obs.flatten())

    def get_partialobs_fitted_params(self, t, y, v, lam=1e-4):
        A = get_kernel_block_ops(
            k=self.kernel,
            ops_left=(eval_k,),
            ops_right=self.basis_operators,
            output_dim=self.dimension,
        )(t, self.time_points)
        V = row_block_diag(v)
        A = V @ A

        K = self.regmat

        M = A.T @ A + lam * K
        M = M + 1e-7 * jnp.diag(M)
        return jnp.linalg.solve(M, A.T @ y)


class CholRKHSInterpolant(TrajectoryModel):
    """
    Args:
        dimension: Dimension of the system
        time_points: time points that we include from basis from canonical feature map
        derivative_orders: Orders of derivatives that we wish to model and include in
        the basis.
    """

    kernel: Kernel
    # dimension: int
    # time_points: jax.Array
    # derivative_orders: tuple[int, ...]
    # num_params: int

    def __init__(
        self,
        kernel=None,
        derivative_orders: tuple[int, ...] = (0, 1),
        nugget=1e-8,
    ) -> None:
        if kernel is None:
            kernel = ConstantKernel(variance=5.0) + ScalarMaternKernel(
                p=5, variance=10.0
            )
        self.kernel = kernel
        self.is_attached = False
        self.derivative_orders = derivative_orders
        self.nugget = nugget

    def __repr__(self):
        return f"""
        Cholesky Parametrized RKHS Trajectory Model
        kernel: {self.kernel.__str__()}
        derivative_orders: {self.derivative_orders}
        nugget: {self.nugget}
        """

    def initialize(
        self,
        t,
        x,
        t_colloc,
        params,
        sigma2_est=None,
    ):
        params["sigma2_est"] = sigma2_est
        self.attach(t_obs=t, x_obs=x, basis_time_points=t_colloc)
        self.system_dim = x.shape[1]
        self.num_basis = len(self.derivative_orders) * len(t_colloc)
        self.tot_params = self.system_dim * self.num_basis
        return params

    def initialize_partialobs(
        self,
        t,
        y,
        v,
        t_colloc,
        params,
        sigma2_est=None,
    ):
        params["sigma2_est"] = sigma2_est
        self.attach_partialobs(t_obs=t, y=y, v=v, basis_time_points=t_colloc)
        self.system_dim = v.shape[1]
        self.num_basis = len(self.derivative_orders) * len(t_colloc)
        self.tot_params = self.system_dim * self.num_basis
        return params

    def attach_partialobs(self, t_obs, y, v, basis_time_points):
        self.dimension = v.shape[1]
        self.time_points = basis_time_points
        self.basis_operators = tuple(
            nth_derivative_operator_1d(n) for n in self.derivative_orders
        )
        self.num_params = (
            len(self.derivative_orders) * len(basis_time_points) * self.dimension
        )
        self.evaluation_kmat = get_kernel_block_ops(
            self.kernel, (eval_k,), self.basis_operators, output_dim=self.dimension
        )
        self.RKHS_mat = get_kernel_block_ops(
            self.kernel, self.basis_operators, self.basis_operators, self.dimension
        )(self.time_points, self.time_points)
        self.RKHS_mat = self.RKHS_mat + self.nugget * diagpart(self.RKHS_mat)
        self.cholT = cholesky(
            self.RKHS_mat + self.nugget * diagpart(self.RKHS_mat), lower=False
        )
        self.regmat = jnp.eye(len(self.RKHS_mat))
        self.is_attached = True

    def attach(self, t_obs, x_obs, basis_time_points):
        self.dimension = x_obs.shape[1]
        self.time_points = basis_time_points
        self.basis_operators = tuple(
            nth_derivative_operator_1d(n) for n in self.derivative_orders
        )
        self.num_params = (
            len(self.derivative_orders) * len(basis_time_points) * self.dimension
        )
        self.evaluation_kmat = get_kernel_block_ops(
            self.kernel, (eval_k,), self.basis_operators, output_dim=self.dimension
        )
        self.RKHS_mat = get_kernel_block_ops(
            self.kernel, self.basis_operators, self.basis_operators, self.dimension
        )(self.time_points, self.time_points)
        self.RKHS_mat = self.RKHS_mat + self.nugget * diagpart(self.RKHS_mat)
        self.cholT = cholesky(
            self.RKHS_mat + self.nugget * diagpart(self.RKHS_mat), lower=False
        )
        self.regmat = jnp.eye(len(self.RKHS_mat))
        self.is_attached = True

    def _evaluate_operator(self, t, z, operator):
        evaluation_matrix = get_kernel_block_ops(
            k=self.kernel,
            ops_left=(operator,),
            ops_right=self.basis_operators,
            output_dim=self.dimension,
        )(t, self.time_points)
        return evaluation_matrix @ solve_triangular(self.cholT, z)

    def predict(self, t, z):
        return self._evaluate_operator(t, z, eval_k).reshape(t.shape[0], self.dimension)

    def __call__(self, t, z) -> Any:
        return self.predict(t, z)

    def derivative(self, t, z, diff_order=1) -> Any:
        return self._evaluate_operator(
            t, z, nth_derivative_operator_1d(diff_order)
        ).reshape(t.shape[0], self.dimension)

    def get_fitted_params(self, t, obs, lam=1e-4):
        K = self.evaluation_kmat(t, self.time_points)
        M = solve_triangular(self.cholT.T, K.T, lower=True).T
        return l2reg_lstsq(M, obs.flatten(), reg=lam)

    def get_partialobs_fitted_params(self, t, y, v, lam=1e-4):
        K = self.evaluation_kmat(t, self.time_points)
        V = row_block_diag(v)
        K = V @ K
        M = solve_triangular(self.cholT.T, K.T, lower=True).T
        return l2reg_lstsq(M, y, reg=lam)


class DataAdaptedRKHSInterpolant(RKHSInterpolant):
    """
    Args:
        dimension: Dimension of the system
        time_points: time points that we include from basis from canonical feature map
        derivative_orders: Orders of derivatives that we wish to model and include in
        the basis.
    """

    def __repr__(self):
        return f"""
        MLE Adapted RKHS Trajectory Model
        kernel: {self.kernel.__str__()}
        derivative_orders: {self.derivative_orders}
        nugget: {self.nugget}
        """

    def initialize(self, t, x, t_colloc, params):
        fitted_kernel, sigma2_est, conv = fit_kernel(
            init_kernel=self.kernel,
            init_sigma2=jnp.var(x) / 20,
            X=t,
            y=x,
            lbfgs_tol=1e-8,
            show_progress=params["show_progress"],
        )
        self.kernel = fitted_kernel
        params["sigma2_est"] = sigma2_est
        self.attach(t_obs=t, x_obs=x, basis_time_points=t_colloc)
        self.system_dim = x.shape[1]
        self.num_basis = len(self.derivative_orders) * len(t_colloc)
        self.tot_params = self.system_dim * self.num_basis
        return params

    def initialize_partialobs(self, t, y, v, t_colloc, params):
        fitted_kernel, sigma2_est, conv = fit_kernel_partialobs(
            init_kernel=self.kernel,
            init_sigma2=jnp.var(y) / 20,
            t=t,
            y=y,
            v=v,
            lbfgs_tol=1e-8,
            show_progress=params["show_progress"],
        )
        self.kernel = fitted_kernel
        params["sigma2_est"] = sigma2_est
        self.attach_partialobs(t_obs=t, y=y, v=v, basis_time_points=t_colloc)
        self.system_dim = v.shape[1]
        self.num_basis = len(self.derivative_orders) * len(t_colloc)
        self.tot_params = self.system_dim * self.num_basis
        return params


class CholDataAdaptedRKHSInterpolant(CholRKHSInterpolant):
    def __repr__(self):
        return f"""
        MLE Adapted Cholesky Parametrized RKHS Trajectory Model
        kernel: {self.kernel.__str__()}
        derivative_orders: {self.derivative_orders}
        nugget: {self.nugget}
        """

    def initialize(self, t, x, t_colloc, params):
        fitted_kernel, sigma2_est, conv = fit_kernel(
            init_kernel=self.kernel,
            init_sigma2=jnp.var(x) / 20,
            X=t,
            y=x,
            lbfgs_tol=1e-8,
            show_progress=False,  # params["show_progress"]
        )
        self.kernel = fitted_kernel
        params["sigma2_est"] = sigma2_est
        self.attach(t_obs=t, x_obs=x, basis_time_points=t_colloc)
        self.system_dim = x.shape[1]
        self.num_basis = len(self.derivative_orders) * len(t_colloc)
        self.tot_params = self.system_dim * self.num_basis
        return params

    def initialize_partialobs(self, t, y, v, t_colloc, params):
        fitted_kernel, sigma2_est, conv = fit_kernel_partialobs(
            init_kernel=self.kernel,
            init_sigma2=jnp.var(y) / 20,
            t=t,
            y=y,
            v=v,
            lbfgs_tol=1e-8,
            show_progress=False,  # params["show_progress"]
        )
        self.kernel = fitted_kernel
        params["sigma2_est"] = sigma2_est
        self.attach_partialobs(t_obs=t, y=y, v=v, basis_time_points=t_colloc)
        self.system_dim = v.shape[1]
        self.num_basis = len(self.derivative_orders) * len(t_colloc)
        self.tot_params = self.system_dim * self.num_basis
        return params
