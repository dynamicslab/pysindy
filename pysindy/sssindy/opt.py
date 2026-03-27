import time
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from functools import partial
from typing import Callable
from typing import cast
from typing import Optional
from warnings import warn

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.linalg import cho_factor
from jax.scipy.linalg import cho_solve
from jax.tree_util import register_dataclass
from tqdm.auto import tqdm
from typing_extensions import Self

from ._skjax import register_scikit_pytree
from .expressions import ObjectiveResidual
from pysindy.optimizers import STLSQ
from pysindy.optimizers.base import _BaseOptimizer
from pysindy.optimizers.base import BaseOptimizer


@partial(
    register_dataclass,
    data_fields=[
        "params",
        "loss",
        "gradnorm",
        "improvement_ratio",
        "step_damping",
        "regularization_loss",
        "lin_sys_rel_resid",
    ],
    meta_fields=[],
)
@dataclass
class OptimizerStepResult:
    """Contains information about a single optimization step."""

    params: jax.Array
    loss: float
    gradnorm: float
    improvement_ratio: float
    step_damping: float
    regularization_loss: float
    lin_sys_rel_resid: float


def print_progress(iteration: int, step_result: OptimizerStepResult):
    print(
        f"Iteration {iteration}, loss = {step_result.loss: .4}, "
        f" gradnorm = {step_result.gradnorm: .4}, "
        f"alpha = {step_result.step_damping: .4}, "
        f" improvement_ratio = {step_result.improvement_ratio: .4}"
    )


@dataclass
class LMSettings:
    """Parameters controlling the behavior of LevenbergMarquardt optimization.

    max_iter: Max number of inner iterations.  Must be positive, by default 501
    atol_gradnorm: Gradient norm stopping condition absolute tolerance
    atol_gn_decrement: Gauss-Newton decrement stopping condition absolute tolerance
    min_improvement: Minimum improvement ratio to accept in backtracking line search,
        as used in the Armijo condition or Armijo-Goldstein condition.
        By default 0.05.
    search_increase_ratio: constant to increase reg strength by in backtracking
        (proximal) search, by default 1.5.
    max_search_iterations: maximum number of backtracking search iterations, by
        default 20
    min_step_damping: minimum weight for penalty term for deviating from
        previous iterates.  Must be nonnegative, by default 1e-9
    max_step_damping : maximum weight for penalty term for deviating from
        previous iterates.  Must be nonnegative, by default 50.
    init_step_damping: starting weight for penalty term for deviating
        from previous iterates.  Must be between ``min_step_damping`` and
        ``max_step_damping`` by default 3.
    step_adapt_multipler: value to use for adapting damping, by default 1.2
    callbacks: Functions that the iteration number and parameters as
        argument and are called every iteration (by default, printing).
    callback_every: How often to use callbacks, by default every 100 iterations.
    track_iterates: Whether to save the optimizer value at each iteration
        in the ConvergenceHistory.  Increases memory usage
    use_jit: Whether to jit functions inside optimization.
    """

    max_iter: int = 501
    atol_gradnorm: float = 1e-6
    atol_gn_decrement: float = 1e-12
    min_improvement: float = 0.05
    search_increase_ratio: float = 2.0
    max_search_iterations: int = 20
    min_step_damping: float = 1e-12
    max_step_damping: float = 100.0
    init_damp: float = 3.0
    step_adapt_multiplier: float = 1.2
    callbacks: tuple[Callable[[int, OptimizerStepResult], object], ...] = field(
        default=(print_progress,)
    )
    callback_every: int = 100
    track_iterates: bool = False
    use_jit: bool = True


@dataclass
class STLSQLMSettings(LMSettings):
    """
    stlsq_max_iter: Number of iterations for sequentially thresholded Ridge regression.
    prox_reg: proximal regularizer strength for trajectory and dynamic params.
    ridge_reg: regularizer strength for dynamics ridge regression.
    threshold: truncation cutoff for dynamic params.
    """

    stlsq_max_iter: int = 50
    prox_reg: float = 1.0
    ridge_reg: float = 1e-2
    threshold: float = 0.5


@dataclass
class ConvergenceHistory:
    track_iterates: bool = False
    loss_vals: list[float] = field(default_factory=list)
    gradnorm: list[float] = field(default_factory=list)
    iterate_history: list = field(default_factory=list)
    improvement_ratios: list[float] = field(default_factory=list)
    damping_vals: list[float] = field(default_factory=list)
    cumulative_time: list = field(default_factory=list)
    linear_system_rel_residual: list = field(default_factory=list)
    regularization_loss_contribution: list = field(default_factory=list)
    convergence_tag: str = "not-yet-run"

    def update(
        self,
        step_results: OptimizerStepResult,
        cumulative_time: float,
    ):
        # Append the new values to the corresponding lists
        self.loss_vals.append(step_results.loss)
        self.gradnorm.append(step_results.gradnorm)
        self.improvement_ratios.append(step_results.improvement_ratio)
        self.damping_vals.append(step_results.step_damping)
        self.cumulative_time.append(cumulative_time)
        self.linear_system_rel_residual.append(step_results.lin_sys_rel_resid)
        self.regularization_loss_contribution.append(step_results.regularization_loss)

        # Conditionally track iterates if enabled
        if self.track_iterates:
            self.iterate_history.append(step_results.params)

    def finish(self, convergence_tag="finished"):
        self.convergence_tag = convergence_tag


@partial(
    register_dataclass,
    meta_fields=[],
    data_fields=["J", "residuals", "loss", "loss_hess_appx", "loss_const_grad"],
)
@dataclass
class ObjEvaluation:
    """The evaluation of an optimization problem at a particular point"""

    J: jax.Array
    residuals: jax.Array
    loss: float
    loss_hess_appx: jax.Array
    loss_const_grad: jax.Array

    def add_regularization(self, other: Self) -> Self:
        """Add evaluation of a regularizer (which does not affect residuals)"""
        return self.__class__(
            self.J,
            self.residuals,
            self.loss + other.loss,
            self.loss_hess_appx + other.loss_hess_appx,
            self.loss_const_grad + other.loss_const_grad,
        )


def _evaluate_objective(params: jax.Array, problem: ObjectiveResidual) -> ObjEvaluation:
    r"""Create a linear least-squares problem from a nonlinear objective.

    It evaluates quantities in the problem:

    .. math::
        \min_u    1/2\cdot\|F(x+u)\|^2

    where :math:`F` is the residual function

    Args:
        params: the current value of the optimization variable, :math:`x`
        problem: The nonlinear objective

    Note:
        The Jacobian is the most expensive part of this computation. If merely
    the residuals or loss are required, it makes sense to call those
    independently.

    Returns:
        An ObjEvaluation with the Jacobian matrix :math:`df(x) / dx`, residuals
    :math:`F(x)`, the loss value :math: `1/2\cdot\|F(x)\|^2`, the approximate hessian
    of the LSQ loss with respect to the parameters, :math:`J^TJ`, and the gradient
    of the total loss (incl regularizer) with respect to a parameter step (at origin).
    """
    J = problem.jac_func(params)
    residuals = problem.resid_func(params)
    loss = cast(float, 0.5 * jnp.sum(residuals**2))
    JtJ = J.T @ J
    rhs = J.T @ residuals
    return ObjEvaluation(J, residuals, loss, JtJ, rhs)


@partial(
    register_scikit_pytree,
    data_fields=[],
    data_fit_fields=["prob", "mat_weight"],
    meta_fields=[],
    meta_fit_fields=[],
)
@dataclass
class _LMRegularizer(ABC):
    """A global regularizer for Levenberg-Marquardt optimization.

    Note that child classes need to be re-registered with jax to be jittable

    Args:
        prox_reg: The scalar regularization strength

    Attributes:
        mat_weight: The weight matrix for the regularizer.  E.g. Gram matrix,
            elliptic norm matrix.  Must be positive definite.
    """

    def fit(self, problem: ObjectiveResidual) -> Self:
        """Assign the parts of the regularizer that do not change during iteration."""
        self.mat_weight = problem.damping_matrix
        self.prob = problem
        return self

    @abstractmethod
    def eval(self, params: jax.Array) -> ObjEvaluation:
        """Evaluate the regularizer at a particular point

        Part of ``ObjEvaluation`` is the Jacobian and residual.  A regularizer
        should supply float zero for the Jacobian and residual components.
        """
        ...

    @abstractmethod
    def step(
        self, params: jax.Array, curr_vals: ObjEvaluation
    ) -> tuple[jax.Array, jax.Array, float]:
        """Calculate the a step of the optimization problem using this regularizer

        Args:
            params: The current value of the optimization variable
            curr_vals: The evaluation of the LSQ part of the optimization problem
                at ``params``
        Returns:
            tuple of the negative parameter step, the estimated negative residual step,
            and the linear system residual
        """
        ...


@partial(
    register_scikit_pytree,
    data_fields=["prox_reg"],
    data_fit_fields=["prob", "mat_weight"],
    meta_fields=[],
    meta_fit_fields=[],
)
@dataclass
class L2CholeskyLMRegularizer(_LMRegularizer):
    prox_reg: float

    @jax.jit
    def eval(self, params: jax.Array) -> ObjEvaluation:
        loss = 0.5 * cast(float, self.prox_reg * params.T @ self.mat_weight @ params)
        loss_hess = self.prox_reg * self.mat_weight
        loss_grad = loss_hess @ params
        return ObjEvaluation(0, 0, loss, loss_hess, loss_grad)  # type: ignore

    @jax.jit
    def step(
        self, params: jax.Array, curr_vals: ObjEvaluation
    ) -> tuple[jax.Array, jax.Array, float]:
        Mchol = cho_factor(curr_vals.loss_hess_appx)
        step = cho_solve(Mchol, curr_vals.loss_const_grad)
        resid_step = curr_vals.J @ step

        linear_residual = (
            curr_vals.J.T @ (resid_step - curr_vals.residuals)
            + curr_vals.loss_hess_appx @ step
            - curr_vals.loss_const_grad
        )
        linear_residual = jnp.linalg.norm(linear_residual) / jnp.linalg.norm(
            curr_vals.loss_const_grad
        )

        return step, resid_step, linear_residual


@partial(
    register_scikit_pytree,
    data_fields=["prox_reg"],
    data_fit_fields=["prob", "mat_weight"],
    meta_fields=["theta_optimizer"],
    meta_fit_fields=["n_proc", "n_theta"],
)
@dataclass
class SINDyAlternatingLMReg(L2CholeskyLMRegularizer):
    theta_optimizer: BaseOptimizer

    def fit(self, problem: ObjectiveResidual) -> Self:
        self.prob = problem
        self.n_proc = self.prob.full_n_process
        self.n_theta = self.prob.full_n_theta
        proc_block = problem.damping_matrix[: self.n_proc, : self.n_proc]
        triagonal_block = jnp.zeros((self.n_proc, self.n_theta))
        theta_block = jnp.zeros((self.n_theta, self.n_theta))
        self.mat_weight = jnp.block(
            [[proc_block, triagonal_block], [triagonal_block.T, theta_block]]
        )
        return self

    def step(
        self, params: jax.Array, curr_vals: ObjEvaluation
    ) -> tuple[jax.Array, jax.Array, float]:
        sys_dim = self.prob.system_dim
        n_meas = self.prob.n_meas

        # jittable?
        def _inner_wrap(self, params, curr_vals):
            n_resid, var_len = curr_vals.J.shape
            proc = lax.dynamic_slice(params, (0,), (self.n_proc,))
            theta = lax.dynamic_slice(params, (self.n_proc,), (self.n_theta,))

            loss_hess_appx = lax.dynamic_slice(
                curr_vals.loss_hess_appx, (0, 0), (self.n_proc, self.n_proc)
            )
            Mchol = cho_factor(loss_hess_appx)
            loss_const_grad = lax.dynamic_slice(
                curr_vals.loss_const_grad, (0,), (self.n_proc,)
            )
            proc_step = cho_solve(Mchol, loss_const_grad)
            new_proc = lax.dynamic_slice(params, (0,), (self.n_proc,)) - proc_step

            J_theta = lax.slice(curr_vals.J, (0, self.n_proc), (n_resid, var_len))
            J_proc = lax.dynamic_slice(curr_vals.J, (0, 0), (n_resid, self.n_proc))
            feat_evals_appx = lax.slice(
                J_theta, (n_meas, 0), (n_resid, self.n_theta), (3, 3)
            )
            derivative_appx = lax.slice(
                J_theta @ theta - J_proc @ (new_proc - proc) - curr_vals.residuals,
                (n_meas,),
                (n_resid,),
            ).reshape((-1, sys_dim))
            return theta, proc_step, feat_evals_appx, derivative_appx

        theta, proc_step, feat_evals_appx, derivative_appx = _inner_wrap(
            self, params, curr_vals
        )
        if jnp.isnan(proc_step).any():
            return (
                jnp.nan * jnp.ones_like(params),
                jnp.nan * jnp.ones_like(curr_vals.residuals),
                jnp.nan,
            )
        feat_evals_appx = np.array(feat_evals_appx)
        derivative_appx = np.array(derivative_appx)
        theta_new = self.theta_optimizer.fit(
            feat_evals_appx, derivative_appx
        ).coef_.T.flatten()
        theta_step = theta - theta_new

        step = jnp.hstack((proc_step, theta_step))
        resid_step = curr_vals.J @ step

        linear_residual = (
            curr_vals.J.T @ (resid_step - curr_vals.residuals)
            + curr_vals.loss_hess_appx @ step
            - curr_vals.loss_const_grad
        )
        linear_residual = jnp.linalg.norm(linear_residual) / jnp.linalg.norm(
            curr_vals.loss_const_grad
        )

        return step, resid_step, linear_residual


def LevenbergMarquardt(
    init_params: jax.Array,
    problem: ObjectiveResidual,
    regularizer: _LMRegularizer,
    opt_settings: LMSettings = LMSettings(),
) -> tuple[jax.Array, ConvergenceHistory]:
    """Adaptively regularized Levenberg Marquardt optimizer
    Parameters
    ----------
    init_params: initial guess
    model :
        Object that contains model.F, and model.jac, and model.damping_matrix
    beta : float
        (global) regularization strength
    reg_weight: Amount of global regularization to apply. Must be positive.
    opt_settings: optimizer settings

    Returns
    -------
    A tuple of the solution and convergence information
    """
    conv_history = ConvergenceHistory(opt_settings.track_iterates)
    params = init_params.copy()
    step_damping = opt_settings.init_damp
    regularizer.fit(problem)

    # Zeroth Step
    if opt_settings:
        _evaluate_objective = jax.jit(globals()["_evaluate_objective"])
    result = cast(ObjEvaluation, _evaluate_objective(params, problem))
    reg_result = regularizer.eval(params)
    result = result.add_regularization(reg_result)
    gradnorm = jnp.linalg.norm(result.loss_const_grad)
    zeroth_step = OptimizerStepResult(
        params, result.loss, gradnorm, 1.0, opt_settings.init_damp, reg_result.loss, 0.0
    )
    conv_history.update(zeroth_step, cumulative_time=0.0)

    start_time = time.time()
    for i in tqdm(range(opt_settings.max_iter), leave=False):
        step_result, succeeded = _LevenbergMarquardtUpdate(
            params, step_damping, problem, regularizer, opt_settings
        )

        step_damping = (
            jnp.maximum(1 / 3, 1 - (2 * step_result.improvement_ratio - 1) ** 3)
            * step_result.step_damping
        )
        if not succeeded:
            warn(f"Search Failed on iteration {i}! Final Iteration Results:")
            conv_history.finish(convergence_tag="failed-line-search")
            return params, conv_history

        params = step_result.params
        model_decrease = (
            conv_history.loss_vals[-1] - step_result.loss
        ) / step_result.improvement_ratio
        conv_history.update(step_result, time.time() - start_time)

        if step_result.gradnorm <= opt_settings.atol_gradnorm:
            conv_history.finish("atol-gradient-norm")
            break
        elif model_decrease * (1 + step_damping) <= opt_settings.atol_gn_decrement:
            conv_history.finish("atol-gauss-newton-decrement")
            break
        if i % opt_settings.callback_every == 0 or i <= 5:
            for callback in opt_settings.callbacks:
                callback(i, step_result)
    else:
        conv_history.finish(convergence_tag="maximum-iterations")

    for callback in opt_settings.callbacks:
        callback(i, step_result)
    return params, conv_history


def _take_prox_step(
    params: jax.Array,
    step_damp: float,
    curr_objdata: ObjEvaluation,
    problem: ObjectiveResidual,
    global_reg: _LMRegularizer,
    gradnorm: float,
) -> OptimizerStepResult:
    local_reg = L2CholeskyLMRegularizer(step_damp)
    local_reg.fit(problem)
    # zeros_like reflects that variable is step, not params
    local_vals = curr_objdata.add_regularization(local_reg.eval(jnp.zeros_like(params)))
    step, resid_step, linear_residual = global_reg.step(params, local_vals)
    # step is negative, because curr_vals.loss_grad gets subtracted to get rhs
    new_params = params - step
    reg_loss = global_reg.eval(new_params).loss
    new_loss = cast(
        float, 0.5 * jnp.sum(problem.resid_func(new_params) ** 2) + reg_loss
    )
    pred_loss = cast(
        float, 0.5 * jnp.sum((resid_step - curr_objdata.residuals) ** 2) + reg_loss
    )
    improvement_ratio = (curr_objdata.loss - new_loss) / (curr_objdata.loss - pred_loss)
    return OptimizerStepResult(
        new_params,
        new_loss,
        gradnorm,
        improvement_ratio,
        step_damp,
        reg_loss,
        linear_residual,
    )


def _LevenbergMarquardtUpdate(
    params: jax.Array,
    init_damp: float,
    problem: ObjectiveResidual,
    global_reg: _LMRegularizer,
    opt_settings: LMSettings,
) -> tuple[OptimizerStepResult, bool]:

    r"""Regularizes and minimizes the local quadratic approximation of a problem

    .. math::
        \min_x   1/2\cdot \|\widetilde F(x)\|^2 + R(x)

    where :math:`\widetilde F` is the linear approximation of the residual.  This
    function enforces locality with an additional damping term based upon distance
    from the previous iterate.  If the new iterate does not improve the loss
    sufficiently, the damping term is increased.
    Args:
        params: Current parametrization value of function to approximate
        init_damp: Starting damping strength.  Larger values shrink the step size.
        problem: The optimization problem to solve,
        global_reg: The regularizer, which should know how to evaluate itself
            at a point and how to iterate when combined with a least-squares
            ObjEvaluation.
        opt_settings: damping adaption and termination criteria.
    """
    # Values that don't change during proximity search
    if opt_settings:
        _evaluate_objective = jax.jit(globals()["_evaluate_objective"])
    curr_objdata = cast(ObjEvaluation, _evaluate_objective(params, problem))
    curr_objdata = curr_objdata.add_regularization(global_reg.eval(params))
    gradnorm = jnp.linalg.norm(curr_objdata.loss_const_grad)
    step_damp = cast(
        float,
        jnp.clip(
            init_damp,
            opt_settings.min_step_damping,
            opt_settings.max_step_damping,
        ),
    )

    for i in range(opt_settings.max_search_iterations):
        step_result = _take_prox_step(
            params, step_damp, curr_objdata, problem, global_reg, gradnorm
        )

        if step_result.improvement_ratio >= opt_settings.min_improvement:
            return step_result, True
        else:
            step_damp = opt_settings.search_increase_ratio * step_damp
    return step_result, False


def STLSQ_solve(u0, theta0, residual_objective, beta, optSettings):
    conv_history = ConvergenceHistory(track_iterates=optSettings.track_iterates)

    @jax.jit
    def F_split(u, theta):
        return residual_objective.F_split(
            [u],
            theta.reshape(
                residual_objective.num_features, residual_objective.system_dim
            ),
        )

    def phi(u, theta):
        return 0.5 * jnp.sum(F_split(u, theta) ** 2)

    # def data_mse(u):
    #     mse = jnp.mean()

    @jax.jit
    def evaluate_objective(u, theta):
        Fval = F_split(u, theta)
        Ju = jax.jacrev(F_split, argnums=0)(u, theta)
        Jtheta = jax.jacrev(F_split, argnums=1)(u, theta)
        return Fval, Ju, Jtheta

    loop_wrapper = tqdm

    max_iter = optSettings.stlsq_max_iter
    rho = optSettings.prox_reg
    alpha = optSettings.ridge_reg
    lam = optSettings.threshold

    u, theta = u0, theta0
    K = residual_objective.state_param_regmat[: len(u), : len(u)]

    loss_vals = []
    for k in loop_wrapper(range(max_iter)):
        u_old = u

        Fval, Ju, Jtheta = evaluate_objective(u, theta)
        rhs_u = rho * K @ u_old - Ju.T @ (Fval - Ju @ u)

        u = jax.scipy.linalg.solve(Ju.T @ Ju + (rho + beta) * K, rhs_u, assume_a="pos")

        rhs_ridge = -Fval + Jtheta @ theta - Ju @ (u - u_old)
        stlsq_opt = STLSQ(threshold=lam, alpha=alpha)
        stlsq_opt.ind_ = np.ones_like(np.array(theta), dtype=int)
        stlsq_opt.fit(x_=np.array(Jtheta), y=np.array(rhs_ridge))
        theta = jnp.array(stlsq_opt.coef_)[0]

        phik = phi(u, theta)
        loss = phik + 0.5 * beta * u.T @ K @ u
        params = jnp.hstack((u, theta))
        result = OptimizerStepResult(params, loss, 0.0, 0.0, 0.0, 0.0, 0.0)
        conv_history.update(result, 0.0)
        loss_vals.append(loss)

    return params, conv_history


class _BaseSSOptimizer(_BaseOptimizer, ABC):
    process_: jax.Array

    @abstractmethod
    def fit(x, y, init_params: Optional[jax.Array] = None) -> Self:
        ...


@dataclass
class STLSQLMSolver(_BaseSSOptimizer):
    beta: float = 1e-12
    optimizer_settings: STLSQLMSettings = field(default_factory=STLSQLMSettings)

    def fit(
        self,
        residual_objective: ObjectiveResidual,
        init_params: Optional[jax.Array] = None,
        *args,
        **kwargs,
    ):
        """
        Arguments:
            residual_objective: A tuple of data residual and dynamics residual
        """
        full_n_kernel = residual_objective.full_n_process
        full_n_theta = residual_objective.full_n_theta
        system_dimension = residual_objective.system_dim

        init_params = jnp.zeros(full_n_kernel + full_n_theta)

        regularizer = L2CholeskyLMRegularizer(self.beta)

        params, history = LevenbergMarquardt(
            init_params,
            residual_objective,
            regularizer,
            self.optimizer_settings,
        )

        u0 = params[:full_n_kernel]
        theta0 = params[full_n_kernel:]

        params, stlsq_history = STLSQ_solve(
            u0=u0,
            theta0=theta0,
            residual_objective=residual_objective,
            beta=self.beta,
            optSettings=self.optimizer_settings,
        )

        self.all_params_ = params
        self.coef_ = np.array(params[full_n_kernel:].reshape(-1, system_dimension).T)
        self.process_ = params[:full_n_kernel]
        self.history_ = history
        self.stlsq_ = stlsq_history

        return self


@dataclass
class LMSolver(_BaseSSOptimizer):
    """A Levenberg-Marquardt solver for single-step SINDy problems.

    It iterates by linearizing the data and dynamics loss around the previous
    iteration's values, adding the process and SINDy regularizers, and keeping
    the next iteration "close" to the previous iteration.  This results in a
    local quadratic approximation to the loss.

    The "closeness" allowed is defined adaptively by a penalty term: when an
    iteration does not improve the true minimum nearly as much as the
    approximator minimum, it tightens the penalty for moving.

    Currently applies an RKHS norm on the process terms and an L-2 norm on the
    SINDy terms.

    Attributes:
        reg_weight: overall regularization coeffficient
        optimizer_settings: Settings for optimizer. See LMSettings for more details
    """

    regularizer: _LMRegularizer = field(
        default_factory=lambda: L2CholeskyLMRegularizer(1e-12)
    )
    optimizer_settings: LMSettings = field(default_factory=LMSettings)

    def fit(
        self,
        residual_objective: ObjectiveResidual,
        init_params: Optional[jax.Array] = None,
    ):
        """
        Arguments:
            residual_objective: A tuple of data residual and dynamics residual
        """
        full_n_process = residual_objective.full_n_process
        full_n_theta = residual_objective.full_n_theta
        system_dimension = residual_objective.system_dim

        if init_params is None:
            init_params = jnp.zeros(full_n_process + full_n_theta)

        params, history = LevenbergMarquardt(
            init_params,
            residual_objective,
            self.regularizer,
            opt_settings=self.optimizer_settings,
        )

        self.all_params_ = params
        self.coef_ = np.array(params[full_n_process:].reshape(-1, system_dimension).T)
        self.process_ = params[:full_n_process]
        if np.isnan(params).any():
            self.coef_ = np.zeros_like(self.coef_)
            raise ValueError("Optimization resulted in Nans, fit is unreliable")
        if jnp.isnan(self.process_).any():
            self.process_ = jnp.zeros_like(self.process_)
        self.history_ = history

        return self
