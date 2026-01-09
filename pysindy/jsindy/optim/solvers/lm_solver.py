import time
from dataclasses import dataclass
from dataclasses import field
from typing import Callable
from typing import Union

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor
from jax.scipy.linalg import cho_solve
from tqdm.auto import tqdm


@dataclass
class LMSettings:
    """
    max_iter : int, optional
        by default 501
    atol_gradnorm : float, optional
        Gradient norm stopping condition absolute tolerance
    atol_gn_decrement: float, optional
        Gauss-Newton decrement stopping condition absolute tolerance
    cmin : float, optional
        Minimum armijo ratio to accept step, by default 0.05
    line_search_increase_ratio : float, optional
        constant to increase reg strength by in backtracking line search, by default 1.5
    max_line_search_iterations : int, optional
        by default 20
    min_alpha : float, optional
        min damping strength, by default 1e-9
    max_alpha : float, optional
        max damping strength, by default 50.
    init_alpha : float, optional
        initial damping strength, by default 3.
    step_adapt_multipler : float, optional
        value to use for adapting alpha, by default 1.2
    callback : callable, optional
        function called to print another loss each iteration, by default None
    print_every : int, optional
        How often to print convergence data, by default 100
    """

    max_iter: int = 501
    atol_gradnorm: float = 1e-8
    cmin: float = 0.05
    line_search_increase_ratio: float = 1.5
    max_line_search_iterations: int = 20
    min_alpha: float = 1e-12
    max_alpha: float = 100.0
    init_alpha: float = 3.0
    step_adapt_multiplier: float = 1.2
    callback: Union[Callable, None] = None
    print_every: int = 200
    track_iterates: bool = False
    show_progress: bool = True
    use_jit: bool = True
    no_tqdm: bool = False


@dataclass
class ConvergenceHistory:
    track_iterates: bool = False
    loss_vals: list = field(default_factory=list)
    gradnorm: list = field(default_factory=list)
    iterate_history: list = field(default_factory=list)
    improvement_ratios: list = field(default_factory=list)
    alpha_vals: list = field(default_factory=list)
    cumulative_time: list = field(default_factory=list)
    linear_system_rel_residual: list = field(default_factory=list)
    regularization_loss_contribution: list = field(default_factory=list)
    convergence_tag: str = "not-yet-run"

    def update(
        self,
        loss,
        gradnorm,
        iterate,
        armijo_ratio,
        alpha,
        cumulative_time,
        linear_system_rel_residual,
        regularization_loss_contribution=0.0,
    ):
        # Append the new values to the corresponding lists
        self.loss_vals.append(loss)
        self.gradnorm.append(gradnorm)
        self.improvement_ratios.append(armijo_ratio)
        self.alpha_vals.append(alpha)
        self.cumulative_time.append(cumulative_time)
        self.linear_system_rel_residual.append(linear_system_rel_residual)
        self.regularization_loss_contribution.append(regularization_loss_contribution)

        # Conditionally track iterates if enabled
        if self.track_iterates:
            self.iterate_history.append(iterate)

    def finish(self, convergence_tag="finished"):
        # Convert lists to JAX arrays
        self.loss_vals = jnp.array(self.loss_vals)
        self.gradnorm = jnp.array(self.gradnorm)
        self.improvement_ratios = jnp.array(self.improvement_ratios)
        self.alpha_vals = jnp.array(self.alpha_vals)
        self.cumulative_time = jnp.array(self.cumulative_time)
        self.linear_system_rel_residual = jnp.array(self.linear_system_rel_residual)
        self.regularization_loss_contribution = jnp.array(
            self.regularization_loss_contribution
        )
        if self.track_iterates:
            self.iterate_history = jnp.array(self.iterate_history)
        self.convergence_tag = convergence_tag


def print_progress(
    i,
    loss,
    gradnorm,
    alpha,
    improvement_ratio,
):
    print(
        f"Iteration {i}, loss = {loss:.4},"
        f" gradnorm = {gradnorm:.4}, alpha = {alpha:.4},"
        f" improvement_ratio = {improvement_ratio:.4}"
    )


def CholeskyLM(init_params, model, beta, optSettings: LMSettings = LMSettings()):
    """Adaptively regularized Levenberg Marquardt optimizer
    Parameters
    ----------
    init_params : jax array
        initial guess
    model :
        Object that contains model.F, and model.jac, and model.damping_matrix
    beta : float
        (global) regularization strength
    optSettings: LMParams
        optimizer settings

    Returns
    -------
    solution
        approximate minimizer
    convergence_history
        ConvergenceHistory tracker
    """
    conv_history = ConvergenceHistory(optSettings.track_iterates)
    start_time = time.time()
    params = init_params.copy()
    J = model.jac_func(params)
    residuals = model.resid_func(params)
    damping_matrix = model.damping_matrix
    alpha = optSettings.init_alpha
    if optSettings.show_progress and optSettings.no_tqdm is False:
        loop_wrapper = tqdm
    else:
        loop_wrapper = lambda x: x

    regularization_contribution = (1 / 2) * beta * params.T @ damping_matrix @ params
    conv_history.update(
        loss=(1 / 2) * jnp.sum(residuals**2) + regularization_contribution,
        gradnorm=jnp.linalg.norm(J.T @ residuals + beta * damping_matrix @ params),
        iterate=params,
        armijo_ratio=1.0,
        alpha=alpha,
        cumulative_time=time.time() - start_time,
        linear_system_rel_residual=0.0,
        regularization_loss_contribution=regularization_contribution,
    )

    def evaluate_objective(params):
        """
        Queries the objective, computing jacobian and residuals at
        current parameters to build a subproblem
        """
        J = model.jac_func(params)
        residuals = model.resid_func(params)
        damping_matrix = model.damping_matrix
        loss = (1 / 2) * jnp.sum(residuals**2) + (
            1 / 2
        ) * beta * params.T @ damping_matrix @ params
        JtJ = J.T @ J
        rhs = J.T @ residuals + beta * damping_matrix @ params
        return J, residuals, damping_matrix, loss, JtJ, rhs

    if optSettings.use_jit is True:
        evaluate_objective = jax.jit(evaluate_objective)

    @jax.jit
    def compute_step(
        params, alpha, J, JtJ, residuals, rhs, previous_loss, damping_matrix
    ):
        """
        Solves subproblem constructed by evaluate_objective
        """
        # Form and solve linear system for step
        M = JtJ + (alpha + beta) * damping_matrix
        # Add small nugget
        M = M + 1e-12 * jnp.diag(jnp.diag(M))
        Mchol = cho_factor(M)
        step = cho_solve(Mchol, rhs)
        Jstep = J @ step

        # Apply 1 step of iterative refinement
        linear_residual = (
            J.T @ (Jstep - residuals)
            + (alpha + beta) * damping_matrix @ step
            - beta * damping_matrix @ params
        )
        step = step - cho_solve(Mchol, linear_residual)

        # Track the linear system residual
        linear_residual = (
            J.T @ (Jstep - residuals)
            + (alpha + beta) * damping_matrix @ step
            - beta * damping_matrix @ params
        )
        linear_system_rel_residual = jnp.linalg.norm(linear_residual) / jnp.linalg.norm(
            rhs
        )

        # Compute step and if we decreased loss
        new_params = params - step
        new_reg_piece = (1 / 2) * beta * new_params.T @ damping_matrix @ new_params
        new_loss = (1 / 2) * jnp.sum(model.resid_func(new_params) ** 2) + new_reg_piece
        predicted_loss = (1 / 2) * jnp.sum((Jstep - residuals) ** 2) + new_reg_piece
        improvement_ratio = (previous_loss - new_loss) / (
            previous_loss - predicted_loss
        )

        return (
            step,
            new_params,
            new_loss,
            improvement_ratio,
            linear_system_rel_residual,
            new_reg_piece,
        )

    def LevenbergMarquadtUpdate(params, alpha):
        r"""Minimizes the local quadratic approximation to a function
        and performs a line search on the proximal regularization alpha
        to ensure sufficient decrease.

        Solves for the negative optimal update, using proximal regularization
        to control how close (in an L2 sense) the update is to zero.
        Optimization variable is :math:`u`, the negative update between
        previous iterate :math:`x^-` and next iterate :math:`x^+`
        .. math::
            \min_u    \|Ju + r\|^2
                        + \text{reg_weight} \|x^--u\|^2_K
                        + \alpha \|step\|^2_K
        where :math:`r` is the residual vector, and the damping matrix
        :math:`damping_matrix` adjusts the L-2 regularization to be an elliptical norm
        (e.g. an RKHS norm)
        Args:
            params: Current parametrization value of function to approximate
            alpha: damping strength.  Larger values shrink the step size.
        """
        J, residuals, damping_matrix, loss, JtJ, rhs = evaluate_objective(params)
        alpha = jnp.clip(alpha, optSettings.min_alpha, optSettings.max_alpha)
        for i in range(optSettings.max_line_search_iterations):
            (
                step,
                new_params,
                new_loss,
                improvement_ratio,
                linear_system_rel_residual,
                new_reg_piece,
            ) = compute_step(
                params, alpha, J, JtJ, residuals, rhs, loss, damping_matrix
            )

            if improvement_ratio >= optSettings.cmin:
                # Check if we get at least some proportion of predicted improvement
                succeeded = True
                return (
                    new_params,
                    new_loss,
                    rhs,
                    improvement_ratio,
                    alpha,
                    linear_system_rel_residual,
                    new_reg_piece,
                    succeeded,
                )
            else:
                alpha = optSettings.line_search_increase_ratio * alpha
            succeeded = False
        return (
            new_params,
            new_loss,
            rhs,
            improvement_ratio,
            alpha,
            linear_system_rel_residual,
            new_reg_piece,
            succeeded,
        )

    for i in loop_wrapper(range(optSettings.max_iter)):
        (
            params,
            loss,
            rhs,
            improvement_ratio,
            alpha,
            linear_system_rel_residual,
            reg_piece,
            succeeded,
        ) = LevenbergMarquadtUpdate(params, alpha)

        # Get new value for alpha
        multiplier = optSettings.step_adapt_multiplier
        if improvement_ratio <= 0.2:
            alpha = multiplier * alpha
        if improvement_ratio >= 0.8:
            alpha = alpha / multiplier

        if not succeeded:
            print("Line Search Failed!")
            print("Final Iteration Results")
            if optSettings.show_progress is True:
                print_progress(
                    i, loss, conv_history.gradnorm[-1], alpha, improvement_ratio
                )
            conv_history.finish(convergence_tag="failed-line-search")
            return params, conv_history
        model_decrease = (conv_history.loss_vals[-1] - loss) / improvement_ratio
        conv_history.update(
            loss=loss,
            gradnorm=jnp.linalg.norm(rhs),
            iterate=params,
            armijo_ratio=improvement_ratio,
            alpha=alpha,
            cumulative_time=time.time() - start_time,
            linear_system_rel_residual=linear_system_rel_residual,
            regularization_loss_contribution=reg_piece,
        )

        if conv_history.gradnorm[-1] <= optSettings.atol_gradnorm:
            conv_history.finish(convergence_tag="atol-gradient-norm")
            if optSettings.show_progress is True:
                print_progress(
                    i, loss, conv_history.gradnorm[-1], alpha, improvement_ratio
                )
            return params, conv_history

        if i > 50:
            gradnorm_stagnate = (
                conv_history.gradnorm[-1] >= 0.99 * conv_history.gradnorm[-25]
            )
            fval_stagnate = (
                conv_history.loss_vals[-1] >= conv_history.loss_vals[-25] - 1e-9
            )
            if gradnorm_stagnate and fval_stagnate:
                conv_history.finish(convergence_tag="stagnation")
                if optSettings.show_progress is True:
                    print_progress(
                        i, loss, conv_history.gradnorm[-1], alpha, improvement_ratio
                    )
                return params, conv_history

        if i % optSettings.print_every == 0 or i <= 5 or i == optSettings.max_iter - 1:
            if optSettings.show_progress is True:
                print_progress(
                    i, loss, conv_history.gradnorm[-1], alpha, improvement_ratio
                )
            if optSettings.callback:
                optSettings.callback(params)
    conv_history.finish(convergence_tag="maximum-iterations")
    return params, conv_history
