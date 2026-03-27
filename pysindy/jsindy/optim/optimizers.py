from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jsindy.optim.solvers.alt_active_set_lm_solver import AlternatingActiveSolve
from jsindy.optim.solvers.lm_solver import CholeskyLM
from jsindy.optim.solvers.lm_solver import LMSettings
from jsindy.trajectory_model import TrajectoryModel
from jsindy.util import full_data_initialize
from jsindy.util import partial_obs_initialize


class LMSolver:
    def __init__(self, beta_reg=1.0, solver_settings=LMSettings()):
        self.solver_settings = solver_settings
        self.beta_reg = beta_reg

    def run(self, model, params):
        # init_params = params["init_params"]
        params["data_weight"] = 1 / (params["sigma2_est"] + 0.01)
        params["colloc_weight"] = 10

        if model.is_partially_observed is False:
            z0, theta0 = full_data_initialize(
                model.t,
                model.x,
                model.traj_model,
                model.dynamics_model,
                sigma2_est=params["sigma2_est"] + 0.01,
                input_orders=model.input_orders,
                ode_order=model.ode_order,
            )
        else:
            z0, theta0 = partial_obs_initialize(
                model.t,
                model.y,
                model.v,
                model.traj_model,
                model.dynamics_model,
                sigma2_est=params["sigma2_est"] + 0.01,
                input_orders=model.input_orders,
                ode_order=model.ode_order,
            )
        z_theta_init = jnp.hstack([z0, theta0.flatten()])

        def resid_func(z_theta):
            z = z_theta[: model.traj_model.tot_params]
            theta = z_theta[model.traj_model.tot_params :].reshape(
                model.dynamics_model.param_shape
            )
            return model.residuals.residual(
                z, theta, params["data_weight"], params["colloc_weight"]
            )

        jac_func = jax.jacrev(resid_func)
        damping_matrix = block_diag(
            model.traj_model.regmat, model.dynamics_model.regmat
        )

        lm_prob = LMProblem(resid_func, jac_func, damping_matrix)
        z_theta, opt_results = CholeskyLM(
            z_theta_init, lm_prob, self.beta_reg, self.solver_settings
        )
        z = z_theta[: model.traj_model.tot_params]
        theta = z_theta[model.traj_model.tot_params :].reshape(
            model.dynamics_model.param_shape
        )

        return z, theta, opt_results, params


class LMProblem:
    def __init__(self, resid_func, jac_func, damping_matrix):
        self.resid_func = resid_func
        self.jac_func = jac_func
        self.damping_matrix = damping_matrix


class AlternatingActiveSetLMSolver:
    def __init__(
        self,
        beta_reg=1.0,
        colloc_weight_scale=100.0,
        fixed_colloc_weight=None,
        fixed_data_weight=None,
        solver_settings=LMSettings(),
        max_inner_iterations=200,
        sparsifier=None,
    ):
        self.solver_settings = solver_settings
        self.beta_reg = beta_reg
        self.colloc_weight_scale = colloc_weight_scale
        self.fixed_colloc_weight = fixed_colloc_weight
        self.fixed_data_weight = fixed_data_weight
        self.max_inner_iterations = max_inner_iterations
        self.sparsifier = sparsifier
        self.params = {}

    def __str__(self):
        return f"""
        Alternating Active Set Optimizer
        beta_reg: {self.beta_reg},
        sparsifier: {self.sparsifier.__str__()}
        data_weight: {self.params['data_weight']}
        colloc_weight: {self.params['colloc_weight']}
        """

    def run(self, model, params):
        if self.fixed_data_weight is not None:
            params["data_weight"] = self.fixed_data_weight
        else:
            params["data_weight"] = 1 / (params["sigma2_est"] + 0.001)
        if self.fixed_colloc_weight is None:
            params["colloc_weight"] = self.colloc_weight_scale * params["data_weight"]
        else:
            params["colloc_weight"] = self.fixed_colloc_weight
        print(params)

        if model.is_partially_observed is False:
            z0, theta0 = full_data_initialize(
                model.t,
                model.x,
                model.traj_model,
                model.dynamics_model,
                sigma2_est=params["sigma2_est"] + 0.01,
                input_orders=model.input_orders,
                ode_order=model.ode_order,
            )
        else:
            z0, theta0 = partial_obs_initialize(
                model.t,
                model.y,
                model.v,
                model.traj_model,
                model.dynamics_model,
                sigma2_est=params["sigma2_est"] + 0.01,
                input_orders=model.input_orders,
                ode_order=model.ode_order,
            )
        z_theta_init = jnp.hstack([z0, theta0.flatten()])

        def resid_func(z_theta):
            z = z_theta[: model.traj_model.tot_params]
            theta = z_theta[model.traj_model.tot_params :].reshape(
                model.dynamics_model.param_shape
            )
            return model.residuals.residual(
                z, theta, params["data_weight"], params["colloc_weight"]
            )

        jac_func = jax.jacrev(resid_func)
        damping_matrix = block_diag(
            model.traj_model.regmat, model.dynamics_model.regmat
        )

        lm_prob = LMProblem(resid_func, jac_func, damping_matrix)
        if self.solver_settings.show_progress:
            print("Warm Start")

        z_theta, lm_opt_results = CholeskyLM(
            z_theta_init, lm_prob, self.beta_reg, self.solver_settings
        )
        z = z_theta[: model.traj_model.tot_params]
        theta = z_theta[model.traj_model.tot_params :].reshape(
            model.dynamics_model.param_shape
        )

        if self.solver_settings.show_progress:
            print("Model after smooth warm start")
            model.print(theta=theta)
            print("Alternating Activeset Sparsifier")

        def F_split(z, theta):
            data_weight = params["data_weight"]
            colloc_weight = params["colloc_weight"]
            return model.residuals.residual(z, theta, data_weight, colloc_weight)

        # fix this later
        aaslm_prob = AASLMProblem(
            system_dim=model.traj_model.system_dim,
            num_features=model.dynamics_model.num_features,
            F_split=F_split,
            t_colloc=model.t_colloc,
            interpolant=model.traj_model,
            state_param_regmat=model.traj_model.regmat,
            model_param_regmat=model.dynamics_model.regmat,
            feature_library=model.dynamics_model.feature_map,
        )

        z, theta, aas_lm_opt_results = AlternatingActiveSolve(
            z0=z,
            theta0=theta,
            residual_objective=aaslm_prob,
            beta=self.beta_reg,
            show_progress=self.solver_settings.show_progress,
            max_inner_iter=self.max_inner_iterations,
            sparsifier=self.sparsifier,
            input_orders=model.input_orders,
            ode_order=model.ode_order,
        )
        theta = theta.reshape(model.dynamics_model.param_shape)
        self.params = params

        return z, theta, [lm_opt_results, aas_lm_opt_results], params


class AnnealedAlternatingActiveSetLMSolver:
    def __init__(
        self,
        beta_reg=1.0,
        colloc_weight_scale=100.0,
        fixed_colloc_weight=None,
        fixed_data_weight=None,
        solver_settings=LMSettings(),
        max_inner_iterations=200,
        sparsifier=None,
        num_annealing_steps=4,
        anneal_colloc_mult=5.0,
        anneal_beta_mult=2.0,
    ):
        self.solver_settings = solver_settings
        self.beta_reg = beta_reg
        self.colloc_weight_scale = colloc_weight_scale
        self.fixed_colloc_weight = fixed_colloc_weight
        self.fixed_data_weight = fixed_data_weight
        self.max_inner_iterations = max_inner_iterations
        self.sparsifier = sparsifier

        self.num_annealing_steps = num_annealing_steps
        self.anneal_colloc_mult = anneal_colloc_mult
        self.anneal_beta_mult = anneal_beta_mult

    def __str__(self):
        return f"""
            Annealed Alternating Active Set Optimizer
            beta_reg: {self.beta_reg},
            sparsifier: {self.sparsifier.__str__()}
            data_weight: {self.fixed_data_weight}
            colloc_weight: {self.fixed_colloc_weight}
            annealing_steps: {self.anneal_colloc_mult}
            anneal_colloc_mult: {self.anneal_colloc_mult}
            anneal_beta_mult: {self.anneal_beta_mult}
            """

    def run(self, model, params):
        sigma2est = params.get("sigma2_est", 0)
        if sigma2est is None:
            # If not using data-adapted interpolant
            sigma2est = 0.0
        if self.fixed_data_weight is not None:
            params["data_weight"] = self.fixed_data_weight
        else:
            params["data_weight"] = 1 / (sigma2est + 0.001)
        if self.fixed_colloc_weight is None:
            params["colloc_weight"] = self.colloc_weight_scale * params["data_weight"]
        else:
            params["colloc_weight"] = self.fixed_colloc_weight
        print(params)
        if model.is_partially_observed is False:
            z0, theta0 = full_data_initialize(
                model.t,
                model.x,
                model.traj_model,
                model.dynamics_model,
                sigma2_est=sigma2est + 0.01,
                input_orders=model.input_orders,
                ode_order=model.ode_order,
            )
        else:
            z0, theta0 = partial_obs_initialize(
                model.t,
                model.y,
                model.v,
                model.traj_model,
                model.dynamics_model,
                sigma2_est=sigma2est + 0.01,
                input_orders=model.input_orders,
                ode_order=model.ode_order,
            )
        z_theta_init = jnp.hstack([z0, theta0.flatten()])

        num_steps = self.num_annealing_steps
        dataweight_vals = [params["data_weight"]] * num_steps
        colloc_weight_vals = [
            params["colloc_weight"] * (self.anneal_colloc_mult ** (i + 1 - num_steps))
            for i in range(num_steps)
        ]
        beta_reg_vals = [
            self.beta_reg * (self.anneal_beta_mult ** (num_steps - i - 1))
            for i in range(num_steps)
        ]
        parameter_sequence = zip(dataweight_vals, colloc_weight_vals, beta_reg_vals)

        def resid_func(z_theta, data_weight, colloc_weight):
            z = z_theta[: model.traj_model.tot_params]
            theta = z_theta[model.traj_model.tot_params :].reshape(
                model.dynamics_model.param_shape
            )
            return model.residuals.residual(z, theta, data_weight, colloc_weight)

        full_lm_opt_results = []

        for data_weight, colloc_weight, beta_reg in parameter_sequence:
            residual_function = partial(
                resid_func, data_weight=data_weight, colloc_weight=colloc_weight
            )
            jac_func = jax.jacrev(residual_function)
            damping_matrix = block_diag(
                model.traj_model.regmat, model.dynamics_model.regmat
            )

            lm_prob = LMProblem(residual_function, jac_func, damping_matrix)
            if self.solver_settings.show_progress:
                print(
                    f"Solving for data_weight = {data_weight}, colloc_weight = {colloc_weight} beta_reg = {beta_reg}"
                )
            z_theta, lm_opt_results = CholeskyLM(
                z_theta_init, lm_prob, self.beta_reg, self.solver_settings
            )
            z_theta_init = z_theta
            full_lm_opt_results.append(lm_opt_results)
        z = z_theta[: model.traj_model.tot_params]
        theta = z_theta[model.traj_model.tot_params :].reshape(
            model.dynamics_model.param_shape
        )

        if self.solver_settings.show_progress:
            print("Model after smooth warm start")
            model.print(theta=theta)
            print("Alternating Activeset Sparsifier")

        def F_split(z, theta):
            data_weight = params["data_weight"]
            colloc_weight = params["colloc_weight"]
            return model.residuals.residual(z, theta, data_weight, colloc_weight)

        # fix this later
        aaslm_prob = AASLMProblem(
            system_dim=model.traj_model.system_dim,
            num_features=model.dynamics_model.num_features,
            F_split=F_split,
            t_colloc=model.t_colloc,
            interpolant=model.traj_model,
            state_param_regmat=model.traj_model.regmat,
            model_param_regmat=model.dynamics_model.regmat,
            feature_library=model.dynamics_model.feature_map,
        )

        z, theta, aas_lm_opt_results = AlternatingActiveSolve(
            z0=z,
            theta0=theta,
            residual_objective=aaslm_prob,
            beta=self.beta_reg,
            show_progress=self.solver_settings.show_progress,
            max_inner_iter=self.max_inner_iterations,
            sparsifier=self.sparsifier,
            input_orders=model.input_orders,
            ode_order=model.ode_order,
        )
        theta = theta.reshape(model.dynamics_model.param_shape)

        return z, theta, [full_lm_opt_results, aas_lm_opt_results], params


@dataclass
class AASLMProblem:
    system_dim: int
    num_features: int
    F_split: callable
    t_colloc: jax.Array
    interpolant: TrajectoryModel
    state_param_regmat: jax.Array
    model_param_regmat: jax.Array
    feature_library: callable
