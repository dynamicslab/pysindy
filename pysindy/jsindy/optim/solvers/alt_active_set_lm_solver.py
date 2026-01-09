import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from pysindy import STLSQ


def norm2(x):
    if len(x) == 0:
        return 0.0
    else:
        return jnp.sum(x**2)


def maxnorm(x):
    if len(x) == 0:
        return 0.0
    else:
        return jnp.max(jnp.abs(x))


@dataclass
class ConvHistory:
    history: dict
    convergence_tag: str


class pySindySparsifier:
    def __init__(self, pysindy_optimizer=None):
        if pysindy_optimizer is None:
            pysindy_optimizer = STLSQ(threshold=0.25, alpha=0.01)
        self.optimizer = pysindy_optimizer

    def __str__(self):
        return self.optimizer.__str__()

    def __call__(self, feat_X, Xdot):
        self.optimizer.fit(feat_X, Xdot)
        theta = jnp.array(self.optimizer.coef_)
        return theta


def AlternatingActiveSolve(
    z0,
    theta0,
    residual_objective,
    beta,
    sparsifier: pySindySparsifier = None,
    show_progress: bool = True,
    max_inner_iter=50,
    input_orders=(0,),
    ode_order=1,
):
    start_time = time.time()
    if sparsifier is None:
        sparsifier = pySindySparsifier()
    ##Initialize with LMSolve results and
    # Set up objective

    z = z0
    theta = theta0

    # init_shape = (residual_objective.system_dim,residual_objective.num_features)
    theta_shape = (residual_objective.num_features, residual_objective.system_dim)

    # theta = theta.flatten()

    # resfunc = jax.jit(lambda z,theta:residual_objective.F_split(z,theta.reshape(init_shape).T))
    resfunc = jax.jit(
        lambda z, theta: residual_objective.F_split(z, theta.reshape(theta_shape))
    )

    jac_z = jax.jit(jax.jacrev(resfunc, argnums=0))
    jac_theta = jax.jit(jax.jacrev(resfunc, argnums=1))

    t_colloc = residual_objective.t_colloc
    interp = residual_objective.interpolant

    tol = 1e-6

    @jax.jit
    def loss(z, theta):
        return (1 / 2) * jnp.sum(resfunc(z, theta) ** 2) + (
            beta / 2
        ) * z.T @ residual_objective.state_param_regmat @ z

    def update_coefs(z):
        X_inputs = jnp.hstack([interp.derivative(t_colloc, z, k) for k in input_orders])
        feat_X = residual_objective.feature_library(X_inputs)

        Xdot = interp.derivative(t_colloc, z, ode_order)
        theta = sparsifier(feat_X, Xdot).T
        return theta.flatten()

    def update_params(z, theta, prox_reg_init=1.0):
        active_set = jnp.where(jnp.abs(theta) > 1e-7)[0]
        m = len(active_set)
        prox_reg = prox_reg_init
        K = residual_objective.state_param_regmat
        I = residual_objective.model_param_regmat[active_set][:, active_set]
        H = jax.scipy.linalg.block_diag(K, I)
        K0 = jax.scipy.linalg.block_diag(K, 0 * I)
        obj_val = loss(z, theta)

        Jz = jac_z(z, theta)
        Jtheta = jac_theta(z, theta)[:, active_set]
        J = jnp.hstack([Jz, Jtheta])
        F = resfunc(z, theta)
        JtJ = J.T @ J
        rhs = J.T @ F + beta * jnp.hstack([K @ z, jnp.zeros(m)])

        loss_vals = [obj_val]
        gnorms = [maxnorm(rhs)]

        max_line_search = 20
        for i in range(max_inner_iter):
            for k in range(max_line_search):
                succeeded = False

                M = JtJ + beta * K0 + prox_reg * H
                step = jnp.linalg.solve(M, rhs)

                dz = step[:-m]
                dtheta = step[-m:]

                cand_z = z - dz
                cand_theta = theta.at[active_set].set(theta[active_set] - dtheta)

                new_obj_val = loss(cand_z, cand_theta)

                predicted_decrease = obj_val - (
                    0.5 * norm2(F - J @ step) + 0.5 * beta * cand_z.T @ K @ cand_z
                )

                true_decrease = obj_val - new_obj_val

                rho = true_decrease / predicted_decrease
                if rho > 0.001 and true_decrease > 0:
                    succeeded = True
                    obj_val = new_obj_val
                    z = cand_z
                    theta = cand_theta
                    break
                else:
                    prox_reg = 2 * prox_reg

            if succeeded is False:
                print("Line search Failed")
                break
            Jz = jac_z(z, theta)
            Jtheta = jac_theta(z, theta)[:, active_set]
            J = jnp.hstack([Jz, Jtheta])
            F = resfunc(z, theta)
            JtJ = J.T @ J
            rhs = J.T @ F + beta * jnp.hstack([K @ z, jnp.zeros(m)])
            prox_reg = jnp.maximum(1 / 3, 1 - (2 * rho - 1) ** 3) * prox_reg
            gnorm = maxnorm(rhs)
            gnorms.append(gnorm)
            loss_vals.append(obj_val)
            if gnorm < tol:
                break

        return z, theta, gnorms, loss_vals, prox_reg

    all_gnorms = []
    all_objval = []
    cum_time = []
    prox_reg = 1.0
    finished = False
    support = jnp.where(jnp.abs(theta) > 1e-7)[0]
    for i in range(20):
        theta = update_coefs(z)
        new_support = jnp.where(jnp.abs(theta) > 1e-7)[0]
        z_new, theta, gnorms, objvals, prox_reg = update_params(z, theta, 2 * prox_reg)
        all_gnorms.append(gnorms)
        all_objval.append(objvals)
        dz = jnp.linalg.norm(z - z_new)
        z = z_new
        if finished == True:
            cum_time.append(time.time() - start_time)
            break

        if set(np.array(support)) == set(np.array(new_support)):
            # Run 1 more iteration just to be sure
            if show_progress:
                print("Active set stabilized")
            convergence_tag = "stable-active-set"
            finished = True
        else:
            sym_diff = set(np.array(support)).symmetric_difference(
                set(np.array(new_support))
            )
            if show_progress:
                print(f"{len(sym_diff)} active coeffs changed")
        support = new_support

        cum_time.append(time.time() - start_time)
        convergence_tag = "maximum-iterations"
    conv_hist = ConvHistory(
        history={
            "gnorms": all_gnorms,
            "objval": all_objval,
            "cumlative_time": cum_time,
        },
        convergence_tag=convergence_tag,
    )
    return z, theta, conv_hist
