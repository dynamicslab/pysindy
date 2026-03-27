import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from tqdm.auto import tqdm


def run_jax_solver(solver, x0):
    state = solver.init_state(x0)
    sol = x0
    values, errors, stepsizes = [state.value], [state.error], [state.stepsize]

    def update(sol, state):
        return solver.update(sol, state)

    jitted_update = jax.jit(update)
    for iter_num in tqdm(range(solver.maxiter)):
        sol, state = jitted_update(sol, state)
        values.append(state.value)
        errors.append(state.error)
        stepsizes.append(state.stepsize)
        if solver.verbose > 0:
            print("Gradient Norm: ", state.error)
            print("Loss Value: ", state.value)
        if state.error <= solver.tol:
            break
        if stepsizes[-1] == 0:
            print("Restart")
            state = solver.init_state(sol)
    convergence_data = {
        "values": np.array(values),
        "gradnorms": np.array(errors),
        "stepsizes": np.array(stepsizes),
    }
    return sol, convergence_data, state


@jit
def l2reg_lstsq(A, y, reg=1e-10):
    U, sigma, Vt = jnp.linalg.svd(A, full_matrices=False)
    return Vt.T @ ((sigma / (sigma**2 + reg)) * (U.T @ y))


def refine_solution(
    params, equation_model, reg_sequence=10 ** (jnp.arange(-4.0, -18, -0.5))
):
    """Refines solution with almost pure gauss newton through SVD"""
    refinement_losses = []
    refined_params = params.copy()
    for reg in tqdm(reg_sequence):
        J = equation_model.jac(refined_params)
        F = equation_model.F(refined_params)
        refined_params = refined_params - l2reg_lstsq(J, F, reg)
        refinement_losses += [equation_model.loss(refined_params)]
    return refined_params, jnp.array(refinement_losses)


def adaptive_refine_solution(
    params, equation_model, initial_reg=1e-4, num_iter=100, mult=0.7
):
    refinement_losses = [equation_model.loss(params)]
    refined_params = params.copy()
    reg_vals = [initial_reg]
    reg = initial_reg
    for i in tqdm(range(num_iter)):
        J = equation_model.jac(refined_params)
        F = equation_model.F(refined_params)
        U, sigma, Vt = jnp.linalg.svd(J, full_matrices=False)

        candidate_regs = [mult * reg, reg, reg / mult]
        candidate_steps = [
            Vt.T @ ((sigma / (sigma**2 + S)) * (U.T @ F)) for S in candidate_regs
        ]

        loss_vals = jnp.array(
            [equation_model.loss(refined_params - step) for step in candidate_steps]
        )
        choice = jnp.argmin(loss_vals)
        reg = candidate_regs[choice]
        step = candidate_steps[choice]
        refined_params = refined_params - step
        refinement_losses.append(loss_vals[choice])
        reg_vals.append(reg)
    return refined_params, jnp.array(refinement_losses), jnp.array(reg_vals)
