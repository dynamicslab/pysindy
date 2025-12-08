import jax
import jax.numpy as jnp


@jax.jit
def l2reg_lstsq(A: jax.Array, y: jax.Array, reg: float = 1e-10):
    r"""Solve the L2-regularized least squares problem

    ..  math:
        \|Ax - b\|^2 + reg * \|x\|^2

    Args:
        A: Explanatory variables/data matrix/regression matrix
        y: Response variables/regression target
        reg
    """
    U, sigma, Vt = jnp.linalg.svd(A, full_matrices=False)
    if jnp.ndim(y) == 2:
        return Vt.T @ ((sigma / (sigma**2 + reg))[:, None] * (U.T @ y))
    else:
        return Vt.T @ ((sigma / (sigma**2 + reg)) * (U.T @ y))
