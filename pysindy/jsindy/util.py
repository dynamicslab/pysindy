import jax
import jax.numpy as jnp


def validate_data_inputs(t, x, y, v):
    if x is None:
        assert y is not None
        assert v is not None
        assert len(t) == len(v)
        assert len(t) == len(y)
    if y is None:
        assert x is not None
        assert len(t) == len(x)
    if x is not None:
        assert y is None
        assert v is None


def check_is_partial_data(t, x, y, v):
    validate_data_inputs(t, x, y, v)
    if v is None:
        return False
    else:
        return True


def get_collocation_points_weights(t, num_colloc=500, bleedout_nodes=1.0):
    min_t = jnp.min(t)
    max_t = jnp.max(t)
    span = max_t - min_t
    lower = min_t - bleedout_nodes * span / num_colloc
    upper = max_t + bleedout_nodes * span / num_colloc
    col_points = jnp.linspace(lower, upper, num_colloc)
    # Scale so that it's consistent to integral, rather than sum to 1.
    col_weights = (upper - lower) / num_colloc * jnp.ones_like(col_points)
    return col_points, col_weights


@jax.jit
def l2reg_lstsq(A, y, reg=1e-10):
    U, sigma, Vt = jnp.linalg.svd(A, full_matrices=False)
    if jnp.ndim(y) == 2:
        return Vt.T @ ((sigma / (sigma**2 + reg))[:, None] * (U.T @ y))
    else:
        return Vt.T @ ((sigma / (sigma**2 + reg)) * (U.T @ y))


def tree_dot(tree, other):
    # Multiply corresponding leaves and sum each product over all its elements.
    vdots = jax.tree.map(lambda x, y: jnp.sum(x * y), tree, other)
    return jax.tree.reduce(lambda x, y: x + y, vdots, initializer=0.0)


def tree_add(tree, other):
    return jax.tree.map(lambda x, y: x + y, tree, other)


def tree_scale(tree, scalar):
    return jax.tree.map(lambda x: scalar * x, tree)


def get_equations(
    coef, feature_names, feature_library, precision: int = 3
) -> list[str]:
    """
    Get the right hand sides of the SINDy model equations.

    Parameters
    ----------
    precision: int, optional (default 3)
        Number of decimal points to include for each coefficient in the
        equation.

    Returns
    -------
    equations: list of strings
        List of strings representing the SINDy model equations for each
        input feature.
    """
    feat_names = feature_library.get_feature_names(feature_names)

    def term(c, name):
        rounded_coef = jnp.round(c, precision)
        if rounded_coef == 0:
            return ""
        else:
            return f"{c:.{precision}f} {name}"

    equations = []
    for coef_row in coef:
        components = [term(c, i) for c, i in zip(coef_row, feat_names)]
        eq = " + ".join(filter(bool, components))
        if not eq:
            eq = f"{0:.{precision}f}"
        equations.append(eq)

    return equations


def full_data_initialize(
    t,
    x,
    traj_model,
    dynamics_model,
    sigma2_est=0.1,
    theta_reg=0.001,
    input_orders=(0,),
    ode_order=1,
):
    t_grid = jnp.linspace(jnp.min(t), jnp.max(t), 500)
    z = traj_model.get_fitted_params(t, x, lam=sigma2_est)

    X_inputs = jnp.hstack([traj_model.derivative(t_grid, z, k) for k in input_orders])
    Xdot_pred = traj_model.derivative(t_grid, z, ode_order)

    theta = dynamics_model.get_fitted_theta(X_inputs, Xdot_pred, lam=theta_reg)
    return z, theta


def partial_obs_initialize(
    t,
    y,
    v,
    traj_model,
    dynamics_model,
    sigma2_est=0.1,
    theta_reg=0.001,
    input_orders=(0,),
    ode_order=1,
):
    t_grid = jnp.linspace(jnp.min(t), jnp.max(t), 500)
    z = traj_model.get_partialobs_fitted_params(t, y, v, lam=sigma2_est)

    X_inputs = jnp.hstack([traj_model.derivative(t_grid, z, k) for k in input_orders])
    Xdot_pred = traj_model.derivative(t_grid, z, ode_order)

    theta = dynamics_model.get_fitted_theta(X_inputs, Xdot_pred, lam=theta_reg)
    return z, theta


def legendre_nodes_weights(n, a, b):
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss(n)
    nodes = jnp.array(nodes)
    weights = jnp.array(weights)
    width = b - a
    nodes = (width) / 2 * nodes + (a + b) / 2
    weights = (width / 2) * weights
    return nodes, weights


def row_block_diag(V):
    n, d = V.shape
    eye = jnp.eye(n)[:, :, None]  # Shape (n, n, 1)
    V_exp = V[None, :, :]  # Shape (1, n, d)
    blocks = eye * V_exp  # Shape (n, n, d)
    return blocks.reshape(n, n * d)  # Shape (n, n*d)
