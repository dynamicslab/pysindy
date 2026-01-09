from logging import getLogger

import jax
import jax.numpy as jnp
from jax.nn import softplus
from jaxopt import LBFGS

from .kernels import softplus_inverse
from .kerneltools import vectorize_kfunc
from .tree_opt import run_gradient_descent
from .tree_opt import run_jaxopt_solver

logger = getLogger(__name__)


def build_neg_marglike(X, y):
    if jnp.ndim(y) == 1:
        m = 1
    elif jnp.ndim(y) == 2:
        m = y.shape[1]
    else:
        raise ValueError("y must be either a 1 or two dimensional array")

    def neg_marginal_likelihood(kernel, sigma2):
        K = vectorize_kfunc(kernel)(X, X)
        identity = jnp.eye(len(X))

        C = jax.scipy.linalg.cholesky(K + sigma2 * identity, lower=True)
        logdet = 2 * jnp.sum(jnp.log(jnp.diag(C)))
        yTKinvY = jnp.sum((jax.scipy.linalg.solve_triangular(C, y, lower=True)) ** 2)
        return m * logdet + yTKinvY

    def loss(params):
        k = params["kernel"]
        sigma2 = softplus(params["transformed_sigma2"])
        return neg_marginal_likelihood(k, sigma2)

    return loss


def build_loocv(X, y):
    def loocv(kernel, sigma2):
        k = vectorize_kfunc(kernel)
        K = k(X, X)
        identity = jnp.eye(len(X))
        P = jnp.linalg.inv(K + sigma2 * identity)
        KP = K @ P
        loo_preds = K @ P @ y - (jnp.diag(KP) / jnp.diag(P)) * (P @ y)
        mse_loo = jnp.mean((loo_preds - y) ** 2)
        return mse_loo

    def loss(params):
        k = params["kernel"]
        sigma2 = softplus(params["transformed_sigma2"])
        return loocv(k, sigma2)

    return loss


def fit_kernel(
    init_kernel,
    init_sigma2,
    X,
    y,
    loss_builder=build_neg_marglike,
    gd_tol=1e-1,
    lbfgs_tol=1e-5,
    max_gd_iter=1000,
    max_lbfgs_iter=1000,
):
    loss = loss_builder(X, y)
    init_params = {
        "kernel": init_kernel,
        "transformed_sigma2": jnp.array(softplus_inverse(init_sigma2)),
    }
    logger.info("Warm starting marginal likelihood with gradient descent")
    params, conv_history_gd = run_gradient_descent(
        loss, init_params, tol=gd_tol, maxiter=max_gd_iter
    )
    solver = LBFGS(loss, maxiter=max_lbfgs_iter, tol=lbfgs_tol)
    logger.info("Solving marginal likelihood with LBFGS")
    params, conv_history_bfgs, state = run_jaxopt_solver(solver, params)
    conv_hist = [conv_history_gd, conv_history_bfgs]

    return params["kernel"], jax.nn.softplus(params["transformed_sigma2"]), conv_hist
