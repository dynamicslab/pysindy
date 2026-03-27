import jax
import jax.numpy as jnp
from jax.nn import softplus
from jaxopt import LBFGS
from jsindy.kerneltools import vectorize_kfunc

from .kernels import softplus_inverse
from .tree_opt import run_gradient_descent
from .tree_opt import run_jaxopt_solver


SIGMA2_FLOOR = 1e-6
def build_neg_marglike(X,y):
    if jnp.ndim(y)==1:
        m = 1
    elif jnp.ndim(y)==2:
        m = y.shape[1]
    else:
        raise ValueError("y must be either a 1 or two dimensional array")
    
    
    def neg_marginal_likelihood(kernel,sigma2):
        K = vectorize_kfunc(kernel)(X,X)
        I = jnp.eye(len(X))

        C = jax.scipy.linalg.cholesky(K + sigma2 * I,lower = True)
        logdet = 2*jnp.sum(jnp.log(jnp.diag(C)))
        yTKinvY = jnp.sum(
            (jax.scipy.linalg.solve_triangular(C,y,lower = True))**2
            )
        return m * logdet + yTKinvY
    
    def loss(params):
        k = params['kernel']
        sigma2 = softplus(params['transformed_sigma2']) + SIGMA2_FLOOR
        return neg_marginal_likelihood(k,sigma2)
    
    return loss

def build_neg_marglike_partialobs(t,y,v):
    if jnp.ndim(y)==1:
        m = 1
    elif jnp.ndim(y)==2:
        m = y.shape[1]
    else:
        raise ValueError("y must be either a 1 or two dimensional array")
    
    def neg_marginal_likelihood(kernel,sigma2):
        Kt = vectorize_kfunc(kernel)(t,t)
        I = jnp.eye(len(t))
        VV = v@v.T
        K = Kt*VV

        C = jax.scipy.linalg.cholesky(K + sigma2 * I,lower = True)
        logdet = 2*jnp.sum(jnp.log(jnp.diag(C)))
        yTKinvY = jnp.sum(
            (jax.scipy.linalg.solve_triangular(C,y,lower = True))**2
            )
        return m * logdet + yTKinvY
    
    def loss(params):
        k = params['kernel']
        sigma2 = softplus(params['transformed_sigma2']) + SIGMA2_FLOOR
        return neg_marginal_likelihood(k,sigma2)
    
    return loss


def build_loocv(X,y):
    def loocv(kernel,sigma2):
        k = vectorize_kfunc(kernel)
        K = k(X,X)
        I = jnp.eye(len(X))
        P = jnp.linalg.inv(K + sigma2*I)
        KP = K@P
        loo_preds = K@P@y - (jnp.diag(KP)/jnp.diag(P))*(P@y)
        mse_loo = jnp.mean((loo_preds - y)**2)
        return mse_loo
    
    def loss(params):
        k = params['kernel']
        sigma2 = softplus(params['transformed_sigma2'])
        return loocv(k,sigma2)
    return loss

def build_random_split_obj(X, y, p=0.2, rng_key=None):
    """
    p: proportion of data to use as validation set (between 0 and 1)
    rng_key: optional JAX
    """
    n = X.shape[0]
    if rng_key is None:
        rng_key = jax.random.key(1)
    perm = jax.random.permutation(rng_key, n)
    n_val = int(jnp.round(p * n))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    Xtrain = X[train_idx]
    ytrain = y[train_idx]
    Xval = X[val_idx]
    yval = y[val_idx]

    def l2_cv(kernel, sigma2):
        K = vectorize_kfunc(kernel)(Xtrain, Xtrain)
        I = jnp.eye(len(ytrain))
        c = jnp.linalg.solve(K + sigma2 * I, ytrain)
        ypred = vectorize_kfunc(kernel)(Xval, Xtrain) @ c
        return jnp.mean((ypred - yval) ** 2)

    def loss(params):
        k = params['kernel']
        sigma2 = softplus(params['transformed_sigma2'])
        return l2_cv(k, sigma2)
    return loss

def build_every_other_obj(X,y):
    Xtrain = X[::2]
    ytrain = y[::2]
    Xval = X[1::2]
    yval = y[1::2]
    def l2_cv(kernel,sigma2):
        K = vectorize_kfunc(kernel)(Xtrain,Xtrain)
        I = jnp.eye(len(ytrain))
        c = jnp.linalg.solve(K + sigma2*I,ytrain)
        ypred = vectorize_kfunc(kernel)(Xval,Xtrain)@c
        return jnp.mean((ypred - yval)**2)

    def loss(params):
        k = params['kernel']
        sigma2 = softplus(params['transformed_sigma2'])
        return l2_cv(k,sigma2)
    return loss

def fit_kernel(
        init_kernel,
        init_sigma2,
        X,
        y,
        loss_builder = build_neg_marglike,
        gd_tol = 1e-4,
        lbfgs_tol = 1e-6,
        max_gd_iter = 3000,
        max_lbfgs_iter = 1000,
        show_progress=True,
        ):
    loss = loss_builder(X,y)
    init_params = {'kernel':init_kernel,
        'transformed_sigma2':jnp.array(softplus_inverse(init_sigma2))
        }

    params,conv_history_gd = run_gradient_descent(
        loss,init_params,tol = gd_tol,
        maxiter = max_gd_iter,
        show_progress=show_progress,
        init_stepsize=1e-4
        )
    solver = LBFGS(loss,maxiter = max_lbfgs_iter,tol = lbfgs_tol)
    params,conv_history_bfgs,state = run_jaxopt_solver(solver,params, show_progress=show_progress)
    conv_hist = [conv_history_gd,conv_history_bfgs]

    return params['kernel'],jax.nn.softplus(params['transformed_sigma2']) + SIGMA2_FLOOR,conv_hist

def fit_kernel_partialobs(
        init_kernel,
        init_sigma2,
        t,y,v,
        gd_tol = 1e-4,
        lbfgs_tol = 1e-6,
        max_gd_iter = 3000,
        max_lbfgs_iter = 1000,
        show_progress=True,
        ):
    loss = build_neg_marglike_partialobs(t,y,v)
    init_params = {'kernel':init_kernel,
        'transformed_sigma2':jnp.array(softplus_inverse(init_sigma2))
        }

    params,conv_history_gd = run_gradient_descent(
        loss,init_params,tol = gd_tol,
        maxiter = max_gd_iter,
        show_progress=show_progress,
        init_stepsize=1e-4
        )
    solver = LBFGS(loss,maxiter = max_lbfgs_iter,tol = lbfgs_tol)
    params,conv_history_bfgs,state = run_jaxopt_solver(solver,params, show_progress=show_progress)
    conv_hist = [conv_history_gd,conv_history_bfgs]

    return params['kernel'],jax.nn.softplus(params['transformed_sigma2']) + SIGMA2_FLOOR,conv_hist
