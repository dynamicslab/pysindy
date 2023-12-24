import random
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax.experimental.ode import odeint
from functools import partial


# Define a hankel matrix
def multi_hankel_matrix2(x, n_tsteps):
    """
    x: (n_dim, time_steps) solution
    n_tsteps: number of time steps to integrate solution
    returns H, a (n_tsteps, n_dim, n_examples) hankel-like matrix
    """

    n_dim = x.shape[1]
    n_examples = x.shape[0] - n_tsteps
    H = np.zeros((n_tsteps, n_dim, n_examples))
    for i in range(n_dim):
        for j in range(n_tsteps):
            H[j, i, :] = x[j : j + n_examples, i]

    return jnp.array(H)


# Simulate lorenz system
# @jax.jit
def lorenz_system(X, t, theta):
    """
    Defines the dynamical system as a function of the state vector x, time t, and
    model parameters theta.
    """
    # Define the differential equation
    dx_dt = theta[0] * (X[1] - X[0])
    dy_dt = X[0] * (theta[1] - X[2]) - X[1]
    dz_dt = X[0] * X[1] - theta[2] * X[2]
    return jnp.array([dx_dt, dy_dt, dz_dt])

"""
Generate the polynomial feature matrix up to degree 2
Input: X (N, 3)
Output: 
    X_ext (N, 1^0 + 3^1 + 3^2)
    Corresponding to 0-order, 1-order, 2-order
"""

def feature_matrix(X):
    # [x, y, z, x^2, xy, xz, yx, y^2, yz, zx, zy, z^2]
    N, dim = X.shape
    features = jnp.zeros((N, 1 + dim + dim**2))
    features[:, 0] = 1
    features[:, 1:4] = X[:, :3]
    features[:, 4] = X[:, 0]*X[:, 0]
    features[:, 5] = X[:, 0]*X[:, 1]
    features[:, 6] = X[:, 0]*X[:, 2]
    features[:, 7] = X[:, 1]*X[:, 0]
    features[:, 8] = X[:, 1]*X[:, 1]
    features[:, 9] = X[:, 1]*X[:, 2]
    features[:, 10] = X[:, 2]*X[:, 0]
    features[:, 11] = X[:, 2]*X[:, 1]
    features[:, 12] = X[:, 2]*X[:, 2]

    return features

def feature_matrix(X):
    N, dim = X.shape
    assert dim >= 3, "Dimension of input X should be at least 3"

    # Create the second order terms
    x2 = X[:, 0, None] * X[:, 0, None]
    xy = X[:, 0, None] * X[:, 1, None]
    xz = X[:, 0, None] * X[:, 2, None]
    y2 = X[:, 1, None] * X[:, 1, None]
    yz = X[:, 1, None] * X[:, 2, None]
    z2 = X[:, 2, None] * X[:, 2, None]

    # Concatenate all the features
    # Shape (N, 1+3+6)
    features = jnp.concatenate([jnp.zeros((N, 1)), X[:, :3], x2, xy, xz, y2, yz, z2], axis=1)
    return features


def predict(x, theta, n_tstep):
    t_pred = np.linspace(0, 0.02002002, n_tstep)
    
    # odeint requires t to be a dummy variable
    def system(x, t, theta):
        x_ext = feature_matrix(x)
        assert x_ext.shape[1] == theta.shape[0], \
            f"the shape of x_ext ({x_ext.shape}) and theta ({theta.shape}) don't match. "
        return x_ext @ theta

    # Solve system ODE with parameters theta, initial condition x0, and time t_pred
    # TODO: this should be more general, we should not be able to know 
    # the structure of the system.

    sol = odeint(system, x, t_pred, theta)
    return sol.transpose([1,2,0])



def cost_function(predicted, data, theta, l1param=None):
    print("data shape = ", data.shape)
    N, dim, step = data.shape
    loss = ((predicted - data)**2).sum() / step / N

    if l1param is not None:
        loss += l1param * jnp.sum(jnp.abs(theta))
    return loss


def old_cost_function(system, data, theta, t, l1param=None):
    # Integrate the system forward in time and evaluate loss
    n_tsteps = data.shape[0]
    n_examples = data.shape[2]
    t_pred = t[:n_tsteps]
    x0 = data[0, :, :]
    print("data shape = ", data.shape)
    print("n_tsteps * n_examples = ", n_tsteps * n_examples)

    # Solve system ODE with parameters theta, initial condition x0, and time t_pred
    x = odeint(system, x0, t_pred, theta)
    loss = jnp.sum((x - data) ** 2) / n_examples / n_tsteps

    if l1param is not None:
        # compute l1 norm of theta (regularization)
        loss += l1param * jnp.sum(jnp.abs(theta))

    return loss

# - What is the role of t?
# x_shape (N, # feature)

x0 = jnp.array([1.0, 1.0, 1.0])
l1param = 1e-4
sigma = 10
beta = 8/3
rho = 28
# DEBUG: the constant term is 0 for debug purpose
Xi_lorenz = np.zeros((10, 3)) # constant term
Xi_lorenz[0, :] = [0, 0, 0] # 1
Xi_lorenz[1, :] = [-sigma, rho, 0] # x
Xi_lorenz[2, :] = [sigma, -1, 0] # y
Xi_lorenz[3, :] = [0, 0, -beta] # z
Xi_lorenz[4, :] = [0, 0, 0] # x^2
Xi_lorenz[5, :] = [0, 0, 1] # xy
Xi_lorenz[6, :] = [0, -1, 0] # xz
Xi_lorenz[7, :] = [0, 0, 0] # y^2
Xi_lorenz[8, :] = [0, 0, 0] # yz
Xi_lorenz[9, :] = [0, 0, 0] #z^2
feature_names = ['1', 'x', 'y', 'z', 'x^2', 'xy', 'xz', 'y^2', 'yz', 'z^2']

for ntime in [1000]:
    t = jnp.linspace(0, 20, ntime)
    sol = odeint(lorenz_system, x0, t, (10, 28, 8 / 3))
    # sol += 1 * (np.random.normal(0, 1, sol.shape) - 0.5)


    for n_tstep in [2, 8]:
        print("ntime, n_tstep = ", ntime, n_tstep)
        # (N, 3, n_tstep)
        H = multi_hankel_matrix2(sol, n_tstep).transpose([2,1,0])

        # optimization parameters
        epochs = 1000
        learning_rate = 0.2
        learning_rate = optax.exponential_decay(
            0.05,
            5000,
            0.95,
        )

        # Define the optimizer
        optimizer = optax.adam(learning_rate)

        # Initialize the parameters
        # params = {"theta": jnp.array([0.0, 0.0, 0.0])}
        params = jnp.zeros((1+3+6, 3))

        # Initialize the optimizer
        opt_state = optimizer.init(params)

        # Define the loss function
        # compute_loss = lambda params, data: cost_function(lorenz_system, data, params['theta'], l1param=0.01)

        # Define predict function

        # SIG predict(x, theta, n_tstep)
        sys_predict = partial(predict, n_tstep=n_tstep)
        # SIG cost_function(predicted, data, theta, l1param=None)
        sys_cost_function = partial(cost_function, l1param=l1param)
        
        def sys_compute_loss(param, x, H):
            pred = sys_predict(x, theta=param)
            loss = sys_cost_function(pred, theta=param, data=H)
            return loss

        # jit compilation
        jit_predict = jax.jit(sys_predict)        
        jit_loss = jax.jit(sys_compute_loss)
        jit_grad_loss = jax.jit(jax.grad(sys_compute_loss))

        train_ratio = 0.25
        # TODO: train test split. There's no such thing right now
        N = H.shape[0]
        batch = max(1, int(train_ratio * N))
        print("batch = ", batch)

        print_iter = 200
        loss = np.zeros(epochs // print_iter)
        t1 = time.time()
        for i in range(epochs):

            # Take subsample of H
            idxs = random.sample(range(N), batch)
            Hsub = H[idxs, :, :]
            x = Hsub[:, :, 0]

            # Update the parameters
            grads = jit_grad_loss(params, x, Hsub)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            if i % print_iter == 0:
                print("Epoch: ", i)
                print("params: ", params)
                loss[i // print_iter] = jit_loss(params, x, Hsub)
                print("Loss :", loss[i // print_iter])
                print("-------")

        metric = ((params - Xi_lorenz)**2).sum()
        print("Metric:", metric)

        t2 = time.time()
        print("Final params: ", params)
        print("Total optimization time = ", t2 - t1)
        plt.figure(2)
        plt.semilogy(
            np.arange(0, epochs, print_iter),
            loss,
            "o",
            label="ntime = " + str(ntime) + ", n_tstep = " + str(n_tstep),
        )
        plt.grid(True)
        plt.xlabel("iterations")
        plt.ylabel("objective")
        plt.legend()
        plt.figure(3)
        plt.semilogy(
            t2 - t1,
            metric,
            "o",
            label="ntime = " + str(ntime) + ", n_tstep = " + str(n_tstep),
        )
        plt.grid(True)
        plt.xlabel("t (s)")
        plt.ylabel("metric")
        plt.legend()
        plt.figure(4)
        for ii in range(3):
            plt.subplot(3, 1, ii + 1)
            plt.plot(
                range(10),
                params[:, ii],
                "o",
                label="ntime = " + str(ntime) + ", n_tstep = " + str(n_tstep),
            )
            plt.grid(True)
            plt.xlabel("Linear coefficients")
            plt.ylabel("Coefficient values")
            plt.legend()
            ax = plt.gca()
            ax.set_xticks(range(10))
            ax.set_xticklabels(feature_names)
plt.figure(4)
for ii in range(3):
    plt.subplot(3, 1, ii + 1)
    plt.plot(
        range(10),
        Xi_lorenz[:, ii],
        "o",
        label="Ground Truth",
    )
    plt.legend()
plt.show()
