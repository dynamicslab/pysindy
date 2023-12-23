import random
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax.experimental.ode import odeint


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

def predict(system, data, theta, t):
    x0 = data[0, :, :]
    n_tsteps = data.shape[0]
    t_pred = t[:n_tsteps]

    # Solve system ODE with parameters theta, initial condition x0, and time t_pred
    # TODO: this should be more general, we should not be able to know 
    # the structure of the system.
    x = odeint(system, x0, t_pred, theta)
    return x



def cost_function(predicted, data, theta, l1param=None):
    error = (predicted - data)**2
    error = jnp.average(error, axis=0, weights=jax.nn.relu(1-(0.05*jnp.arange(error.shape[0]))))
    loss = jnp.average(error, axis=1).sum()
    # from IPython import embed
    # embed() or exit(0)
    # loss = jnp.mean((predicted-data)**2, axis=[0, 2]).sum()
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
for ntime in [5000]:
    t = jnp.linspace(0, 20, ntime)
    sol = odeint(lorenz_system, x0, t, (10, 28, 8 / 3))

    for n_tstep in [2, 10, 100]:
        print("ntime, n_tstep = ", ntime, n_tstep)
        H = multi_hankel_matrix2(sol, n_tstep)

        # optimization parameters
        epochs = 1000
        learning_rate = 0.2

        # Define the optimizer
        optimizer = optax.adam(learning_rate)

        # Initialize the parameters
        params = {"theta": jnp.array([0.0, 0.0, 0.0])}

        # Initialize the optimizer
        opt_state = optimizer.init(params)

        # Define the loss function
        # compute_loss = lambda params, data: cost_function(lorenz_system, data, params['theta'], l1param=0.01)

        # Define predict function
        sys_predict = lambda params, data: predict(lorenz_system, data, params['theta'], t=t)
        # Define the loss function
        compute_loss = lambda params, data: cost_function(
            sys_predict(params, data),
            data,
            params['theta'],
            l1param=0.0
        )

        # jit compilation
        jit_predict = jax.jit(sys_predict)        
        jit_loss = jax.jit(compute_loss)
        jit_grad_loss = jax.jit(jax.grad(compute_loss))

        train_ratio = 0.1
        nsamp = H.shape[2]
        batch = max(1, int(train_ratio * nsamp))
        print("batch = ", batch)

        print_iter = 50
        loss = np.zeros(epochs // print_iter)
        t1 = time.time()
        for i in range(epochs):

            # Take subsample of H
            idxs = random.sample(range(nsamp), batch)
            Hsub = H[:, :, idxs]

            # # Get predictions
            # if i > epochs-10: 
            #     preds = jit_predict(params, Hsub)
            #     error = ((preds - Hsub)[1]**2).mean(axis=0)
            #     plt.scatter(idxs, error)
            #     plt.show()
            # if i == epochs-2:
            #     exit(0)

            # Update the parameters
            grads = jit_grad_loss(params, Hsub)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            if i % print_iter == 0:
                # print("Epoch: ", i)
                # print("params: ", params["theta"])
                loss[i // print_iter] = jit_loss(params, Hsub)
                # print("Loss :", loss[i // print_iter])
                # print("-------")

        metric = ((params['theta'] - jnp.array([10, 28, 8 / 3]))**2).sum()
        print("Metric:", metric)

        t2 = time.time()
        print("Final params: ", params["theta"])
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
plt.show()
