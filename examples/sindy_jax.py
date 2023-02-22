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


def cost_function(system, data, theta, ntime, t, l1param=None):
    # Integrate the system forward in time and evaluate loss
    n_tsteps = data.shape[0]
    n_examples = ntime
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


x0 = jnp.array([1.0, 1.0, 1.0])
for ntime in [1000, 10000]:
    t = jnp.linspace(0, 20, ntime)
    sol = odeint(lorenz_system, x0, t, (10, 28, 8 / 3))
    sys_cost_function = lambda H, coefs: cost_function(
        lorenz_system, H, coefs, l1param=0.0, ntime=ntime, t=t
    )

    for n_tstep in [2, 10, 100]:
        print("ntime, n_tstep = ", ntime, n_tstep)
        H = multi_hankel_matrix2(sol, n_tstep)
        cf_jit = jax.jit(sys_cost_function)

        # optimization parameters
        epochs = 4000
        learning_rate = 0.2

        # Define the optimizer
        optimizer = optax.adam(learning_rate)

        # Initialize the parameters
        params = {"theta": jnp.array([0.0, 0.0, 0.0])}

        # Initialize the optimizer
        opt_state = optimizer.init(params)

        # Define the loss function
        # compute_loss = lambda params, data: cost_function(lorenz_system, data, params['theta'], l1param=0.01)

        # Define the loss function
        compute_loss = lambda params, data: cf_jit(data, params["theta"])
        cl_jit = jax.jit(compute_loss)
        grads_jit = jax.jit(jax.grad(cl_jit))

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

            # Update the parameters
            grads = grads_jit(params, Hsub)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            if i % print_iter == 0:
                print("Epoch: ", i)
                print("params: ", params["theta"])
                loss[i // print_iter] = cl_jit(params, Hsub)
                print("Loss :", loss[i // print_iter])
                print("-------")

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
            loss[-1],
            "o",
            label="ntime = " + str(ntime) + ", n_tstep = " + str(n_tstep),
        )
        plt.grid(True)
        plt.xlabel("t (s)")
        plt.ylabel("objective")
        plt.legend()
plt.show()
