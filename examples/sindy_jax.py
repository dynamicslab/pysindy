import random
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax.experimental.ode import odeint
from functools import partial
import math
from itertools import combinations_with_replacement
from pysindy import PolynomialLibrary

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
# Feature library
# TODO: Can I reuse the polynomial library from sklearn?
# Questions: is it a good practice to use numpy and then convert it to jnp?
def feature_matrix(X):
    # TODO: this should target any size, as long as it only extends the last axis
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
    # TODO: the step size here is not adaptive
    # Question: what is the role of time here
    t_pred = np.linspace(0, 0.02002002*(n_tstep-1), n_tstep)
    
    # odeint requires t to be a dummy variable
    def system(x, t, theta):
        x_ext = feature_matrix(x)
        assert x_ext.shape[1] == theta.shape[0], \
            f"the shape of x_ext ({x_ext.shape}) and theta ({theta.shape}) don't match. "
        return x_ext @ theta

    sol = odeint(system, x, t_pred, theta)
    return sol.transpose([1,2,0])



def cost_function(predicted, data, theta, l1param=None):
    print("data shape = ", data.shape)
    N, dim, step = data.shape
    # DEBUG: I need to add "/3" to match the performance
    # of sindy_jax_update.py. I am not sure why, but this 
    # doesn't seem to be a large problem. Still, I want
    # to know why
    loss = ((predicted - data)**2).sum() / step / N / 3 
    

    if l1param is not None:
        loss += l1param * jnp.sum(jnp.abs(theta))
    return loss


# x_shape (t, # feature)

x0 = jnp.array([1.0, 1.0, 1.0])
l1param = 1e-4
sigma = 10
beta = 8/3
rho = 28
# DEBUG: the constant term is 0 for debug purpose
def get_GT():
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
    return Xi_lorenz
Xi_lorenz = get_GT()
print("Ground Truth")
print(np.array2string(Xi_lorenz, formatter={'float_kind':lambda x: f"{x:.2f}"}))


feature_names = ['1', 'x', 'y', 'z', 'x^2', 'xy', 'xz', 'y^2', 'yz', 'z^2']

# TODO: I need to implement a function that returns the coefficients
# given the training data


def multi_step_ridge_regression(H, n_targets, n_tstep, verbose=False):
    """
    Parameters
    ----------
        H : Hankel matrix, shape (num_traj, timestep, n_features)
        n_targets : the number of coordinate
        n_tstep: the number of steps

    Returns
    -------
        coef : the coefficient
    """

    num_traj, timestep, n_features = H.shape

    # optimization parameters
    l1param = 1e-4
    epochs = 1000
    learning_rate = 0.2
    learning_rate = optax.exponential_decay(
        0.05,
        5000,
        0.95,
    )
    # Define the optimizer
    optimizer = optax.adam(learning_rate)

    # Initialize the parameters and optimizer
    params = jnp.zeros((n_features, n_targets))
    opt_state = optimizer.init(params)

    def sys_predict(x, theta):
        # TODO: used n_tstep,n_targets from the larger context
        t_pred = np.linspace(0, 0.02002002*(n_tstep-1), n_tstep)
        x_coord = x[..., 1:1+n_targets]
        # DEBUG: here is incorrect
        grad_func = lambda x_coord, t, x, theta: x @ theta
        x_pred = odeint(grad_func, x_coord, t_pred, x, theta)
        return x_pred.transpose([1,0,2])

    # Define predict function
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
    batch = max(1, int(train_ratio * num_traj))
    if verbose: print("batch = ", batch)

    print_iter = 200
    loss = np.zeros(epochs // print_iter)
    t1 = time.time()
    for i in range(epochs):

        # Take subsample of H
        idxs = random.sample(range(num_traj), batch)
        Hsub = H[idxs, :, :]
        x = Hsub[:, 0, :]

        # Update the parameters
        grads = jit_grad_loss(params, x, Hsub[..., 1:1+n_targets])
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if i % print_iter == 0:
            print("Epoch: ", i)
            print("params: ", params)
            loss[i // print_iter] = jit_loss(params, x, Hsub[..., 1:1+n_targets])
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
    return params



class PolynomialFeatureTransformer:

    def __init__(self, input_dim, degree):
        self.input_dim = input_dim
        self.degree = degree
        self.output_dim = 0
        for i in range(degree + 1):
            self.output_dim += math.comb(input_dim + i - 1, i)
    
    def transform(self, x):
        ext_x = []
        # DEBUG: no constant terms for now
        ext_x.append(jnp.zeros(x.shape[:-1])[..., None])  # Add the bias term (degree 0)

        for d in range(1, self.degree + 1):
            for terms in combinations_with_replacement(range(self.input_dim), d):
                term = x[..., terms[0]]
                for t in terms[1:]:
                    term = term * x[..., t]
                ext_x.append(term[..., None])

        return jnp.concatenate(ext_x, axis=-1)

# TODO: different dimension
def mse_loss(y_true, y_pred):
    return jnp.mean((y_true - y_pred) ** 2)

class SINDy_trainer:

    """
        feature_transform: extend the dimension of the last axis of 
        the input matrix
    """
    def __init__(
            self, 
            feature_transformer,
            l1param=1e-7,
            epochs=2000,
            learning_rate=0.5,
        ):
        self.feature_transformer = feature_transformer
        self.n_features = feature_transformer.output_dim
        self.n_targets = feature_transformer.input_dim

        # optimization parameters
        self.l1param = l1param
        self.epochs = epochs
        # learning_rate = optax.exponential_decay(
        #     learning_rate,
        #     5000,
        #     0.95,
        # )

        # Define the optimizer
        self.optimizer = optax.adam(learning_rate)

        # Initialize the parameters and optimizer
        self.params = jnp.zeros((self.n_features, self.n_targets))
        # self.params = Xi_lorenz
        self.opt_state = self.optimizer.init(self.params)

    def fit(self, X, t_pred):
        def loss_func(params, x):
            loss = mse_loss(x, self.predict(x[..., 0, :], t_pred, params))
            loss += self.l1param * jnp.sum(jnp.abs(params))
            return loss

        value_and_grad_func = jax.value_and_grad(jax.jit(loss_func))

        print_ter = 200
        self.loss_history = []
        for i in range(self.epochs):
            # TODO: mini batch?
            loss, grads = value_and_grad_func(self.params, X)
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
            self.params = optax.apply_updates(self.params, updates)

            if i%print_ter==0:
                print(f"Epoch {i}: Loss {loss}")
            # print(self.params)
            self.loss_history.append(loss)

    # @staticmethod #TODO: add this
    def predict(self, x, t_pred, params):
        def sys(x, t):
            x = self.feature_transformer.transform(x)
            return x @ params
        
        sol = odeint(sys, x, t_pred)
        return sol.transpose([1, 0, 2])
    
    def plot_loss(self):
        if not self.loss_history:
            print("No losses recorded")
            return
        
        plt.plot(self.loss_history)
        plt.yscale("log")
        plt.xlabel("Epochs")
        plt.ylabel("Loss (log)")
        plt.show()



def main():

    for ntime in [5000]:
        timestamps = jnp.linspace(0, 30, ntime)
        sol = odeint(lorenz_system, x0, timestamps, (10, 28, 8 / 3))
        # sol += 1 * (np.random.normal(0, 1, sol.shape) - 0.5)


        for n_tstep in [2]:
            print("ntime, n_tstep = ", ntime, n_tstep)
            # x = feature_matrix(sol)
            H = multi_hankel_matrix2(sol, n_tstep).transpose([2,0,1])
            t_pred = timestamps[:n_tstep]

            feature_transformer = PolynomialFeatureTransformer(input_dim=3, degree=2)

            trainer = SINDy_trainer(feature_transformer, epochs=5000)
            trainer.fit(H, t_pred)
            trainer.plot_loss()

            # param = multi_step_ridge_regression(H, 3, n_tstep, verbose=True)
            print(np.array2string(trainer.params, formatter={'float_kind':lambda x: f"{x:.2f}"}))


if __name__=="__main__":
    main()