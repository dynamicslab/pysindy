# Generate dynamical system data
import numpy as np
from scipy.integrate import odeint


def f(x, t):
    return [np.sin(2 * x[1]), 1 - np.cos(x[0])]


dt = 0.01
t_train = np.arange(0, 10, dt)
x0_train = [1, 2]
x_train = odeint(f, x0_train, t_train)

with open("data.npy", "wb") as file:
    np.save(file, x_train)
