import numpy as np
from scipy.integrate import solve_ivp

from pysindy.utils import linear_damped_SHO
from pysindy.utils import lorenz

n_spectral = 1
fd_order = 1


def gen_data_sine(noise_level: float):
    # True data
    x = np.linspace(0, 2 * np.pi / 15, 15)
    y = np.sin(x)
    y_dot = np.cos(x)

    # Add noise
    seed = 111
    np.random.seed(seed)
    y_noisy = y + noise_level * np.random.randn(len(y))
    return x, y, y_noisy, y_dot


def gen_data_step(noise_level: float):
    # True data
    x = np.linspace(-1, 1, 20)
    y = np.abs(x)
    y_dot = np.sign(x)

    # Add noise
    seed = 111
    np.random.seed(seed)
    y_noisy = y + noise_level * np.random.randn(len(y))
    return x, y, y_dot, y_noisy


def gen_data_sho(noise_level: float, integrator_keywords: dict):
    dt = 0.01
    t_train = np.arange(0, 0.5, dt)
    t_train_span = (t_train[0], t_train[-1])
    x0_train = [2, 0]
    x_train = solve_ivp(
        linear_damped_SHO, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
    ).y.T
    x_train_noisy = x_train + noise_level * np.random.randn(*x_train.shape)
    return dt, t_train, x_train, x_train_noisy


def gen_data_lorenz(noise_level: float, integrator_keywords: dict):
    dt = 0.002
    t_train = np.arange(0, 0.5, dt)
    t_train_span = (t_train[0], t_train[-1])
    x0_train = [-8, 8, 27]
    x_train = solve_ivp(
        lorenz, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
    ).y.T
    x_train_noisy = x_train + noise_level * np.random.randn(*x_train.shape)
    return dt, t_train, x_train, x_train_noisy
