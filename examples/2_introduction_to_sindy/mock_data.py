import numpy as np


def gen_data1():
    t = np.linspace(0, 0.1, 10)
    x = 3 * np.exp(-2 * t)
    y = 0.5 * np.exp(t)
    return t, x, y


def gen_data2():
    x0 = 6
    y0 = -0.1
    t_test = np.linspace(0, 0.1, 10)
    x_test = x0 * np.exp(-2 * t_test)
    y_test = y0 * np.exp(t_test)
    return x0, y0, t_test, x_test, y_test
