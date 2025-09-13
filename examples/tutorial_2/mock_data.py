from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp

import pysindy as ps


def get_models():
    t = np.linspace(0, 0.5, 20)
    rhs = ps.utils.lorenz
    x = solve_ivp(rhs, (t[0], t[-1]), (10, 10, 10), t_eval=t).y.T
    good_model = ps.SINDy().fit(x, t=t, feature_names=["x", "y", "z"])
    ok_model = deepcopy(good_model)
    ok_model.optimizer.coef_[1, 1] = 4
    bad_model = deepcopy(good_model)
    bad_model.optimizer.coef_[1, 1] = 10
    return t, x, good_model, ok_model, bad_model
