import numpy as np
from scipy.integrate import solve_ivp

import pysindy as ps


def get_models():
    t = np.linspace(0, 10, 500)
    rhs = ps.utils.lorenz
    x = solve_ivp(rhs, (t[0], t[-1]), (10, 10, 10), t_eval=t).y.T
    good_model = ps.SINDy().fit(x, t=t, feature_names=["x", "y", "z"])
    skip = 4
    ok_model = ps.SINDy().fit(x[::skip], t=t[::skip], feature_names=["x", "y", "z"])
    bad_model = ps.SINDy().fit(
        x[:: skip**2], t=t[:: skip**2], feature_names=["x", "y", "z"]
    )
    return t, x, good_model, ok_model, bad_model
