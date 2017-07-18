import numpy as np
from scipy.integrate import odeint

from sparsereg.model import SINDy

def rhs_harmonic_oscillator(y, t):
    dy0 = y[1]
    dy1 = -y[0]
    return [dy0, dy1]

x0 = [0, 1]
t = np.linspace(0, 10, 1000)

x = odeint(rhs_harmonic_oscillator, x0, t)

kw = dict(fit_intercept=True, normalize=False)
model = SINDy(dt=t[1]-t[0], degree=5, threshold=0.5, alpha=1.0, n_jobs=-1, kw=kw).fit(x)

for i, eq in enumerate(model.equations()):
    print("dx_{} / dt = ".format(i), eq)
