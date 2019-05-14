import warnings

import numpy as np
from scipy.integrate import odeint
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.utils import check_random_state

from sindy.model import SINDy


def rhs_harmonic_oscillator(y, t):
    dy0 = y[1]
    dy1 = -0.3 * y[0]
    return [dy0, dy1]


x0 = [0, 1]
t = np.linspace(0, 10, 1000)
x = odeint(rhs_harmonic_oscillator, x0, t)
x_train, x_test = x[:750], x[750:]
kw = dict(fit_intercept=True, normalize=False)
model = SINDy(dt=t[1] - t[0], degree=2, alpha=0.3, kw=kw)
rng = check_random_state(42)
cv = KFold(n_splits=5, random_state=rng, shuffle=False)
params = {"alpha": [0.1, 0.2, 0.3, 0.4, 0.5], "threshold": [0.1, 0.3, 0.5]}
grid = GridSearchCV(model, params, cv=cv)
with warnings.catch_warnings():  # suppress matrix illconditioned warning
    warnings.filterwarnings("ignore")
    grid.fit(x_train)
selected_model = grid.best_estimator_
print("Score on test data ", selected_model.score(x_test))
print(
    "Selected hyperparameter (alpha, threshold): ",
    selected_model.alpha,
    selected_model.threshold,
)
for i, eq in enumerate(selected_model.equations()):
    print("dx_{} / dt = ".format(i), eq)
print(
    "Complexity of the model (sum of coefficients and \
intercetps bigger than the threshold): ",
    selected_model.complexity,
)
