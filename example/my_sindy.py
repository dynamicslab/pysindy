"""
Some examples of what we think typical package use will look like.
"""
import numpy as np
from scipy.integrate import odeint
from sklearn.model_selection import KFold, GridSearchCV

from sindy import SINDy


# 
# Generate training data
# 

def rhs_harmonic_oscillator(y, t):
    dy0 = y[1]
    dy1 = -0.3 * y[0]
    return [dy0, dy1]

x0 = [0, 1]
t = np.linspace(0, 10, 1000)
x = odeint(rhs_harmonic_oscillator, x0, t)
x_train, x_test = x[:750], x[750:]

# 
# Model training
# 


# Most basic version
model = SINDy()
model.fit(x_train, t)

# Pass in derivative
model = SINDy()
model.fit(x_train, x_dot=x_dot_train)

# Define custom library for SINDy to use
# Similar functionality to sklearn.preprocessing.PolynomialFeatures
library_1 = Library(library_type='polynomial', degree=2)
library_2 = Library(library_type='fourier', degree=2)
library   = library_1 + library_2
model     = SINDy(library=library)
model.fit(x_train)

# Select best threshold with cross-validation
model      = SINDy()
cv         = KFold(n_splits=3, shuffle=False)
params     = {'threshold': [0.1, 0.5, 1.0]}
grid       = GridSearchCV(model, params, cv=cv)
grid.fit(x_train)
best_model = grid.best_estimator_


# 
# Features to add in later (after v0)
# 

# Fit to data with some parameters specified
kwargs = dict(threshold=0.1,
              max_iterations=10,
              normalize=False,
              differentiation_method='TV',
              optimizer='sr3',
              time_delay=False,
              n_jobs=5,
        )
model = SINDy(dt=0.1, library_type='polynomial', degree=3, **kwargs)
model.fit(x_train)


# Higher order systems (derivatives)? Later feature?
model = SINDy(order=3)
model.fit(x_train)



# 
# Model evaluation (after the model has been fit)
# 

# Print learned equations
print(model.equations())

# Get coefficients in learned equations
coeffs = model.coefficients()

# Use learned model to make predictions (i.e. solve an IVP)
prediction = model.predict(x_0, integrator='adaptive')

# Get model score
score = model.score(x_test, type='mse')