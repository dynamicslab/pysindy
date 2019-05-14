# Some examples of what we think typical package use will look like.


# 
# Model training
# 

# Most basic version
model = SINDy()
model.fit(X_train)

# Pass in derivative
model = SINDy()
model.fit(X_train, X_dot=X_dot_train)

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
model.fit(X_train)

# Define custom library for SINDy to use
# Similar functionality to sklearn.preprocessing.PolynomialFeatures
library_1 = Library(library_type='polynomial', degree=2)
library_2 = Library(library_type='fourier', degree=2)
library   = library_1 + library_2
model     = SINDy(library=library)
model.fit(X_train)

# Select best threshold with cross-validation
model      = SINDy()
cv         = KFold(n_splits=3, shuffle=False)
params     = {'threshold': [0.1, 0.5, 1.0]}
grid       = GridSearchCV(model, params, cv=cv)
grid.fit(X_train)
best_model = grid.best_estimator_

# Higher order systems (derivatives)? Later feature?
model = SINDy(order=3)
model.fit(X_train)



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