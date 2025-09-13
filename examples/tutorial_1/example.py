# %% [markdown]
# # Generating data and fitting pysindy models.
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dynamicslab/pysindy/v2.0?filepath=examples/tutorial_1.ipynb)
#
# This notebook discusses the relationship between SINDy and PySINDy,
# using a brief example showing how different objects in the SINDy method are represented in PySINDy.
# It may help to be familiar with the basics of SINDy, as presented in the original paper
# or [summarized here](https://pysindy.readthedocs.org/en/latest/summary.html).
#
# Suppose we have measurements of the position of a particle obeying the following dynamical system at different points in time:
#
# $$
#     \frac{d}{dt} \begin{bmatrix} x \\ y \end{bmatrix}
#     = \begin{bmatrix} -2x \\ y \end{bmatrix}
#     = \begin{bmatrix} -2 & 0 \\ 0 & 1 \end{bmatrix}
#     \begin{bmatrix} x \\ y \end{bmatrix}
# $$
#
# Note that this system of differential equations decouples into two differential equations whose solutions are simply $x(t) = x_0e^{-2t}$ and $y(t) = y_0e^t$, where $x_0 = x(0)$ and $y_0=y(0)$ are the initial conditions.
#
# Using the initial conditions $x_0 = 3$ and $y_0 = \tfrac{1}{2}$, we construct the data matrix $X$.
# %%
import numpy as np

import pysindy as ps

if __name__ != "testing":
    from example_data import gen_data1
    from example_data import gen_data2
else:
    from mock_data import gen_data1
    from mock_data import gen_data2

# %% [markdown]
# `pysindy` requires training data, typically observations of time series data.
# Observation arrays follow the following axis conventions:
# `(spatial_1, ..., spatial_n, time, coordinate)`.
# For ODEs (no spatial dependence), that means the first axis is time,
# the second axis is coordinate.
# `pysindy` also requires the timepoints of the observations.
# While there are several ways to pass this information, the most straightfowrads
# is a 1-D array of timepoints.

# %%


t, x, y = gen_data1()
X = np.stack((x, y), axis=-1)  # First column is x, second is y
print(f"Data is shape: {X.shape}")
print(f"time is shape: {t.shape}")

# %% [markdown]
# ## Creating the `model` object
#
# The main model object is `pysindy.SINDy`, which has several components.
# Each component controls a different part of the SINDy process.

# %% [markdown]
# We first select a differentiation method from the `differentiation` submodule,
# which are also exposed as part of the top-level API.
# Here, we choose `FiniteDifference`, the default, which is good when observations are noiseless.
# Each differentiation method has different kwargs to control smoothness, accuracy, etc.
# Here, we set `order=2` for second-order finite differences

# %%
differentiation_method = ps.FiniteDifference(order=2)

# %% [markdown]
# The candidate library can be specified with an object from the `feature_library` submodule.
# `PolynomialLibrary` is a good default; after all, most functions can be approximated by a polynomial.
# Moreover, most functions we study are polynomial differential equations.
#
# Each feature library has parameters to control exactly which features are added.
# Here, for example, we set the polynomial degree to 3, which includes terms such as
# $x^3$, $xy$, $y$, etc. as well as a constant term.  See the docs for more details of the parameters

# %%
feature_library = ps.PolynomialLibrary(degree=3)

# %% [markdown]
# Next we select which optimizer should be used.
# Sequentially-thresholded least squares is the default choice as used in the
# original paper.
# Without too much detail, it works by calculating a regularized regression, then dropping
# terms smaller than a cutoff threhsold.

# %%
optimizer = ps.STLSQ(threshold=0.1)

# %% [markdown]
# Finally, we bring these three components together in one `SINDy` object.

# %%
model = ps.SINDy(
    differentiation_method=differentiation_method,
    feature_library=feature_library,
    optimizer=optimizer,
)

# %% [markdown]
# Following the `scikit-learn` workflow, we first instantiate a `SINDy` class object with the desired properties, then fit it to the data in separate step.

# %%
model.fit(X, t=t, feature_names=["x", "y"])

# %% [markdown]
# We can inspect the governing equations discovered by the model and check whether they seem reasonable with the `print` function.

# %%
model.print()

# %% [markdown]
# ## Alternate ways of passing data to `fit()`
#
# **Uniform timestep**
#
# If all observations are separated by a uniform timestep, you can simply pass a scalar
# $dt$ in place of an array or list of observation times

# %%
dt = t[1] - t[0]
model.fit(X, t=dt, feature_names=["x", "y"])
model.print()

# %% [markdown]
# **Multiple Trajectories**
#
# Depending on your use case, you may have multiple trajectories of the same system.
# It is possible to fit multiple trajectories by passing `x` and `t` as lists of arrays.
# In this case, each entry of the `x` list must have a corresponding number of time points
# as the same entry in the `t` list, or you need to pass the scalar `t=dt`.  To wit:

# %%
_, _, t2, x2, y2, x2_dot, y2_dot = gen_data2()

X2 = np.stack((x2, y2), axis=-1)

print(f"First trajectory has {len(t)} timepoints")
print(f"Second trajectory has {len(t2)} timepoints")
print(
    f"Since time intervals are not the same (dt1={t[1]-t[0]:.4f},"
    f"dt2={t2[1]-t2[0]:.4f}), we can't just pass t=dt"
)

model.fit([X, X2], [t, t2], feature_names=["x", "y"])
model.print()

# %% [markdown]
# **Known `x_dot`**
#
# The final data format you can use to fit a model is explicitly passing `x_dot`.
# This can be useful if you measure the derivative (e.g. doppler, accelerometer),
# if you calculate the derivative manually from another package,
# or if you know the true derivative values and are evaluating the performance of SINDy
# in a best-case scenario.
# The array format is the same for `x_dot` as for `x`, and can be combined with
# the multiple trajectories approach.

# %%
X2_dot = np.stack((x2_dot, y2_dot), axis=-1)
model.fit(X2, t2, x_dot=X2_dot, feature_names=["x", "y"])
model.print()

# %% [markdown]
# There are more things you can do with a fitted model beyond printing;
# You can predict, simulate, and more depending upon the components used in
# the model.
# You can also examine `feature_library` and `optimizer`, which have additional information
# about the fitting process and problem shape, e.g. `feature_library.n_features_out_`.
#
# The next tutorial will cover evaluating and visualizing a model fit.
