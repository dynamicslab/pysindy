# %% [markdown]
# # Differentiators in PySINDy
#
# This notebook explores the differentiation methods available in PySINDy. Most of the methods are powered by the [derivative](https://pypi.org/project/derivative/) package. While this notebook explores these methods on temporal data, these apply equally well to the computation of spatial derivatives for SINDy for PDE identification (see example Jupyter notebooks 10 and 12, on PDEs and weak forms).
# %% [markdown]
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dynamicslab/pysindy/v1.7?filepath=examples/5_differentiation.ipynb)
# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np

import pysindy as ps

# ignore user warnings
warnings.filterwarnings("ignore", category=UserWarning)

integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12

from utils import (
    compare_methods,
    print_equations,
    compare_coefficient_plots,
    plot_sho,
    plot_lorenz,
)

if __name__ != "testing":
    from example_data import (
        gen_data_sine,
        gen_data_step,
        gen_data_sho,
        gen_data_lorenz,
        n_spectral,
        fd_order,
    )
else:
    from mock_data import (
        gen_data_sine,
        gen_data_step,
        gen_data_sho,
        gen_data_lorenz,
        n_spectral,
        fd_order,
    )


# %% [markdown]
# In the cell below we define all the available differentiators. Note that the different options in `SINDyDerivative` all originate from `derivative`.
#
# * `FiniteDifference` - First order (forward difference) or second order (centered difference) finite difference methods with the ability to drop endpoints. Does *not* assume a uniform time step. Appropriate for smooth data.
# * `finite_difference` - Central finite differences of any order. Assumes a uniform time step. Appropriate for smooth data.
# * `Smoothed Finite Difference` - `FiniteDifference` with a smoother (default is Savitzky Golay) applied to the data before differentiation. Appropriate for noisy data.
# * `savitzky_golay` - Perform a least-squares fit of a polynomial to the data, then compute the derivative of the polynomial. Appropriate for noisy data.
# * `spline` - Fit the data with a spline (of arbitrary order) then perform differentiation on the spline. Appropriate for noisy data.
# * `trend_filtered` - Use total squared variations to fit the data (computes a global derivative that is a piecewise combination of polynomials of a chosen order). Set `order=0` to obtain the total-variational derivative. Appropriate for noisy data
# * `spectral` - Compute the spectral derivative of the data via Fourier Transform. Appropriate for very smooth (i.e. analytic) data. There is an in-house PySINDy version for speed but this is also included in the derivative package.

# %%
diffs = [
    ("PySINDy Finite Difference", ps.FiniteDifference()),
    ("Finite Difference", ps.SINDyDerivative(kind="finite_difference", k=1)),
    ("Smoothed Finite Difference", ps.SmoothedFiniteDifference()),
    (
        "Savitzky Golay",
        ps.SINDyDerivative(kind="savitzky_golay", left=0.5, right=0.5, order=3),
    ),
    ("Spline", ps.SINDyDerivative(kind="spline", s=1e-2)),
    ("Trend Filtered", ps.SINDyDerivative(kind="trend_filtered", order=0, alpha=1e-2)),
    ("Spectral", ps.SINDyDerivative(kind="spectral")),
    ("Spectral, PySINDy version", ps.SpectralDerivative()),
]

# %% [markdown]
# ## Compare differentiation methods directly
# First we'll use the methods to numerically approximate derivatives to measurement data directly, without bringing SINDy into the picture. We'll compare the different methods' accuracies when working with clean data ("approx" in the plots) and data with a small amount of white noise ("noisy").

# %%
noise_level = 0.01

# %% [markdown]
# ### Sine

# %%
# True data
x, y, y_noisy, y_dot = gen_data_sine(noise_level)
axs = compare_methods(diffs, x, y, y_noisy, y_dot)

# %% [markdown]
# ### Absolute value

# %%
# Shrink window for Savitzky Golay method
diffs[3] = (
    "Savitzky Golay",
    ps.SINDyDerivative(kind="savitzky_golay", left=0.1, right=0.1, order=3),
)

x, y, y_dot, y_noisy = gen_data_step(noise_level)

axs = compare_methods(diffs, x, y, y_noisy, y_dot)

# %% [markdown]
# ## Compare differentiators when used in PySINDy
# We got some idea of the performance of the differentiation options applied to raw data. Next we'll look at how they work as a single component of the SINDy algorithm.

# %% [markdown]
# ### Linear oscillator
# $$ \frac{d}{dt} \begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix} -0.1 & 2 \\ -2 & -0.1 \end{bmatrix} \begin{bmatrix}x \\ y\end{bmatrix} $$

# %%
noise_level = 0.1

# %%
# Generate training data

dt, t_train, x_train, x_train_noisy = gen_data_sho(noise_level, integrator_keywords)

# %%
figure = plt.figure(figsize=[5, 5])
plot_sho(x_train, x_train_noisy)

# %%
# Allow Trend Filtered method to work with linear functions
diffs[5] = (
    "Trend Filtered",
    ps.SINDyDerivative(kind="trend_filtered", order=1, alpha=1e-2),
)

equations_clean = {}
equations_noisy = {}
coefficients_clean = {}
coefficients_noisy = {}
input_features = ["x", "y"]
threshold = 0.5

for name, method in diffs:
    model = ps.SINDy(
        differentiation_method=method,
        optimizer=ps.STLSQ(threshold=threshold),
        t_default=dt,
        feature_names=input_features,
    )

    model.fit(x_train, quiet=True)
    equations_clean[name] = model.equations()
    coefficients_clean[name] = model.coefficients()

    model.fit(x_train_noisy, quiet=True)
    equations_noisy[name] = model.equations()
    coefficients_noisy[name] = model.coefficients()

# %%
print_equations(equations_clean, equations_noisy)

# %%
feature_names = model.get_feature_names()
compare_coefficient_plots(
    coefficients_clean,
    coefficients_noisy,
    input_features=input_features,
    feature_names=feature_names,
)

# %% [markdown]
# ### Lorenz system
#
# $$ \begin{aligned} \dot x &= 10(y-x)\\ \dot y &= x(28 - z) - y \\ \dot z &= xy - \tfrac{8}{3} z, \end{aligned} $$
#

# %%
noise_level = 0.5

# %%
# Generate measurement data
dt, t_train, x_train, x_train_noisy = gen_data_lorenz(noise_level, integrator_keywords)

# %%
fig = plt.figure(figsize=(8, 8))
plot_lorenz(x_train, x_train_noisy)
fig.show()

# %%
equations_clean = {}
equations_noisy = {}
coefficients_clean = {}
coefficients_noisy = {}
input_features = ["x", "y", "z"]

threshold = 0.5

for name, method in diffs:
    model = ps.SINDy(
        differentiation_method=method,
        optimizer=ps.STLSQ(threshold=threshold),
        t_default=dt,
        feature_names=input_features,
    )

    model.fit(x_train, quiet=True)
    equations_clean[name] = model.equations()
    coefficients_clean[name] = model.coefficients()

    model.fit(x_train_noisy, quiet=True)
    equations_noisy[name] = model.equations()
    coefficients_noisy[name] = model.coefficients()

# %%
print_equations(equations_clean, equations_noisy)

# %%
feature_names = model.get_feature_names()
compare_coefficient_plots(
    coefficients_clean,
    coefficients_noisy,
    input_features=input_features,
    feature_names=feature_names,
)

# %%
import timeit

N_spectral = np.logspace(1, 8, n_spectral, dtype=int)
spectral_times = np.zeros((n_spectral, 2))
for i in range(n_spectral):
    # True data
    x = np.linspace(0, 2 * np.pi, N_spectral[i])
    y = np.sin(x)
    y_dot = np.cos(x)
    noise_level = 0.05
    y_noisy = y + noise_level * np.random.randn(len(y))

    start = timeit.default_timer()
    spectral1 = ps.SINDyDerivative(kind="spectral")(y_noisy, x)
    stop = timeit.default_timer()
    spectral_times[i, 0] = stop - start

    start = timeit.default_timer()
    spectral2 = ps.SpectralDerivative(y_noisy, x)
    stop = timeit.default_timer()
    spectral_times[i, 1] = stop - start

# %%
plt.figure()
plt.grid(True)
plt.semilogy(spectral_times[:, 0], label="derivative python package")
plt.semilogy(spectral_times[:, 1], label="In-house spectral derivative")
plt.ylabel("Time (s)")
plt.xlabel("Matrix size in powers of 10")
plt.legend()
plt.show()

# %%
# Check error improves as order increases
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x) - x**5
y_dot = np.cos(x) - 5 * x**4
err = np.zeros(9)
for order in range(1, fd_order + 1):
    diff = ps.FiniteDifference(d=1, order=order)
    diff = diff(y, x)
    err[order - 1] = np.sum(np.abs(y_dot - diff))
plt.figure()
plt.plot(range(1, 10), err)
plt.grid(True)
plt.ylabel("Derivative error")
plt.xlabel("Finite difference order")
plt.show()
