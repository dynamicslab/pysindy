# %% [markdown]
# # Feature Overview
#
# This notebook provides a simple overview of the basic functionality of PySINDy. In addition to demonstrating the basic usage for fitting a SINDy model, we demonstrate several means of customizing the SINDy fitting procedure. These include different forms of input data, different optimization methods, different differentiation methods, and custom feature libraries.
#
# An interactive version of this notebook is available on binder.
# %% [markdown]
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dynamicslab/pysindy/v1.7.3?filepath=examples/1_feature_overview.ipynb)
# %%
import warnings
from contextlib import contextmanager
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

import pysindy as ps
from pysindy.utils import enzyme
from pysindy.utils import lorenz
from pysindy.utils import lorenz_control

if __name__ != "testing":
    t_end_train = 10
    t_end_test = 15
else:
    t_end_train = 0.04
    t_end_test = 0.04

data = (Path() / "../data").resolve()


@contextmanager
def ignore_specific_warnings():
    filters = copy(warnings.filters)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=LinAlgWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    yield
    warnings.filters = filters


if __name__ == "testing":
    import sys
    import os

    sys.stdout = open(os.devnull, "w")
# %%
# Seed the random number generators for reproducibility
np.random.seed(100)

# %% [markdown]
# ### A note on solve_ivp vs odeint before we continue
# The default solver for `solve_ivp` is a Runga-Kutta method (RK45) but this seems to work quite poorly on a number of these examples, likely because they are multi-scale and chaotic. Instead, the LSODA method seems to perform quite well (ironically this is the default for the older `odeint` method). This is a nice reminder that system identification is important to get the right model, but a good integrator is still needed at the end!

# %%
# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "LSODA"
integrator_keywords["atol"] = 1e-12

# %% [markdown]
# ## Basic usage
# We will fit a SINDy model to the famous Lorenz system:
# $$ \dot{x} = \sigma (y - x),$$
# $$ \dot{y} = x(\rho - z) - y, $$
# $$ \dot{z} = x y - \beta z. $$
#

# %% [markdown]
# ### Train the model

# %%
# Generate measurement data
dt = 0.002

t_train = np.arange(0, t_end_train, dt)
x0_train = [-8, 8, 27]
t_train_span = (t_train[0], t_train[-1])
x_train = solve_ivp(
    lorenz, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
).y.T

# %%
# Instantiate and fit the SINDy model
model = ps.SINDy()
model.fit(x_train, t=dt)
model.print()

# %% [markdown]
# ### Assess results on a test trajectory

# %%
# Evolve the Lorenz equations in time using a different initial condition
t_test = np.arange(0, t_end_test, dt)
x0_test = np.array([8, 7, 15])
t_test_span = (t_test[0], t_test[-1])
x_test = solve_ivp(
    lorenz, t_test_span, x0_test, t_eval=t_test, **integrator_keywords
).y.T

# Compare SINDy-predicted derivatives with finite difference derivatives
print("Model score: %f" % model.score(x_test, t=dt))

# %% [markdown]
# ### Predict derivatives with learned model

# %%
# Predict derivatives using the learned model
x_dot_test_predicted = model.predict(x_test)

# Compute derivatives with a finite difference method, for comparison
x_dot_test_computed = model.differentiate(x_test, t=dt)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_dot_test_computed[:, i], "k", label="numerical derivative")
    axs[i].plot(t_test, x_dot_test_predicted[:, i], "r--", label="model prediction")
    axs[i].legend()
    axs[i].set(xlabel="t", ylabel=r"$\dot x_{}$".format(i))
fig.show()

# %% [markdown]
# ### Simulate forward in time

# %%
# Evolve the new initial condition in time with the SINDy model
x_test_sim = model.simulate(x0_test, t_test)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_test[:, i], "k", label="true simulation")
    axs[i].plot(t_test, x_test_sim[:, i], "r--", label="model simulation")
    axs[i].legend()
    axs[i].set(xlabel="t", ylabel="$x_{}$".format(i))

fig = plt.figure(figsize=(10, 4.5))
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2], "k")
ax1.set(xlabel="$x_0$", ylabel="$x_1$", zlabel="$x_2$", title="true simulation")

ax2 = fig.add_subplot(122, projection="3d")
ax2.plot(x_test_sim[:, 0], x_test_sim[:, 1], x_test_sim[:, 2], "r--")
ax2.set(xlabel="$x_0$", ylabel="$x_1$", zlabel="$x_2$", title="model simulation")

fig.show()

# %% [markdown]
# ## Different forms of input data
#
# Here we explore different types of input data accepted by the the `SINDy` class.

# %% [markdown]
# ### Single trajectory, pass in collection times

# %%
model = ps.SINDy()
model.fit(x_train, t=t_train)
model.print()

# %% [markdown]
# ### Single trajectory, set default timestep
# Since we used a uniform timestep when defining `t_train` we can tell set a default timestep to be used whenever `t` isn't passed in.

# %%
model = ps.SINDy(t_default=dt)
model.fit(x_train)
model.print()

# %% [markdown]
# ### Single trajectory, pass in pre-computed derivatives

# %%
x_dot_true = np.zeros(x_train.shape)
for i in range(t_train.size):
    x_dot_true[i] = lorenz(t_train[i], x_train[i])

model = ps.SINDy()
model.fit(x_train, t=t_train, x_dot=x_dot_true)
model.print()

# %% [markdown]
# ### Multiple trajectories
# We use the Lorenz equations to evolve multiple different initial conditions forward in time, passing all the trajectories into a single `SINDy` object. Note that `x_train_multi` is a list of 2D numpy arrays.

# %%
if __name__ != "testing":
    n_trajectories = 20
    sample_range = (500, 1500)
else:
    n_trajectories = 2
    sample_range = (10, 15)
x0s = np.array([36, 48, 41]) * (np.random.rand(n_trajectories, 3) - 0.5) + np.array(
    [0, 0, 25]
)
x_train_multi = []
for i in range(n_trajectories):
    x_train_multi.append(
        solve_ivp(
            lorenz, t_train_span, x0s[i], t_eval=t_train, **integrator_keywords
        ).y.T
    )

model = ps.SINDy()
model.fit(x_train_multi, t=dt)
model.print()

# %% [markdown]
# ### Multiple trajectories, different lengths of time
# This example is similar to the previous one, but each trajectory consists of a different number of measurements.

# %%
x0s = np.array([36, 48, 41]) * (np.random.rand(n_trajectories, 3) - 0.5) + np.array(
    [0, 0, 25]
)
x_train_multi = []
t_train_multi = []
for i in range(n_trajectories):
    n_samples = np.random.randint(*sample_range)
    t = np.arange(0, n_samples * dt, dt)
    t_span = (t[0], t[-1])
    x_train_multi.append(
        solve_ivp(lorenz, t_span, x0s[i], t_eval=t, **integrator_keywords).y.T
    )
    t_train_multi.append(t)

model = ps.SINDy()
model.fit(x_train_multi, t=t_train_multi)
model.print()

# %% [markdown]
# ### Discrete time dynamical system (map)


# %%
def f(x):
    return 3.6 * x * (1 - x)


if __name__ != "testing":
    n_steps = 1000
else:
    n_steps = 10
eps = 0.001  # Noise level
x_train_map = np.zeros((n_steps))
x_train_map[0] = 0.5
for i in range(1, n_steps):
    x_train_map[i] = f(x_train_map[i - 1]) + eps * np.random.randn()
model = ps.SINDy(discrete_time=True)
model.fit(x_train_map)

model.print()

# %% [markdown]
# ### Pandas DataFrame

# %%
import pandas as pd

# Create a dataframe with entries corresponding to measurements and
# indexed by the time at which the measurements were taken
df = pd.DataFrame(data=x_train, columns=["x", "y", "z"], index=t_train)

# The column names can be used as feature names
model = ps.SINDy(feature_names=df.columns)

# Everything needs to be converted to numpy arrays to be passed in
model.fit(df.values, t=df.index.values)
model.print()

# %% [markdown]
# ## Optimization options
# In this section we provide examples of different parameters accepted by the built-in sparse regression optimizers `STLSQ`, `SR3`, `ConstrainedSR3`, `MIOSR`, `SSR`, and `FROLS`. The `Trapping` optimizer is not straightforward to use; please check out Example 8 for some examples. We also show how to use a scikit-learn sparse regressor with PySINDy.

# %% [markdown]
# ### STLSQ - change parameters

# %%
stlsq_optimizer = ps.STLSQ(threshold=0.01, alpha=0.5)

model = ps.SINDy(optimizer=stlsq_optimizer)
model.fit(x_train, t=dt)
model.print()

# %% [markdown]
# ### STLSQ - verbose (print out optimization terms at each iteration)
# The same verbose option is available with all the other optimizers. For optimizers that use the CVXPY
# package, there is additional boolean flag, verbose_cvxpy, that decides whether or not CVXPY solves will also be verbose.

# %%
stlsq_optimizer = ps.STLSQ(threshold=0.01, alpha=0.5, verbose=True)

model = ps.SINDy(optimizer=stlsq_optimizer)
model.fit(x_train, t=dt)
model.print()

# %% [markdown]
# ### SR3

# %%
sr3_optimizer = ps.SR3(threshold=0.1, thresholder="l1")

model = ps.SINDy(optimizer=sr3_optimizer)
model.fit(x_train, t=dt)
model.print()

# %% [markdown]
# ### SR3 - with trimming
# `SR3` is capable of automatically trimming outliers from training data. Specifying the parameter `trimming_fraction` tells the method what fraction of samples should be trimmed.

# %%
corrupted_inds = np.random.randint(0, len(x_train), size=len(x_train) // 30)
x_train_corrupted = x_train.copy()
x_train_corrupted[corrupted_inds] += np.random.standard_normal((len(corrupted_inds), 3))

# Without trimming
sr3_optimizer = ps.SR3()
with ignore_specific_warnings():
    model = ps.SINDy(optimizer=sr3_optimizer).fit(x_train_corrupted, t=dt)
print("Without trimming")
model.print()

# With trimming
sr3_optimizer = ps.SR3(trimming_fraction=0.1)
with ignore_specific_warnings():
    model = ps.SINDy(optimizer=sr3_optimizer).fit(x_train_corrupted, t=dt)
print("\nWith trimming")
model.print()

# %% [markdown]
# ### SR3 regularizers
# The default regularizer with SR3 is the L0 norm, but L1 and L2 are allowed. Note that the pure L2 option is notably less sparse than the other options.

# %%
sr3_optimizer = ps.SR3(threshold=0.1, thresholder="l0")
with ignore_specific_warnings():
    model = ps.SINDy(optimizer=sr3_optimizer).fit(x_train, t=dt)
print("L0 regularizer: ")
model.print()

sr3_optimizer = ps.SR3(threshold=0.1, thresholder="l1")
with ignore_specific_warnings():
    model = ps.SINDy(optimizer=sr3_optimizer).fit(x_train, t=dt)
print("L1 regularizer: ")
model.print()

sr3_optimizer = ps.SR3(threshold=0.1, thresholder="l2")
with ignore_specific_warnings():
    model = ps.SINDy(optimizer=sr3_optimizer).fit(x_train, t=dt)
print("L2 regularizer: ")
model.print()

# %% [markdown]
# ### SR3 - with variable thresholding
# `SR3` and its variants (ConstrainedSR3, TrappingSR3, SINDyPI) can use a matrix of thresholds to set different thresholds for different terms.

# %%
# Without thresholds matrix
sr3_optimizer = ps.SR3(threshold=0.1, thresholder="l0")
model = ps.SINDy(optimizer=sr3_optimizer).fit(x_train, t=dt)
print("Threshold = 0.1 for all terms")
model.print()

# With thresholds matrix
thresholds = 2 * np.ones((10, 3))
thresholds[4:, :] = 0.1
sr3_optimizer = ps.SR3(thresholder="weighted_l0", thresholds=thresholds)
model = ps.SINDy(optimizer=sr3_optimizer).fit(x_train, t=dt)
print("Threshold = 0.1 for quadratic terms, else threshold = 1")
model.print()

# %% [markdown]
# It can be seen that the x1 term in the second equation correctly gets truncated with these thresholds.
#
# ### ConstrainedSR3
# We can impose linear equality and inequality constraints on the coefficients in the `SINDy` model using the `ConstrainedSR3` class. Below we constrain the x0 coefficient in the second equation to be exactly 28 and the x0 and x1 coefficients in the first equations to be negatives of one another. See this [notebook](https://github.com/dynamicslab/pysindy/blob/master/examples/7_plasma_examples.ipynb) for examples.

# %%
# Figure out how many library features there will be
library = ps.PolynomialLibrary()
library.fit([ps.AxesArray(x_train, {"ax_sample": 0, "ax_coord": 1})])
n_features = library.n_output_features_
print(f"Features ({n_features}):", library.get_feature_names())

# %%
# Set constraints
n_targets = x_train.shape[1]
constraint_rhs = np.array([0, 28])

# One row per constraint, one column per coefficient
constraint_lhs = np.zeros((2, n_targets * n_features))

# 1 * (x0 coefficient) + 1 * (x1 coefficient) = 0
constraint_lhs[0, 1] = 1
constraint_lhs[0, 2] = 1

# 1 * (x0 coefficient) = 28
constraint_lhs[1, 1 + n_features] = 1

optimizer = ps.ConstrainedSR3(
    constraint_rhs=constraint_rhs, constraint_lhs=constraint_lhs
)
model = ps.SINDy(optimizer=optimizer, feature_library=library).fit(x_train, t=dt)
model.print()

# %%
# Try with normalize columns (doesn't work with constraints!!!)
optimizer = ps.ConstrainedSR3(
    constraint_rhs=constraint_rhs,
    constraint_lhs=constraint_lhs,
    normalize_columns=True,
    threshold=10,
)
with ignore_specific_warnings():
    model = ps.SINDy(optimizer=optimizer, feature_library=library).fit(x_train, t=dt)

model.print()

# %%
# Repeat with inequality constraints, need CVXPY installed
try:
    import cvxpy  # noqa: F401

    run_cvxpy = True
except ImportError:
    run_cvxpy = False
    print("No CVXPY package is installed")

# %%
if run_cvxpy:
    # Repeat with inequality constraints
    eps = 1e-5
    constraint_rhs = np.array([eps, eps, 28])

    # One row per constraint, one column per coefficient
    constraint_lhs = np.zeros((3, n_targets * n_features))

    # 1 * (x0 coefficient) + 1 * (x1 coefficient) <= eps
    constraint_lhs[0, 1] = 1
    constraint_lhs[0, 2] = 1

    # -eps <= 1 * (x0 coefficient) + 1 * (x1 coefficient)
    constraint_lhs[1, 1] = -1
    constraint_lhs[1, 2] = -1

    # 1 * (x0 coefficient) <= 28
    constraint_lhs[2, 1 + n_features] = 1

    optimizer = ps.ConstrainedSR3(
        constraint_rhs=constraint_rhs,
        constraint_lhs=constraint_lhs,
        inequality_constraints=True,
        thresholder="l1",
        tol=1e-7,
        threshold=10,
        max_iter=10000,
    )
    model = ps.SINDy(optimizer=optimizer, feature_library=library).fit(x_train, t=dt)
    model.print()
    print(optimizer.coef_[0, 1], optimizer.coef_[0, 2])

# %% [markdown]
# ### MIOSR
# Mixed-integer optimized sparse regression (MIOSR) is an optimizer which solves the NP-hard subset selection problem to provable optimality using techniques from mixed-integer optimization. This optimizer is expected to be most performant compared to heuristics in settings with limited data or on systems with small coefficients. See Bertsimas, Dimitris, and Wes Gurnee. "Learning Sparse Nonlinear Dynamics via Mixed-Integer Optimization." arXiv preprint arXiv:2206.00176 (2022). for details.
#
# Note, MIOSR requires `gurobipy` as a dependency. Please either `pip install gurobipy` or `pip install pysindy[miosr]`.
#
# Gurobi is a private company, but a limited academic license is available when you pip install. If you have previously installed `gurobipy` but your license has expired, `import gurobipy` will work but using the functionality will throw a `GurobiError`. See [here](https://support.gurobi.com/hc/en-us/articles/360038967271-How-do-I-renew-my-free-individual-academic-or-free-trial-license-) for how to renew a free academic license.

# %%
try:
    import gurobipy

    run_miosr = True
    GurobiError = gurobipy.GurobiError
except ImportError:
    run_miosr = False

# %% [markdown]
# MIOSR can handle sparsity constraints in two ways: dimensionwise sparsity where the algorithm is fit once per each dimension, and global sparsity, where all dimensions are fit jointly to respect the overall sparsity constraint. Additionally, MIOSR can handle constraints and can be adapted to implement custom constraints by advanced users.
#
# ### MIOSR target_sparsity vs. group_sparsity

# %%
if run_miosr:
    try:
        miosr_optimizer = ps.MIOSR(target_sparsity=7)
        model = ps.SINDy(optimizer=miosr_optimizer)
        model.fit(x_train, t=dt)
        model.print()
    except GurobiError:
        print("User has an expired gurobi license")

# %%
if run_miosr:
    try:
        miosr_optimizer = ps.MIOSR(group_sparsity=(2, 3, 2), target_sparsity=None)

        model = ps.SINDy(optimizer=miosr_optimizer)
        model.fit(x_train, t=dt)
        model.print()
    except GurobiError:
        print("User does not have a gurobi license")

# %% [markdown]
# ### MIOSR (verbose) with equality constraints

# %%
if run_miosr:
    try:
        # Set constraints
        n_targets = x_train.shape[1]
        constraint_rhs = np.array([0, 28])

        # One row per constraint, one column per coefficient
        constraint_lhs = np.zeros((2, n_targets * n_features))

        # 1 * (x0 coefficient) + 1 * (x1 coefficient) = 0
        constraint_lhs[0, 1] = 1
        constraint_lhs[0, 2] = 1

        # 1 * (x0 coefficient) = 28
        constraint_lhs[1, 1 + n_features] = 1

        miosr_optimizer = ps.MIOSR(
            constraint_rhs=constraint_rhs,
            constraint_lhs=constraint_lhs,
            verbose=True,  # print the full gurobi log
            target_sparsity=7,
        )
        model = ps.SINDy(optimizer=miosr_optimizer, feature_library=library)
        model.fit(x_train, t=dt)
        model.print()
        print(optimizer.coef_[0, 1], optimizer.coef_[0, 2])
    except GurobiError:
        print("User does not have a gurobi license")

# %% [markdown]
# See the [gurobi documentation](https://www.gurobi.com/documentation/9.5/refman/mip_logging.html) for more information on how to read the log output and this [tutorial](https://www.gurobi.com/resource/mip-basics/) on the basics of mixed-integer optimization.

# %% [markdown]
# ### SSR (greedy algorithm)
# Stepwise sparse regression (SSR) is a greedy algorithm which solves the problem (defaults to ridge regression) by iteratively truncating the smallest coefficient during the optimization. There are many ways one can decide to truncate terms. We implement two popular ways, truncating the smallest coefficient at each iteration, or chopping each coefficient, computing N - 1 models, and then choosing the model with the lowest residual error. See this [notebook](https://github.com/dynamicslab/pysindy/blob/master/examples/11_SSR_FROLS.ipynb) for examples.

# %%
ssr_optimizer = ps.SSR(alpha=0.05)

model = ps.SINDy(optimizer=ssr_optimizer)
model.fit(x_train, t=dt)
model.print()

# %% [markdown]
# The alpha parameter is the same here as in the STLSQ optimizer. It determines the amount of L2 regularization to use, so if alpha is nonzero, this is doing Ridge regression rather than least-squares regression.

# %%
ssr_optimizer = ps.SSR(alpha=0.05, criteria="model_residual")
model = ps.SINDy(optimizer=ssr_optimizer)
model.fit(x_train, t=dt)
model.print()

# %% [markdown]
# The kappa parameter determines how sparse a solution is desired.

# %%
ssr_optimizer = ps.SSR(alpha=0.05, criteria="model_residual", kappa=1e-3)
model = ps.SINDy(optimizer=ssr_optimizer)
model.fit(x_train, t=dt)
model.print()

# %% [markdown]
# ### FROLS (greedy algorithm)
# Forward Regression Orthogonal Least Squares (FROLS) is another greedy algorithm which solves the least-squares regression problem (actually default is to solve ridge regression) with $L_0$ norm by iteratively selecting the most correlated function in the library. At each step, the candidate functions are orthogonalized with respect to the already-selected functions. The selection criteria is the Error Reduction Ratio, i.e. the normalized increase in the explained output variance due to the addition of a given function to the basis. See this [notebook](https://github.com/dynamicslab/pysindy/blob/master/examples/11_SSR_FROLS.ipynb) for examples.

# %%
optimizer = ps.FROLS(alpha=0.05)
model = ps.SINDy(optimizer=optimizer)
model.fit(x_train, t=dt)
model.print()

# %% [markdown]
# The kappa parameter determines how sparse a solution is desired.

# %%
optimizer = ps.FROLS(alpha=0.05, kappa=1e-7)
model = ps.SINDy(optimizer=optimizer)
model.fit(x_train, t=dt)
model.print()

# %% [markdown]
# ### LASSO
# In this example we use a third-party Lasso implementation (from scikit-learn) as the optimizer.

# %%
lasso_optimizer = Lasso(alpha=2, max_iter=2000, fit_intercept=False)

model = ps.SINDy(optimizer=lasso_optimizer)
model.fit(x_train, t=dt)
model.print()

# %% [markdown]
# ### Ensemble methods
# One way to improve SINDy performance is to generate many models by sub-sampling the time series (ensemble) or sub-sampling the candidate library $\mathbf{\Theta}$ (library ensemble). The resulting models can then be synthesized by taking the average (bagging), taking the median (this is the recommended because it works well in practice), or some other post-processing. See this [notebook](https://github.com/dynamicslab/pysindy/blob/master/examples/13_ensembling.ipynb) for more examples.

# %%
# Default is to sample the entire time series with replacement, generating 10 models on roughly
# 60% of the total data, with duplicates.

# Custom feature names
np.random.seed(100)
feature_names = ["x", "y", "z"]

ensemble_optimizer = ps.EnsembleOptimizer(
    ps.STLSQ(threshold=1e-3, normalize_columns=False),
    bagging=True,
    n_subset=int(0.6 * x_train.shape[0]),
)

model = ps.SINDy(optimizer=ensemble_optimizer, feature_names=feature_names)
model.fit(x_train, t=dt)
ensemble_coefs = ensemble_optimizer.coef_list
mean_ensemble = np.mean(ensemble_coefs, axis=0)
std_ensemble = np.std(ensemble_coefs, axis=0)

# Now we sub-sample the library. The default is to omit a single library term.
library_ensemble_optimizer = ps.EnsembleOptimizer(
    ps.STLSQ(threshold=1e-3, normalize_columns=False), library_ensemble=True
)
model = ps.SINDy(optimizer=library_ensemble_optimizer, feature_names=feature_names)

model.fit(x_train, t=dt)
library_ensemble_coefs = library_ensemble_optimizer.coef_list
mean_library_ensemble = np.mean(library_ensemble_coefs, axis=0)
std_library_ensemble = np.std(library_ensemble_coefs, axis=0)

# Plot results
xticknames = model.get_feature_names()
for i in range(10):
    xticknames[i] = "$" + xticknames[i] + "$"
plt.figure(figsize=(10, 4))
colors = ["b", "r", "k"]
plt.subplot(1, 2, 1)
plt.title("ensembling")
for i in range(3):
    plt.errorbar(
        range(10),
        mean_ensemble[i, :],
        yerr=std_ensemble[i, :],
        fmt="o",
        color=colors[i],
        label=r"Equation for $\dot{" + feature_names[i] + r"}$",
    )
ax = plt.gca()
plt.grid(True)
ax.set_xticks(range(10))
ax.set_xticklabels(xticknames, verticalalignment="top")
plt.subplot(1, 2, 2)
plt.title("library ensembling")
for i in range(3):
    plt.errorbar(
        range(10),
        mean_library_ensemble[i, :],
        yerr=std_library_ensemble[i, :],
        fmt="o",
        color=colors[i],
        label=r"Equation for $\dot{" + feature_names[i] + r"}$",
    )
ax = plt.gca()
plt.grid(True)
plt.legend()
ax.set_xticks(range(10))
ax.set_xticklabels(xticknames, verticalalignment="top")

# %% [markdown]
# ## Differentiation options

# %% [markdown]
# ### Pass in pre-computed derivatives
# Rather than using one of PySINDy's built-in differentiators, you can compute numerical derivatives using a method of your choice then pass them directly to the `fit` method. This option also enables you to use derivative data obtained directly from experiments.

# %%
x_dot_precomputed = ps.FiniteDifference()._differentiate(x_train, t_train)

model = ps.SINDy()
model.fit(x_train, t=t_train, x_dot=x_dot_precomputed)
model.print()

# %% [markdown]
# ### Drop end points from finite difference computation
# Many methods of numerical differentiation exhibit poor performance near the endpoints of the data. The `FiniteDifference` and `SmoothedFiniteDifference` methods allow one to easily drop the endpoints for improved accuracy.

# %%
fd_drop_endpoints = ps.FiniteDifference(drop_endpoints=True)

model = ps.SINDy(differentiation_method=fd_drop_endpoints)
model.fit(x_train, t=t_train)
model.print()

# %% [markdown]
# ### Differentiation along specific array axis
# For partial differential equations (PDEs), you may have spatiotemporal data in a multi-dimensional array. For this case, the `FiniteDifference` method allows one to only differential along a specific axis, so one can easily differentiate in a specific spatial direction.

# %%
from scipy.io import loadmat

# Load the data stored in a matlab .mat file
kdV = loadmat(data / "kdv.mat")
t = np.ravel(kdV["t"])
X = np.ravel(kdV["x"])
x = np.real(kdV["usol"])
dt_kdv = t[1] - t[0]

# Plot x and x_dot
plt.figure()
plt.pcolormesh(t, X, x)
plt.xlabel("t", fontsize=16)
plt.ylabel("X", fontsize=16)
plt.title(r"$u(x, t)$", fontsize=16)
plt.figure()
x_dot = ps.FiniteDifference(axis=1)._differentiate(x, t=dt_kdv)

plt.pcolormesh(t, X, x_dot)
plt.xlabel("t", fontsize=16)
plt.ylabel("x", fontsize=16)
plt.title(r"$\dot{u}(x, t)$", fontsize=16)
plt.show()

# %% [markdown]
# ### Smoothed finite difference
# This method, designed for noisy data, applies a smoother (the default is `scipy.signal.savgol_filter`) before performing differentiation.

# %%
smoothed_fd = ps.SmoothedFiniteDifference()

model = ps.SINDy(differentiation_method=smoothed_fd)
model.fit(x_train, t=t_train)
model.print()

# %% [markdown]
# ### More differentiation options
# PySINDy is compatible with any of the differentiation methods from the [derivative](https://pypi.org/project/derivative/) package. They are explored in detail in [this notebook](https://github.com/dynamicslab/pysindy/blob/master/examples/5_differentiation.ipynb).
#
# PySINDy defines a `SINDyDerivative` class for interfacing with `derivative` methods. To use a differentiation method provided by `derivative`, simply pass into `SINDyDerivative` the keyword arguments you would give the [dxdt](https://derivative.readthedocs.io/en/latest/api.html#derivative.differentiation.dxdt) method.

# %%
spline_derivative = ps.SINDyDerivative(kind="spline", s=1e-2)

model = ps.SINDy(differentiation_method=spline_derivative)
model.fit(x_train, t=t_train)
model.print()

# %% [markdown]
# ## Feature libraries

# %% [markdown]
# ### Custom feature names

# %%
feature_names = ["x", "y", "z"]
model = ps.SINDy(feature_names=feature_names)
model.fit(x_train, t=dt)
model.print()

# %% [markdown]
# ### Custom left-hand side when printing the model

# %%
model = ps.SINDy()
model.fit(x_train, t=dt)
model.print(lhs=["dx0/dt", "dx1/dt", "dx2/dt"])

# %% [markdown]
# ### Customize polynomial library
# Omit interaction terms between variables, such as $x_0 x_1$.

# %%
poly_library = ps.PolynomialLibrary(include_interaction=False)

model = ps.SINDy(feature_library=poly_library, optimizer=ps.STLSQ(threshold=0.5))
model.fit(x_train, t=dt)
model.print()

# %% [markdown]
# ### Fourier library

# %%
fourier_library = ps.FourierLibrary(n_frequencies=3)

model = ps.SINDy(feature_library=fourier_library, optimizer=ps.STLSQ(threshold=4))
model.fit(x_train, t=dt)
model.print()

# %% [markdown]
# ### Fully custom library
# The `CustomLibrary` class gives you the option to pass in function names to improve the readability of the printed model. Each function "name" should itself be a function.

# %%
library_functions = [
    lambda x: np.exp(x),
    lambda x: 1.0 / x,
    lambda x: x,
    lambda x, y: np.sin(x + y),
]
library_function_names = [
    lambda x: "exp(" + x + ")",
    lambda x: "1/" + x,
    lambda x: x,
    lambda x, y: "sin(" + x + "," + y + ")",
]
custom_library = ps.CustomLibrary(
    library_functions=library_functions, function_names=library_function_names
)

model = ps.SINDy(feature_library=custom_library)
with ignore_specific_warnings():
    model.fit(x_train, t=dt)
model.print()

# %% [markdown]
# ### Fully custom library, default function names
# If no function names are given, default ones are given: `f0`, `f1`, ...

# %%
library_functions = [
    lambda x: np.exp(x),
    lambda x: 1.0 / x,
    lambda x: x,
    lambda x, y: np.sin(x + y),
]
custom_library = ps.CustomLibrary(library_functions=library_functions)

model = ps.SINDy(feature_library=custom_library)
with ignore_specific_warnings():
    model.fit(x_train, t=dt)
model.print()

# %% [markdown]
# ### Identity library
# The `IdentityLibrary` leaves input data untouched. It allows the flexibility for users to apply custom transformations to the input data before feeding it into a `SINDy` model.

# %%
identity_library = ps.IdentityLibrary()

model = ps.SINDy(feature_library=identity_library)
model.fit(x_train, t=dt)
model.print()

# %% [markdown]
# ### Concatenate two libraries
# Two or more libraries can be combined via the `+` operator.

# %%
identity_library = ps.IdentityLibrary()
fourier_library = ps.FourierLibrary()
combined_library = identity_library + fourier_library

model = ps.SINDy(feature_library=combined_library, feature_names=feature_names)
model.fit(x_train, t=dt)
model.print()
model.get_feature_names()

# %% [markdown]
# ### Tensor two libraries together
# Two or more libraries can be tensored together via the `*` operator.

# %%
identity_library = ps.PolynomialLibrary(include_bias=False)
fourier_library = ps.FourierLibrary()
combined_library = identity_library * fourier_library

model = ps.SINDy(feature_library=combined_library, feature_names=feature_names)
model.fit(x_train, t=dt)
# model.print()  # prints out long and unobvious model
print("Feature names:\n", model.get_feature_names())

# %%
# the model prediction is quite bad of course
# because the library has mostly useless terms
x_dot_test_predicted = model.predict(x_test)

# Compute derivatives with a finite difference method, for comparison
x_dot_test_computed = model.differentiate(x_test, t=dt)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_dot_test_computed[:, i], "k", label="numerical derivative")
    axs[i].plot(t_test, x_dot_test_predicted[:, i], "r--", label="model prediction")
    axs[i].legend()
    axs[i].set(xlabel="t", ylabel=r"$\dot x_{}$".format(i))
fig.show()

# %% [markdown]
# ### Generalized library
#
# Create the most general and flexible possible library by combining and tensoring as many libraries as you want, and you can even specify which input variables to use to generate each library! A much more advanced example is shown further below for PDEs.
# One can specify:
# <br>
# 1. N different libraries to add together
# 2. A list of inputs to use for each library. For two libraries with four inputs this
#  would look like inputs_per_library = [[0, 1, 2, 3], [0, 1, 2, 3]] and to avoid using
#  the first two input variables in the second library, you would change it to
#  inputs_per_library = [[0, 1, 2, 3], [2, 3]].
#
# 3. A list of libraries to tensor together and add to the overall library. For four libraries, we could make three tensor libraries by using tensor_array = [[1, 0, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]]. The first sub-array takes the tensor product of libraries 0, 2, 3, the second takes the tensor product of all of them, and the last takes the tensor product of the libraries 2 and 3. This is a silly example since the [1, 1, 1, 1] tensor product already contains all the possible terms.
# 4. A list of library indices to exclude from the overall library. The first N libraries correspond to the input libraries and the subsequent indices correspond to the tensored libraries. For two libraries, exclude_libraries=[0,1] and tensor_array=[[1,1]] would result in a library consisting of only the tensor product.
# <br>
# <br>
# Note that using this is a more advanced feature, but with the benefit that it allows any SINDy library you want. <br>

# %%
# Initialize two libraries
poly_library = ps.PolynomialLibrary(include_bias=False)
fourier_library = ps.FourierLibrary()

# Initialize the default inputs, but
# don't use the x0 input for generating the Fourier library
inputs_per_library = [(0, 1, 2), (1, 2)]

# Tensor all the polynomial and Fourier library terms together
tensor_array = [[1, 1]]

# Initialize this generalized library, all the work hidden from the user!
generalized_library = ps.GeneralizedLibrary(
    [poly_library, fourier_library],
    tensor_array=tensor_array,
    exclude_libraries=[1],
    inputs_per_library=inputs_per_library,
)

# Fit the model and print the library feature names to check success
model = ps.SINDy(feature_library=generalized_library, feature_names=feature_names)
model.fit(x_train, t=dt)
model.print()
print("Feature names:\n", model.get_feature_names())

# %% [markdown]
# ## SINDy with control (SINDYc)
# SINDy models with control inputs can also be learned. Here we learn a Lorenz control model:
# $$ \dot{x} = \sigma (y - x) + u_0$$
# $$ \dot{y} = x(\rho - z) - y $$
# $$ \dot{z} = x y - \beta z - u_1$$

# %% [markdown]
# ### Train the model


# %%
# Control input
def u_fun(t):
    return np.column_stack([np.sin(2 * t), t**2])


# Generate measurement data
dt = 0.002

t_train = np.arange(0, t_end_train, dt)
t_train_span = (t_train[0], t_train[-1])
x0_train = [-8, 8, 27]
x_train = solve_ivp(
    lorenz_control,
    t_train_span,
    x0_train,
    t_eval=t_train,
    args=(u_fun,),
    **integrator_keywords,
).y.T
u_train = u_fun(t_train)

# %%
# Instantiate and fit the SINDYc model
model = ps.SINDy()
model.fit(x_train, u=u_train, t=dt)
model.print()

# %% [markdown]
# ### Assess results on a test trajectory

# %%
# Evolve the Lorenz equations in time using a different initial condition
t_test = np.arange(0, t_end_test, dt)
t_test_span = (t_test[0], t_test[-1])
u_test = u_fun(t_test)
x0_test = np.array([8, 7, 15])
x_test = solve_ivp(
    lorenz_control,
    t_test_span,
    x0_test,
    t_eval=t_test,
    args=(u_fun,),
    **integrator_keywords,
).y.T
u_test = u_fun(t_test)

# Compare SINDy-predicted derivatives with finite difference derivatives
print("Model score: %f" % model.score(x_test, u=u_test, t=dt))

# %% [markdown]
# ### Predict derivatives with learned model

# %%
# Predict derivatives using the learned model
x_dot_test_predicted = model.predict(x_test, u=u_test)

# Compute derivatives with a finite difference method, for comparison
x_dot_test_computed = model.differentiate(x_test, t=dt)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_dot_test_computed[:, i], "k", label="numerical derivative")
    axs[i].plot(t_test, x_dot_test_predicted[:, i], "r--", label="model prediction")
    axs[i].legend()
    axs[i].set(xlabel="t", ylabel=r"$\dot x_{}$".format(i))
fig.show()

# %% [markdown]
# ### Simulate forward in time (control input function known)
# When working with control inputs `SINDy.simulate` requires a *function* to be passed in for the control inputs, `u`, because the default integrator used in `SINDy.simulate` uses adaptive time-stepping. We show what to do in the case when you do not know the functional form for the control inputs in the example following this one.

# %%
# Evolve the new initial condition in time with the SINDy model
x_test_sim = model.simulate(x0_test, t_test, u=u_fun)

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(7, 9))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test, x_test[:, i], "k", label="true simulation")
    axs[i].plot(t_test, x_test_sim[:, i], "r--", label="model simulation")
    axs[i].legend()
    axs[i].set(xlabel="t", ylabel="$x_{}$".format(i))

fig = plt.figure(figsize=(10, 4.5))
ax1 = fig.add_subplot(121, projection="3d")
ax1.plot(x_test[:, 0], x_test[:, 1], x_test[:, 2], "k")
ax1.set(xlabel="$x_0$", ylabel="$x_1$", zlabel="$x_2$", title="true simulation")

ax2 = fig.add_subplot(122, projection="3d")
ax2.plot(x_test_sim[:, 0], x_test_sim[:, 1], x_test_sim[:, 2], "r--")
ax2.set(xlabel="$x_0$", ylabel="$x_1$", zlabel="$x_2$", title="model simulation")

fig.show()

# %% [markdown]
# ### Simulate forward in time (unknown control input function)
# If you only have a vector of control input values at the times in `t_test` and do not know the functional form for `u`, the `simulate` function will internally form an interpolating function based on the vector of control inputs. As a consequence of this interpolation procedure, `simulate` will not give a state estimate for the last time point in `t_test`. This is because the default integrator, `scipy.integrate.solve_ivp` (with LSODA as the default solver), is adaptive and sometimes attempts to evaluate the interpolant outside the domain of interpolation, causing an error.

# %%
u_test = u_fun(t_test)

# %%
x_test_sim = model.simulate(x0_test, t_test, u=u_test)

# Note that the output is one example short of the length of t_test
print("Length of t_test:", len(t_test))
print("Length of simulation:", len(x_test_sim))

fig, axs = plt.subplots(x_test.shape[1], 1, sharex=True, figsize=(12, 4))
for i in range(x_test.shape[1]):
    axs[i].plot(t_test[:-1], x_test[:-1, i], "k", label="true simulation")
    axs[i].plot(t_test[:-1], x_test_sim[:, i], "r--", label="model simulation")
    axs[i].set(xlabel="t", ylabel="$x_{}$".format(i))

fig.show()

# %% [markdown]
# ## Implicit ODEs
# How would we use SINDy to solve an implicit ODE? In other words, now the LHS can depend on x and x_dot,
# $$\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \dot{\mathbf{x}})$$
#
#
# In order to deal with this, we need a library $\Theta(\mathbf{x}, \dot{\mathbf{x}})$. SINDy parallel implicit (SINDy-PI) is geared to solve these problems. It solves the optimization problem,
# $$argmin_\mathbf{\Xi} (\|\Theta(\mathbf{X}, \dot{\mathbf{X}}) - \Theta(\mathbf{X}, \dot{\mathbf{X}})\mathbf{\Xi}\| + \lambda \|\mathbf{\Xi}\|_1)$$
# such that diag$(\mathbf{\Xi}) = 0$. So for every candidate library term it generates a different model. With N state variables, we need to choose N of the equations to solve for the system evolution. See the [SINDy-PI notebook](https://github.com/dynamicslab/pysindy/blob/master/examples/9_sindypi_with_sympy.ipynb) for more details.
#
# Here we illustrate successful identification of the 1D Michelson-Menten enzyme model
# $$\dot{x} = 0.6 - \frac{1.5 x}{0.3 + x}.$$
# Or, equivalently
# $$\dot{x} = 0.6 - 3 x - \frac{10}{3} x\dot{x}.$$
#
# Note that some of the model fits fail. This is usually because the LHS term being fitted is a poor match to the data, but this error can also be caused by CVXPY not liking the parameters passed to the solver.
#

# %%
if run_cvxpy:
    # define parameters
    r = 1
    dt = 0.001
    if __name__ != "testing":
        T = 4
    else:
        T = 0.02
    t = np.arange(0, T + dt, dt)
    t_span = (t[0], t[-1])
    x0_train = [0.55]
    x_train = solve_ivp(enzyme, t_span, x0_train, t_eval=t, **integrator_keywords).y.T

    # Initialize custom SINDy library so that we can have
    # x_dot inside it.
    x_library_functions = [
        lambda x: x,
        lambda x, y: x * y,
        lambda x: x**2,
        lambda x, y, z: x * y * z,
        lambda x, y: x * y**2,
        lambda x: x**3,
        lambda x, y, z, w: x * y * z * w,
        lambda x, y, z: x * y * z**2,
        lambda x, y: x * y**3,
        lambda x: x**4,
    ]
    x_dot_library_functions = [lambda x: x]

    # library function names includes both the x_library_functions
    # and x_dot_library_functions names
    library_function_names = [
        lambda x: x,
        lambda x, y: x + y,
        lambda x: x + x,
        lambda x, y, z: x + y + z,
        lambda x, y: x + y + y,
        lambda x: x + x + x,
        lambda x, y, z, w: x + y + z + w,
        lambda x, y, z: x + y + z + z,
        lambda x, y: x + y + y + y,
        lambda x: x + x + x + x,
        lambda x: x,
    ]

    # Need to pass time base to the library so can build the x_dot library from x
    sindy_library = ps.SINDyPILibrary(
        library_functions=x_library_functions,
        x_dot_library_functions=x_dot_library_functions,
        t=t,
        function_names=library_function_names,
        include_bias=True,
    )

    # Use the SINDy-PI optimizer, which relies on CVXPY.
    # Note that if LHS of the equation fits the data poorly,
    # CVXPY often returns failure.
    sindy_opt = ps.SINDyPI(
        threshold=1e-6,
        tol=1e-8,
        thresholder="l1",
        max_iter=20000,
    )
    model = ps.SINDy(
        optimizer=sindy_opt,
        feature_library=sindy_library,
        differentiation_method=ps.FiniteDifference(drop_endpoints=True),
    )
    model.fit(x_train, t=t)
    model.print()

# %% [markdown]
# ## SINDy with control parameters (SINDyCP)
# The control input in PySINDy can be used to discover equations parameterized by control parameters in conjunction with the `ParameterizedLibrary`. We demonstrate on the logistic map
# $$ x_{n+1} = r x_n(1-x_n)$$
# which depends on a single parameter $r$.

# %%
# Iterate the map and drop the initial 500-step transient. The behavior is chaotic for r>3.6.
if __name__ != "testing":
    num = 1000
    N = 1000
    N_drop = 500
else:
    num = 20
    N = 20
    N_drop = 10
r0 = 3.5
rs = r0 + np.arange(num) / num * (4 - r0)
xss = []
for r in rs:
    xs = []
    x = 0.5
    for n in range(N + N_drop):
        if n >= N_drop:
            xs = xs + [x]
        x = r * x * (1 - x)
    xss = xss + [xs]

plt.figure(figsize=(4, 4), dpi=100)
for ind in range(num):
    plt.plot(np.ones(N) * rs[ind], xss[ind], ",", alpha=0.1, c="black", rasterized=True)
plt.xlabel("$r$")
plt.ylabel("$x_n$")
plt.show()

# %% [markdown]
# We construct a `parameter_library` and a `feature_library` to act on the input data `x` and the control input `u` independently. The `ParameterizedLibrary` is composed of products of the two libraries output features. This enables fine control over the library features, which is especially useful in the case of PDEs like those arising in pattern formation modeling. See this [notebook](https://github.com/dynamicslab/pysindy/blob/master/examples/17_parameterized_pattern_formation/17_parameterized_pattern_formation.ipynb) for examples.

# %%
# use four parameter values as training data
rs_train = [3.6, 3.7, 3.8, 3.9]
xs_train = [np.array(xss[np.where(np.array(rs) == r)[0][0]]) for r in rs_train]

feature_lib = ps.PolynomialLibrary(degree=3, include_bias=True)
parameter_lib = ps.PolynomialLibrary(degree=1, include_bias=True)
lib = ps.ParameterizedLibrary(
    feature_library=feature_lib,
    parameter_library=parameter_lib,
    num_features=1,
    num_parameters=1,
)
opt = ps.STLSQ(threshold=1e-1, normalize_columns=False)
model = ps.SINDy(
    feature_library=lib, optimizer=opt, feature_names=["x", "r"], discrete_time=True
)
model.fit(xs_train, u=rs_train)
model.print()

# %% [markdown]
# ## PDEFIND Feature Overview
# PySINDy now supports SINDy for PDE identification (PDE-FIND) (Rudy, Samuel H., Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz. "Data-driven discovery of partial differential equations." Science Advances 3, no. 4 (2017): e1602614.). We illustrate a basic example on Burgers' equation:
# $$u_t = -uu_x + 0.1u_{xx}$$
#
# Note that for noisy PDE data, the most robust method is to use the weak form of PDEFIND (see below)!

# %%
from scipy.io import loadmat

# Load data
data = loadmat(data / "burgers.mat")
t = np.ravel(data["t"])
x = np.ravel(data["x"])
u = np.real(data["usol"])
dt = t[1] - t[0]
dx = x[1] - x[0]
u_dot = ps.FiniteDifference(axis=-1)._differentiate(u, t=dt)

# Plot the spatiotemporal data
plt.figure()
plt.imshow(u, aspect="auto")
plt.colorbar()
plt.figure()
plt.imshow(u_dot, aspect="auto")
plt.colorbar()
u = np.reshape(u, (len(x), len(t), 1))

# Define quadratic library with up to third order derivatives
# on a uniform spatial grid
library_functions = [lambda x: x, lambda x: x * x]
library_function_names = [lambda x: x, lambda x: x + x]
pde_lib = ps.PDELibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    derivative_order=3,
    spatial_grid=x,
    diff_kwargs={"is_uniform": True, "periodic": True},
)

optimizer = ps.STLSQ(threshold=0.1, alpha=1e-5, normalize_columns=True)
model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
# Note that the dimensions of u are reshaped internally,
# according to the dimensions in spatial_grid
model.fit(u, t=dt)
model.print()

# %% [markdown]
# ### Weak formulation system identification improves robustness to noise.
# PySINDy also supports weak form PDE identification following Reinbold et al. (2019).

# %%
# Same library but using the weak formulation
library_functions = [lambda x: x, lambda x: x * x]
library_function_names = [lambda x: x, lambda x: x + x]
X, T = np.meshgrid(x, t)
XT = np.array([X, T]).T
pde_lib = ps.WeakPDELibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    derivative_order=3,
    spatiotemporal_grid=XT,
    is_uniform=True,
)

# %%
optimizer = ps.STLSQ(threshold=0.01, alpha=1e-5, normalize_columns=True)
model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)

# Note that reshaping u is done internally
model.fit(u, t=dt)
model.print()

# %% [markdown]
# ### GeneralizedLibrary
# The `GeneralizedLibrary` is meant for identifying ODEs/PDEs the depend on the spatial and/or temporal coordinates and/or nonlinear functions of derivative terms.
#
# Often, especially for PDEs, there is some explicit spatiotemporal dependence such as through an external potential. For instance, a well known PDE is the Poisson equation for the electric potential in 2D:
# $$ (\partial_{xx} + \partial_{yy})\phi(x, y) = \rho(x,y).$$
#
#
# **Note that all other SINDy libraries implemented in PySINDy only allow for functions of $\phi(x, y)$ on the RHS of the SINDy equation.** This method allows for functions of the spatial and temporal coordinates like $\rho(x, y)$, as well as anything else you can imagine.
#
# Let's suppose we have a distribution like the following
# $$ \rho(x, y) = x^2 + y^2$$
# We can actually directly input $(\partial_{xx} + \partial_{yy})\phi(x, y)$ as "x_dot" in the SINDy fit, functionally replacing the normal left-hand-side (LHS) of the SINDy equation. Then we can build a candidate library of terms to discover the correct charge density matching this data.
#
# In the following, we will build three different libraries, each using their own subset of inputs, tensor a subset of them together, and fit a model. This is total overkill for this problem but hopefully is illustrative.

# %%
# Define the spatial grid
if __name__ != "testing":
    nx = 50
    ny = 100
else:
    nx = 6  # must be even
    ny = 10
Lx = 1
Ly = 1
x = np.linspace(0, Lx, nx)
dx = x[1] - x[0]
y = np.linspace(0, Ly, ny)
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing="ij")

# Define rho
rho = X**2 + Y**2
plt.figure(figsize=(20, 3))
plt.subplot(1, 5, 1)
plt.imshow(rho, aspect="auto", origin="lower")
plt.title(r"$\rho(x, y)$")
plt.colorbar()

# Generate the PDE data for phi by fourier transforms
# since this is homogeneous PDE
# and we assume periodic boundary conditions
nx2 = int(nx / 2)
ny2 = int(ny / 2)
# Define Fourier wavevectors (kx, ky)
kx = (2 * np.pi / Lx) * np.hstack(
    (np.linspace(0, nx2 - 1, nx2), np.linspace(-nx2, -1, nx2))
)
ky = (2 * np.pi / Ly) * np.hstack(
    (np.linspace(0, ny2 - 1, ny2), np.linspace(-ny2, -1, ny2))
)

# Get 2D mesh in (kx, ky)
KX, KY = np.meshgrid(kx, ky, indexing="ij")
K2 = KX**2 + KY**2
K2[0, 0] = 1e-5

# Generate phi data by solving the PDE and plot results
phi = np.real(np.fft.ifft2(-np.fft.fft2(rho) / K2))
plt.subplot(1, 5, 2)
plt.imshow(phi, aspect="auto", origin="lower")
plt.title(r"$\phi(x, y)$")
plt.colorbar()

# Make del^2 phi and plot various quantities
phi_xx = ps.FiniteDifference(d=2, axis=0)._differentiate(phi, dx)
phi_yy = ps.FiniteDifference(d=2, axis=1)._differentiate(phi, dy)
plt.subplot(1, 5, 3)
plt.imshow(phi_xx, aspect="auto", origin="lower")
plt.title(r"$\phi_{xx}(x, y)$")
plt.subplot(1, 5, 4)
plt.imshow(phi_yy, aspect="auto", origin="lower")
plt.title(r"$\phi_{yy}(x, y)$")
plt.subplot(1, 5, 5)
plt.imshow(
    phi_xx + phi_yy + abs(np.min(phi_xx + phi_yy)),
    aspect="auto",
    origin="lower",
)
plt.title(r"$\phi_{xx}(x, y) + \phi_{yy}(x, y)$")
plt.colorbar()

# Define a PolynomialLibrary, FourierLibrary, and PDELibrary
poly_library = ps.PolynomialLibrary(include_bias=False)
fourier_library = ps.FourierLibrary()
X_mesh, Y_mesh = np.meshgrid(x, y)
pde_library = ps.PDELibrary(
    derivative_order=1, spatial_grid=np.asarray([X_mesh, Y_mesh]).T
)

# Inputs are going to be all the variables [phi, X, Y].
# Remember we can use a subset of these input variables to generate each library
data = np.transpose(np.asarray([phi, X, Y]), [1, 2, 0])

# The 'x_dot' terms will be [phi_xx, X, Y]
# Remember these are the things that are being fit in the SINDy regression
Laplacian_phi = phi_xx + phi_yy + abs(np.min(phi_xx + phi_yy))
data_dot = np.transpose(np.asarray([Laplacian_phi, X, Y]), [1, 2, 0])

# Tensor polynomial library with the PDE library
tensor_array = [[1, 0, 1]]

# Remove X and Y from PDE library terms because why would we take these derivatives
inputs_per_library = [(0, 1, 2), (0, 1, 2), (0,)]

# Fit a generalized library of 3 feature libraries + 1 internally
# generated tensored library and only use the input variable phi
# for the PDELibrary. Note that this holds true both for the
# individual PDELibrary and any tensored libraries constructed from it.
generalized_library = ps.GeneralizedLibrary(
    [poly_library, fourier_library, pde_library],
    tensor_array=tensor_array,
    inputs_per_library=inputs_per_library,
)
optimizer = ps.STLSQ(threshold=8, alpha=1e-3, normalize_columns=True)
model = ps.SINDy(feature_library=generalized_library, optimizer=optimizer)
model.fit(data, x_dot=data_dot)

# Note scale of phi is large so some coefficients >> 1
# --> would want to rescale phi with eps_0 for a harder problem
model.print()

# %%
# Get prediction of rho and plot results
# predict expects a time axis...so add one and ignore it...
data_shaped = data.reshape((data.shape[0], data.shape[1], 1, data.shape[2]))
rho_pred = model.predict(data_shaped)[:, :, 0, :]
if __name__ != "testing":
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    plt.title(r"True $\rho$")
    plt.imshow(rho, aspect="auto", origin="lower")
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title(r"Predicted $\rho_p$")
    plt.imshow(rho_pred[:, :, 0], aspect="auto", origin="lower")
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title(r"Residual errors $\rho - \rho_p$")
    plt.imshow(rho - rho_pred[:, :, 0], aspect="auto", origin="lower")
    plt.colorbar()
    print("Feature names:\n", model.get_feature_names())
