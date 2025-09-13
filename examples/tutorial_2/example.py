# %% [markdown]
# # Evaluating a `SINDy` model
#
# There's several broad ways to evaluate pysindy models:
# * The functional form, or coefficient evaluation
# * The instantaneous derivative prediction
# * The simulation forwards in time
#
# Each of these methods may be the best for a given use case,
# but each also have unique challenges.
# This notebook will take you through different ways you can  evaluate a pysindy model.
# It will illustrate the use cases and the drawbacks of each.
#
#
# Let's take a look at some models.  Some will be good, some will be bad
# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

if __name__ != "testing":
    from example_data import get_models
else:
    from mock_data import get_models

from utils import compare_coefficient_plots

# %%
t, x, good_model, ok_model, bad_model = get_models()

# %% [markdown]
# Our data is the Lorenz system, and each model is trained on progressively less data:

# %%
ax = plt.figure().add_subplot(projection="3d")
ax.plot(x[:, 0], x[:, 1], x[:, 2])

# %% [markdown]
# ## Coefficient evaluation
#
# The most straightforwards way to evaluate a model is by seeing how well it compares
# to the true equations.  The model coefficients are stored in `model.optimizer.coef_`

# %%
print("Good:\n")
good_model.print()
print("\nSo-so:")
ok_model.print()
print("\nBad:")
bad_model.print()

# %%
good_model.optimizer.coef_

# %% [markdown]
# This as an array, with one equation per row.  Each column represents a term from
# `model.get_feature_names()`, so it is possible to construct an array representing
# the true coefficients, then compare the difference.
#

# %%
features = good_model.get_feature_names()
features

# %%
lorenz_coef = np.array(
    [
        [0, -10, 10, 0, 0, 0, 0, 0, 0, 0],
        [0, 28, -1, 0, 0, 0, -1, 0, 0, 0],
        [0, 0, 0, -8.0 / 3, 0, 1, 0, 0, 0, 0],
    ],
    dtype=float,
)
print(f"Good model MSE: {mean_squared_error(lorenz_coef, good_model.optimizer.coef_)}")
print(f"Good model MAE: {mean_absolute_error(lorenz_coef, good_model.optimizer.coef_)}")
print(f"Ok model MSE: {mean_squared_error(lorenz_coef, ok_model.optimizer.coef_)}")
print(f"Ok model MAE: {mean_absolute_error(lorenz_coef, ok_model.optimizer.coef_)}")
print(f"bad model MSE: {mean_squared_error(lorenz_coef, bad_model.optimizer.coef_)}")
print(f"bad model MAE: {mean_absolute_error(lorenz_coef, bad_model.optimizer.coef_)}")

# %% [markdown]
# These coefficients can also be compared visually, e.g. using `pyplot.imshow` or
# `seaborn.heatmap`.  The pysindy-experiments package has some plotting and comparison
# utilities which have been copied into an adjacent utility module:

# %%
fig = plt.figure(figsize=[8, 4])
axes = fig.subplots(1, 4)
fig = compare_coefficient_plots(
    good_model.optimizer.coef_,
    lorenz_coef,
    input_features=["x", "y", "z"],
    feature_names=good_model.get_feature_names(),
    axs=axes[:2],
)
axes[1].set_title("Good model")
fig = compare_coefficient_plots(
    ok_model.optimizer.coef_,
    lorenz_coef,
    input_features=["x", "y", "z"],
    feature_names=ok_model.get_feature_names(),
    axs=[axes[0], axes[2]],
)
axes[2].set_title("OK model")
fig = compare_coefficient_plots(
    bad_model.optimizer.coef_,
    lorenz_coef,
    input_features=["x", "y", "z"],
    feature_names=bad_model.get_feature_names(),
    axs=[axes[0], axes[3]],
)
axes[3].set_title("Bad model")
plt.tight_layout()

# %% [markdown]
# Not all coefficients are equivalent, however.
# E.g. a small coefficient in front of an $x^5$ term may mean more to you and your
# use case than a larger constant coefficient.
# There are different ways of evaluating how important each feature is,
# but they all end up as weights in a call to a scoring metric.
# In this example, weights are calculated as root mean square values of each feature.

# %%
weights = np.sqrt(np.sum(good_model.feature_library.transform(x) ** 2, axis=0) / len(x))
print(
    "weights are ", {feat: f"{weight:.0f}" for feat, weight in zip(features, weights)}
)

good_weighted_mse = mean_squared_error(
    lorenz_coef.T, good_model.optimizer.coef_.T, sample_weight=weights
)
good_weighted_mae = mean_absolute_error(
    lorenz_coef.T, good_model.optimizer.coef_.T, sample_weight=weights
)
print(f"Good model weighted MSE: {good_weighted_mse}")
print(f"Good model weighted MAE: {good_weighted_mae}")
ok_weighted_mse = mean_squared_error(
    lorenz_coef.T, ok_model.optimizer.coef_.T, sample_weight=weights
)
ok_weighted_mae = mean_absolute_error(
    lorenz_coef.T, ok_model.optimizer.coef_.T, sample_weight=weights
)
print(f"Ok model weighted MSE: {ok_weighted_mse}")
print(f"Ok model weighted MAE: {ok_weighted_mae}")
bad_weighted_mse = mean_squared_error(
    lorenz_coef.T, bad_model.optimizer.coef_.T, sample_weight=weights
)
bad_weighted_mae = mean_absolute_error(
    lorenz_coef.T, bad_model.optimizer.coef_.T, sample_weight=weights
)
print(f"Bad model weighted MSE: {bad_weighted_mse}")
print(f"Bad model weighted MAE: {bad_weighted_mae}")

# %% [markdown]
# There are other ways of evaluating model coefficients beyond these metrics,
# the most popular being ways of mathematically analyzing the stability of the discovered model.
# These are beyond the scope of the tutorial, but look at the notebooks on `StabilizedLinearSR3`
# and `TrappingSR3`.

# %% [markdown]
# ## Prediction
#
#

# %% [markdown]
# Sometimes, there's no simple way to evaluate the functional form of the discovered model.
# Some use cases, such as model predictive control, care about immediate prediction
# but not the analytic differences between functions.
# In these cases, it makes the most sense to evaluate the predictive capability.
#
# A great example is the nonlinear pendulum.
# If a pendulum is swinging close to origin, $f(x)=x$ and $f(x)=\sin(x)$
# are very close together.
# Even though coefficient metrics would say that these are different functions,
# they yield very similar predictions, as shown below:

# %%
# +/- 30 degrees from bottom dead center
pendulum_angles = np.pi / 180 * np.linspace(-30, 30, 21)
plt.plot(pendulum_angles, np.sin(pendulum_angles), label=r"$f(x)=\sin(x)$")
plt.plot(pendulum_angles, pendulum_angles, label=r"$f(x)=x$")
plt.legend()

# %% [markdown]
# This occurs because the features are nearly collinear.
# Understanding and compensating for collinearity of features in the function library
# is a challenge.  We can avoid that difficulty if we just score models based upon prediction.
#
# Fortunately, `model.score` depends upon `model.predict()`, which makes evaluating models
# based upon prediction more straightforwards than evaluating the coefficients of the discovered
# model.

# %%
good_model.score(x, t)

# %% [markdown]
# We can inspect the predicted vs observed phase space and gradient plots for visual equivalence
# as well as look for systemic bias in prediction

# %%
fig = plt.figure(figsize=[6, 3])
axes = fig.subplots(1, 3)
for ax, model, name in zip(
    axes, (good_model, ok_model, bad_model), ("good", "ok", "bad")
):
    x_dot_pred = model.predict(x)
    x_dot_true = model.differentiation_method(x, t[1] - t[0])
    ax.scatter(x_dot_true, x_dot_pred - x_dot_true)
    ax.set_title(name)
fig.suptitle("Is there a systemic bias?")
axes[1].set_xlabel("'True' value")
axes[0].set_ylabel("Prediction error")
fig.tight_layout()

# %% [markdown]
# **WARNING!**  All of the predictive measurements compare predictions with an
# ostensibly 'true' $\dot x$, which is typically not available.
# All these examples use the x_dot calculated in the first step of SINDy as "true".
# Even `score()` does so internally.
# If using a differentiation method that oversmooths (in the limit, to a constant line),
# models that predict smaller values of $\dot x$ (in the limit, $\dot x=0$)
# will score the best.

# %% [markdown]
# ## Simulation
#
# If we want to understand the behavior of a system over time, there's no substitute for simulation.
# However, nolinear models of the type of SINDy are not guaranteed to have trajectories
# beyond a certain duration.
# Attempts to simulate near or beyond that duration may bog down or fail the solver.
# Here's an example of a system that explodes in finite time, and cannot be simulated
# beyond it: $$ \dot x = x^2 $$
#
# There is no straightforwards way to guarantee
# that a system will not blow up or go off to infinity,
# but a good prediction or coefficient score is a decent indication.

# %%
x_sim_good = good_model.simulate(x[0], t)
x_sim_ok = ok_model.simulate(x[0], t)

# %% [markdown]
# The bad model becomes stiff, potentially blowing up.
# A stiffer solver may be able to integrate in cases where blow-up occurs,
# but these cases require individual attention and are beyond the scope of this
# tutorial.  Run the following cell to see the kinds of warnings that occur,
# but you will likely have to interrupt the kernel.

# %%
# x_sim_bad = bad_model.simulate(x[0], t)

# %% [markdown]
# The second problem with simulation is that the simulated dynamics may capture some essential
# aspect of the problem, but the numerical difference suggests a poor model.
# This simultaneous correctness and incorrectness can occur if the model recreates the
# exact period and shape of an oscillation, but is out of phase
# (e.g. confusing the predator and prey in lotka volterra).
# It can also occur in chaotic systems, which may mirror the true system very well for a time
# but must eventually diverge dramatically.  Our model and true Lorenz system are chaotic:
#

# %%
fig, axs = plt.subplots(3, 2)
for col, (x_sim, name) in enumerate(zip((x_sim_good, x_sim_ok), ("good", "ok"))):
    axs[0, col].set_title(f"{name} model")
    for coord_ind in range(3):
        axs[coord_ind, col].plot(t, x[:, coord_ind])
        axs[coord_ind, col].plot(t, x_sim[:, coord_ind])
fig.suptitle("Even an accurate model of a chaotic system will look bad in time")

# %% [markdown]
# However, chaotic systems have a useful property.
# Their aperiodic trajectories sweep out a probability distribution.
# And though this distribution is complex and potentially low-or-fractal-dimensional,
# we can estimate a distribution from the data and see how much the simulation diverges from
# that distribution.
# The log likelihood measures divergence in nats, a continuous quantity from information theory.
#
# Here, create the reference distribution from a Gaussian KDE of the true data:

# %%
from sklearn.neighbors import KernelDensity

kde = KernelDensity(kernel="gaussian").fit(x)
base_likelihood = kde.score_samples(x).sum()

# %%
good_excess_loss = base_likelihood - kde.score_samples(x_sim_good).sum()
ok_excess_loss = base_likelihood - kde.score_samples(x_sim_ok).sum()
print(f"Our best model loses {good_excess_loss} nats of information")
print(f"Our ok model loses {ok_excess_loss} nats of information")

# %% [markdown]
# ## What to do with this information
#
# These methods all help you evaluate whether a model is fitting well,
# and given several models, help you evaluate the best for a given application.
# If a model is performing poorly, it may be possible to construct a better SINDy model.
# The next tutorial describes how to choose better components for the SINDy model.
