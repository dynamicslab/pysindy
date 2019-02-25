import copy
import re
import warnings
from functools import partial
from itertools import chain
from itertools import repeat

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate
import seaborn as sns
import sympy
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from sparsereg.model import STRidge
from sparsereg.model.base import equation


class STRidge2(STRidge):
    def _expand(self, x, y):
        residuals = y - self.predict(x)
        small_ind = np.abs(self.coef_) < self.threshold
        res_model = clone(self)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            res_model.fit(x.copy()[:, small_ind], residuals)
            # res_model.fit(x.copy(), residuals)

        c = np.zeros_like(self.coef_)
        c[small_ind] = res_model.coef_
        # c = res_model.coef_

        self.intercept_ += res_model.intercept_
        self.coef_ += c
        self.iters += res_model.iters
        self.history_.extend(res_model.history_)

        if self.unbias and self.alpha > 0:
            self._unbias(x, y)

    def refit(self, x, y):
        check_is_fitted(self, "coef_")
        self.ind_ = np.abs(self.coef_) >= self.threshold
        self._reduce(x, y)
        self._expand(x, y)
        self.intercept_ = self.intercept_ if abs(self.intercept_) >= self.threshold else 0

        return self


def integrate_estimator(t, x0, est):
    def dy(x, t):
        return est.predict(x.reshape(1, -1))[0]

    return scipy.integrate.odeint(dy, x0, t)


def jacobi(model):
    var = [sympy.sympify(f"x_{i}") for i in range(len(model.estimators_))]
    input_fmt = lambda s: s.replace(" ", "*")
    eqs = [
        equation(est, input_features=[str(v) for v in var], input_fmt=input_fmt) for est in model.estimators_
    ]
    dx = sympy.Matrix([sympy.sympify(eq) for eq in eqs])
    mat = dx.jacobian(var)
    jac = sympy.lambdify(var, mat, "numpy")
    return jac


def biggest_eigenvalue(jac, x):
    w = np.linalg.eigvals(jac(*x))
    return np.max(np.real(w))


def get_horizon(model, y, delta_y, dt, t0, t_max):
    """
    :param y: measured trajectory
    :param delta_y: threshold for absolute difference between model prediction and measurement
    :param dt:
    :param model: sklearn regressor
    :returns: time T until model and mesurement diverge
    """
    jac = jacobi(model)

    y = iter(y)
    y0 = next(y)
    dy = lambda t, x: model.predict(x.reshape(1, -1))[0]

    agrees = lambda yobs, yhat: np.linalg.norm(yobs - yhat) < np.linalg.norm(delta_y)

    ode = scipy.integrate.ode(dy)
    ode.set_initial_value(y0, t0)

    t = t0

    yobs = y0
    actual_error = [0]
    lyaps = [biggest_eigenvalue(jac, yobs)]
    while t < t_max - dt and agrees(yobs, ode.y):
        ode.integrate(ode.t + dt)
        yobs = next(y)
        actual_error.append(np.linalg.norm(ode.y - yobs))
        lyaps.append(biggest_eigenvalue(jac, yobs))
        t += dt
    return t - t0, np.array(actual_error), np.mean(lyaps)


def rolling(a, size):
    from numpy.lib.stride_tricks import as_strided

    return as_strided(a, shape=(a.shape[0] - size + 1, size), strides=a.strides * 2)


def refit(model, x, y):
    for yi, est in zip(y.T, model.estimators_):
        est._final_estimator.refit(est.steps[0][1].transform(x), yi)
    return model


def analysis(grid, x, dx, t, t_model, t_update, t_error, error_threshold, alpha=2):
    stats = []
    get_index = lambda t_: np.argmax(t > t_)
    model_history = {}

    num_points_to_model = get_index(t_model)

    x_train, dx_train = x[:num_points_to_model], dx[:num_points_to_model]
    with warnings.catch_warnings():  # filter sparsereg user warnings for aggressive thresholding
        warnings.filterwarnings("ignore")
        grid.fit(x_train, dx_train)
    model = grid.best_estimator_

    model_history[(0, t[num_points_to_model])] = model

    internal_time = t_model
    internal_time += t_error

    print("\n    initial model")
    for i, est in enumerate(model.estimators_):
        print("dx_{}/dt = ".format(i), equation(est))

    while internal_time < t[-1]:
        # print(internal_time)
        index = get_index(internal_time)
        horizon, ie, lyap = get_horizon(
            model, x[index:], error_threshold, t[1] - t[0], t[index], min(internal_time + t_error, t[-1])
        )

        magic_number = np.mean(np.log(np.linalg.norm(ie / error_threshold))) / horizon
        stats.append((internal_time, lyap, magic_number))

        if horizon >= t_error:
            internal_time += t_error
        elif lyap < alpha * magic_number:
            if internal_time + t_update > t[-1]:
                break

            print("\n   remodeling at ", internal_time + horizon, "\n")

            start = index
            end = get_index(internal_time + t_update)

            model = refit(copy.deepcopy(model), x[start:end], dx[start:end])
            for i, est in enumerate(model.estimators_):
                print("dx_{}/dt = ".format(i), equation(est))
            model_history[(internal_time + horizon, internal_time + t_update)] = model
            internal_time += t_update
        else:
            internal_time += t_error
    return model_history, stats


def make_targets(grid, dims):
    feats = list(map(clean, grid.best_estimator_.estimators_[0].steps[0][1].get_feature_names()))
    for d in dims:
        for k in d:
            if k != "offset":
                assert k in feats, "k:{} features:{}".format(k, feats)
    return [[d.get(n, 0) for n in chain(feats, ["offset"])] for d in dims]


def error_in_coefs(metric, agg, model, targets):
    return agg(
        [
            metric(
                np.array(target),
                np.fromiter(chain(est._final_estimator.coef_, [est._final_estimator.intercept_]), dtype=float),
            )
            for est, target in zip(model.estimators_, targets)
        ]
    )


def metric(a, b):
    return np.sum(np.abs(a - b))


my_error = partial(error_in_coefs, metric, np.sum)


def frist_smaller(lst, alpha):
    try:
        return next(x[0] for x in enumerate(lst) if alpha < x[1]) - 1
    except StopIteration:
        return len(lst) - 1


def select_model(model_history, t):
    keys = sorted(model_history.keys())
    stamps = [k[1] for k in keys]

    index = frist_smaller(stamps, t)
    model = model_history.get(keys[index]) if index != -1 else None
    return model


def get_error(model_history, t, get_targets):
    models = [select_model(model_history, t_) for t_ in t]
    error = [my_error(model, get_targets(t_)) if model else -1 for model, t_ in zip(models, t)]
    return error


def plot_error(ax, model_history, t, get_targets):
    error = get_error(model_history, t, get_targets)
    ax.plot(t, error)
    ax.set_ylabel(r"$|\xi - \hat{\xi} |$")
    ax.set_xlabel(r"$t_{update}$")
    scale = max(t) - min(t)
    ax.set_xlim(min(t) - 0.03 * scale, max(t) + 0.03 * scale)


def clean(eq):
    rules = [("+-", "-"), ("+ -", "-"), ("*", ""), (" ", ""), ("**", "^"), ("1.0*", "")]
    for rule in rules:
        eq = eq.replace(*rule)
    return eq


def to_latex(names):
    return ["${}$".format(name) for name in map(clean, names)]


def str_to_eq(eqs):
    pre = r"$\begin{aligned}"
    post = r"\end{aligned}$"
    return pre + r"|".join(eqs) + post


def make_table(path, model_history, varnames=None, precision=2):
    any_model = list(model_history.values())[0]
    dim = len(any_model.estimators_)
    varnames = varnames or ["x_{}".format(i) for i in range(dim)]
    records = []
    for times, model in model_history.items():
        model_detected, model_fitted = map(round, times, repeat(precision))
        eq = []
        for i, (est, v) in enumerate(zip(model.estimators_, varnames)):
            eq_ = equation(est, input_features=varnames, precision=precision)
            eq_ = clean(eq_)
            eq.append(r"\dot{{{}}} & = ".format(v) + eq_)
        record = {
            "$t_{\text{detected}}$": model_detected,
            "$t_{\text{update}}$": model_fitted,
            "Equations": str_to_eq(eq),
        }
        records.append(record)
    pd.set_option("display.max_colwidth", -1)
    df = pd.DataFrame.from_dict(records)
    content = df.to_latex(index=False, escape=False, bold_rows=True, multicolumn=False)
    content = (
        content.replace("\\\\\n\\bottomrule", "\\bottomrule")
        .replace("\\\\", "\\\\ \\hline")
        .replace("|", "\\\\")
        .replace("\\bottomrule", "\\\\\n\\bottomrule")
    )
    with open(path, "w") as f:
        f.write(content)
    return df


def lyap_plot(ax, *stats):
    times, lyaps, magics = zip(*stats)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("inverse timescale")
    ax.set_yscale("symlog", linthreshy=1.0)

    times, lyaps, magics = zip(*stats)
    ax.plot(times, lyaps, "-^", label=r"$\bar{\lambda}(t)$")
    ax.plot(times, magics, "-o", label=r"$\frac{\log(\bar{\Delta}(t)) - \log(\Delta x)}{T(t)}$")
    ax.legend(loc="lower right", bbox_to_anchor=(1.05, 1.05), ncol=2)
