import jax
import jax.numpy as jnp
import jax.random as jrp
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from sklearn.kernel_ridge import KernelRidge

from pysindy import STLSQ
from pysindy.sssindy import JaxPolyLib
from pysindy.sssindy import JointObjective
from pysindy.sssindy import LMSolver
from pysindy.sssindy import SSSINDy
from pysindy.sssindy.interpolants import GaussianRBFKernel
from pysindy.sssindy.interpolants import get_gaussianRBF
from pysindy.sssindy.interpolants import RKHSInterpolant
from pysindy.sssindy.interpolants.base import MockInterpolant
from pysindy.sssindy.opt import _evaluate_objective
from pysindy.sssindy.opt import _LMRegularizer
from pysindy.sssindy.opt import L2CholeskyLMRegularizer
from pysindy.sssindy.opt import LMSettings
from pysindy.sssindy.opt import SINDyAlternatingLMReg
from pysindy.sssindy.sssindy import _initialize_params
from pysindy.utils.odes import lorenz

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices()[0])


# Todo: when merging with pysindy, most of the data fixtures are clones
# from that repo's conftest.py
@pytest.fixture(scope="session")
def data_1d():
    t = np.linspace(0, 1, 12)
    x = 0.2 * t.reshape(-1, 1)
    return x, t


@pytest.fixture(scope="session")
def data_lorenz():
    t = np.linspace(0, 1, 12)
    x0 = [8, 27, -7]
    x = solve_ivp(lorenz, (t[0], t[-1]), x0, t_eval=t).y.T

    return x, t


def test_print(data_1d, capsys):
    x = jnp.array(data_1d[0])
    t = jnp.array(data_1d[1])
    model = SSSINDy(optimizer=LMSolver(optimizer_settings=LMSettings(max_iter=30)))
    model.fit(x, t)
    model.print()
    out, _ = capsys.readouterr()
    model.predict(x)
    model.score(x, t, x)
    model.simulate(x[0], t)

    assert len(out) > 0
    assert " = " in out


@pytest.mark.parametrize(
    "reg",
    [
        (L2CholeskyLMRegularizer(1e-12)),
        (SINDyAlternatingLMReg(1e-12, theta_optimizer=STLSQ())),
    ],
    ids=type,
)
def test_lm_regularizers(data_lorenz, reg: _LMRegularizer):
    interp = RKHSInterpolant(get_gaussianRBF(0.2), (0,), 1e-5)
    exp = JointObjective(1, 1, JaxPolyLib(), interp)
    x = [jnp.array(data_lorenz[0])]
    t = [jnp.array(data_lorenz[1])]
    objective = exp.fit(x, t).transform(x, t)
    reg.fit(objective)
    interp.fit_time(x[0].shape[-1], t[0])
    ez_coeff = jnp.linalg.inv(interp.evaluation_kmat(t[0], t[0])) @ x[0].reshape(
        (-1, 1)
    )
    params = jnp.hstack((ez_coeff.flatten(), jnp.zeros(objective.full_n_theta)))

    origin_val = _evaluate_objective(params, objective)
    local_val = L2CholeskyLMRegularizer(1).fit(objective).eval(params)
    curr_val = origin_val.add_regularization(local_val)
    reg.eval(params)
    result = reg.step(params, curr_val)
    assert not jnp.isnan(result[0]).any()


def test_expression(data_lorenz):
    exp = JointObjective(1, 1, JaxPolyLib(), MockInterpolant())
    x = [jnp.array(data_lorenz[0])]
    t = [jnp.array(data_lorenz[1])]
    objective = exp.fit(x, t).transform(x, t)
    vector_len = objective.full_n_process + objective.full_n_theta
    params = jnp.zeros(vector_len)
    _evaluate_objective(params, objective)
    jax.jit(_evaluate_objective)(params, objective)


def test_multiple_trajectories(data_lorenz):
    x = [jnp.array(data_lorenz[0]), jnp.array(data_lorenz[0])]
    t = [jnp.array(data_lorenz[1]), jnp.array(data_lorenz[1])]
    model = SSSINDy(optimizer=LMSolver(optimizer_settings=LMSettings(max_iter=30)))
    model.fit(x, t)
    model.x_predict(t[0])
    model.predict(x[0][0])
    model.score(x, t, x)


@pytest.mark.parametrize("data", ["sin_data", "data_lorenz"], ids=["sin", "lorenz"])
@pytest.mark.parametrize(
    "init_strategy",
    [
        jrp.key(5),
        "zeros",
        "ones",
        STLSQ(),
        [LMSolver(optimizer_settings=LMSettings(max_iter=2, use_jit=False))],
    ],
    ids=type,
)
def test_init_params(data, init_strategy, request):
    exp = JointObjective(1, 1, JaxPolyLib(), RKHSInterpolant(get_gaussianRBF(0.2)))
    x, t = request.getfixturevalue(data)
    t = t.flatten()
    obj = exp.fit([x], [t], t_coloc=[t]).transform(x, t, t)
    params = _initialize_params(init_strategy, exp, obj, [x], [t], [t])
    assert len(params) == obj.full_n_theta + obj.full_n_process


@pytest.fixture()
def sin_data():
    t = jnp.arange(0, 6, 1, dtype=float)
    x = jnp.sin(t).reshape((-1, 1))
    return x, t


@pytest.fixture()
def twod_sin_data():
    t = jnp.arange(0, 6, 1, dtype=float)
    x = jnp.sin(t).reshape((-1, 1))
    return jnp.hstack((x, -x)), t


@pytest.fixture()
def threed_sin_data():
    t = jnp.arange(0, 6, 1, dtype=float)
    x = jnp.sin(t).reshape((-1, 1))
    return jnp.hstack((x, -x, 3 * x)), t


@pytest.mark.parametrize(
    "k_constructor", [get_gaussianRBF, GaussianRBFKernel], ids=["old", "new"]
)
@pytest.mark.parametrize(
    "data",
    ["sin_data", "twod_sin_data", "threed_sin_data"],
    ids=["1d", "2d", "3d"],
)
def test_rbf_kernel(request, data, k_constructor):
    x, t_obs = request.getfixturevalue(data)
    gamma = 1
    dt = t_obs[1] - t_obs[0]
    t_pred = jnp.arange(t_obs.min(), t_obs.max(), dt / 2, dtype=float).reshape((-1, 1))

    sk_kernel = KernelRidge(alpha=0, kernel="rbf", gamma=gamma)
    sk_kernel.fit(t_obs.reshape((-1, 1)), x)
    sk_pred = sk_kernel.predict(t_pred)

    our_gamma = jnp.sqrt(1 / (2 * gamma))
    our_interp = RKHSInterpolant(
        nugget=0,
        kernel=k_constructor(our_gamma),
        derivative_orders=(0,),
    )
    our_interp.fit_time(x.shape[-1], t_obs.flatten())
    params = our_interp.fit_obs(t_obs.flatten(), x, noise_var=0)
    our_pred = our_interp.__call__(t_pred, params)

    # Because interpolants can cross the axis at slightly different times,
    # np/jnp.allclose will raise false positives
    rel_error = jnp.linalg.norm((sk_pred - our_pred)) / jnp.linalg.norm(sk_pred)
    assert rel_error < 0.01
    vec_align = jnp.sum(sk_pred * our_pred) / jnp.linalg.norm(sk_pred) ** 2
    assert 0.99 < vec_align < 1.01

    our_pred_again = our_interp.interpolate(x, t_obs.flatten(), t_pred, 0)
    assert jnp.allclose(our_pred_again, our_pred, atol=1e-8)
