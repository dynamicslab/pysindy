import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pysindy.utils.odes import bacterial
from pysindy.utils.odes import burgers_galerkin
from pysindy.utils.odes import cubic_damped_SHO
from pysindy.utils.odes import cubic_oscillator
from pysindy.utils.odes import double_pendulum
from pysindy.utils.odes import duffing
from pysindy.utils.odes import enzyme
from pysindy.utils.odes import hopf
from pysindy.utils.odes import kinematic_commonroad
from pysindy.utils.odes import linear_3D
from pysindy.utils.odes import linear_damped_SHO
from pysindy.utils.odes import lorenz
from pysindy.utils.odes import lorenz_control
from pysindy.utils.odes import lotka
from pysindy.utils.odes import meanfield
from pysindy.utils.odes import mhd
from pysindy.utils.odes import oscillator
from pysindy.utils.odes import pendulum_on_cart
from pysindy.utils.odes import rossler
from pysindy.utils.odes import van_der_pol
from pysindy.utils.odes import yeast


@pytest.mark.parametrize(
    "ode_params",
    [
        (bacterial, 2),
        (cubic_damped_SHO, 2),
        (cubic_oscillator, 2),
        (double_pendulum, 4),
        (duffing, 2),
        (enzyme, 1),
        (hopf, 2),
        (kinematic_commonroad, 5, 2),
        (linear_3D, 3),
        (linear_damped_SHO, 2),
        (lorenz, 3),
        (lorenz_control, 3, 2),
        (lotka, 2),
        (meanfield, 3),
        (mhd, 6),
        (oscillator, 3),
        (pendulum_on_cart, 4),
        (rossler, 3),
        (van_der_pol, 2),
        (yeast, 7),
    ],
)
def test_odes(ode_params):
    def u_fun(t):
        return np.column_stack([np.sin(2 * t), t ** 2])

    t = np.linspace(0, 10, 100)
    x0 = np.random.rand(ode_params[1])
    if np.shape(ode_params)[0] == 3:
        x_sim = solve_ivp(ode_params[0], (t[0], t[-1]), x0, t_eval=t, args=(u_fun,)).y.T
    else:
        x_sim = solve_ivp(ode_params[0], (t[0], t[-1]), x0, t_eval=t).y.T
    assert np.max(abs(x_sim)) <= 1e5  # avoided unbounded growth


# define analytic galerkin model for quadratic nonlinear systems
def galerkin_model(a, L, Q):
    """RHS of POD-Galerkin model, for time integration"""
    return (L @ a) + np.einsum("ijk,j,k->i", Q, a, a)


def test_galerkin_models():
    # get analytic L and Q operators and galerkin model
    L, Q = burgers_galerkin()

    def rhs(t, a):
        return galerkin_model(a, L, Q)

    # Generate initial condition from unstable eigenvectors
    lamb, Phi = np.linalg.eig(L)
    idx = np.argsort(-np.real(lamb))
    lamb, Phi = lamb[idx], Phi[:, idx]
    a0 = np.real(1e-4 * Phi[:, :2] @ np.random.random((2)))

    # define parameters
    dt = 1e-3
    t_sim = np.arange(0, 300, dt)
    x_sim = solve_ivp(rhs, (t_sim[0], t_sim[-1]), a0, t_eval=t_sim).y.T
    assert np.max(abs(x_sim)) <= 1e5  # avoided unbounded growth
