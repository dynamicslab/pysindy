import numpy as np
import pytest
from scipy.integrate import odeint

from pysindy.utils.odes import bacterial
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
    t = np.linspace(0, 10, 100)
    x0 = np.random.rand(ode_params[1])
    if np.shape(ode_params)[0] == 3:
        u = np.column_stack([np.sin(2 * t), t ** 2])
        x_sim = odeint(ode_params[0], x0, t, args=(u,))
    else:
        x_sim = odeint(ode_params[0], x0, t)
    assert np.max(abs(x_sim)) <= 1e5  # avoided unbounded growth
