from typing import Callable
from typing import cast
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.prng import PRNGKeyArray
from jax.random import normal
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

from .expressions import JaxPolyLib
from .expressions import JointObjective
from .expressions import ObjectiveResidual
from .interpolants import get_gaussianRBF
from .interpolants import RKHSInterpolant
from .opt import _BaseSSOptimizer
from .opt import LMSolver
from pysindy._core import _adapt_to_multiple_trajectories
from pysindy._core import _BaseSINDy
from pysindy._core import _check_multiple_trajectories
from pysindy._core import TrajectoryType
from pysindy.optimizers import BaseOptimizer

StrategySpec = (
    str | jax.Array | np.ndarray | PRNGKeyArray | BaseOptimizer | _BaseSSOptimizer
)


class SSSINDy(_BaseSINDy):

    optimizer: _BaseSSOptimizer
    expression: JointObjective
    feature_library: JaxPolyLib
    feature_names: Optional[list[str]]
    init_strategy: StrategySpec

    def __init__(
        self,
        expression: JointObjective = JointObjective(
            data_weight=3.0,
            dynamics_weight=1.0,
            lib=JaxPolyLib(2),
            interp_template=RKHSInterpolant(get_gaussianRBF(0.2)),
        ),
        optimizer: _BaseSSOptimizer = LMSolver(),
        init_strategy: StrategySpec = "zeros",
        feature_names: Optional[list[str]] = None,
    ):
        super().__init__()
        self.expression = expression
        self.optimizer = optimizer
        self.feature_library = expression.lib
        self.feature_names = feature_names
        self.init_strategy = init_strategy

    def fit(
        self,
        x: list[jax.Array],
        t: list[jax.Array],
        t_coloc: Optional[list[jax.Array]] = None,
    ):
        if t_coloc is None:
            t_coloc = t
        if not _check_multiple_trajectories(x, None, None):
            x, t, t_coloc, u = _adapt_to_multiple_trajectories(x, t, t_coloc, None)
        t_coloc = cast(list[jax.Array], t_coloc)

        self.n_control_features_ = 0  # cannot yet fit control features
        self.model = Pipeline(
            [("expression", self.expression), ("optimizer", self.optimizer)]
        )

        objective = self.expression.fit(x, t, t_coloc=t_coloc).transform(x, t, t_coloc)

        init_params = _initialize_params(
            self.init_strategy, self.expression, objective, x, t, t_coloc
        )

        self.optimizer.fit(objective, init_params=init_params)
        self.fitted = True
        self.n_control_features_ = 0
        self._fit_shape()

    def x_predict(self, t):
        return [
            interp(t, params=self.optimizer.process_[slise])
            for interp, slise in zip(
                self.expression.traj_interps, self.expression.traj_coef_slices_
            )
        ]

    def predict(self, x: jax.Array) -> jax.Array:
        r"""Predict the time derivative \dot{x} = f(x). Later include time dependence"""
        feats = self.feature_library.transform(
            x.reshape(-1, self.feature_library.n_features_in_)  # type: ignore
        )
        return feats @ self.optimizer.coef_.T  # type: ignore

    def coefficients(self):
        if hasattr(self, "fitted") and self.fitted:
            return self.optimizer.coef_
        else:
            raise ValueError("Must run fit() first.")

    def simulate(self, x0: jax.Array, t: jax.Array, **integrator_kws) -> jax.Array:
        def rhs(t, x):
            return self.predict(x[np.newaxis, :])[0]

        return ((solve_ivp(rhs, (t[0], t[-1]), x0, t_eval=t, **integrator_kws)).y).T

    def score(
        self,
        x: TrajectoryType,
        t: TrajectoryType,
        x_dot: TrajectoryType,
        metric: Callable[[jax.Array, jax.Array], float] = r2_score,  # type: ignore
    ) -> float:
        if not isinstance(x, list):
            x = [x]  # type: ignore
        if not isinstance(t, list):
            t = [t]  # type: ignore
        if not isinstance(x_dot, list):
            x_dot = [x_dot]  # type: ignore
        if x_dot is None:
            x_dot = jnp.vstack([self.predict(x_i) for x_i in x])
        x_dot_predict = jnp.vstack(x_dot)
        x_dot_interp = jnp.vstack(
            [
                interp.derivative(t_i, self.optimizer.process_[slise])
                for t_i, interp, slise in zip(
                    t, self.expression.traj_interps, self.expression.traj_coef_slices_
                )
            ]
        )

        return metric(x_dot_interp, x_dot_predict)


def _initialize_params(
    init_strategy: StrategySpec,
    expression: JointObjective,
    objective: ObjectiveResidual,
    x: Optional[list[jax.Array]] = None,
    t: Optional[list[jax.Array]] = None,
    t_colloc: Optional[list[jax.Array]] = None,
) -> jax.Array:
    """Initialize the optimization parameter using a variety of strategies.

    Args:
        init_strategy: The strategy to use for initialization. It can be a:
            - string: "zeros" or "ones"
            - jax.Array: A jax array of initial values.
            - np.ndarray: A numpy array of initial values.
            - PRNGKeyArray: A jax random key for generating normally
                distributed random values.
            - BaseOptimizer: When sending a BaseOptimizer, the data residual
                is first used to fit the process coefficients, then the BaseOptimizer
                is used to fit the SINDy dynamics coefficients.
            - _BaseSSOptimizer: Fit using an initial joint SINDy optimizer.
        expression: The expression object.  Depending on the strategy, this
            may be safely passed as None.
        objective: The objective residual object.  Depending on the strategy, this
            may be safely passed as None.  Otherwise, it needs to be consistent with
            the expression.
        t_coloc: The collocation points.
    Returns:
        Initial values for the optimization parameter.
    """
    if init_strategy == "zeros":
        init_params = jnp.zeros(objective.full_n_process + objective.full_n_theta)
    elif init_strategy == "ones":
        init_params = jnp.ones(objective.full_n_process + objective.full_n_theta)
    elif isinstance(init_strategy, PRNGKeyArray):
        # Match order matters here: PRNGKeyArray is a jax.Array, so it would match
        # the next condition if we don't check for it first.
        init_params = normal(
            init_strategy, shape=(objective.full_n_process + objective.full_n_theta,)
        )
    elif isinstance(init_strategy, jax.Array):
        init_params = init_strategy
    elif isinstance(init_strategy, np.ndarray):
        init_params = jnp.array(init_strategy)
    elif isinstance(init_strategy, BaseOptimizer):

        if not (
            isinstance(x, list) and isinstance(t, list) and isinstance(t_colloc, list)
        ):
            raise ValueError(
                "If init_strategy is a BaseOptimizer, x, t, and t_coloc must be "
                "provided as lists."
            )
        init_proc = []
        init_colloc = []
        init_colloc_d = []
        for traj, traj_x, traj_t, traj_tc in zip(
            expression.traj_interps, x, t, t_colloc
        ):
            traj.fit_time(expression.system_dimension, traj_t)
            tparams = traj.fit_obs(traj_t, traj_x, noise_var=1e-5)
            init_proc.append(tparams.flatten())
            init_colloc.append(traj(traj_tc, tparams))
            init_colloc_d.append(traj.derivative(traj_tc, tparams))

        init_proc = jnp.hstack(init_proc)
        x_ = np.vstack(expression.lib.transform(init_colloc))
        y_ = np.vstack(init_colloc_d)
        init_theta = init_strategy.fit(x_, y_).coef_
        init_params = jnp.hstack((init_proc, init_theta.T.flatten()))
    elif isinstance(init_strategy, list):
        params = jnp.zeros(objective.full_n_process + objective.full_n_theta)
        if not init_strategy:
            raise ValueError("If init_strategy is a list, it must not be empty. ")
        for strat in init_strategy:
            if not isinstance(strat, _BaseSSOptimizer):
                raise ValueError(
                    f"If init_strategy is a list, all elements must be "
                    f"of type _BaseSSOptimizer. Got {type(strat)}."
                )
            strat.fit(objective)
            params = jnp.hstack((strat.process_, strat.coef_.flatten()))
        init_params = params
    else:
        raise ValueError(f"init_strategy: {init_strategy} not understood. ")
    return init_params
