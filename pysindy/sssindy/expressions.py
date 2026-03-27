from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Any
from typing import Callable
from typing import Optional

import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from jax.tree_util import register_dataclass
from sklearn.base import BaseEstimator
from sklearn.base import check_is_fitted
from sklearn.base import TransformerMixin
from typing_extensions import Self

import pysindy as ps
from ._typing import Float1D
from ._typing import Float2D
from .interpolants.base import TrajectoryInterpolant
from pysindy.feature_library.base import x_sequence_or_item


@partial(
    register_dataclass,
    data_fields=["model_param_regmat", "state_param_regmat"],
    meta_fields=[
        "data_residual_func",
        "dynamics_residual_func",
        "n_meas",
        "full_n_process",
        "full_n_theta",
        "system_dim",
        "num_features",
        "traj_coef_slices",
    ],
)
@dataclass
class ObjectiveResidual:
    """
    Arguments returned when calling KernelObjective.transform().

    Warning: This dataclass generates a hash based upon container identity,
    assuming immutability.

    Args:
    -----
    data_residual_func: residual function for only the data loss.
    dynamics_residual_func: residual function for only the dynamics loss.
    n_meas: Total number of measurements taken (across all trajectories).
    full_n_process: total number of coefficients on kernel Ansatz for approximation
                   to trajectories.
    full_n_theta: Total number of coefficients on dynamics, or SINDy, approximation
                  to true governing dynamics.
    system_dim: Dimension of governing ode system.
    num_features: Number of features in the feature library.
    traj_coef_slices: list of slices to access the coefficients for each trajectory.
        Each indexes an array of shape ``(full_n_process + full_n_theta,)``.

    Attributes:
        resid_func (JitWrapper): univariate residual function for the full loss.
        jac_func (JitWrapper): univariate jacobian of the residual function.
        damping_matrix (jax.Array): Matrix reflecting the natural parameter metric
            for inner products and norms.
    """

    data_residual_func: Callable[[list[jax.Array]], Any]
    dynamics_residual_func: Callable[[list[jax.Array], jax.Array], Any]
    model_param_regmat: jax.Array
    state_param_regmat: jax.Array
    n_meas: int
    full_n_process: int
    full_n_theta: int
    system_dim: int
    num_features: int
    traj_coef_slices: list[slice]

    def __post_init__(self):
        self.resid_func = self.F_stacked
        self.jac_func = jax.jacrev(self.F_stacked)
        self.damping_matrix = block_diag(
            self.state_param_regmat, self.model_param_regmat
        )

    def extract_state_params(self, stacked_flattened_params):
        state_params = [
            stacked_flattened_params[traj_slice] for traj_slice in self.traj_coef_slices
        ]
        return state_params

    def extract_model_params(self, stacked_flattened_params):
        theta_model = stacked_flattened_params[self.full_n_process :].reshape(
            self.num_features, self.system_dim
        )
        return theta_model

    def F_split(self, state_params, model_params):
        return jnp.hstack(
            [
                self.data_residual_func(state_params),
                self.dynamics_residual_func(state_params, model_params),
            ]
        )

    def F_stacked(self, stacked_flattened_params):
        """
        This stacks the input variables (state_params + model_params) and returns
        the stacked
        """
        state_params = self.extract_state_params(stacked_flattened_params)
        theta_model = self.extract_model_params(stacked_flattened_params)
        return self.F_split(state_params=state_params, model_params=theta_model)

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        return self is other


def _make_data_residual_f(
    interpolant: TrajectoryInterpolant,
    measurement_times: Float1D,
    z_measurements: Float2D,
):
    normalization = jnp.sqrt(
        jnp.prod(jnp.array(z_measurements.shape))
    )  # Normalization factor

    def residual_func(state_params):
        return (
            jnp.array(z_measurements)
            - interpolant(measurement_times, params=state_params)
        ).flatten() / normalization

    return residual_func


class JaxPolyLib(ps.PolynomialLibrary):
    @x_sequence_or_item
    def fit(self, x: list[jax.Array]):
        super().fit(x)

    @x_sequence_or_item
    def transform(self, x: list[jax.Array]):
        xforms = []
        for dataset in x:
            terms = [
                jnp.prod(dataset**exps, axis=1, keepdims=True)
                for exps in self.powers_
            ]
            xforms.append(jnp.concatenate(terms, axis=-1))
        return xforms

    def __call__(self, X):
        return self.transform(X)


def _make_dynamics_residual_f(
    interpolant: TrajectoryInterpolant, t_coloc: Float1D, feature_lib: JaxPolyLib
):
    def residual_func(state_params, theta_model):
        state_estimates = interpolant(t_coloc, state_params)
        features = feature_lib(state_estimates)

        derivative_estimates = interpolant.derivative(
            t_coloc, state_params, diff_order=1
        )
        dynamic_residuals = (features @ theta_model - derivative_estimates).flatten()
        return dynamic_residuals / jnp.sqrt(len(dynamic_residuals))

    return residual_func


@dataclass
class JointObjective(BaseEstimator, TransformerMixin):
    """Single-step SINDy loss expression, specialized for a kernel basis

    TODO: Eventually we'll want to eliminate the discrepancy
    between this and residual objective, or explicitly make this
    a residual-factory

    An expression conceptualizes the combination of data and process misfit.
    Upon instantiation, it concretizes all elements of the loss function except
    the data and the optimization variables.

    When fit, it also identifies the shape of the optimization variables

    Attributes:

        traj_coef_slices_: list[slice]
            indexes to each trajectory's coefficients in the process variable
    """

    data_weight: float
    dynamics_weight: float
    lib: JaxPolyLib
    interp_template: TrajectoryInterpolant

    def fit(
        self,
        z_meas: list[jax.Array],
        t_meas: list[jax.Array],
        *,
        t_coloc: Optional[list[jax.Array]] = None,
    ) -> Self:
        """Determine the residual functions for data and colocation

        Arguments:
            z_meas: Measurements of trajectories, axes following pysindy convention
            t_meas: time of these measurements
            t_coloc: time points for each trajectory to encourage fit of
                SINDy coefficients

        """
        if t_coloc is None:
            t_coloc = t_meas
        self.t_meas_ = t_meas
        self.t_coloc_ = t_coloc
        self.lib.fit(z_meas)
        self.n_meas_ = [t.shape[0] for t in t_meas]
        self.n_coloc_ = [t.shape[0] for t in t_coloc]
        self.system_dimension = z_meas[0].shape[-1]
        self.n_features_ = self.lib.n_output_features_
        # Ike/Alex, we should see if we can keep variables in their native shapes as
        # long as possible, rather than pre-flattening them here and in _make_*_residual
        self.full_n_theta_ = self.n_features_ * self.system_dimension

        self.n_trajectories_ = len(z_meas)
        self.traj_interps = [
            copy(self.interp_template).fit_time(
                dimension=self.system_dimension,
                time_points=t,
            )
            for t in self.t_coloc_
        ]  # Later generalize to higher orders

        traj_coef_start = 0
        self.traj_coef_slices_ = []
        for traj in self.traj_interps:
            traj_coef_end = traj_coef_start + traj.num_params
            self.traj_coef_slices_.append(slice(traj_coef_start, traj_coef_end))

        self.full_n_process = sum(interp.num_params for interp in self.traj_interps)
        self.data_resid_funcs_ = [
            _make_data_residual_f(interp, t, z)  # type: ignore
            for interp, z, t in zip(self.traj_interps, z_meas, t_meas, strict=True)
        ]
        self.dyna_resid_funcs_ = [
            _make_dynamics_residual_f(interp, t, self.lib)  # type: ignore
            for interp, t in zip(self.traj_interps, t_coloc, strict=True)
        ]
        return self

    @x_sequence_or_item
    def transform(self, *args, **kwargs) -> ObjectiveResidual:
        """Convert the data into a residual function.

        Arguments are retained only for sklearn compatibility

        Returns:
            A tuple of residual function for the loss, a matrix defining the
            RKHS norm (and by extension, the trajectory basis functions and number
            of basis coefficients), and the number of total measurements, .
            The residual function accepts stacked trajectory estimates and
            sindy coefficient estimates; the matrix defining the RKHS also
            defines the basis for the trajectory estimates
        """
        check_is_fitted(self)

        def stacked_data_residual_fun(state_params: list[jax.Array]):
            return jnp.hstack(
                [
                    jnp.sqrt(self.data_weight) * res(coef)
                    for coef, res in zip(state_params, self.data_resid_funcs_)
                ]
            )

        def stacked_dyna_residual_fun(
            state_params: list[jax.Array], theta_model: jax.Array
        ):
            return jnp.hstack(
                [
                    jnp.sqrt(self.dynamics_weight) * dyna_resid(coef, theta_model)
                    for coef, dyna_resid in zip(state_params, self.dyna_resid_funcs_)
                ]
            )

        # This one should be instantiated by the feature library
        model_param_regmat = jnp.eye(self.full_n_theta_)

        # State
        state_param_regmat = block_diag(*[traj.gram_mat for traj in self.traj_interps])

        residual_objective = ObjectiveResidual(
            data_residual_func=stacked_data_residual_fun,
            dynamics_residual_func=stacked_dyna_residual_fun,
            model_param_regmat=model_param_regmat,
            state_param_regmat=state_param_regmat,
            n_meas=sum(self.n_meas_) * self.system_dimension,
            full_n_process=self.full_n_process,
            full_n_theta=self.full_n_theta_,
            system_dim=self.system_dimension,
            traj_coef_slices=self.traj_coef_slices_,
            num_features=self.lib.n_output_features_,
        )

        # convert to ObjectiveResidual and update where appropriate
        return residual_objective

    def get_feature_names(self, input_features: Optional[Sequence[str]] = None):
        return self.lib.get_feature_names(input_features=input_features)
