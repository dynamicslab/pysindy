import jax
import jax.numpy as jnp
from jsindy.dynamics_model import FeatureLinearModel
from jsindy.trajectory_model import TrajectoryModel


class FullDataTerm:
    def __init__(self, t, x, trajectory_model: TrajectoryModel):
        self.t = t
        self.x = x
        self.trajectory_model = trajectory_model
        self.system_dim = x.shape[1]
        self.num_obs = len(t)
        self.total_size = self.num_obs * self.system_dim

    def residual(self, z):
        # TODO: Code optimization, directly adapt trajectoy_model
        # To the observation locations
        return self.x - self.trajectory_model(self.t, z)

    def residual_flat(self, z):
        return self.residual(z).flatten()


class PartialDataTerm:
    def __init__(self, t, y, v, trajectory_model: TrajectoryModel):
        self.t = t
        self.y = y
        self.v = v
        self.trajectory_model = trajectory_model
        self.system_dim = v.shape[1]
        self.num_obs = len(t)
        self.total_size = len(t)

    def residual(self, z):
        pred_y = jnp.sum(self.trajectory_model(self.t, z) * self.v, axis=1)
        return self.y - pred_y

    def residual_flat(self, z):
        return self.residual(z)


class CollocationTerm:
    def __init__(
        self,
        t_colloc,
        w_colloc,
        trajectory_model: TrajectoryModel,
        dynamics_model: FeatureLinearModel,
        input_orders=(0,),
        ode_order=1,
    ):
        self.t_colloc = t_colloc
        self.w_colloc = w_colloc
        assert len(t_colloc) == len(w_colloc)
        self.num_colloc = len(t_colloc)
        self.system_dim = trajectory_model.system_dim
        self.trajectory_model = trajectory_model
        self.dynamics_model = dynamics_model
        self.input_orders = input_orders
        self.ode_order = ode_order

    def residual(self, z, theta):
        X_inputs = jnp.hstack(
            [
                self.trajectory_model.derivative(self.t_colloc, z, k)
                for k in self.input_orders
            ]
        )

        Xdot_pred = self.dynamics_model(X_inputs, theta)
        Xdot_true = self.trajectory_model.derivative(
            self.t_colloc, z, diff_order=self.ode_order
        )
        return jnp.sqrt(self.w_colloc[:, None]) * (Xdot_true - Xdot_pred)

    def residual_flat(self, z, theta):
        return self.residual(z, theta).flatten()


class JointResidual:
    def __init__(
        self, data_term: FullDataTerm | PartialDataTerm, colloc_term: CollocationTerm
    ):
        self.data_term = data_term
        self.colloc_term = colloc_term

    def data_residual(self, z):
        return self.data_term.residual_flat(z)

    def colloc_residual(self, z, theta):
        return self.colloc_term.residual_flat(z, theta)

    def residual(self, z, theta, data_weight, colloc_weight):
        return jnp.hstack(
            [
                jnp.sqrt(data_weight) * self.data_residual(z),
                jnp.sqrt(colloc_weight) * self.colloc_residual(z, theta),
            ]
        )
