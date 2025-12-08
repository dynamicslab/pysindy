from abc import ABC
from abc import abstractmethod
from typing import Any

import jax
import jax.numpy as jnp
from typing_extensions import Self


class TrajectoryInterpolant(ABC):
    """Model for a trajectory estimate, represents system state as a function of time"""

    num_params: int

    @abstractmethod
    def fit_time(self, dimension: int, time_points: jax.Array) -> Self:
        """Establish the shape and internal structure of the interpolant."""
        pass

    @abstractmethod
    def fit_obs(self, t: jax.Array, x: jax.Array, noise_var: float) -> jax.Array:
        """Discover coefficients of internal model given observations data.

        Args:
            x: observation data, in shape (n_time, system_dimension)
            noise_var: the variance in the measurement noise

        Returns:
            The parameters used to evaluate the interpolant and its derivatives.
        """
        pass

    @abstractmethod
    def interpolate(
        self, x: jax.Array, t: jax.Array, t_colloc: jax.Array, diff_order=0
    ) -> jax.Array:
        """Fit a copy of this interpolant to observations x at time t

        This does not mutate the original interpolant.

        Arguments:
            x: Observations of the system at time t
            t: Time points of the observations
            t_colloc: Points at which to interpolate
            diff_order: Order of the derivative to evaluate

        Returns:
            An nth-order derivative that interpolates the data
        """
        pass

    @abstractmethod
    def __call__(self, t, params) -> Any:
        pass

    @abstractmethod
    def derivative(self, t, params, diff_order=1) -> Any:
        pass


class LSQInterpolant(TrajectoryInterpolant):
    gram_mat: jax.Array


class MockInterpolant(TrajectoryInterpolant):
    """Don't interpolate any data, just return it back.  Say all derivatives are zero"""

    def __init__(self):
        pass

    def fit_time(self, dimension, time_points):
        self.time_points = time_points
        self.dimension = dimension

        self.num_params = self.dimension * len(self.time_points)
        self.gram_mat = jnp.diag(jnp.ones((self.num_params)))
        return self

    def __call__(self, t, params) -> Any:
        return params.reshape(t.shape[0], self.dimension)

    def derivative(self, t, params, diff_order=1) -> Any:
        return jnp.zeros((t.shape[0], self.dimension))

    def fit_obs(self, t: jax.Array, x: jax.Array, noise_var: float) -> jax.Array:
        if len(t) != len(x):
            raise ValueError("I'm a mock interpolant, I don't do any interpolating")
        return x

    def interpolate(
        self, x: jax.Array, t: jax.Array, t_colloc: jax.Array, diff_order=0
    ) -> jax.Array:
        if diff_order == 0:
            return x
        else:
            return jnp.zeros_like(x)
