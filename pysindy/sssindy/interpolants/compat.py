import pysindy as ps
from .base import TrajectoryInterpolant


class InterpolantDifferentiation(ps.BaseDifferentiation):
    """Use the new interpolation methods for differentiation in classic SINDy.

    Args:
        interpolant: The interpolant to use for differentiation.
        d: The order of the derivative to compute.
        noise_var: the measurement noise variance
    """

    def __init__(
        self, interpolant: TrajectoryInterpolant, d: int = 1, noise_var: float = 0
    ):
        self.interpolant = interpolant
        self.d = d
        self.noise_var = noise_var

    def _differentiate(self, x, t):
        self.interpolant.fit_time(x.shape[-1], t)
        interp_params = self.interpolant.fit_obs(t, x, noise_var=self.noise_var)
        self.smoothed_x_ = self.interpolant(t, interp_params)
        x_dot = self.interpolant.derivative(t, interp_params, diff_order=self.d)
        return x_dot
