import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn import softplus

from .base_kernels import Kernel
from .base_kernels import softplus_inverse
from .matern import build_matern_core


class TranslationInvariantKernel(Kernel):
    """
    Not used for anything yet, but maybe unifies some of the other kernels
    Kernels defined by k(x,y) = var * h( (x-y)/ls )
    """

    core_func: callable
    raw_variance: jax.Array
    raw_lengthscale: jax.Array

    min_lengthscale: jax.Array = eqx.field(static=True)
    fix_variance: bool = eqx.field(static=True)
    fix_lengthscale: bool = eqx.field(static=True)

    def __init__(
        self,
        core_func,
        lengthscale,
        variance,
        min_lengthscale,
        fix_variance=False,
        fix_lengthscale=False,
    ):
        self.raw_variance = softplus_inverse(jnp.array(variance))
        if lengthscale < min_lengthscale:
            raise ValueError("Initial lengthscale below minimum")
        self.raw_lengthscale = softplus_inverse(
            jnp.array(lengthscale) - min_lengthscale
        )
        self.min_lengthscale = min_lengthscale
        self.fix_variance = fix_variance
        self.fix_lengthscale = fix_lengthscale
        self.core_func = core_func

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        var = softplus(self.raw_variance)
        if self.fix_variance is True:
            var = jax.lax.stop_gradient(var)

        ls = softplus(self.raw_lengthscale) + self.min_lengthscale
        if self.fix_lengthscale is True:
            ls = jax.lax.stop_gradient(ls)

        scaled_diff = (y - x) / ls
        return var * self.core_func(scaled_diff)


class ScalarMaternKernel(Kernel):
    """
    Scalar half-integer order matern kernel
    order = p+(1/2)

    Parameters:
        p: int
        variance > 0
        lengthscale > 0
    Internally stored as "raw_" after applying softplus_inverse.
    """

    core_matern: callable = eqx.field(static=True)
    p_order: int = eqx.field(static=True)
    raw_variance: jax.Array
    raw_lengthscale: jax.Array
    min_lengthscale: jax.Array = eqx.field(static=True)

    def __init__(self, p, lengthscale=1.0, variance=1.0, min_lengthscale=0.01):
        self.raw_variance = softplus_inverse(jnp.array(variance))
        # if lengthscale<min_lengthscale:
        #     raise ValueError("Initial lengthscale below minimum")
        self.raw_lengthscale = softplus_inverse(
            jnp.array(lengthscale) - min_lengthscale
        )
        self.core_matern = build_matern_core(p)
        self.min_lengthscale = min_lengthscale
        self.p_order = p

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        var = softplus(self.raw_variance)
        ls = softplus(self.raw_lengthscale) + self.min_lengthscale
        scaled_diff = (y - x) / ls
        return var * self.core_matern(scaled_diff)

    def scale(self, c):
        new_raw_var = softplus_inverse(c * softplus(self.raw_variance))
        return eqx.tree_at(lambda x: x.raw_variance, self, new_raw_var)

    def __str__(self):
        var = softplus(self.raw_variance)
        ls = softplus(self.raw_lengthscale) + self.min_lengthscale
        return f"{var:.2f}Matern({self.p_order},{ls:.2f})"


class GaussianRBFKernel(Kernel):
    """
    RBF (squared exponential) kernel:
        k(x, y) = variance * exp(-||x - y||^2 / (2*lengthscale^2))

    Parameters:
        variance > 0
        lengthscale > 0
    Internally stored as "raw_" after applying softplus_inverse.
    """

    raw_variance: jax.Array
    raw_lengthscale: jax.Array
    min_lengthscale: jax.Array = eqx.field(static=True)

    def __init__(self, lengthscale=1.0, variance=1.0, min_lengthscale=0.01):
        # Convert user-supplied positive parameters to unconstrained domain
        if lengthscale < min_lengthscale:
            raise ValueError("Initial lengthscale below minimum")
        self.raw_variance = softplus_inverse(jnp.array(variance))
        self.raw_lengthscale = softplus_inverse(
            jnp.array(lengthscale) - min_lengthscale
        )
        self.min_lengthscale = min_lengthscale

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        var = softplus(self.raw_variance)
        ls = softplus(self.raw_lengthscale) + self.min_lengthscale
        sqdist = jnp.sum((x - y) ** 2)
        return var * jnp.exp(-0.5 * sqdist / (ls**2))

    def scale(self, c):
        new_raw_var = softplus_inverse(c * softplus(self.raw_variance))
        return eqx.tree_at(lambda x: x.raw_variance, self, new_raw_var)

    def __str__(self):
        var = softplus(self.raw_variance)
        ls = softplus(self.raw_lengthscale) + self.min_lengthscale
        return f"{var:.2f}GRBF({ls:.2f})"


class RationalQuadraticKernel(Kernel):
    """
    Rational Quadratic kernel:
      k(x, y) = variance * [1 + (||x - y||^2 / (2 * alpha * lengthscale^2))]^(-alpha)

    Parameters:
        variance > 0
        lengthscale > 0
        alpha > 0
    Internally stored as "raw_" after applying softplus_inverse.
    """

    raw_variance: jax.Array
    raw_lengthscale: jax.Array
    raw_alpha: jax.Array
    min_lengthscale: jax.Array = eqx.field(static=True)

    def __init__(self, lengthscale=1.0, alpha=1.0, variance=1.0, min_lengthscale=0.01):
        self.raw_variance = softplus_inverse(jnp.array(variance))
        self.raw_lengthscale = softplus_inverse(jnp.array(lengthscale))
        self.raw_alpha = softplus_inverse(jnp.array(alpha))
        self.min_lengthscale = min_lengthscale

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        var = softplus(self.raw_variance)
        ls = softplus(self.raw_lengthscale) + self.min_lengthscale
        a = softplus(self.raw_alpha)

        sqdist = jnp.sum((x - y) ** 2)
        factor = 1.0 + (sqdist / (2.0 * a * ls**2))
        return var * jnp.power(factor, -a)

    def scale(self, c):
        new_raw_var = softplus_inverse(c * softplus(self.raw_variance))
        return eqx.tree_at(lambda x: x.raw_variance, self, new_raw_var)

    def __str__(self):
        var = softplus(self.raw_variance)
        a = softplus(self.raw_alpha)
        ls = softplus(self.raw_lengthscale) + self.min_lengthscale
        return f"{var:.2f}RQ({a},{ls:.2f})"


class SpectralMixtureKernel(Kernel):
    """
    Spectral Mixture kernel for scalar inputs:
      k(x, y) = sum_{m=1..M} w_m * exp(-2 * (pi*sigma_m)^2 * (x-y)^2) * cos(2 pi (x-y) * periods_m)
    where tau = x - y.

    Internally stored as "raw_" after applying softplus_inverse.
    """

    raw_weights: jnp.ndarray
    raw_freq_sigmas: jnp.ndarray
    periods: jnp.ndarray

    def __init__(self, key, num_mixture=20, period_variance=10.0):
        key1, key2, key3 = jax.random.split(key, 3)
        self.raw_weights = jax.random.normal(key1, shape=(num_mixture,))
        self.raw_freq_sigmas = jax.random.normal(key2, shape=(num_mixture,))
        self.periods = jnp.sqrt(period_variance) * jax.random.normal(
            key3, shape=(num_mixture,)
        )

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        tau = x - y
        weights = softplus(self.raw_weights)
        freq_sigmas = softplus(self.raw_freq_sigmas)

        kernel_components = jnp.exp(
            -2.0 * (jnp.pi * freq_sigmas) ** 2 * tau**2
        ) * jnp.cos(2.0 * jnp.pi * tau * self.periods)
        return jnp.sum(weights * kernel_components)

    def scale(self, c):
        new_raw_weights = softplus_inverse(c * softplus(self.raw_weights))
        return eqx.tree_at(lambda x: x.raw_weights, self, new_raw_weights)

    def __print__(self):
        weights = softplus(self.raw_weights)
        return f"{jnp.sum(weights):.2f}SpecMix(n={len(self.periods)})"


class LinearKernel(Kernel):
    """
    Linear Kernel k(x, y) = v* <x,y>

    Params:
        variance, variance
    Internally stored as "raw_" after applying softplus_inverse.
    """

    raw_variance: jnp.ndarray

    def __init__(self, variance: float = 1.0):
        """
        :param constant: A positive float specifying the kernel's variance
        """
        if variance <= 0:
            raise ValueError("LinearKernel requires a strictly positive constant.")
        # Store an unconstrained parameter via softplus-inverse
        self.raw_variance = softplus_inverse(jnp.array(variance))

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        v = softplus(self.raw_variance)  # guaranteed positive
        return v * jnp.dot(x, y)

    def scale(self, c):
        new_raw_var = softplus_inverse(c * softplus(self.raw_variance))
        return eqx.tree_at(lambda x: x.raw_variance, self, new_raw_var)

    def __str__(self):
        v = softplus(self.raw_variance)
        return f"{v:.2f}Lin()"


class PolynomialKernel(Kernel):
    """
    Polynomial Kernel k(x, y) = v * (<x,y>+c)^p

    Params:
        variance, variance
    Internally stored as "raw_" after applying softplus_inverse.
    """

    raw_variance: jnp.ndarray
    degree: int = eqx.field(static=True)
    c: jnp.ndarray

    def __init__(self, variance: float = 1.0, c: float = 1.0, degree: int = 2):
        if variance <= 0:
            raise ValueError("LinearKernel requires a strictly positive constant.")
        self.raw_variance = softplus_inverse(jnp.array(variance))
        self.c = jnp.array(c)
        self.degree = degree

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        v = softplus(self.raw_variance)  # guaranteed positive
        return v * jnp.pow(jnp.dot(x, y) + self.c, self.degree)

    def scale(self, c):
        new_raw_var = softplus_inverse(c * softplus(self.raw_variance))
        return eqx.tree_at(lambda x: x.raw_variance, self, new_raw_var)

    def __print__(self):
        v = softplus(self.raw_variance)  # guaranteed positive
        return f"{v:.2f}Poly({self.c},{self.degree})"
