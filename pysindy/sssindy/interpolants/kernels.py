from abc import ABC
from abc import abstractmethod
from typing import Callable
from warnings import warn

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn import softplus
from jax.tree_util import Partial as partial

from .matern import build_matern_core


def softplus_inverse(y: jnp.ndarray) -> jnp.ndarray:
    return y + jnp.log1p(-jnp.exp(-y))


class Kernel(eqx.Module, ABC):
    """Abstract base class for kernels in JAX + Equinox."""

    @abstractmethod
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Compute k(x, y). Must be overridden by subclasses."""
        pass

    @abstractmethod
    def pformat(self) -> str:
        """Format the kernel as a string.

        All internal scaling returned to user parameter space."""
        pass

    def __add__(self, other: "Kernel"):
        """
        Overload the '+' operator so we can do k1 + k2.
        Internally, we return a SumKernel object containing both.
        Also handles the case if `other` is already a SumKernel, in
        which case we combine everything into one big sum.
        """
        if isinstance(other, SumKernel):
            # Combine self with an existing SumKernel's list
            return SumKernel(*([self] + list(other.kernels)))
        elif isinstance(other, Kernel):
            return SumKernel(self, other)
        else:
            return NotImplemented

    def __prod__(self, other: "Kernel"):
        """
        Overload the '*' operator so we can do k1 * k2.
        Internally, we return a ProductKernel object containing both.
        Also handles the case if `other` is already a ProductKernel, in
        which case we combine everything into one big sum.
        """
        if isinstance(other, ProductKernel):
            return ProductKernel(*([self] + list(other.kernels)))
        elif isinstance(other, Kernel):
            return ProductKernel(self, other)
        else:
            return NotImplemented

    def transform(f):
        """
        Creates a transformed kernel, returning a kernel function
        k_transformed(x,y) = k(f(x),f(y))
        """


class TransformedKernel(Kernel):
    """
    Transformed kernel, representing the
    composition of a kernel with another
    fixed function
    """

    kernel: Kernel
    transform: Callable = eqx.field(static=True)

    def __init__(self, kernel, transform):
        self.kernel = kernel
        self.transform = transform

    def __call__(self, x, y):
        return self.kernel(self.transform(x), self.transform(y))

    def pformat(self):
        return (
            f"TransformedKernel(transform={self.transform.__name__}\n"
            f"\tapplied to {self.kernel.pformat()})"
        )


class SumKernel(Kernel):
    """
    Represents the sum of multiple kernels:
      k_sum(x, y) = sum_{k in kernels} k(x, y)
    """

    kernels: tuple[Kernel, ...]

    def __init__(self, *kernels: Kernel):
        self.kernels = kernels

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return sum(k(x, y) for k in self.kernels)

    def __add__(self, other: "Kernel"):
        """
        If we do (k1 + k2) + k3, the left side is a SumKernel, so
        we define its __add__ to merge again into one SumKernel.
        """
        if isinstance(other, SumKernel):
            return SumKernel(*(list(self.kernels) + list(other.kernels)))
        elif isinstance(other, Kernel):
            return SumKernel(*(list(self.kernels) + [other]))
        else:
            return NotImplemented

    def pformat(self):
        kstrings = ["\n\t" + kernel.pformat() for kernel in self.kernels]
        return "Sum of (" + ", ".join(kstrings) + "\n)"


class ProductKernel(Kernel):
    """
    Represents the sum of multiple kernels:
      k_sum(x, y) = prod_{k in kernels} k(x, y)
    """

    kernels: tuple[Kernel, ...]

    def __init__(self, *kernels: Kernel):
        self.kernels = kernels

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.prod(jnp.array([k(x, y) for k in self.kernels]))

    def __prod__(self, other: "Kernel"):
        """
        If we do (k1*k2)*k3, the left side is a ProductKernel, so
        we define its __prod__ to merge again into one ProductKernel.
        """
        if isinstance(other, SumKernel):
            return ProductKernel(*(list(self.kernels) + list(other.kernels)))
        elif isinstance(other, Kernel):
            return ProductKernel(*(list(self.kernels) + [other]))
        else:
            return NotImplemented

    def pformat(self):
        return f"Product of ({[kernel.pformat() for kernel in self.kernels]})"


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

    core_matern: Callable = eqx.field(static=True)
    p: int = eqx.field(static=True)
    raw_variance: jax.Array
    raw_lengthscale: jax.Array
    min_lengthscale: jax.Array = eqx.field(static=True)

    def __init__(self, p: int, lengthscale=1.0, variance=1.0, min_lengthscale=0.01):
        self.raw_variance = softplus_inverse(jnp.array(variance))
        if lengthscale < min_lengthscale:
            raise ValueError("Initial lengthscale below minimum")
        self.raw_lengthscale = softplus_inverse(
            jnp.array(lengthscale) - min_lengthscale
        )
        self.p = p
        self.core_matern = build_matern_core(p)
        self.min_lengthscale = min_lengthscale

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        var = softplus(self.raw_variance)
        ls = softplus(self.raw_lengthscale) + self.min_lengthscale
        scaled_diff = (y - x) / ls
        return var * self.core_matern(scaled_diff)

    def pformat(self):
        return (
            f"Matern kernel: order={self.p}, "
            f"variance={softplus(self.raw_variance)}, "
            f"lengthscale={softplus(self.raw_lengthscale) + self.min_lengthscale}"
        )


class GaussianRBFKernel(Kernel):
    """
    RBF (squared exponential) kernel:
        k(x, y) = variance * exp(-||x - y||^2 / (2*lengthscale^2))

    Parameters:
        variance > 0
        lengthscale > 0

    Internally stored as "raw_" after applying softplus_inverse.  Note that
    lengthscale is 1/sqrt(2 * gamma), where gamma is what sklearn uses.
    """

    raw_variance: jax.Array
    raw_lengthscale: jax.Array
    min_lengthscale: jax.Array = eqx.field(static=True)

    def __init__(self, lengthscale=1.0, variance=1.0, min_lengthscale=0.01):
        # Convert user-supplied positive parameters to unconstrained domain
        if lengthscale < min_lengthscale:
            raise ValueError("Initial lengthscale below minimum")
        self.raw_variance = softplus_inverse(jnp.array(variance))
        self.raw_lengthscale = softplus_inverse(jnp.array(lengthscale))
        self.min_lengthscale = min_lengthscale

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        var = softplus(self.raw_variance)
        ls = softplus(self.raw_lengthscale) + self.min_lengthscale
        sqdist = jnp.sum((x - y) ** 2)
        return var * jnp.exp(-0.5 * sqdist / (ls**2))

    def pformat(self):
        return (
            f"GaussianRBF kernel: variance={softplus(self.raw_variance)}, "
            f"lengthscale={softplus(self.raw_lengthscale) + self.min_lengthscale}"
        )


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

    def pformat(self):
        return (
            f"RationalQuadratic kernel: variance={softplus(self.raw_variance)}, "
            f"lengthscale={softplus(self.raw_lengthscale) + self.min_lengthscale}, "
            f"alpha={softplus(self.raw_alpha)}"
        )


class SpectralMixtureKernel(Kernel):
    r"""
    Spectral Mixture kernel for scalar inputs:

    .. math::
      k(\tau) = \sum_{m=1}^M w_m * \exp(-2 * (\pi*\sigma_m)^2
        * (x-y)^2) * \cos(2 \pi (x-y) * \text{periods_m})

    where :math:`\tau = x - y`.

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

    def pformat(self):
        return (
            f"SpectralMixture kernel: "
            f"weights={softplus(self.raw_weights)}, "
            f"freq_sigmas={softplus(self.raw_freq_sigmas)}, "
            f"periods={self.periods}"
        )


class ConstantKernel(Kernel):
    """
    Constant kernel k(x, y) = c for all x, y.

    Params:
        c, variance of the constant shift
    Internally stored as "raw_" after applying softplus_inverse.
    """

    raw_constant: jnp.ndarray

    def __init__(self, variance: float = 1.0):
        """
        :param constant: A positive float specifying the kernel's constant value.
        """
        if variance <= 0:
            raise ValueError("ConstantKernel requires a strictly positive constant.")
        # Store an unconstrained parameter via softplus-inverse
        self.raw_constant = softplus_inverse(jnp.array(variance))

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        c = softplus(self.raw_constant)  # guaranteed positive
        return c

    def pformat(self):
        return f"Constant kernel: constant={softplus(self.raw_constant)}, "


def get_gaussianRBF(gamma: float) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """
    Builds an RBF kernel function.

    Args:
        gamma (double): Length scale of the RBF kernel.

    Returns:
        function: This function returns the RBF kernel with fixed parameter gamma.
    """
    warn(
        "Instead of the functional API, how about a nice cuppa GaussianRBFKernel?",
        DeprecationWarning,
    )
    return partial(gaussian_rbf, gamma=gamma)


def gaussian_rbf(x, y, *, gamma: float):
    return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * gamma**2))
