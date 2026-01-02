from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn import softplus

def softplus_inverse(y: jnp.ndarray) -> jnp.ndarray:
    return y + jnp.log1p(-jnp.exp(-y))

class Kernel(eqx.Module):
    """Abstract base class for kernels in JAX + Equinox."""

    @abstractmethod
    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Compute k(x, y). Must be overridden by subclasses."""
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
            return SumKernel(*( [self] + list(other.kernels) ))
        elif isinstance(other, Kernel):
            return SumKernel(self, other)
        else:
            return NotImplemented
    
    def __mul__(self,other:"Kernel"):
        """
        Overload the '*' operator so we can do k1 * k2.
        Internally, we return a ProductKernel object containing both.
        Also handles the case if `other` is already a ProductKernel, in
        which case we combine everything into one big sum.
        """
        if isinstance(other, ProductKernel):
            return ProductKernel(*( [self] + list(other.kernels) ))
        elif isinstance(other, Kernel):
            return ProductKernel(self, other)
        else:
            return NotImplemented
        
    def transform(self,f):
        """
        Creates a transformed kernel, returning a kernel function 
        k_transformed(x,y) = k(f(x),f(y))
        """
        return TransformedKernel(self,f)

    def scale(self,c):
        """
        returns a kernel rescaled by a constant factor c
            really should be implemented better
            but the abstract Kernel doesn't include the variances yet
        Thus, we return a product kernel with the constant kernel,
        abusing the __mul__ overloading
        """
        kc = ConstantKernel(c)
        return kc * self
    

class TransformedKernel(Kernel):
    """
    Transformed kernel, representing the 
    composition of a kernel with another
    fixed function
    """
    kernel: Kernel
    transform: callable = eqx.field(static=True)

    def __init__(self,kernel,transform):
        self.kernel = kernel
        self.transform = transform

    def __call__(self, x, y):
        return self.kernel(self.transform(x),self.transform(y))
    
    def __str__(self):
        return f"Transformed({self.kernel.__str__()})"

        
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
    
    def scale(self,c):
        """
        Push scaling down a level
        """
        return SumKernel(*[k.scale(c) for k in self.kernels])
    
    def __str__(self):
        component_str = [k.__str__() for k in self.kernels]
        return f"{" + ".join(component_str)}"

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
    
    def scale(self,c):
        """
        Scale the first kernel
        """        
        return ProductKernel(*([self.kernels[0].scale(c)] + [self.kernels[1:]]))
    
    def __str__(self):
        component_str = ["(" + k.__str__() + ")" for k in self.kernels]
        return f"{"*".join(component_str)}"

class FrozenKernel(Kernel):
    kernel:Kernel
    def __init__(self,kernel):
        self.kernel = kernel

    def __call__(self, x, y):
        return jax.lax.stop_gradient(self.kernel)(x, y)

    def __str__(self):
        return self.kernel.__str__()

class ConstantKernel(Kernel):
    """
    Constant kernel k(x, y) = c for all x, y.

    Params:
        variance, variance of the constant shift
    Internally stored as "raw_" after applying softplus_inverse.
    """
    raw_variance: jnp.ndarray

    def __init__(self, variance: float = 1.0):
        """
        :param variance: A positive float specifying the kernel's constant value.
        """
        if variance <= 0:
            raise ValueError("ConstantKernel requires a strictly positive constant.")
        # Store an unconstrained parameter via softplus-inverse
        self.raw_variance = softplus_inverse(jnp.array(variance))

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        v = softplus(self.raw_variance)
        return v
    
    def scale(self,c):
        return ConstantKernel(c*softplus(self.raw_variance))

    def __str__(self):
        v = softplus(self.raw_variance)
        return f"{v:.3f}"
