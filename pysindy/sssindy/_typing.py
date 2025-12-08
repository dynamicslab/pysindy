from typing import Callable
from typing import TypeAlias
from typing import TypeVar

import jax
import numpy as np
from numpy.typing import NBitBase


Float1D = np.ndarray[tuple[int], np.dtype[np.floating[NBitBase]]]
Float2D = np.ndarray[tuple[int, int], np.dtype[np.floating[NBitBase]]]
ArrayType = TypeVar("ArrayType", np.ndarray, jax.Array, covariant=True)
AnyArray = np.ndarray | jax.Array
KernelFunc: TypeAlias = Callable[[jax.Array, jax.Array], jax.Array]
TrajOrList = TypeVar("TrajOrList", list[jax.Array], jax.Array)
