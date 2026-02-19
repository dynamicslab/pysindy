from typing import TypeAlias

import numpy as np
import numpy.typing as npt

# In python 3.12, use type statement
# https://docs.python.org/3/reference/simple_stmts.html#the-type-statement
NpFlt: TypeAlias = np.float64
FloatDType: TypeAlias = np.dtype[NpFlt]
Int1D: TypeAlias = np.ndarray[tuple[int], np.dtype[np.int_]]
Float1D: TypeAlias = np.ndarray[tuple[int], FloatDType]
Float2D: TypeAlias = np.ndarray[tuple[int, int], FloatDType]
Float3D: TypeAlias = np.ndarray[tuple[int, int, int], FloatDType]
Float4D: TypeAlias = np.ndarray[tuple[int, int, int, int], FloatDType]
Float5D: TypeAlias = np.ndarray[tuple[int, int, int, int, int], FloatDType]
FloatND: TypeAlias = npt.NDArray[NpFlt]
