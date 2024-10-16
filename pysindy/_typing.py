import numpy as np
import numpy.typing as npt

# In python 3.12, use type statement
# https://docs.python.org/3/reference/simple_stmts.html#the-type-statement
NpFlt = np.floating[npt.NBitBase]
FloatDType = np.dtype[np.floating[npt.NBitBase]]
Int1D = np.ndarray[tuple[int], np.dtype[np.int_]]
Float1D = np.ndarray[tuple[int], FloatDType]
Float2D = np.ndarray[tuple[int, int], FloatDType]
Float3D = np.ndarray[tuple[int, int, int], FloatDType]
Float4D = np.ndarray[tuple[int, int, int, int], FloatDType]
Float5D = np.ndarray[tuple[int, int, int, int, int], FloatDType]
FloatND = npt.NDArray[NpFlt]
