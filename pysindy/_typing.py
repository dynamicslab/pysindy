import numpy as np
import numpy.typing as npt

# In python 3.12, use type statement
# https://docs.python.org/3/reference/simple_stmts.html#the-type-statement
NpFlt = np.floating[npt.NBitBase]
Float2D = np.ndarray[tuple[int, int], np.dtype[NpFlt]]
