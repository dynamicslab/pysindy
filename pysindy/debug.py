from typing import Self
from weakref import WeakValueDictionary

import numpy as np

class Seq1D:
    weakrefs: WeakValueDictionary[slice, list["Seq1D"]]
    length: int | None
    mult_buffer: np.ndarray | float

    def __init__(self) -> None:
        self.weakrefs = WeakValueDictionary()
        self.length = None
        self.mult_buffer = 1.0

    def __getitem__(self, obj: slice) -> "Seq1D":
        selection = Seq1D()
        self.weakrefs[obj] = selection

        return selection

    def __add__(self, other: Self) -> "Seq1D":
        if self is other:
            self.mult_buffer += 1
        if not self.length:
            self.length = other.length
        elif self.length != other.length:
            raise ValueError("Incompatible shapes for multiplication")
        return self

    def __sub__(self, other: Self) -> "Seq1D":
        if not self.length:
            self.length = other.length
        elif self.length != other.length:
            raise ValueError("Incompatible shapes for multiplication")
        other.mult_buffer *= -1
        return self

    def __mul__(self, other: np.ndarray) -> "Seq1D":
        if not self.length:
            self.length = other.shape[0]
        elif self.length != other.shape[0]:
            raise ValueError("Incompatible shapes for multiplication")
        self.mult_buffer = self.mult_buffer * other
        return self

    def finalize(self) -> np.ndarray:
        ...