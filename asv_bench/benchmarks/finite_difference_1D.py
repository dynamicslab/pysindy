# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import numpy as np
from derivative import FiniteDifference


class FiniteDifferenceBM:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        self.differentiator = FiniteDifference(k=1)
        self.t = np.arange(0, 2 * np.pi, 0.01)

    def time_derivative(self):
        self.differentiator.d(X=self.t, t=self.t, axis=0)

    def peakmem_derivative(self):
        self.differentiator.d(X=self.t, t=self.t, axis=0)
