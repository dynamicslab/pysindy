import numpy as np

from pysindy import FiniteDifference


class FiniteDifferenceBM:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        self.differentiator = FiniteDifference()
        self.t = np.arange(0, 2 * np.pi, 0.01)

    def time_derivative(self):
        self.differentiator(self.t, t=self.t)

    def peakmem_derivative(self):
        self.differentiator(self.t, t=self.t)
