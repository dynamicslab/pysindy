import numpy as np

import pysindy as ps


class InferenceBM:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        t = np.linspace(0, 1, 100)
        x = 3 * np.exp(-2 * t)
        y = 0.5 * np.exp(t)
        X = np.stack((x, y), axis=-1)  # First column is x, second is y

        self.model = ps.SINDy()
        self.model.fit(X, t=t, feature_names=["x", "y"])
        rng = np.random.default_rng(42)
        self.inputs = rng.random((1000, 2))

    def time_a_predict(self):
        self.model.predict(self.inputs[:1])

    def peakmem_a_predict(self):
        self.model.predict(self.inputs[:1])

    def time_batch_predict(self):
        self.model.predict(self.inputs)

    def peakmem_batch_predict(self):
        self.model.predict(self.inputs)
