import numpy as np

from sparsereg.efs import EFS

np.random.seed(42)
x = np.random.normal(size=(1000, 2))
y = x[:, 0] * x[:, 1]

model = EFS()
model.fit(x, y)