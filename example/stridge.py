import numpy as np
from sklearn.datasets import make_regression

from sparsereg.model.base import STRidge

x, y = make_regression(
    n_samples=1000, n_features=10, n_informative=10, n_targets=1, random_state=42
)
print(STRidge().fit(x, y).coef_)
print(STRidge(unbias=False).fit(x, y).coef_)
