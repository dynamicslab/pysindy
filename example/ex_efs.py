import numpy as np

from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sparsereg.efs import EFS

x, y = make_regression(n_samples=100, n_features=10, n_informative=3, n_targets=1)

steps = ("transformer", EFS(time=10, gen=50)),  ("estimator", Lasso())
model = Pipeline(steps)

print(cross_val_score(model, x, y, cv=10, n_jobs=-1))

