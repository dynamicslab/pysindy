import numpy as np

from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sparsereg.efs import EFS

x, y = make_regression(n_samples=100, n_features=10, n_informative=10, n_targets=3)

steps = ("scaler", StandardScaler()), ("estimator", EFS(time=5*60, mu=1, q=4, gen=1000, alpha=0.3))
model = MultiOutputRegressor(Pipeline(steps))

model.fit(x, y)

print(cross_val_score(model, x, y, cv=10, n_jobs=-1))

