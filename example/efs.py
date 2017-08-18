import numpy as np

from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sparsereg.model.efs import EFS

x, y = make_regression(n_samples=1000, n_features=10, n_informative=10, n_targets=3)

x_train, x_test, y_train, y_test = train_test_split(x, y)

steps = ("scaler", StandardScaler()), ("estimator", EFS(mu=1, q=3, max_stall_iter=5))
model = MultiOutputRegressor(Pipeline(steps))

model.fit(x_train, y_train)

print(model.score(x_test, y_test))
