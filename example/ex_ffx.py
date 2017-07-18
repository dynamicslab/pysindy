from sklearn.datasets import load_boston
import numpy as np

from sparsereg.model.ffx import run_ffx
from sparsereg.model._base import equation


np.random.seed(42)
x = np.random.normal(size=(1000, 2))
y = x[:, 0] * x[:, 1]


exponents = [1, 2]
operators = {}
max_iter = 1000
l1_ratios = [0.95, 0.8]

front = run_ffx(x, y, exponents, operators, max_iter=max_iter, l1_ratios=l1_ratios, n_jobs=-1, num_alphas=30)
for model in front:
    print(model.test_score_, model.complexity_, equation(model))
