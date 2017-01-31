from sklearn.datasets import load_boston
import numpy as np

from sparsereg.ffx import run_ffx

data = load_boston()
x, y = data.data, data.target


exponents = [1, 2]
operators = {"sin": np.sin, "cos": np.cos}
l1_ratio = 0.95

front = run_ffx(x, y, exponents, operators, l1_ratio=l1_ratio)
for model in front:
    print(pprint(model), model.score_, model.complexity)
