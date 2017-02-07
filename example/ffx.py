from sklearn.datasets import load_boston
import numpy as np

from sparsereg.ffx import run_ffx

#data = load_boston()
#x, y = data.data, data.target
#print(x.shape)
np.random.seed(42)
x = np.random.normal(size=(1000, 2))
y = x[:, 0] * x[:, 1]




exponents = [1, 2]
operators = {"sin": np.sin}
max_iter = 5000
l1_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.95]

front = run_ffx(x, y, exponents, operators, max_iter=max_iter, l1_ratios=l1_ratios)

for model in front:
    print(model.pprint(), model.score_, model.complexity)
