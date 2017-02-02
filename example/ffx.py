from sklearn.datasets import load_boston
import numpy as np

from sparsereg.ffx import run_ffx

data = load_boston()
x, y = data.data, data.target

#x = np.random.normal(size=(1000, 1))*2*np.pi
#y = np.sin(x[:, 0])


exponents = [1, 2]
operators = {"sin": np.sin, "cos": np.cos}
l1_ratio = 0.95
eps = 1e-30
n_alphas = 100

front = run_ffx(x, y, exponents, operators, l1_ratio=l1_ratio, eps=eps, n_alphas=n_alphas)
for model in front:
    print(model.pprint(), model.score_, model.complexity)
