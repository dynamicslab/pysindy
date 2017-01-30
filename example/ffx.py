from sklearn.datasets import load_boston

from sparsereg.ffx import run_ffx

data = load_boston()
x, y = data.data, data.target


exponents = [1, 2]
operators = {}

front = run_ffx(x, y, exponents, operators)
for model in front:
    print(model, model.score_, model.complexity)
