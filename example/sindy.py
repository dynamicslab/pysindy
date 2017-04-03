from sklearn.datasets import load_boston
import numpy as np
import symfeat as sf

from sparsereg.sindy import SINDy

data = load_boston()
x, y = data.data, data.target

operators = {}
exponents = [1]
sym = sf.SymbolicFeatures(exponents=exponents, operators=operators)
features = sym.fit_transform(x)

estimator = SINDy(knob=1)
estimator.fit(features, y)
print(estimator.coef_)
print(estimator.score(features, y))
