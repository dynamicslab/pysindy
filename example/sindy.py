from sklearn.datasets import load_boston
import numpy as np
import symfeat as sf

from sparsereg.sindy import SINDy

data = load_boston()
x, y = data.data, data.target

#operators = {"sin": np.sin}
operators = {}
exponents = [1, 2, 0.5]
sym = sf.SymbolicFeatures(exponents=exponents, operators=operators)
features = sym.fit_transform(x)

estimator = SINDy()
estimator.fit(features, y)
print(estimator.score(features, y))
