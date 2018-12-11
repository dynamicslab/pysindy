import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from sparsereg.model.bayes import JMAP
from sparsereg.model.bayes import scale_sigma

size = 10000
scale = 0.1
x = 3 * np.sort(np.random.normal(size=(size, 1)), axis=0)

y = x[:, 0] + 2.5 * x[:, 0] ** 2 + np.random.normal(scale=scale, size=size)

normalize = True
degree = 2
poly = PolynomialFeatures(degree=degree, include_bias=False)
xfeat = poly.fit_transform(x, y)
print("JMAP")
model = JMAP(normalize=normalize)
model.fit(xfeat, y)
# print("ve", model.ve_, model.ve_.shape)
# print("vf", model.vf_, np.sqrt(model.vf_))
print("lambda", model.lambda_)
print("alpha", model.alpha_)
print("coef", model.coef_, model.std_coef_)

from sklearn.linear_model import BayesianRidge

print("SKLEARN BayesianRidge")
model = BayesianRidge(normalize=normalize)
model.fit(xfeat, y)

print("lambda", model.lambda_)
print("alpha", model.alpha_)
print("coef", model.coef_, scale_sigma(model, model.X_offset_, model.X_scale_)[1])
