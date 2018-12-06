import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from sparsereg.model.jmap import JMAP


size = 10000
scale = 0.05
x = np.random.normal(size=(size, 2))
y = x[:, 0] + 2.5 * x[:, 1] * x[:, 0] + np.random.normal(scale=scale, size=size)


model = JMAP()
model.fit(x, y)
print(model.print_model())
print(model.sigma2_)

poly = PolynomialFeatures(degree=2, include_bias=False)
xfeat = poly.fit_transform(x, y)

model = JMAP()
model.fit(xfeat, y)
print(model.print_model(input_features=poly.get_feature_names()))
print(model.sigma2_)
