from collections import defaultdict

from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import train_test_split

from sparsereg.model.group_lasso import SparseGroupLasso

rng = check_random_state(42)


x = rng.normal(size=(1000, 2))
y = 12*x[:, 0]**2 + 5*x[:, 0]**3+ 0.01*rng.normal(size=1000)
print(y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=rng)

poly = PolynomialFeatures(degree=3, include_bias=False).fit(x_train)
features_train = poly.transform(x_train)
features_test = poly.transform(x_test)


km = KMeans(n_clusters=2, random_state=rng).fit(features_train.T)

labels = defaultdict(list)
for k, v in zip(poly.get_feature_names(), km.labels_):
    labels[v].append(k)

print(labels)

alpha = 0.1
normalize = True
sgl = SparseGroupLasso(groups=km.labels_, alpha=alpha, rho=.3, normalize=normalize).fit(features_train, y_train)
l = Lasso(alpha=alpha, normalize=normalize).fit(features_train, y_train)


for model in [sgl, l]:
    print(model.score(features_test, y_test))
    print(model.coef_)
