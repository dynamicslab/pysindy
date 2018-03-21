from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import Lasso
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, explained_variance_score

from sparsereg.model.group_lasso import SparseGroupLasso
from sparsereg.preprocessing.symfeat import SymbolicFeatures
from sparsereg.model.base import _print_model

rng = check_random_state(42)
x = rng.normal(size=(10000, 1))
y = np.cos(x[:, 0]) + x[:, 0] ** 2 + x[:, 0] ** 3  # + 0.01*rng.normal(size=1000)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=rng)
pre = SymbolicFeatures(exponents=[1, 2], operators={"sin": np.sin, "cos": np.cos}).fit(
    x_train
)
features_train = pre.transform(x_train)
features_test = pre.transform(x_test)
km = AgglomerativeClustering(n_clusters=4).fit(features_train.T)
labels = defaultdict(list)
for k, v in zip(pre.get_feature_names(), km.labels_):
    labels[v].append(k)
print(labels)
params = {"alpha": [0.001, 0.01, 0.02, 0.05], "normalize": [True]}
scorer = make_scorer(explained_variance_score)
sgl = SparseGroupLasso(groups=km.labels_, rho=0.3, alpha=0.02)
l = Lasso()
for model in [sgl, l]:
    grid = GridSearchCV(model, params, n_jobs=1, scoring=scorer, error_score=0).fit(
        features_train, y_train
    )
    print(grid.score(features_test, y_test))
    print(
        _print_model(
            grid.best_estimator_.coef_,
            pre.get_feature_names(),
            grid.best_estimator_.intercept_,
        )
    )
