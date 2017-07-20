from sklearn.model_selection import train_test_split
import numpy as np

from sparsereg.model.ffx import run_ffx, FFX


np.random.seed(42)
x = np.random.normal(size=(1000, 2))
y = x[:, 0] / ( 1 + x[:, 1] )

# exponents = [1, 2]
# operators = {}
# max_iter = 1000
# l1_ratios = [0.95, 0.8, 0.5, 0.2]
# x_train, x_test, y_train, y_test = train_test_split(x, y)
#
# front = run_ffx(x_train, x_test, y_train, y_test, exponents, operators, max_iter=max_iter, l1_ratios=l1_ratios, n_jobs=-1, num_alphas=30, eps=1e-20)
# for model in front:
#     print(model.test_score_, model.complexity_, model.print_model())

model = FFX(n_jobs=-1, decision="weight", max_fit_time=1)
model.fit(x, y)
print(model.print_model())
