import numpy as np

from sparsereg.model.ffx import FFX

np.random.seed(42)
x = np.random.normal(size=(500, 2))
y = x[:, 0] / ( 1 + x[:, 1] )

model = FFX(n_jobs=1, l1_ratios=(0.2, 0.8, 0.95,), max_complexity=10, eps=1e-3, num_alphas=10, alpha_max=33)
model.fit(x, y)
print(model.print_model())
