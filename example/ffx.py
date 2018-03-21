import numpy as np

from sparsereg.model.ffx import FFX

np.random.seed(42)
x = np.random.normal(size=(33333, 2))
y = x[:, 0] / (1 + x[:, 1])
model = FFX(
    n_jobs=-1,
    l1_ratios=(0.8, 0.9, 0.95),
    exponents=[1, 2],
    target_score=1e-5,
    max_complexity=250,
    num_alphas=1000,
    eps=1E-70,
    rational=True,
)
model.fit(x, y)
print(model.print_model())
print(model.score(x, y))
