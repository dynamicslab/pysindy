# Can likely be deleted

import numpy as np

from sindy.model.ffx import FFX
import datetime

np.random.seed(42)
x = np.random.normal(scale=3.0, size=(33333, 3))

print("Actual function: x_0^2 + sin(x_1) + x_2 cos(x_1)")
y = x[:, 0]**2 + np.sin(x[:,1]) + x[:,2] * np.cos(x[:, 1])
model = FFX(
    n_jobs=-1,
    l1_ratios=(0.001, 0.01, 0.5, 0.9, 0.999),
    exponents=[1, 2],
    operators={'sin': np.sin, 'cos': np.cos},
    target_score=1e-6,
    max_complexity=250,
    num_alphas=1000,
    eps=1E-70,
    rational=False,
)
t0 = datetime.datetime.utcnow()
model.fit(x, y)
dur = datetime.datetime.utcnow() - t0
dur = dur.total_seconds()
print(f"Fit result: {model.print_model()}")
print(f"fit score (on training data): {model.score(x, y)}")
print(f"fitting took {dur:.2f} seconds.")