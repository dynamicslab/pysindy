import warnings

from sklearn.model_selection import train_test_split
import numpy as np

from sparsereg.model.ffx import run_ffx, FFX


np.random.seed(42)
x = np.random.normal(size=(1000, 2))
y = x[:, 0] / ( 1 + x[:, 1] )

model = FFX(decision="weight")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    model.fit(x, y)
print(model.print_model())
