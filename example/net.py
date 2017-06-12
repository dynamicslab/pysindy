from sparsereg.net import net

from sklearn.datasets import load_boston
import numpy as np
import symfeat as sf

from sparsereg.sindy import SINDy
from sklearn.linear_model import Lasso

data = load_boston()
x, y = data.data, data.target

models = net(Lasso, x, y, "alpha")
print([(k, min(v, key=lambda x: getattr(x, "alpha"))) for k, v in models.items()])