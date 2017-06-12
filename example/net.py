from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso, Ridge
import matplotlib.pyplot as plt
import numpy as np
import symfeat as sf
from sparsereg.sindy import SINDy

from sparsereg.net import net
data = load_boston()
x, y = data.data, data.target

ests = [Lasso, Ridge, SINDy]
attrs = ["alpha", "alpha", "knob"]
names = ["Lasso", "Ridge", "SINDy"]


for i, (est, attr, name) in enumerate(zip(ests, attrs, names)):

    models = net(Lasso, x, y, "alpha")

    m = sorted(models)
    scores = np.array([models[k].score(x, y) for k in m])

    plt.plot(m, scores + i*0.05 , 'o--', label=name)

plt.legend()
plt.xlabel("# coefficient")
plt.ylabel(r"$R^2$")
plt.gca().invert_xaxis()
plt.show()