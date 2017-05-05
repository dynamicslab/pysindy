from sklearn.datasets import load_boston
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from joblib import delayed, Parallel
import matplotlib.pyplot as plt
import numpy as np
import symfeat as sf

from sparsereg.sindy import SINDy


def get_it(x, y, **kw):
    estimator = SINDy(**kw)
    try:
        estimator.fit(x, y)
        iters = estimator.iters
    except ConvergenceWarning:
        iters = estimator.max_iters
    except FitFailedWarning:
        iters = estimator.iters

    return iters


if __name__ == "__main__":
    data = load_boston()
    x, y = data.data, data.target

    operators = {}
    exponents = [1]
    sym = sf.SymbolicFeatures(exponents=exponents, operators=operators)
    features = sym.fit_transform(x)

    for knob in [0, 0.01, 0.05, 0.1, 0.5, 1, 5]:
        l1 = np.logspace(-2, 2, 20)
        iters = Parallel(n_jobs=-1)(delayed(get_it)(features.copy(), y.copy(), l1=l, knob=knob) for l in l1)
        plt.semilogx(l1, iters, "-o",label=r"$\lambda_s={}$".format(knob))
    plt.ylabel(r"$n_{iter}$")
    plt.xlabel(r"$\lambda_r$")
    plt.legend()
    plt.show()
