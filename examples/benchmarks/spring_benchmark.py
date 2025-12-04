#!/usr/bin/env python3
"""
Benchmark PySINDy optimizers on a nonlinear spring system.
System:
    xdot = v
    vdot = -k * x - c * v + F * sin(x**2)
Simulates with RK4, evaluates multiple optimizers, prints metrics and equations,
and saves a plot comparing true vs predicted test trajectories.
"""
import time
import warnings
from typing import List, Tuple

import numpy as np

from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import (
    STLSQ,
    SR3,
    FROLS,
    SSR,
)
# Optional optimizers
try:
    from pysindy.optimizers import TorchOptimizer  # type: ignore
except Exception:
    TorchOptimizer = None  # type: ignore

try:
    from pysindy.optimizers import SBR
except Exception:
    SBR = None  # type: ignore


def myspring(t, x, k=-4.518, c=0.372, F0=9.123):
    """
    Example nonlinear dynamical system.
    xdot = v
    vdot = - k x - v c + F sin(x**2)
    """
    return np.array([x[1], k * x[0] - c * x[1] + F0 * np.sin(x[0] ** 2)])


def rk4_system(f, x0: np.ndarray, t0: float, t1: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(t0, t1 + 1e-12, dt)
    X = np.zeros((t.size, x0.size), dtype=float)
    X[0, :] = np.array(x0, dtype=float)
    for i in range(1, t.size):
        ti = t[i - 1]
        xi = X[i - 1, :]
        k1 = f(ti, xi)
        k2 = f(ti + dt / 2.0, xi + dt * k1 / 2.0)
        k3 = f(ti + dt / 2.0, xi + dt * k2 / 2.0)
        k4 = f(ti + dt, xi + dt * k3)
        X[i, :] = xi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    dX_dt = np.vstack([f(tt, xx) for tt, xx in zip(t, X)])
    return t, X, dX_dt


def split(arr: np.ndarray, ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    n = arr.shape[0]
    n_tr = int(np.floor(ratio * n))
    return arr[:n_tr], arr[n_tr:]


def build_library() -> PolynomialLibrary:
    # Include sin via generalized or custom library? Approximate with polynomials up to degree 5
    return PolynomialLibrary(degree=5, include_interaction=True)


def run_optimizer(name: str, optimizer, X_tr: np.ndarray, dt: float, library: PolynomialLibrary,
                  X_te: np.ndarray, dX_dt_te: np.ndarray):
    model = SINDy(optimizer=optimizer, feature_library=library)
    t0 = time.perf_counter()
    model.fit(X_tr, t=dt)
    fit_time = time.perf_counter() - t0
    score = model.score(X_te, t=dt)
    dX_pred_te = model.predict(X_te)
    mse = float(np.mean((dX_pred_te - dX_dt_te) ** 2))
    complexity = getattr(optimizer, 'complexity', None)
    try:
        equations = model.equations()
    except Exception:
        equations = None
    return {
        "name": name,
        "fit_time_s": fit_time,
        "score": float(score),
        "mse": mse,
        "complexity": int(complexity) if complexity is not None else None,
        "equations": equations,
        "model": model,
    }


def nonlinear_spring_example() -> None:
    warnings.filterwarnings("ignore")
    # Simulate
    t0, t1, dt = 0.0, 10.0, 0.01
    x0 = np.array([0.4, 1.6], dtype=float)
    t, X, dX_dt = rk4_system(myspring, x0, t0, t1, dt)
    # Train/test split
    ratio = 0.67
    X_tr, X_te = split(X, ratio)
    dX_dt_tr, dX_dt_te = split(dX_dt, ratio)
    # Build library
    library = build_library()

    optimizers: List[Tuple[str, object]] = []
    optimizers.append(("STLSQ", STLSQ(threshold=0.1, alpha=0.01, max_iter=30)))
    optimizers.append(("SR3-L0", SR3(reg_weight_lam=0.05, regularizer="L0", relax_coeff_nu=1.0, max_iter=100)))
    optimizers.append(("FROLS", FROLS()))
    optimizers.append(("SSR", SSR()))
    if SBR is not None:
        try:
            optimizers.append(("SBR", SBR()))
        except Exception:
            pass
    if TorchOptimizer is not None:
        try:
            optimizers.append((
                "TorchOptimizer",
                TorchOptimizer(
                    seed=0,
                ),
            ))
        except Exception:
            pass

    dt_scalar = dt
    results = []
    for name, opt in optimizers:
        try:
            res = run_optimizer(name, opt, X_tr, dt_scalar, library, X_te, dX_dt_te)
        except Exception as e:
            res = {"name": name, "error": str(e)}
        results.append(res)

    # Print summary
    header = f"{'Optimizer':<15} {'Score':>10} {'MSE':>12} {'Time (s)':>12} {'Complexity':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['name']:<15} ERROR: {r['error']}")
        else:
            print(f"{r['name']:<15} {r['score']:>10.4f} {r['mse']:>12.4e} {r['fit_time_s']:>12.4f} {str(r['complexity']):>12}")
    for r in results:
        if "error" in r:
            print(f"{r['name']:<15} ERROR: {r['error']}")
        else:
            print(f"{r['name']:<15} {r['score']:>10.4f} {r['mse']:>12.4e} {r['fit_time_s']:>12.4f} {str(r['complexity']):>12}")
            eqs = r.get("equations")
            if eqs:
                for eq in eqs:
                    print(f"    {eq}")
            else:
                print("    (equations unavailable)")

if __name__ == "__main__":
    nonlinear_spring_example()
