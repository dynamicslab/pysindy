#!/usr/bin/env python3
"""
Benchmark PySINDy optimizers on the Lorenz system.
Runs a quick RK4 simulation, constructs a SINDy model with a PolynomialLibrary,
then evaluates multiple optimizers (including TorchOptimizer if torch is available)
for runtime, model score, and sparsity (complexity).
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
    WrappedOptimizer,
)

# Optional optimizers guarded in __init__ (may be None if dependency missing)
try:
    from pysindy.optimizers import TorchOptimizer  # type: ignore
except Exception:
    TorchOptimizer = None  # type: ignore

try:
    from pysindy.optimizers import SBR  # may require cvxpy or extra deps
except Exception:
    SBR = None  # type: ignore


def lorenz(t, x, sigma: float = 10.0, beta: float = 8.0 / 3.0, rho: float = 28.0):
    u, v, w = x
    up = -sigma * (u - v)
    vp = rho * u - v - u * w
    wp = -beta * w + u * v
    return np.array([up, vp, wp], dtype=float)


def rk4_lorenz(t0: float = 0.0, t1: float = 10.0, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(t0, t1 + 1e-12, dt)
    X = np.zeros((t.size, 3), dtype=float)
    X[0, :] = np.array([1.0, 1.0, 1.0], dtype=float)
    # RK4 integrator
    for i in range(1, t.size):
        ti = t[i - 1]
        xi = X[i - 1, :]
        k1 = lorenz(ti, xi)
        k2 = lorenz(ti + dt / 2.0, xi + dt * k1 / 2.0)
        k3 = lorenz(ti + dt / 2.0, xi + dt * k2 / 2.0)
        k4 = lorenz(ti + dt, xi + dt * k3)
        X[i, :] = xi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # Drop a short burn-in to avoid transient initialization effects
    burn = int(1.0 / dt)  # drop first 1s
    t = t[burn:]
    X = X[burn:, :]

    # Exact RHS at sample points using analytic lorenz
    dX_dt = np.vstack([lorenz(tt, xx) for tt, xx in zip(t, X)])
    return t, X, dX_dt


def build_library() -> PolynomialLibrary:
    # Standard SINDy polynomial library for Lorenz (degree 2 is typical)
    return PolynomialLibrary(degree=2, include_interaction=True)


def run_optimizer(name: str, optimizer, X: np.ndarray, dt: float, library: PolynomialLibrary, dX_dt: np.ndarray):
    model = SINDy(optimizer=optimizer, feature_library=library)
    t0 = time.perf_counter()
    model.fit(X, t=dt)
    fit_time = time.perf_counter() - t0
    score = model.score(X, t=dt)
    # Predict derivatives from learned model, compare to analytic RHS
    dX_pred = model.predict(X)
    mse = float(np.mean((dX_pred - dX_dt) ** 2))
    complexity = optimizer.complexity if hasattr(optimizer, "complexity") else None
    # Gather equations as strings
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
    }


def system_lorenz_example() -> None:
    """Simulate the Lorenz attractor and test PySINDy + our solvers on it.

    Uses a simple RK4 integrator (no extra deps). To keep runtime reasonable,
    solver options are reduced compared to defaults but can be adjusted.
    """
    warnings.filterwarnings("ignore")
    t, X, dX_dt = rk4_lorenz()
    dt = t[1] - t[0]
    library = build_library()

    optimizers = []  # type: List[Tuple[str, object]]

    # STLSQ: classic baseline
    optimizers.append(("STLSQ", STLSQ(threshold=0.1, alpha=0.05, max_iter=20)))

    # SR3: relaxed regularized regression, L0 prox behaves like hard thresholding
    optimizers.append(("SR3-L0", SR3(reg_weight_lam=0.1, regularizer="L0", relax_coeff_nu=1.0, max_iter=50)))

    # FROLS & SSR: forward regression variants
    optimizers.append(("FROLS", FROLS()))
    optimizers.append(("SSR", SSR()))

    # SBR if available
    if SBR is not None:
        try:
            optimizers.append(("SBR", SBR()))
        except Exception:
            pass

    # Torch-based optimizer if available
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

    results = []
    for name, opt in optimizers:
        try:
            res = run_optimizer(name, opt, X, dt, library, dX_dt)
        except Exception as e:
            res = {"name": name, "error": str(e)}
        results.append(res)

    # Pretty print summary
    header = f"{'Optimizer':<15} {'Score':>10} {'MSE':>12} {'Time (s)':>12} {'Complexity':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        if "error" in r:
            print(f"{r['name']:<15} ERROR: {r['error']}")
        else:
            print(f"{r['name']:<15} {r['score']:>10.4f} {r['mse']:>12.4e} {r['fit_time_s']:>12.4f} {str(r['complexity']):>12}")
            # Print discovered system equations
    for  r in results:
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
    system_lorenz_example()
