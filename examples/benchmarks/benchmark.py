#!/usr/bin/env python3
"""
Generalized benchmark for PySINDy optimizers on multiple nonlinear systems.

Features
- Unified runner for many ODE systems (Lorenz, Rossler, Van der Pol, Duffing, Chua,
  Pendulum, Kuramoto, Hindmarsh–Rose (reduced), FitzHugh–Nagumo, Thomas, Sprott A,
  and a nonlinear spring).
- Simple RK4 integrator (no external dependencies).
- Evaluates available optimizers (STLSQ, SR3, FROLS, SSR, optional SBR, optional TorchOptimizer).
- Reports runtime, score, MSE against analytic RHS, complexity, and discovered equations.
- CLI flags to choose system, integration params, library degree, and optimizers.
- Supports running all systems in a single invocation with --system all.

Usage examples
- Run Lorenz with defaults:
  python examples/benchmarks/benchmark.py --system lorenz --dt 0.01 --t1 10
- List systems:
  python examples/benchmarks/benchmark.py --list
- Run all systems on all solvers:
  python examples/benchmarks/benchmark.py --system all --dt 0.01 --t1 5
"""
import argparse
import time
import traceback
import warnings
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary
from pysindy.optimizers import FROLS
from pysindy.optimizers import SR3
from pysindy.optimizers import SSR
from pysindy.optimizers import STLSQ

try:
    from pysindy.optimizers import SBR
except Exception:
    SBR = None  # type: ignore
try:
    from pysindy.optimizers import TorchOptimizer
except Exception:
    TorchOptimizer = None  # type: ignore


# ------------------------------- Systems ------------------------------------


def lorenz_rhs(t, x, sigma=10.0, beta=8.0 / 3.0, rho=28.0):
    u, v, w = x
    return np.array(
        [
            -sigma * (u - v),
            rho * u - v - u * w,
            -beta * w + u * v,
        ],
        dtype=float,
    )


def rossler_rhs(t, x, a=0.2, b=0.2, c=5.7):
    x1, x2, x3 = x
    return np.array(
        [
            -(x2 + x3),
            x1 + a * x2,
            b + x3 * (x1 - c),
        ],
        dtype=float,
    )


def vdp_rhs(t, x, mu=3.0):
    x1, x2 = x
    return np.array(
        [
            x2,
            mu * (1 - x1**2) * x2 - x1,
        ],
        dtype=float,
    )


def duffing_rhs(t, x, delta=0.2, alpha=-1.0, beta=1.0, gamma=0.3, omega=1.2):
    # Unforced variant (set gamma=0) or keep small forcing
    x1, x2 = x
    return np.array(
        [
            x2,
            -delta * x2 - alpha * x1 - beta * x1**3 + gamma * np.cos(omega * t),
        ],
        dtype=float,
    )


def chua_rhs(t, x, alpha=15.6, beta=28.0, m0=-1.143, m1=-0.714):
    x1, x2, x3 = x
    h = m1 * x1 + 0.5 * (m0 - m1) * (np.abs(x1 + 1) - np.abs(x1 - 1))
    return np.array(
        [
            alpha * (x2 - x1 - h),
            x1 - x2 + x3,
            -beta * x2,
        ],
        dtype=float,
    )


def pendulum_rhs(t, x, g=9.81, L=1.0, q=0.0, F=0.0, omega_d=0.0):
    # Damped/forced nonlinear pendulum; default undamped, unforced
    theta, omega = x
    return np.array(
        [
            omega,
            -(g / L) * np.sin(theta) - q * omega + F * np.sin(omega_d * t),
        ],
        dtype=float,
    )


def kuramoto_rhs(t, x, K=0.5):
    # Small network (3 oscillators) with identical natural freq=0
    # x are phases; coupling via sine differences
    n = x.shape[0]
    dx = np.zeros(n)
    for i in range(n):
        dx[i] = (K / n) * np.sum(np.sin(x - x[i]))
    return dx


def hindmarsh_rose_rhs(t, x):
    # Reduced 2D form for benchmarking
    x1, x2 = x
    a = 1.0
    b = 3.0
    c = 1.0
    d = 5.0
    return np.array(
        [
            x2 - a * x1**3 + b * x1**2 + 1.0,  # simplified variant
            c - d * x1**2 - x2,  # simplified variant
        ],
        dtype=float,
    )


def fitzhugh_nagumo_rhs(t, x, a=0.7, b=0.8, tau=12.5, Ia=0.5):
    v, w = x
    return np.array(
        [
            v - v**3 / 3 - w + Ia,
            (v + a - b * w) / tau,
        ],
        dtype=float,
    )


def thomas_rhs(t, x, b=0.208186):
    x1, x2, x3 = x
    return np.array(
        [
            np.sin(x2) - b * x1,
            np.sin(x3) - b * x2,
            np.sin(x1) - b * x3,
        ],
        dtype=float,
    )


def sprott_a_rhs(t, x):
    x1, x2, x3 = x
    return np.array(
        [
            x2,
            x3,
            -x1 + x2**2 - x3,
        ],
        dtype=float,
    )


def myspring_rhs(t, x, k=-4.518, c=0.372, F0=9.123):
    return np.array([x[1], k * x[0] - c * x[1] + F0 * np.sin(x[0] ** 2)], dtype=float)


SYSTEMS: Dict[str, Tuple[Callable, np.ndarray]] = {
    "lorenz": (lorenz_rhs, np.array([1.0, 1.0, 1.0], dtype=float)),
    "rossler": (rossler_rhs, np.array([0.1, 0.1, 0.1], dtype=float)),
    "vanderpol": (vdp_rhs, np.array([2.0, 0.0], dtype=float)),
    "duffing": (duffing_rhs, np.array([0.1, 0.0], dtype=float)),
    "chua": (chua_rhs, np.array([0.1, 0.0, 0.0], dtype=float)),
    "pendulum": (pendulum_rhs, np.array([0.5, 0.0], dtype=float)),
    "kuramoto3": (kuramoto_rhs, np.array([0.2, -0.3, 0.1], dtype=float)),
    "hindmarsh_rose2": (hindmarsh_rose_rhs, np.array([0.0, 0.0], dtype=float)),
    "fitzhugh_nagumo": (fitzhugh_nagumo_rhs, np.array([0.0, 0.0], dtype=float)),
    "thomas": (thomas_rhs, np.array([0.1, 0.2, 0.3], dtype=float)),
    "sprott_a": (sprott_a_rhs, np.array([0.1, 0.1, 0.1], dtype=float)),
    "myspring": (myspring_rhs, np.array([0.4, 1.6], dtype=float)),
}


def rk4(
    f: Callable, x0: np.ndarray, t0: float, t1: float, dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate an ODE x' = f(t, x) using classical RK4.

    Parameters
    - f: callable(t, x) -> np.ndarray, RHS function.
    - x0: initial state vector.
    - t0: start time.
    - t1: end time.
    - dt: time step.

    Returns
    - t: time array of shape (N,).
    - X: state trajectory of shape (N, d).
    - dX_dt: analytic RHS evaluated along trajectory, shape (N, d).
    """
    t = np.linspace(t0, t1, int(round((t1 - t0) / dt)) + 1)
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


def build_library(degree: int = 3) -> PolynomialLibrary:
    """Construct a polynomial feature library.

    Parameters
    - degree: maximum polynomial degree; interactions enabled.

    Returns
    - PolynomialLibrary instance.
    """
    return PolynomialLibrary(degree=degree, include_interaction=True)


def run_optimizer(
    name: str,
    optimizer,
    X: np.ndarray,
    dt: float,
    library: PolynomialLibrary,
    dX_dt: np.ndarray,
):
    """Fit a SINDy model and compute metrics for an optimizer.

    Returns a dict with name, runtime, score, MSE vs analytic RHS, complexity,
    equations (if available), and the fitted model.
    """
    model = SINDy(optimizer=optimizer, feature_library=library)
    t0 = time.perf_counter()
    model.fit(X, t=dt)
    fit_time = time.perf_counter() - t0
    score = model.score(X, t=dt)
    dXdt_pred = model.predict(X)
    mse = float(np.mean((dXdt_pred - dX_dt) ** 2))
    complexity = optimizer.complexity if hasattr(optimizer, "complexity") else None
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


def main():
    """Entry point for the generalized benchmark runner.

    Parses CLI, builds the library, iterates over selected systems, runs selected
    optimizers, prints a summary table, and highlights the best optimizer per system
    (lowest MSE), including its discovered equations.
    """
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(
        description="Generalized nonlinear systems benchmark for PySINDy."
    )
    parser.add_argument(
        "--system",
        type=str,
        default="all",
        choices=sorted(list(SYSTEMS.keys()) + ["all"]),
    )
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--t1", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--degree", type=int, default=3, help="Polynomial library degree"
    )
    parser.add_argument(
        "--optimizers", type=str, default="all", help="Comma-separated list or 'all'"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available systems and exit"
    )
    args = parser.parse_args()

    if args.list:
        print("Available systems:")
        for name in sorted(SYSTEMS.keys()):
            print(f" - {name}")
        print(" - all (run every system)")
        return

    systems_to_run = (
        [(args.system, SYSTEMS[args.system])]
        if args.system != "all"
        else list(SYSTEMS.items())
    )
    library = build_library(args.degree)

    # Select optimizers
    opt_defs: List[Tuple[str, object]] = []
    opt_defs.append(("STLSQ", STLSQ(threshold=0.1, alpha=0.05, max_iter=20)))
    opt_defs.append(
        (
            "SR3-L0",
            SR3(reg_weight_lam=0.1, regularizer="L0", relax_coeff_nu=1.0, max_iter=50),
        )
    )
    opt_defs.append(("FROLS", FROLS()))
    opt_defs.append(("SSR", SSR()))
    if SBR is not None:
        try:
            opt_defs.append(("SBR", SBR()))
        except Exception:
            traceback.print_exc()
    if TorchOptimizer is not None:
        try:
            opt_defs.append(
                (
                    "TorchOptimizer",
                    TorchOptimizer(
                        threshold=0.05,
                        alpha_l1=1e-3,
                        step_size=1e-2,
                        max_iter=200,
                        optimizer="cadamw",
                        seed=0,
                        unbias=True,
                        early_stopping_patience=50,
                        min_delta=1e-8,
                    ),
                )
            )
        except Exception:
            traceback.print_exc()

    if args.optimizers != "all":
        names = {n.strip() for n in args.optimizers.split(",")}
        opt_defs = [pair for pair in opt_defs if pair[0] in names]

    # Run per system
    for sys_name, (rhs, x0) in systems_to_run:
        print(f"\n=== System: {sys_name} ===")
        t, X, dX_dt = rk4(rhs, x0, args.t0, args.t1, args.dt)
        results = []
        for name, opt in opt_defs:
            try:
                res = run_optimizer(name, opt, X, args.dt, library, dX_dt)
            except Exception as e:
                res = {"name": name, "error": str(e)}
            results.append(res)

        header = f"{'Optimizer':<15} {'Score':>10} {'MSE':>12} {'Time (s)':>12} {'Complexity':>12}"
        print(header)
        print("-" * len(header))
        for r in results:
            if "error" in r:
                print(f"{r['name']:<15} ERROR: {r['error']}")
            else:
                print(
                    f"{r['name']:<15} {r['score']:>10.4f} {r['mse']:>12.4e} {r['fit_time_s']:>12.4f} {str(r['complexity']):>12}"
                )

        # Select and print best optimizer by lowest MSE among successful runs
        successful = [r for r in results if "error" not in r]
        if successful:
            best = min(successful, key=lambda r: r["mse"])  # type: ignore
            print(
                f"\n>>> Best optimizer: {best['name']} | Score={best['score']:.4f} | MSE={best['mse']:.4e} | Time={best['fit_time_s']:.4f}s | Complexity={best['complexity']}"
            )
            eqs = best.get("equations")
            if eqs:
                print("Discovered equations:")
                for eq in eqs:
                    print(f"    {eq}")
            else:
                print("(equations unavailable)")
        else:
            print("\n>>> No successful optimizer runs for this system.")


if __name__ == "__main__":
    main()
