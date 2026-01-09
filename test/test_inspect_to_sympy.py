import pytest

try:
    from dysts.flows import Lorenz
except Exception:  # pragma: no cover - skip if dysts not installed
    pytest.skip(
        "dysts not available; skipping inspect_to_sympy tests", allow_module_level=True
    )

from asv_bench.benchmarks.inspect_to_sympy import dynsys_to_sympy


def test_lorenz_to_sympy():
    lor = Lorenz()
    symbols, exprs, lambda_rhs = dynsys_to_sympy(lor, func_name="_rhs")
    assert len(symbols) == lor.dimension
    # evaluate lambda with simple numeric values
    vals = tuple(float(i + 1) for i in range(lor.dimension))
    mat = lambda_rhs(*vals)
    assert mat.shape[0] == lor.dimension
