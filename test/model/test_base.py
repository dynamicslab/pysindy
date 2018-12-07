import pytest

from sparsereg.model.base import print_model


@pytest.mark.parametrize(
    "case, res",
    [
        (
            dict(
                coef=[1, 2, 0.001], errors=[0.01, 0.2, 0.003], input_features=["x0", "x1", "x2"], precision=2
            ),
            "(1.00 ± 0.01) x0 + (2.00 ± 0.20) x1",
        ),
        (
            dict(
                coef=[1, 2, 0.001], errors=[0.01, 0.2, 0.003], input_features=["x0", "x1", "x2"], precision=3
            ),
            "(1.000 ± 0.010) x0 + (2.000 ± 0.200) x1 + (0.001 ± 0.003) x2",
        ),
        (
            dict(
                coef=[1, 2, 0.001],
                input_features=["x0", "x1", "x2"],
                intercept=0.1,
                sigma_intercept=0.1,
                precision=2,
            ),
            "1.00 x0 + 2.00 x1 + (0.10 ± 0.10)",
        ),
        (
            dict(coef=[1, 2, 0.001], input_features=["x0", "x1", "x2"], intercept=0.1, precision=2),
            "1.00 x0 + 2.00 x1 + 0.10",
        ),
        (dict(coef=[1, 2, 0.001], input_features=["x0", "x1", "x2"], precision=2), "1.00 x0 + 2.00 x1"),
        (dict(coef=[0, 0, 0.001], input_features=["x0", "x1", "x2"], precision=2), "0.00"),
    ],
)
def test_print_model(case, res):
    assert print_model(**case) == res
