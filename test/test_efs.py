import pytest

from sparsereg.efs import *

size_cases = (
    ("x_1", 1),
    ("x_2", 1),
    ("log(x_1)", 2),
    ("div(x_3, add(x_1, x_2))", 5),
)

@pytest.mark.parametrize("case", size_cases)
def test_size(case):
    name, res = case
    assert size(name) == res
