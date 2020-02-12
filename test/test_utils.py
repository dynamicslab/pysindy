import pytest
from ffx.api import FFXRegressor
from sklearn.linear_model import Lasso

from pysindy.optimizers import SR3
from pysindy.optimizers import STLSQ
from pysindy.utils import supports_multiple_targets


@pytest.mark.parametrize(
    "cls, support", [(Lasso, True), (STLSQ, True), (SR3, True), (FFXRegressor, False)]
)
def test_supports_multiple_targets(cls, support):
    assert supports_multiple_targets(cls()) == support
