import numpy as np

import pytest

from sparsereg.util.pipeline import ColumnSelector

cases = ((np.random.random(10), 0),
         (np.random.random(size=(10, 2)), 1),
         (np.random.random(size=(10, 2)), slice(None))
        )

@pytest.mark.parametrize("case", cases)
def test_ColumnSelector(case):
    x, index = case
    assert len(ColumnSelector(index).fit_transform(x).shape) == 2
