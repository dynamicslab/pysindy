import pytest
import numpy as np


@pytest.fixture
def data():
    x = 2*np.random.random(size=(100, 2)) + 5
    y = 2 * x[:, 1]
    return x, y
