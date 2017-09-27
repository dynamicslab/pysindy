import pytest
import numpy as np


@pytest.fixture
def data():
    x = 2 * np.random.random(size=(100, 2)) + 5
    y = 2 * x[:, 1] + 3
    return x, y


@pytest.fixture
def data_full_rank():
    x = 2 * np.random.random(size=(100, 2)) + 5
    y = np.sum(x, axis=1)
    return x, y
