import os
import sys
import glob
import inspect

import pytest

THIS_FILES_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
slow = pytest.mark.skipif(not pytest.config.getoption("--runslow"), reason="need --runslow option to run")

@pytest.mark.parametrize("file", glob.glob(THIS_FILES_DIR + "/../example/sindy.py"))
#@slow
def test_symbolic_regression_example(file):
    error_code = os.system("{} {}".format(sys.executable, file))
    assert error_code == 0
