import os
import sys
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

notebook_dir = Path(__file__).parent.parent / "examples"
notebooks = [nb.name for nb in notebook_dir.glob("*.ipynb")]


def _load_notebook(filename: Path) -> str:
    with open(filename, "r") as fh:
        notebook = nbformat.read(fh, as_version=4)
    return notebook


def _create_kernel():
    from ipykernel import kernelapp as app

    kernel_name = sys.executable.replace("/", ".").replace("\\", ".").replace(":", ".")
    app.launch_new_instance(argv=["install", "--user", "--name", kernel_name])
    return kernel_name


@pytest.mark.parametrize("filename", notebooks)
@pytest.mark.slow
def test_notebook(filename):
    nb = _load_notebook(notebook_dir / filename)
    os.chdir(notebook_dir)
    ExecutePreprocessor(timeout=-1, kernel=_create_kernel).preprocess(nb)
    os.chdir(Path(__file__).parent)
