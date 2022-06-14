import os
import sys
from pathlib import Path

import nbformat
import pytest
from ipykernel import kernelapp as app
from nbconvert.preprocessors import ExecutePreprocessor

# Find all example notebooks
notebook_dir = Path(__file__).parent.parent / "examples"
notebooks = [nb.name for nb in notebook_dir.glob("*.ipynb")]


def _load_notebook(filename: Path) -> nbformat.NotebookNode:
    """Load Jupyter Notebook as NotebookNode object."""
    with open(filename, "r") as fh:
        notebook = nbformat.read(fh, as_version=4)
    return notebook


def _create_kernel() -> str:
    """Create a IPython Kernel for the current executable."""
    kernel_name = sys.executable.replace("/", ".").replace("\\", ".").replace(":", ".")
    app.launch_new_instance(argv=["install", "--user", "--name", kernel_name])
    return kernel_name


@pytest.mark.parametrize("filename", notebooks)
@pytest.mark.slow
def test_notebook(filename):
    nb = _load_notebook(notebook_dir / filename)
    # Need to run in notebook directory to find data via relative path
    os.chdir(notebook_dir)
    # Run the notebook
    ExecutePreprocessor(timeout=-1, kernel=_create_kernel).preprocess(nb)
    os.chdir(Path(__file__).parent)
