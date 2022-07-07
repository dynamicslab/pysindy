import os
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path

import nbformat
import pytest
from ipykernel import kernelapp as app
from nbconvert.preprocessors import ExecutePreprocessor

# Find all example notebooks
notebook_dir = Path(__file__).parent.parent / "examples"
notebooks = [nb.name for nb in notebook_dir.glob("*.ipynb")]
notebook_scripts = []
for x in notebook_dir.iterdir():
    if x.is_dir() and (x / "example.py").exists():
        notebook_scripts.append(x.name)


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


@contextmanager
def _cwd(directory):
    """Modify working directory and sys.path temporarily"""
    cwd = Path.cwd()
    os.chdir(directory)
    sys.path.insert(0, str(directory))
    yield
    sys.path.pop(0)
    os.chdir(cwd)


@pytest.mark.parametrize("directory", notebook_scripts)
def test_notebook_script(directory: Path):
    # Run in native directory with modified sys.path for imports to work
    with _cwd(notebook_dir / directory):
        runpy.run_path(str(notebook_dir / directory / "example.py"), run_name="testing")


@pytest.mark.parametrize("filename", notebooks)
@pytest.mark.slow
def test_notebook(filename):
    nb = _load_notebook(notebook_dir / filename)
    # Run in native directory to find data via relative path
    with _cwd(notebook_dir):
        ExecutePreprocessor(timeout=-1, kernel=_create_kernel).preprocess(nb)
