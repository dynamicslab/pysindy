import os
from argparse import ArgumentParser
from pathlib import Path

import jupytext
import nbformat
from ipykernel import kernelapp as app
from nbconvert.preprocessors import ExecutePreprocessor

notebook_dir = Path(__file__).parent

argparser = ArgumentParser(
    description="Create and run a .ipynb file from a script named 'example.py'."
)
argparser.add_argument("target", help="Directory containing example.py")


def _create_kernel() -> str:
    """Create a IPython Kernel for the current executable."""
    kernel_name = __name__
    app.launch_new_instance(argv=["install", "--user", "--name", kernel_name])
    return kernel_name


if __name__ == "__main__":
    args = argparser.parse_args()
    tgt_dir = Path(args.target)
    tgt_file = tgt_dir.resolve() / "example.py"
    # convert notebook-formatted python script to notebook
    nb = jupytext.read(tgt_file)
    cwd = Path.cwd()
    os.chdir(tgt_dir)
    # Run the notebook
    ExecutePreprocessor(timeout=-1, kernel=_create_kernel).preprocess(nb)
    os.chdir(cwd)
    # Save the notebook
    with open(str(tgt_dir / "example.ipynb"), "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
