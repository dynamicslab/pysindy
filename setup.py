import pathlib
import sys

from setuptools import find_packages
from setuptools import setup


assert sys.version_info >= (3, 7, 0), "PySINDy requires Python 3.7+"

NAME = "pysindy"
DESCRIPTION = "Sparse Identification of Nonlinear Dynamics"
URL = "https://github.com/dynamicslab/pysindy"
EMAIL = "bdesilva@uw.edu, kpchamp@uw.edu, info@markusqua.de, akaptano@uw.edu"
AUTHOR = "Brian de Silva, Kathleen Champion, Markus Quade, Alan Kaptanoglu"
PYTHON = ">=3.7"
LICENSE = "MIT"
CLASSIFIERS = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering :: Mathematics",
]

here = pathlib.Path(__file__).parent

with open(here / "requirements.txt", "r") as f:
    REQUIRED = f.readlines()

with open(here / "README.rst", "r") as f:
    LONG_DESCRIPTION = f.read()


setup(
    name=NAME,
    use_scm_version=True,
    setup_requires=["setuptools_scm", "setuptools_scm_git_archive"],
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/x-rst",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=["test", "examples"]),
    install_requires=REQUIRED,
    python_requires=PYTHON,
    license=LICENSE,
    classifiers=CLASSIFIERS,
)
