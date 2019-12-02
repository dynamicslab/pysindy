import sys

assert sys.version_info >= (3, 6, 0), "sindy requires Python 3.6+"
import pathlib

from setuptools import find_packages, setup

NAME = "sindy"
DESCRIPTION = "Sparse Identification of Nonlinear Dynamics"
URL = "https://github.com/briandesilva/sindy"
EMAIL = "bdesilva@uw.edu"
AUTHOR = "Brian de Silva"
PYTHON = ">=3.6"
LICENSE = "MIT"
CLASSIFIERS = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Development Status :: 4 - Beta",
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
    version="0.0.1",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=["test", "example"]),
    install_requires=REQUIRED,
    python_requires=PYTHON,
    license=LICENSE,
    classifiers=CLASSIFIERS,
)
