import sys
assert sys.version_info >= (3, 6, 0), "sparsereg requires Python 3.6+"
import pathlib

import versioneer
from setuptools import find_packages, setup

NAME = "sparsereg"
DESCRIPTION = "Modern sparse linear regression"
URL = "https://github.com/ohjeah/sparsereg"
EMAIL = "info@markusqua.de"
AUTHOR = "Markus Quade"
PYTHON = ">=3.6"
LICENSE = "MIT"
CLASSIFIERS = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    "Topic :: Scientific/Engineering :: Mathematics",
]

here = pathlib.Path(__file__).parent

with open(here / "requirements.txt", "r") as f:
    REQUIRED = f.readlines()

with open(here / "README.md", "r") as f:
    LONG_DESCRIPTION = f.read()


setup(
    name=NAME,
    version=versioneer.get_version(),
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=["test", "example"]),
    install_requires=REQUIRED,
    python_requires=PYTHON,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    cmdclass=versioneer.get_cmdclass(),
)
