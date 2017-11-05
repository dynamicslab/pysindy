"""
   _________  ____ ______________  ________  ____ _
  / ___/ __ \/ __ `/ ___/ ___/ _ \/ ___/ _ \/ __ `/
 (__  ) /_/ / /_/ / /  (__  )  __/ /  /  __/ /_/ /
/____/ .___/\__,_/_/  /____/\___/_/   \___/\__, /
    /_/                                   /____/
"""

import io
import os

import versioneer
from setuptools import find_packages, setup

NAME = "sparsereg"
DESCRIPTION = "Modern sparse linear regression"
URL = "https://github.com/ohjeah/sparsereg"
EMAIL = "info@markusqua.de"
AUTHOR = "Markus Quade"
PYTHON = ">=3.5"
LICENSE = "LGPL"
CLASSIFIERS = [
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    "Topic :: Scientific/Engineering :: Mathematics",    
]

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements.txt"), "r") as f:
    REQUIRED = f.readlines()

with io.open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
    LONG_DESCRIPTION = "\n" + f.read()


setup(
    name=NAME,
    version=versioneer.get_version(),
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
    cmdclass=versioneer.get_cmdclass(),
)
