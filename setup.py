import os
import sys
from os.path import dirname

from setuptools import setup, find_packages

here = os.path.abspath(dirname(__file__))

with open(os.path.join(here, 'README.rst')) as f:
    long_description = '\n' + f.read()

base_dir = os.path.dirname(__file__)

about = {}
with open(os.path.join(base_dir, "sparsereg", "__version__.py")) as f:
    exec(f.read(), about)

with open("requirements.txt", "r") as f:
    required = f.readlines()

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist bdist_wheel upload")
    sys.exit()

setup(
    name='sparsereg',
    version=about['__version__'],
    description='Modern sparse linear regression',
    long_description=long_description,
    author='Markus Quade',
    author_email='info@markusqua.de',
    url='https://github.com/ohjeah/sparsereg',
    packages=find_packages(exclude=["test", "example"]),
    install_requires=required,
    license='MIT',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
