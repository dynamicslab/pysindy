"""
   _________  ____ ______________  ________  ____ _
  / ___/ __ \/ __ `/ ___/ ___/ _ \/ ___/ _ \/ __ `/
 (__  ) /_/ / /_/ / /  (__  )  __/ /  /  __/ /_/ /
/____/ .___/\__,_/_/  /____/\___/_/   \___/\__, /
    /_/                                   /____/
"""

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

NAME = "sparsereg"
DESCRIPTION = 'Modern sparse linear regression'
URL = 'https://github.com/ohjeah/sparsereg'
EMAIL = 'info@markusqua.de'
AUTHOR = 'Markus Quade'

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "requirements.txt"), "r") as f:
    REQUIRED = f.readlines()

with io.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    LONG_DESCRIPTION = '\n' + f.read()

about = {}
with open(os.path.join(here, NAME, '__version__.py')) as f:
    exec(f.read(), about)


class PublishCommand(Command):
    """Support setup.py publish."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds ...')
            rmtree(os.path.join(here, 'dist'))
        except FileNotFoundError:
            pass

        self.status('Building Source and Wheel (universal) distribution...')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPi via Twine...')
        os.system('twine upload dist/*')

        sys.exit()


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=["test", "example"]),
    install_requires=REQUIRED,
    license='MIT',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    cmdclass={
        'publish': PublishCommand,
    },
)
