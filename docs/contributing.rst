Contributing to pysindy
================================
We welcome all sorts of contributions!
There are generally three different types of contributions.
Please be aware of how to conduct yourself in an online, professional environment;
We have a standard contributor `code of conduct`_.

.. _code of conduct: https://github.com/dynamicslab/pysindy/blob/main/docs/contributor_covenant.md



Three types of contributions
--------------------------------
The pysindy repository combines several different types of users and use cases:

*   You want to discover an ODE for your particular data, potentially as a research paper
*   You want to publish a new methods of learning differential equations from data,
    making it available to other users
*   You are learning about equation discovery from reading papers, and this repo is
    an educational supplement to existing papers/textbooks.

There are several different ways people contribute:

*   Sharing a novel use case/research notebook with the wider public
*   Adding features or fixing bugs
*   Improving documentation.

Contributing examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We love seeing examples of PySINDy being used to solve interesting problems!
If you would like to contribute an example to the documentation,
reach out to us by creating an issue.

Examples are external repositories that
`follow a structure <https://github.com/dynamicslab/pysindy-example>`_.
They tend to be pinned to a set of dependencies and may not be kept up to date
with breaking API changes.
Pysindy has a sphinx extension that allows incorporating externally-built docs,
so long as it follows that structure.
Once your example repository is fully built, create a PR here that adds your repo
to examples/external.yml and examples/README.rst as:


.. code-block:: rst
    :caption: examples/README.rst

    .. pysindy-example::
        :key: your-example-key
        :title: Title of your example

        A description of your example

.. code-block:: yaml
    :caption: external.yml

    - name: "your-example-key"
      user: "user-or-org-owning-repo"
      repo: "repo-name"
      ref: "git hash of commit"
      dir: "name of folder with examples in it, e.g. 'examples' or '.'"


.. hint::

    The most frequent issue in linking external documentation is header levels.
    The sphinx directive adjusts section levels to fit correctly within our
    documentation.
    This means that notebooks need to have exactly one highest-level header as
    the document link text.
    Subordinate levels should proceed in a rational way, e.g. level two headers
    following level one headers, etc.

If you want to keep your example up to date with pysindy main branch,
i.e. across breaking changes, that is also possible.
*This repository* has the CI job ``notify-experiments`` that sends a
repository-dispatch action to repos that want to be notfied of changes to pysindy.
The example repository has a CI job (``pysindy.yml``) that can be cued by such dispatch.
You will have to add your repository information to the ``notify-experiments``
and add a `fine-grained PAT <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens>`_
with permissions ``contents: read & write`` and ``metadata: read only``,
then record that in pysindy's github secrets.
Alternatively, you could trigger your builds based upon cron timing.

Updates to an example are not automitcally used in the pysindy doc build.
To update an example, you need to change the commit information
in ``examples/external.yml``.

Contritubing documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We of course love documentation improvements!
Begin by opening an issue;
sometimes the problem is that the documentation is correct, but the code has a bug.
Other times, the documentation is merely out of date.
We attempt to follow the `divio format`_ of
tutorials, how-to, reference, and explanation.
We also have a section for research paper examples as a very specific form of how-to.

When making a documentation PR, it helps to pass CI if you build the check that the
documentation builds locally first.
See `Building docs`_


.. _divio format: https://docs.divio.com/documentation-system/

Contributing code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Similar to other open-source projects, most contributions start with a github issue
in order to solicit concurrence from project maintainers.
Then, the general process is:

1. Fork the repository
2. Step through the `Set up development environment`_.
3. Hack Hack Hack according to `Coding Guidelines`_.
4. Open a pull request
5. Respond to reviews

A good place to get started learning this process is to follow one of the issues labeled
`good first issue`_.

.. _good first issue: https://github.com/dynamicslab/pysindy/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22

Development process
--------------------------------

Set up development environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Clone the repo, then create a project isolation in your environment manager of choice
(e.g. venv, conda, virtualenv).
Install the project in an editable way with all optional dependencies, e.g.

.. code-block:: bash

    pip install -e .[dev,miosr,cvxpy,docs,sbr]

This will allow you to run unit tests and use code-formatting tools.

Install our pre-commit script via

.. code-block:: bash

    pre-commit install

Pre-commit will automatically check all future commits for code style.
To be accepted your code should conform to PEP8 and pass all unit tests.
Code can be tested by invoking

.. code-block:: bash

    pytest --jax-pcc

The ``jax-pcc`` flag is optional and will speed up the jax tests on repeated runs
by caching compiled functions to disk.


Coding Guidelines
^^^^^^^^^^^^^^^^^^^^

Hack away, following PEP 484, PEP 8, and other relevant guidelines.
Automated tooling (``pre-commit``) will hopefully keep you on track here.

Stage your changes with:

.. code-block:: bash

    git add path/to/changed/file.py

Pre-commit will then automatically run all checks against your committed code. If you want to trigger this manually, you can run the following to automatically reformat your staged code

.. code-block:: bash

    pre-commit

Note that you will then need to re-stage any changes ``pre-commit`` made to your code.

Make sure to write useful commit messages.
Consider that developers years from now will use ``git log -S`` and ``git blame``
to discover when certain changes took place and unearth why you made
the choices you did.

For good guidance on commit messages, see `what makes a good git commit`_ or
`conventional commits`_.  TL;DR: Subjects in imperative tense, less than 72 characters.
If necessary, body describes why the change was made, e.g. previous behavior,
spooky action at a distance. If the commit resolves
a github issue, use github's automatic issue-\ `closing words`_.
Lines in the commit body should not exceed 80 characters.

.. _conventional commits: https://www.conventionalcommits.org/en/v1.0.0/
.. _what makes a good git commit: https://www.simplethread.com/what-makes-a-good-git-commit/
.. _closing words: https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue#linking-a-pull-request-to-an-issue-using-a-keyword

Building docs
^^^^^^^^^^^^^^^^^^
With all optional dependencies installed, run:

.. code-block:: bash

    python -m sphinx -TEWb html docs docs/_build

Or check the build step in the most recent CI run or [RTD build](https://readthedocs.org/projects/pysindy/builds/).

Community
------------
This repository is a fork from `original work <https://github.com/Ohjeah/sparsereg>`_ by `Markus Quade <https://github.com/Ohjeah>`_.

Thanks to the members of the community who have contributed to PySINDy!

Desired changes
-----------------------

Major changes that will affect the `type system <object model>`_ include:

1. Promoting weak and discrete SINDy into subclasses of ``_BaseSINDy``.  This will remove the need for feature libraries to do so much, and prevent incompatibility errors.
2. Making feature libraries array-implementation agnostic in order to be compatible with sympy, jax, cvxpy.
3. Standardize the differentiation API, so that order & axis behave the same for each method, so that the smoothed trajectory is also returned, and so that smoothing can be consistent across multiple spatial dimensions.
4. Add support for single-step methods that do differentiation and fitting the dynamics at the same time.
5. Make the statistical optimizers (``EnsembleOptimizer``, ``SBR``) a separate type so that they can be more easily compared (and prevent the incompatible wrapping of one with the other).

There are a number of SINDy variants and advanced functionality that would be great to implement in future releases:

1. Bayesian SINDy, for instance that from Hirsh, Seth M., David A. Barajas-Solano, and J. Nathan Kutz. "Sparsifying Priors for Bayesian Uncertainty Quantification in Model Discovery." arXiv preprint arXiv:2107.02107 (2021).
2. Tensor SINDy, using the methods in Gelß, Patrick, et al. "Multidimensional approximation of nonlinear dynamical systems." Journal of Computational and Nonlinear Dynamics 14.6 (2019).
3. Stochastic SINDy, using the methods in Brückner, David B., Pierre Ronceray, and Chase P. Broedersz. "Inferring the dynamics of underdamped stochastic systems." Physical review letters 125.5 (2020): 058103.
4. Integration of PySINDy with a Python model-predictive control (MPC) code.
5. The PySINDy weak formulation is based on the work in Reinbold, Patrick AK, Daniel R. Gurevich, and Roman O. Grigoriev. "Using noisy or incomplete data to discover models of spatiotemporal dynamics." Physical Review E 101.1 (2020): 010203. It might be useful to additionally implement the weak formulation from Messenger, Daniel A., and David M. Bortz. "Weak SINDy for partial differential equations." Journal of Computational Physics (2021): 110525. The weak formulation in PySINDy is also fairly slow and computationally intensive, so finding ways to speed up the code would be great.
6. The blended conditional gradients (BCG) algorithm for solving the constrained LASSO problem, Carderera, Alejandro, et al. "CINDy: Conditional gradient-based Identification of Non-linear Dynamics--Noise-robust recovery." arXiv preprint arXiv:2101.02630 (2021).
