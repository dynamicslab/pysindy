PySINDy
=========

|BuildCI| |RTD| |PyPI| |Codecov| |JOSS|

**PySINDy** is a sparse regression package with several implementations for the Sparse Identification of Nonlinear Dynamical systems (SINDy) method introduced in Brunton et al. (2016a), including the unified optimization approach of Champion et al. (2019) and SINDy with control from Brunton et al. (2016b). A comprehensive literature review is given in de Silva et al. (2020).

System identification
---------------------
System identification refers to the process of leveraging measurement data to infer governing equations, in the form of dynamical systems, describing the data. Once discovered, these equations can make predictions about future states, can inform control inputs, or can enable the theoretical study using analytical techniques.
Dynamical systems are a flexible, well-studied class of mathematical objects for modeling systems evolving in time.
SINDy is a model discovery method which uses *sparse regression* to infer nonlinear dynamical systems from measurement data.
The resulting models are inherently *interpretable* and *generalizable*.

How it works
^^^^^^^^^^^^
Suppose, for some physical system of interest, we have measurements of state variables ``x(t)`` (a vector of length n) at different points in time. Examples of state variables include the position, velocity, or acceleration of objects; lift, drag, or angle of attack of aerodynamic objects; and concentrations of different chemical species. If we suspect that the system could be well-modeled by a dynamical system of the form

.. code-block:: text

    x'(t) = f(x(t)),

then we can use SINDy to learn ``f(x)`` from the data (``x'(t)`` denotes the time derivative of ``x(t)``). Note that both ``f(x)`` and ``x(t)`` are typically vectors. The fundamental assumption SINDy employs is that each component of ``f(x)``, ``f_i(x)`` can be represented as a *sparse* linear combination of basis functions ``theta_j(x)``

.. code-block:: text

    f_i(x) = theta_1(x) * xi_{1,i} + theta_2(x) * xi_{2,i} + ... + theta_k * xi{k,i}

Concatenating all the objects into matrices (denoted with capitalized names) helps to simplify things.
To this end we place all measurements of the state variables into a data matrix ``X`` (with a row per time measurement and a column per variable), the derivatives of the state variables into a matrix ``X'``, all basis functions evaluated at all points in time into a matrix ``Theta(X)`` (each basis function gets a column), and all coefficients into a third matrix ``Xi`` (one column per state variable).
The approximation problem to be solved can then be compactly written as

.. code-block:: text

    X' = Theta(X) * Xi.

Each row of this matrix equation corresponds to one coordinate function of ``f(x)``.
SINDy employs sparse regression techniques to find a solution ``Xi`` with sparse column vectors.
For a more in-depth look at the mathematical foundations of SINDy, please see our `introduction to SINDy <https://pysindy.readthedocs.io/en/latest/examples/2_introduction_to_sindy.html>`__.

Relation to PySINDy
^^^^^^^^^^^^^^^^^^^
The PySINDy class revolves around the ``SINDy`` class which consists of three primary components; one for each term in the above matrix approximation problem.

* ``differentiation_method``: computes ``X'``, though if derivatives are known or measured directly, they can be used instead
* ``feature_library``: specifies the candidate basis functions to be used to construct ``Theta(X)``
* ``optimizer``: implements a sparse regression method for solving for ``Xi``

Once a ``SINDy`` object has been created it must be fit to measurement data, similar to a ``scikit-learn`` model. It can then be used to predict derivatives given new measurements, evolve novel initial conditions forward in time, and more. PySINDy has been written to be as compatible with ``scikit-learn`` objects and methods as possible.

Example
^^^^^^^
Suppose we have measurements of the position of a particle obeying the following dynamical system at different points in time

.. code-block:: text

  x' = -2x
  y' = y

Note that this system of differential equations decouples into two differential equations whose solutions are simply ``x(t) = x_0 * exp(-2 * t)`` and ``y(t) = y_0 * exp(t)``, where ``x_0 = x(0)`` and ``y_0 = y(0)`` are the initial conditions.

Using the initial conditions ``x_0 = 3`` and ``y_0 = 0.5``, we construct the data matrix ``X``.

.. code-block:: python

  import numpy as np
  import pysindy as ps

  t = np.linspace(0, 1, 100)
  x = 3 * np.exp(-2 * t)
  y = 0.5 * np.exp(t)
  X = np.stack((x, y), axis=-1)  # First column is x, second is y

To instantiate a ``SINDy`` object with the default differentiation method, feature library, and optimizer and then fit it to the data, we invoke

.. code-block:: python

  model = ps.SINDy(feature_names=["x", "y"])
  model.fit(X, t=t)

We use the ``feature_names`` argument so that the model prints out the correct labels for ``x`` and ``y``. We can inspect the governing equations discovered by the model and check whether they seem reasonable with the ``print`` function.

.. code-block:: python

  model.print()

which prints the following

.. code-block:: text

  x' = -2.000 x
  y' = 1.000 y

PySINDy provides numerous other features not shown here. We recommend the `feature overview <https://pysindy.readthedocs.io/en/latest/examples/1_feature_overview.html>`__ section of the documentation for a more exhaustive summary of additional features.

Installation
------------

Installing with pip
^^^^^^^^^^^^^^^^^^^

If you are using Linux or macOS you can install PySINDy with pip:

.. code-block:: bash

  pip install pysindy

Installing from source
^^^^^^^^^^^^^^^^^^^^^^
First clone this repository:

.. code-block:: bash

  git clone https://github.com/dynamicslab/pysindy

Then, to install the package, run

.. code-block:: bash

  pip install .

If you do not have pip you can instead use

.. code-block:: bash

  python setup.py install

If you do not have root access, you should add the ``--user`` option to the above lines.

Documentation
-------------
The documentation site for PySINDy can be found `here <https://pysindy.readthedocs.io/en/latest/>`__. There are numerous `examples <https://pysindy.readthedocs.io/en/latest/examples/index.html>`_ of PySINDy in action to help you get started. Examples are also available as `Jupyter notebooks <https://github.com/dynamicslab/pysindy/tree/master/examples>`__.

Community guidelines
--------------------

Contributing code
^^^^^^^^^^^^^^^^^
We welcome contributions to PySINDy. To contribute a new feature please submit a pull request. To get started we recommend installing the packages in ``requirements-dev.txt`` via

.. code-block:: bash

    pip install -r requirements-dev.txt

This will allow you to run unit tests and automatically format your code. To be accepted your code should conform to PEP8 and pass all unit tests. Code can be tested by invoking

.. code-block:: bash

    pytest

We recommed using ``pre-commit`` to format your code. Once you have staged changes to commit

.. code-block:: bash

    git add path/to/changed/file.py

you can run the following to automatically reformat your staged code

.. code-block:: bash

    pre-commit -a -v

Reporting issues or bugs
^^^^^^^^^^^^^^^^^^^^^^^^
If you find a bug in the code or want to request a new feature, please open an issue.

Getting help
^^^^^^^^^^^^
For help using PySINDy please consult the `documentation <https://pysindy.readthedocs.io/en/latest/>`__ and/or our `examples <https://github.com/dynamicslab/pysindy/tree/master/examples>`__, or create an issue.

References
----------------------
-  de Silva, Brian M., Kathleen Champion, Markus Quade,
   Jean-Christophe Loiseau, J. Nathan Kutz, and Steven L. Brunton.
   *PySINDy: a Python package for the sparse identification of
   nonlinear dynamics from data.* arXiv preprint arXiv:2004.08424 (2020)
   `[arXiv] <https://arxiv.org/abs/2004.08424>`_

-  Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz.
   *Discovering governing equations from data by sparse identification
   of nonlinear dynamical systems.* Proceedings of the National
   Academy of Sciences 113.15 (2016): 3932-3937.
   `[DOI] <http://dx.doi.org/10.1073/pnas.1517384113>`__

-  Champion, Kathleen, Peng Zheng, Aleksandr Y. Aravkin, Steven L.
   Brunton, and J. Nathan Kutz. *A unified sparse optimization
   framework to learn parsimonious physics-informed models from
   data.* arXiv preprint arXiv:1906.10612 (2019).
   `[arXiv] <https://arxiv.org/abs/1906.10612>`__
   
-  Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz.
   *Sparse identification of nonlinear dynamics with control (SINDYc).*
   IFAC-PapersOnLine 49.18 (2016): 710-715.
   `[DOI] <https://doi.org/10.1016/j.ifacol.2016.10.249>`_


.. |BuildCI| image:: https://github.com/dynamicslab/pysindy/workflows/Build%20CI/badge.svg
    :target: https://github.com/dynamicslab/pysindy/actions?query=workflow%3A%22Build+CI%22

.. |RTD| image:: https://readthedocs.org/projects/pysindy/badge/?version=latest
    :target: https://pysindy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |PyPI| image:: https://badge.fury.io/py/pysindy.svg
    :target: https://badge.fury.io/py/pysindy

.. |Codecov| image:: https://codecov.io/gh/dynamicslab/pysindy/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/dynamicslab/pysindy

.. |JOSS| image:: https://joss.theoj.org/papers/82d080bbe10ac3ab4bc03fa75f07d644/status.svg
    :target: https://joss.theoj.org/papers/82d080bbe10ac3ab4bc03fa75f07d644
