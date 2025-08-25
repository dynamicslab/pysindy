PySINDy
=========

|BuildCI| |RTD| |PyPI| |Codecov| |JOSS1| |JOSS2| |DOI|

**PySINDy** is a sparse regression package with several implementations for the
Sparse Identification of Nonlinear Dynamical systems (SINDy) method introduced
in Brunton et al. (2016a).
It also includes a variety of other methods from related literature.
A comprehensive literature review is given in de Silva et al. (2020) and Kaptanoglu, de Silva et al. (2021).

System identification
---------------------
System identification refers to the process of leveraging measurement data to infer governing equations, in the form of dynamical systems, describing the data.
Once discovered, these equations can make predictions about future states, can inform control inputs, or can enable the theoretical study using analytical techniques.
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
For a more in-depth look at the mathematical foundations of SINDy, please see our `introduction to SINDy <https://pysindy.readthedocs.io/en/latest/examples/2_introduction_to_sindy/example.html>`__.

Relation to PySINDy
^^^^^^^^^^^^^^^^^^^
The PySINDy package revolves around the ``SINDy`` class which consists of three primary components; one for each term in the above matrix approximation problem.

* ``differentiation_method``: computes ``X'``, though if derivatives are known or measured directly, they can be used instead
* ``feature_library``: specifies the candidate basis functions to be used to construct ``Theta(X)``
* ``optimizer``: implements a sparse regression method for solving for ``Xi``

Once a ``SINDy`` object has been created it must be fit to measurement data, similar to a ``scikit-learn`` model.
It can then be used to predict derivatives given new measurements, evolve novel initial conditions forward in time, and more.

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

PySINDy provides numerous other features not shown here. We recommend the `feature overview <https://pysindy.readthedocs.io/en/latest/examples/1_feature_overview/example.html>`__ section of the documentation for a more exhaustive summary of additional features.

Installation
------------

The preferred way to install is with pip or conda e.g. ``pip install pysindy``.
You may have to add ``--user`` option.
Pysindy also provides several extras:

cvxpy
  Convex optimizer ``SR3`` and subclasses

miosr
  Branch-and-bound optimizer for L0-constraint, ``MIOSR``

sbr
  Bayesian regression optimizer yielding posteriors, ``SBR``.


Documentation
-------------
The documentation site for PySINDy can be found `here <https://pysindy.readthedocs.io/en/latest/>`__.
There are numerous `examples <https://pysindy.readthedocs.io/en/latest/examples/index.html>`_ of PySINDy in action to help you get started.
Examples are also available as `Jupyter notebooks <https://github.com/dynamicslab/pysindy/tree/master/examples>`__.
A video overview of PySINDy can be found on `Youtube <https://www.youtube.com/watch?v=DvbbXX8Bd90>`__.
We have also created a `video playlist <https://www.youtube.com/playlist?list=PLN90bHJU-JLoOfEk0KyBs2qLTV7OkMZ25>`__ with practical PySINDy tips.

If something is unclear, please open an issue.  To discuss your particular dynamics problem, open a discussion.  Make sure to format your example as python code in github!

PySINDy implements a lot of advanced functionality that may be overwhelming for new users or folks who are unfamiliar with these methods. Below (see here if image does not render https://github.com/dynamicslab/pysindy/blob/master/docs/JOSS2/Fig3.png), we provide a helpful flowchart for figuring out which methods to use, given the characteristics of your dataset:

.. image:: https://github.com/dynamicslab/pysindy/blob/master/docs/JOSS2/Fig3.png

This flow chart summarizes how ``PySINDy`` users can start with a dataset and systematically choose the proper candidate library and sparse regression optimizer that are tailored for a specific scientific task. The ``GeneralizedLibrary`` class allows for tensoring, concatenating, and otherwise combining many different candidate libraries.

Contributions:
-----------------------------
See `Contributor guide <https://pysindy.readthedocs.io/en/latest/contributing.html>`_.


Citing PySINDy
-----------------
See `Academic use <https://pysindy.readthedocs.io/en/latest/academic.html>`_.


Related packages
----------------
* `Deeptime <https://github.com/deeptime-ml/deeptime>`_ - A Python library for the analysis of time series data with methods for dimension reduction, clustering, and Markov model estimation.
* `PyDMD <https://github.com/mathLab/PyDMD/>`_ - A Python package using the Dynamic Mode Decomposition (DMD) for a data-driven model simplification based on spatiotemporal coherent structures. DMD is a great alternative to SINDy.
* `PySINDyGUI <https://github.com/hyumo/pysindy-gui>`_ - A slick-looking GUI for PySINDy.
* `SEED <https://github.com/M-Vause/SEED2.0>`_ - Software for the Extraction of Equations from Data: a GUI for many of the methods provided by PySINDy.
* `SymINDy <https://github.com/andreikitaitsev/SymINDy/>`_ - A Python package combining SINDy with genetic programming-based symbolic regression, used for the functions library optimization.
* `derivative <https://derivative.readthedocs.io/en/latest/>`_ - A Python package for differentiating (and optionally, smoothing) data, used by pysindy


.. |BuildCI| image:: https://github.com/dynamicslab/pysindy/actions/workflows/main.yml/badge.svg
    :target: https://github.com/dynamicslab/pysindy/actions/workflows/main.yml?query=branch%3Amaster

.. |RTD| image:: https://readthedocs.org/projects/pysindy/badge/?version=latest
    :target: https://pysindy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |PyPI| image:: https://badge.fury.io/py/pysindy.svg
    :target: https://badge.fury.io/py/pysindy

.. |Codecov| image:: https://codecov.io/gh/dynamicslab/pysindy/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/dynamicslab/pysindy

.. |JOSS1| image:: https://joss.theoj.org/papers/82d080bbe10ac3ab4bc03fa75f07d644/status.svg
    :target: https://joss.theoj.org/papers/82d080bbe10ac3ab4bc03fa75f07d644

.. |JOSS2| image:: https://joss.theoj.org/papers/10.21105/joss.03994/status.svg
    :target: https://doi.org/10.21105/joss.03994

.. |DOI| image:: https://zenodo.org/badge/186055899.svg
   :target: https://zenodo.org/badge/latestdoi/186055899
