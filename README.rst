PySINDy
=========

|BuildCI| |RTD| |PyPI| |Codecov| |JOSS1| |JOSS2| |DOI|

**PySINDy** is a package for system identification, primarily revolving around the method of
Sparse Identification of Nonlinear Dynamical systems (SINDy) method introduced
in Brunton et al. (2016a).
It also includes other methods from related literature.

*System identification* refers to the process of using measurement data to infer the governing dynamics.
Once discovered, these equations can make predictions about future states, can inform control inputs, or can enable the theoretical study using analytical techniques.
The resulting models are inherently *interpretable* and *generalizable*.


First Steps
------------------

Installation
^^^^^^^^^^^^^^

The preferred way to install is with pip or conda e.g. ``pip install pysindy``.
You may have to add the ``--user`` option.
Pysindy also provides several extras, e.g. ``pip install pysindy[miosr]``:

cvxpy
  Convex optimizer ``SR3`` and subclasses

miosr
  Branch-and-bound optimizer for L0-constraint, ``MIOSR``

sbr
  Bayesian regression optimizer yielding posteriors, ``SBR``.

Example
^^^^^^^^^^^
Suppose we have measurements of the position of a particle obeying the following dynamical system at different points in time

.. math::

  x' &= -2 x \\
  y' &= y

Note that this system of differential equations decouples into two differential equations whose solutions are simply

.. math::

  x(t) &= x_0 * exp(-2 * t) \\
  y(t) &= y_0 * exp(t)

This example uses the initial conditions ``x_0 = 3`` and ``y_0 = 0.5``.
It then fits and prints the discovered model

.. code-block:: python

  import numpy as np
  import pysindy as ps

  t = np.linspace(0, 1, 100)
  x = 3 * np.exp(-2 * t)
  y = 0.5 * np.exp(t)
  X = np.stack((x, y), axis=-1)  # First column is x, second is y

  model = ps.SINDy()
  model.fit(X, t=t, feature_names=["x", "y"])
  model.print()

which correctly results in

.. code-block:: text

  x' = -2.000 x
  y' = 1.000 y

PySINDy provides numerous other features not shown here. We have a variety of tutorials and examples, starting with
`generating data and fitting models <examples/tutorial_1/example.html>`_.


Getting Help
-------------

* If you have a **question** or find a **bug**, please open an `issue <https://github.com/dynamicslab/pysindy/issues>`_ on github.
* The **documentation** site for PySINDy can be found `here <https://pysindy.readthedocs.io/en/latest/>`__.
  A video overview of PySINDy can be found on `Youtube <https://www.youtube.com/watch?v=DvbbXX8Bd90>`__.
  We have also created a `video playlist <https://www.youtube.com/playlist?list=PLN90bHJU-JLoOfEk0KyBs2qLTV7OkMZ25>`__ with practical PySINDy tips.
* To understand more about the **types of objects** in pysindy, see the `object model <https://pysindy.readthedocs.io/en/latest/objects>`_.
* If you want to fix a problem, add a feature, or share an example, check the `**Contributor** Guide <https://pysindy.readthedocs.io/en/latest/contributing.html>`_.
* If you are using pysindy in **academic** work, please see `Academic Use <https://pysindy.readthedocs.io/en/latest/academic.html>`_ for recommendations, including **citations**.


Related packages
----------------
* `Deeptime <https://github.com/deeptime-ml/deeptime>`_ - A Python library for the analysis of time series data with methods for dimension reduction, clustering, and Markov model estimation.
* `PyDMD <https://github.com/mathLab/PyDMD/>`_ - A Python package using the Dynamic Mode Decomposition (DMD) for a data-driven model simplification based on spatiotemporal coherent structures. DMD is a great alternative to SINDy.
* `PySINDyGUI <https://github.com/hyumo/pysindy-gui>`_ - A slick-looking GUI for PySINDy.
* `SEED <https://github.com/M-Vause/SEED2.0>`_ - Software for the Extraction of Equations from Data: a GUI for many of the methods provided by PySINDy.
* `SymINDy <https://github.com/andreikitaitsev/SymINDy/>`_ - A Python package combining SINDy with genetic programming-based symbolic regression, used for the functions library optimization.
* `derivative <https://derivative.readthedocs.io/en/latest/>`_ - A Python package for differentiating (and optionally, smoothing) data, used by pysindy


.. |BuildCI| image:: https://github.com/dynamicslab/pysindy/actions/workflows/main.yml/badge.svg
    :target: https://github.com/dynamicslab/pysindy/actions/workflows/main.yml?query=branch%3Amain

.. |RTD| image:: https://readthedocs.org/projects/pysindy/badge/?version=latest
    :target: https://pysindy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |PyPI| image:: https://badge.fury.io/py/pysindy.svg
    :target: https://badge.fury.io/py/pysindy

.. |Codecov| image:: https://codecov.io/gh/dynamicslab/pysindy/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/dynamicslab/pysindy

.. |JOSS1| image:: https://joss.theoj.org/papers/82d080bbe10ac3ab4bc03fa75f07d644/status.svg
    :target: https://joss.theoj.org/papers/82d080bbe10ac3ab4bc03fa75f07d644

.. |JOSS2| image:: https://joss.theoj.org/papers/10.21105/joss.03994/status.svg
    :target: https://doi.org/10.21105/joss.03994

.. |DOI| image:: https://zenodo.org/badge/186055899.svg
   :target: https://zenodo.org/badge/latestdoi/186055899
