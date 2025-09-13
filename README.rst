PySINDy
=========

|BuildCI| |RTD| |PyPI| |Codecov| |JOSS1| |JOSS2| |DOI|

**PySINDy** is a sparse regression package with several implementations for the Sparse Identification of Nonlinear Dynamical systems (SINDy) method introduced in Brunton et al. (2016a), including the unified optimization approach of Champion et al. (2019), SINDy with control from Brunton et al. (2016b), Trapping SINDy from Kaptanoglu et al. (2021), SINDy-PI from Kaheman et al. (2020), PDE-FIND from Rudy et al. (2017), and so on. A comprehensive literature review is given in de Silva et al. (2020) and Kaptanoglu, de Silva et al. (2021).

.. contents:: Table of contents

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
For a more in-depth look at the mathematical foundations of SINDy, please see our `introduction to SINDy <https://pysindy.readthedocs.io/en/latest/examples/2_introduction_to_sindy/example.html>`__.

Relation to PySINDy
^^^^^^^^^^^^^^^^^^^
The PySINDy package revolves around the ``SINDy`` class which consists of three primary components; one for each term in the above matrix approximation problem.

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

PySINDy provides numerous other features not shown here. We recommend the `feature overview <https://pysindy.readthedocs.io/en/latest/examples/1_feature_overview/example.html>`__ section of the documentation for a more exhaustive summary of additional features.

Installation
------------

The preferred way to install is with pip or conda e.g. ``pip install pysindy``

You may have to add ``--user`` option to the above lines.
Pysindy also provides several extras:

cvxpy
  Convex optimizer ``SR3`` and subclasses

miosr
  Branch-and-bound optimizer for L0-constraint, ``MIOSR``

sbr
  Bayesian regression optimizer yielding posteriors, ``SBR``.


Documentation
-------------
The documentation site for PySINDy can be found `here <https://pysindy.readthedocs.io/en/latest/>`__. There are numerous `examples <https://pysindy.readthedocs.io/en/latest/examples/index.html>`_ of PySINDy in action to help you get started. Examples are also available as `Jupyter notebooks <https://github.com/dynamicslab/pysindy/tree/master/examples>`__. A video overview of PySINDy can be found on `Youtube <https://www.youtube.com/watch?v=DvbbXX8Bd90>`__. We have also created a `video playlist <https://www.youtube.com/playlist?list=PLN90bHJU-JLoOfEk0KyBs2qLTV7OkMZ25>`__ with practical PySINDy tips.

PySINDy implements a lot of advanced functionality that may be overwhelming for new users or folks who are unfamiliar with these methods. Below (see here if image does not render https://github.com/dynamicslab/pysindy/blob/master/docs/JOSS2/Fig3.png), we provide a helpful flowchart for figuring out which methods to use, given the characteristics of your dataset:

.. image:: https://github.com/dynamicslab/pysindy/blob/master/docs/JOSS2/Fig3.png

This flow chart summarizes how ``PySINDy`` users can start with a dataset and systematically choose the proper candidate library and sparse regression optimizer that are tailored for a specific scientific task. The ``GeneralizedLibrary`` class allows for tensoring, concatenating, and otherwise combining many different candidate libraries.

Community guidelines
--------------------

Contributing
^^^^^^^^^^^^^^^^^^^^^
See `Contributor guide <https://pysindy.readthedocs.io/en/latest/contributing.html>`_

Reporting issues or bugs
^^^^^^^^^^^^^^^^^^^^^^^^
If you find a bug in the code or want to request a new feature, please open an issue.

Getting help
^^^^^^^^^^^^
For help using PySINDy please consult the `documentation <https://pysindy.readthedocs.io/en/latest/>`__ and/or our `examples <https://github.com/dynamicslab/pysindy/tree/master/examples>`__, or create an issue.

Citing PySINDy
--------------
PySINDy has been published in the Journal of Open Source Software (JOSS). The paper can be found `here <https://joss.theoj.org/papers/10.21105/joss.02104>`__.

If you use PySINDy in your work, please cite it using the following two references:

Brian M. de Silva, Kathleen Champion, Markus Quade, Jean-Christophe Loiseau, J. Nathan Kutz, and Steven L. Brunton., (2020). *PySINDy: A Python package for the sparse identification of nonlinear dynamical systems from data.* Journal of Open Source Software, 5(49), 2104, https://doi.org/10.21105/joss.02104

Kaptanoglu et al., (2022). PySINDy: A comprehensive Python package for robust sparse system identification. Journal of Open Source Software, 7(69), 3994, https://doi.org/10.21105/joss.03994

Bibtex:

.. code-block:: text

    @article{desilva2020,
    doi = {10.21105/joss.02104},
    url = {https://doi.org/10.21105/joss.02104},
    year = {2020},
    publisher = {The Open Journal},
    volume = {5},
    number = {49},
    pages = {2104},
    author = {Brian de Silva and Kathleen Champion and Markus Quade and Jean-Christophe Loiseau and J. Kutz and Steven Brunton},
    title = {PySINDy: A Python package for the sparse identification of nonlinear dynamical systems from data},
    journal = {Journal of Open Source Software}
    }

Bibtex:

.. code-block:: text

    @article{Kaptanoglu2022,
    doi = {10.21105/joss.03994},
    url = {https://doi.org/10.21105/joss.03994},
    year = {2022},
    publisher = {The Open Journal},
    volume = {7},
    number = {69},
    pages = {3994},
    author = {Alan A. Kaptanoglu and Brian M. de Silva and Urban Fasel and Kadierdan Kaheman and Andy J. Goldschmidt and Jared Callaham and Charles B. Delahunt and Zachary G. Nicolaou and Kathleen Champion and Jean-Christophe Loiseau and J. Nathan Kutz and Steven L. Brunton},
    title = {PySINDy: A comprehensive Python package for robust sparse system identification},
    journal = {Journal of Open Source Software}
    }


References
----------------------
-  de Silva, Brian M., Kathleen Champion, Markus Quade,
   Jean-Christophe Loiseau, J. Nathan Kutz, and Steven L. Brunton.
   *PySINDy: a Python package for the sparse identification of
   nonlinear dynamics from data.* arXiv preprint arXiv:2004.08424 (2020)
   `[arXiv] <https://arxiv.org/abs/2004.08424>`__

-  Kaptanoglu, Alan A., Brian M. de Silva, Urban Fasel, Kadierdan Kaheman, Andy J. Goldschmidt
   Jared L. Callaham, Charles B. Delahunt, Zachary G. Nicolaou, Kathleen Champion,
   Jean-Christophe Loiseau, J. Nathan Kutz, and Steven L. Brunton.
   *PySINDy: A comprehensive Python package for robust sparse system identification.*
   arXiv preprint arXiv:2111.08481 (2021).
   `[arXiv] <https://arxiv.org/abs/2111.08481>`__

-  Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz.
   *Discovering governing equations from data by sparse identification
   of nonlinear dynamical systems.* Proceedings of the National
   Academy of Sciences 113.15 (2016): 3932-3937.
   `[DOI] <http://dx.doi.org/10.1073/pnas.1517384113>`__

-  Champion, K., Zheng, P., Aravkin, A. Y., Brunton, S. L., & Kutz, J. N. (2020).
   *A unified sparse optimization framework to learn parsimonious physics-informed
   models from data.* IEEE Access, 8, 169259-169271.
   `[DOI] <https://doi.org/10.1109/ACCESS.2020.3023625>`__

-  Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz.
   *Sparse identification of nonlinear dynamics with control (SINDYc).*
   IFAC-PapersOnLine 49.18 (2016): 710-715.
   `[DOI] <https://doi.org/10.1016/j.ifacol.2016.10.249>`__

-  Kaheman, K., Kutz, J. N., & Brunton, S. L. (2020).
   *SINDy-PI: a robust algorithm for parallel implicit sparse identification
   of nonlinear dynamics.* Proceedings of the Royal Society A, 476(2242), 20200279.
   `[DOI] <https://doi.org/10.1098/rspa.2020.0279>`__

-  Kaptanoglu, A. A., Callaham, J. L., Aravkin, A., Hansen, C. J., & Brunton, S. L. (2021).
   *Promoting global stability in data-driven models of quadratic nonlinear dynamics.*
   Physical Review Fluids, 6(9), 094401.
   `[DOI] <https://doi.org/10.1103/PhysRevFluids.6.094401>`__


Related packages
----------------
* `Deeptime <https://github.com/deeptime-ml/deeptime>`_ - A Python library for the analysis of time series data with methods for dimension reduction, clustering, and Markov model estimation.
* `PyDMD <https://github.com/mathLab/PyDMD/>`_ - A Python package using the Dynamic Mode Decomposition (DMD) for a data-driven model simplification based on spatiotemporal coherent structures. DMD is a great alternative to SINDy.
* `PySINDyGUI <https://github.com/hyumo/pysindy-gui>`_ - A slick-looking GUI for PySINDy.
* `SEED <https://github.com/M-Vause/SEED2.0>`_ - Software for the Extraction of Equations from Data: a GUI for many of the methods provided by PySINDy.
* `SymINDy <https://github.com/andreikitaitsev/SymINDy/>`_ - A Python package combining SINDy with genetic programming-based symbolic regression, used for the functions library optimization.

Contributors
------------
This repository is a fork from `original work <https://github.com/Ohjeah/sparsereg>`_ by `Markus Quade <https://github.com/Ohjeah>`_.

Thanks to the members of the community who have contributed to PySINDy!

+-------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `billtubbs <https://github.com/kopytjuk>`_            | Bug fix `#68 <https://github.com/dynamicslab/pysindy/issues/68>`_                                                                                          |
+-------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `kopytjuk <https://github.com/kopytjuk>`_             | Concatenation feature for libraries `#72 <https://github.com/dynamicslab/pysindy/pull/72>`_                                                                |
+-------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| `andgoldschmidt <https://github.com/andgoldschmidt>`_ | `derivative <https://derivative.readthedocs.io/en/latest/>`_ package for numerical differentiation `#85 <https://github.com/dynamicslab/pysindy/pull/85>`_ |
+-------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+

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
