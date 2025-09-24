Tutorials
=============================
These are introductory guides to get familiar with how to use pysindy in your project.
Tutorials in the pysindy repo are also available as `Jupyter notebooks <https://github.com/dynamicslab/pysindy/tree/master/examples>`_.


`Getting Started fitting a model <./tutorial_1/example.ipynb>`_
---------------------------------------------------------------------------------------------------------------------
We recommend that people new to SINDy start here.
We show how to use PySINDy to learn a model for a simple linear differential equation.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./tutorial_1/example

`Evaluating a model fit <./tutorial_2/example.ipynb>`_
---------------------------------------------------------------------------------------------------------------------

It can be tricky to evaluate how well a model is fit.
This tutorial shows different ways to evaluate and visualize a model,
depending on the use case.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./tutorial_2/example


`Feature overview <./1_feature_overview/example.ipynb>`_
-----------------------------------------------------------------------------------------------------------
This notebook gives an almost exhaustive overview of the different features available in PySINDy. It's a good reference for how to set various options and work with different types of datasets.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./1_feature_overview/example


How-To Guides
==========================================
The following examples are both more specific and more advanced.
Some are copied from another repository that contains dependency information and
potentially a greater description.



`Scikit-learn compatibility <./4_scikit_learn_compatibility/example.ipynb>`_
-------------------------------------------------------------------------------------------------------------------------------
Shows how PySINDy interfaces with various Scikit-learn objects.

* Cross-validation
* Sparse regressors

.. toctree::
    :hidden:
    :maxdepth: 1

    ./4_scikit_learn_compatibility/example

`Differentiation <./5_differentiation/example.ipynb>`_
---------------------------------------------------------------------------------------------------------
Explore the differentiation methods available in PySINDy on pure differentiation problems and as components in the SINDy algorithm.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./5_differentiation/example


`Deeptime compatibility <./6_deeptime_compatibility/example.ipynb>`_
------------------------------------------------------------------------------------------------------------------------
See a demonstration of PySINDy objects designed to conform to the `Deeptime <https://deeptime-ml.github.io/latest/index.html>`_ API.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./6_deeptime_compatibility/example

`SINDyPI <./9_sindypi_with_sympy/example.ipynb>`_
----------------------------------------------------------------------------------------------
This notebook applies the ``SINDyPI`` optimizer to a simple implicit ODE and was originally proposed in this paper: Kaheman, Kadierdan, J. Nathan Kutz, and Steven L. Brunton. "SINDy-PI: a robust algorithm for parallel implicit sparse identification of nonlinear dynamics." Proceedings of the Royal Society A 476.2242 (2020): 20200279.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./9_sindypi_with_sympy/example

`PDEFIND <./10_PDEFIND_examples/example.ipynb>`_
----------------------------------------------------------------------------------------------
This notebook applies the PDEFIND algorithm (SINDy for PDE identification) to a number of PDEs, and was originally proposed in this paper: Rudy, Samuel H., et al. "Data-driven discovery of partial differential equations." Science Advances 3.4 (2017): e1602614.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./10_PDEFIND_examples/example

`Greedy Algorithms <./11_SSR_FROLS_examples/example.ipynb>`_
-----------------------------------------------------------------------------------------------------
This notebook uses the step-wise sparse regression (SSR) and forward-regression orthogonal least-squares (FROLS) algorithms, which are greedy algorithms that iteratively truncate (or add) one nonzero coefficient at each algorithm iteration.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./11_SSR_FROLS_examples/example

`Weak formulation SINDy <./12_weakform_SINDy_examples/example.ipynb>`_
--------------------------------------------------------------------------------------------------------------
This notebook uses SINDy to identify the weak-formulation of a system of ODEs or PDEs, adding significant robustness against noise in the data.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./12_weakform_SINDy_examples/example

`Model ensembles <./13_ensembling/example.ipynb>`_
----------------------------------------------------------------------------------------------
This notebook uses sub-sampling of the data and sub-sampling of the SINDy library to generate many models, and the user can choose how to average or otherwise combine these models together. This tends to make SINDy more robust against noisy data.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./13_ensembling/example

`PySINDy lectures <./15_pysindy_lectures/example.ipynb>`_
----------------------------------------------------------------------------------------------
Lectures on PySINDy usage from Alan Kaptanoglu's Youtube channel: https://www.youtube.com/playlist?list=PLN90bHJU-JLoOfEk0KyBs2qLTV7OkMZ25.

.. toctree::
    :hidden:
    :maxdepth: 1

    ./15_pysindy_lectures/example

.. pysindy-example::
    :key: sindy_cp
    :title: Discover parameters for stable pattern formation using SINDy-CP.

    While parameters can be considered as constant control signals for spatiotemporal
    dynamics, combining the PDEFIND and SINDYc approaches to discover parameterized
    spatiotemporal dynamics poses a significant challenge. Here, we develop the
    system identification for nonlinear dynamics with control parameters (SINDyCP)
    approach to discover such parameterized dynamics.

.. pysindy-example::
    :key: stable_linear_sr3
    :title: Discovering stable linear models: ``StabilizedLinearSR3``

    While trapping SINDy handles a more complex type of stability (Lyapunov) for
    nonlinear dynamics, the linear case is simpler.  Here are several demonstrations
    of the ``StabilizedLinearSR3`` optimizer for POD and state-space models.

.. pysindy-example::
    :key: sbr
    :title: Post-analysis of Bayesian SINDy

    Demonstrates using Arviz to analyze the results of Bayesian SINDy.

External Research Examples
============================
These examples demonstrate very specific applications of pysindy, being used to
run experiments for research papers.
They are copied from repositories that contain dependency information and
potentially a greater description.  Most are meant to be run on a specific version
of pysindy and are not necessarily kept up to date with major version bumps.
They may require substantial computing resources.

.. pysindy-example::
    :key: original
    :title: Original Paper

    This repository recreates the results from the `original SINDy paper <https://www.pnas.org/content/pnas/113/15/3932.full.pdf>`_.
    It applies SINDy to the following problems:

    * Linear 2D ODE
    * Cubic 2D ODE
    * Linear 3D ODE
    * Lorenz system
    * Fluid wake behind a cylinder
    * Logistic map
    * Hopf system


.. pysindy-example::
    :key: plasma
    :title: Plasma physics

    Use the ``ConstrainedSR3`` optimizer to build a constrained model for the temporal
    POD modes of a plasma simulation.

.. pysindy-example::
    :key: trapping
    :title: Discovering stable quadratic models: ``TrappingSR3``

    The Trapping SINDy notebook applies the ``TrappingSR3`` optimizer to various
    canonical fluid systems, proposed in this paper:
    Kaptanoglu, Alan A., et al. "Promoting global stability in data-driven models of
    quadratic nonlinear dynamics." Physical Review Fluids 6.9 (2021): 094401.
    A preprint is found here `<https://arxiv.org/abs/2105.01843>`_.
    Other examples in this folder explore the extension to local trapping and enstrophy
    in Peng, Mai et al. "Local stability guarantees for data-driven quadratically
    nonlinear models."

.. pysindy-example::
    :key: cavity-flow
    :title: Cavity flow

    Demonstrates the use of SINDy to learn a model for the quasiperiodic dynamics in a
    shear-driven cavity at Re=7500, following Callaham, Brunton, and Loiseau (2021),
    preprint available here `<https://arxiv.org/pdf/2106.02409>`_.

.. pysindy-example::
    :key: benchmarks
    :title: Noise robustness benchmarking

    Benchmarking PySINDy on low dimensional chaotic systems subject to noise, following
    Kaptanoglu, Zhang, Nicolaou, Fasel, & Brunton, (2023),
    `preprint available <https://arxiv.org/abs/2302.10787>`_.
