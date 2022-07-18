PySINDy Examples
================

This directory showcases the following examples of PySINDy in action.

`Feature overview <https://pysindy.readthedocs.io/en/latest/examples/1_feature_overview.html>`_
-----------------------------------------------------------------------------------------------------------
This notebook gives an almost exhaustive overview of the different features available in PySINDy. It's a good reference for how to set various options and work with different types of datasets.

`Introduction to SINDy <./2_introduction_to_sindy/example.ipynb>`_
---------------------------------------------------------------------------------------------------------------------
We recommend that people new to SINDy start here. We give a gentle introduction to the SINDy method and how different steps in the algorithm are represented in PySINDy. We also show how to use PySINDy to learn a model for a simple linear differential equation.

`Original paper <https://pysindy.readthedocs.io/en/latest/examples/3_original_paper.html>`_
-------------------------------------------------------------------------------------------------------
This notebook uses PySINDy to reproduce the examples in the `original SINDy paper <https://www.pnas.org/content/pnas/113/15/3932.full.pdf>`_. Namely, it applies PySINDy to the following problems:

* Linear 2D ODE
* Cubic 2D ODE
* Linear 3D ODE
* Lorenz system
* Fluid wake behind a cylinder
* Logistic map
* Hopf system

`Scikit-learn compatibility <https://pysindy.readthedocs.io/en/latest/examples/4_scikit_learn_compatibility.html>`_
-------------------------------------------------------------------------------------------------------------------------------
Shows how PySINDy interfaces with various Scikit-learn objects.

* Cross-validation
* Sparse regressors

`Differentiation <./5_differentation/example.ipynb>`_
---------------------------------------------------------------------------------------------------------
Explore the differentiation methods available in PySINDy on pure differentiation problems and as components in the SINDy algorithm.

`Deeptime compatibility <https://pysindy.readthedocs.io/en/latest/examples/6_deeptime_compatibility.html>`_
------------------------------------------------------------------------------------------------------------------------
See a demonstration of PySINDy objects designed to conform to the `Deeptime <https://deeptime-ml.github.io/latest/index.html>`_ API.

`Plasma physics <https://pysindy.readthedocs.io/en/latest/examples/7_plasma_example.html>`_
----------------------------------------------------------------------------------------------
Use the ``ConstrainedSR3`` optimizer to build a constrained model for the temporal POD modes of a plasma simulation.


`Trapping SINDy <https://pysindy.readthedocs.io/en/latest/examples/8_trapping_sindy_paper_examples.html>`_
-----------------------------------------------------------------------------------------------------------
This notebook applies the ``TrappingSR3`` optimizer to various canonical fluid systems., proposed in this paper: Kaptanoglu, Alan A., et al. "Promoting global stability in data-driven models of quadratic nonlinear dynamics." Physical Review Fluids 6.9 (2021): 094401. A preprint is found here `<https://arxiv.org/abs/2105.01843>`_.

`SINDyPI <https://pysindy.readthedocs.io/en/latest/examples/9_sindypi_with_sympy.html>`_
----------------------------------------------------------------------------------------------
This notebook applies the ``SINDyPI`` optimizer to a simple implicit ODE and was originally proposed in this paper: Kaheman, Kadierdan, J. Nathan Kutz, and Steven L. Brunton. "SINDy-PI: a robust algorithm for parallel implicit sparse identification of nonlinear dynamics." Proceedings of the Royal Society A 476.2242 (2020): 20200279.

`PDEFIND <https://pysindy.readthedocs.io/en/latest/examples/10_PDEFIND_examples.html>`_
----------------------------------------------------------------------------------------------
This notebook applies the PDEFIND algorithm (SINDy for PDE identification) to a number of PDEs, and was originally proposed in this paper: Rudy, Samuel H., et al. "Data-driven discovery of partial differential equations." Science Advances 3.4 (2017): e1602614.

`Greedy Algorithms <https://pysindy.readthedocs.io/en/latest/examples/11_SSR_FROLS_examples.html>`_
-----------------------------------------------------------------------------------------------------
This notebook uses the step-wise sparse regression (SSR) and forward-regression orthogonal least-squares (FROLS) algorithms, which are greedy algorithms that iteratively truncate (or add) one nonzero coefficient at each algorithm iteration.

`Weak formulation SINDy <https://pysindy.readthedocs.io/en/latest/examples/12_weakform_SINDy_examples.html>`_
--------------------------------------------------------------------------------------------------------------
This notebook uses SINDy to identify the weak-formulation of a system of ODEs or PDEs, adding significant robustness against noise in the data.

`Model ensembles <https://pysindy.readthedocs.io/en/latest/examples/13_ensembling.html>`_
----------------------------------------------------------------------------------------------
This notebook uses sub-sampling of the data and sub-sampling of the SINDy library to generate many models, and the user can choose how to average or otherwise combine these models together. This tends to make SINDy more robust against noisy data.

`Cavity flow <https://pysindy.readthedocs.io/en/latest/examples/14_cavity_flow.html>`_
----------------------------------------------------------------------------------------------
Demonstrates the use of SINDy to learn a model for the quasiperiodic dynamics in a shear-driven cavity at Re=7500, following Callaham, Brunton, and Loiseau (2021), preprint available here `<https://arxiv.org/pdf/2106.02409>`_.


Full table of contents
----------------------
.. toctree::
    :glob:
    :hidden:

    **
