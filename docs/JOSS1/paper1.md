---
title: 'PySINDy: A Python package for the sparse identification of nonlinear dynamical systems from data'
tags:
  - Python
  - dynamical systems
  - sparse regression
  - model discovery
  - system identification
  - machine learning
authors:
  - name: Brian M. de Silva
    affiliation: 1
  - name: Kathleen Champion
    affiliation: 1
  - name: Markus Quade
    affiliation: 2
  - name: Jean-Christophe Loiseau
    affiliation: 3
  - name: J. Nathan Kutz
    affiliation: 1
  - name: Steven L. Brunton
    affiliation: "4, 1"
affiliations:
 - name: Department of Applied Mathematics, University of Washington
   index: 1
 - name: Ambrosys GmbH
   index: 2
 - name: École Nationale Supérieure des Arts et Métiers
   index: 3
 - name: Department of Mechanical Engineering, University of Washington
   index: 4
date: 11 February 2020
bibliography: paper.bib
---

# Summary

Scientists have long quantified empirical observations by developing mathematical models that characterize the observations, have some measure of interpretability, and are capable of making predictions.
Dynamical systems models in particular have been widely used to study, explain, and predict system behavior in a wide range of application areas, with examples ranging from Newton's laws of classical mechanics to the Michaelis-Menten kinetics for modeling enzyme kinetics.
While governing laws and equations were traditionally derived by hand, the current growth of available measurement data and resulting emphasis on data-driven modeling motivates algorithmic approaches for model discovery.
A number of such approaches have been developed in recent years and have generated widespread interest, including Eureqa [@Schmidt81], sure independence screening and sparsifying operator [@PhysRevMaterials.2.083802], and the sparse identification of nonlinear dynamics (SINDy) [@brunton2016pnas].
Maximizing the impact of these model discovery methods requires tools to make them widely accessible to scientists across domains and at various levels of mathematical expertise.

`PySINDy` is a Python package for the discovery of governing dynamical systems models from data.
In particular, `PySINDy` provides tools for applying the SINDy approach to model discovery [@brunton2016pnas].
Given data in the form of state measurements $\mathbf{x}(t) \in \mathbb{R}^n$, the SINDy method seeks a function $\mathbf{f}$ such that
$$\frac{d}{dt}\mathbf{x}(t) = \mathbf{f}(\mathbf{x}(t)).$$
SINDy poses this model discovery as a sparse regression problem, wherein relevant terms in $\mathbf{f}$ are selected from a library of candidate functions.
Thus, SINDy models balance accuracy and efficiency, resulting in parsimonious models that avoid overfitting while remaining interpretable and generalizable.
This approach is straightforward to understand and can be readily customized using different sparse regression algorithms or library functions.

The `PySINDy` package is aimed at researchers and practitioners alike, enabling anyone with access to measurement data to engage in scientific model discovery.
The package is designed to be accessible to inexperienced practitioners, while also including options that allow more advanced users to customize it to their needs.
A number of popular SINDy variants are implemented, but `PySINDy` is also designed to enable further extensions for research and experimentation.
The package follows object-oriented design and is `scikit-learn` compatible.

The SINDy method has been widely applied for model identification in applications such as chemical reaction dynamics [@Hoffmann2018], nonlinear optics [@Sorokina2016oe], thermal fluids [@Loiseau2019data], plasma convection [@Dam2017pf], numerical algorithms [@Thaler2019jcp], and structural modeling [@lai2019sparse].
It has also been extended to handle  more complex modeling scenarios such as partial differential equations [@Schaeffer2017prsa;@Rudy2017sciadv], systems with inputs or control [@Kaiser2018prsa], corrupt or limited data [@tran2017exact;@schaeffer2018extracting], integral formulations [@Schaeffer2017pre;@Reinbold2020pre], physical constraints [@Loiseau2017jfm], tensor representations [@Gelss2019mindy], and stochastic systems [@boninsegna2018sparse].
However, there is not a definitive standard implementation or package for applying SINDy.
Versions of SINDy have been implemented within larger projects such as `sparsereg` [@markus_quade_sparsereg], but no specific implementation has emerged as the most widely adopted and most versions implement only a limited set of features.
Researchers have thus typically written their own implementations, resulting in duplicated effort and a lack of standardization.
This not only makes it more difficult to apply SINDy to scientific data sets, but also makes it more challenging to benchmark extensions to the method against the original and makes such extensions less accessible to end users.
The `PySINDy` package provides a dedicated central codebase where many of the basic SINDy features are implemented, allowing for easy use and standardization.
This also makes it straightforward for users to extend the package in a way such that new developments are available to a wider user base.


# Features
The core object in the `PySINDy` package is the `SINDy` model class, which is implemented as a `scikit-learn` estimator.
This design was chosen to make the package simple to use for a wide user base, as many potential users will be familiar with `scikit-learn`.
It also expresses the `SINDy` model object at the appropriate level of abstraction so that users can embed it into more complicated pipelines in `scikit-learn`, such as tools for parameter tuning and model selection.

Applying `SINDy` involves making several modeling decisions, namely: which numerical differentiation method is used, which functions make up the feature library, and which sparse regression algorithm is applied to learn the model.
The core `SINDy` object uses a set of default options but can be easily customized using a number of common approaches implemented in `PySINDy`.
The package provides a few standard options for numerical differentiation (finite difference and smoothed finite difference), feature libraries (polynomial and Fourier libraries, as well as a class for creating custom libraries), and sparse regression techniques (sequentially thresholded least squares [@brunton2016pnas], LASSO [@10.2307/2346178], and sparse relaxed regularized regression [@zheng2018ieee]).
Users can also create their own differentiation, sparse regression, or feature library objects for further customization.

The software package includes tutorials in the form of Jupyter notebooks.
These tutorials demonstrate the usage of various features in the package and reproduce the examples from the original SINDy paper [@brunton2016pnas].


# Acknowledgments

This project is a fork of [`sparsereg`](https://github.com/Ohjeah/sparsereg) [@markus_quade_sparsereg].
SLB acknowledges funding support from the Air Force Office of Scientific Research (AFOSR FA9550-18-1-0200) and the Army Research Office (ARO W911NF-19-1-0045).
JNK acknowledges support from the Air Force Office of Scientific Research (AFOSR FA9550-17-1-0329).
This material is based upon work supported by the National Science Foundation Graduate Research Fellowship under Grant Number DGE-1256082.

# References
