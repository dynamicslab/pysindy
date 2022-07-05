---
title: 'PySINDy: A comprehensive Python package for robust sparse system identification'
tags:
  - Python
  - dynamical systems
  - sparse regression
  - model discovery
  - system identification
  - machine learning
authors:
  - name: Alan A. Kaptanoglu
    affiliation: 1
  - name: Brian M. de Silva
    affiliation: 2
  - name: Urban Fasel
    affiliation: 3
  - name: Kadierdan Kaheman
    affiliation: 3
  - name: Andy J. Goldschmidt
    affiliation: 1
  - name: Jared Callaham
    affiliation: 3
  - name: Charles B. Delahunt
    affiliation: 2
  - name: Zachary G. Nicolaou
    affiliation: 2
  - name: Kathleen Champion
    affiliation: 2
  - name: Jean-Christophe Loiseau
    affiliation: 4
  - name: J. Nathan Kutz
    affiliation: 2
  - name: Steven L. Brunton
    affiliation: 3
affiliations:
 - name: Department of Physics, University of Washington
   index: 1
 - name: Department of Applied Mathematics, University of Washington
   index: 2
 - name: Department of Mechanical Engineering, University of Washington
   index: 3
 - name: Arts et Métiers Institute of Technology, CNAM, DynFluid, HESAM Université
   index: 4
date: 21 October 2021
output: bookdown::html_document2
bibliography: paper.bib
---

# Summary
Automated data-driven modeling, the process of directly discovering the governing equations of a system from data, is increasingly being used across the scientific community. `PySINDy` is a Python package that provides tools for applying the sparse identification of nonlinear dynamics (SINDy) approach to data-driven model discovery. In this major update to `PySINDy`, we implement several advanced features that enable the discovery of more general differential equations from noisy and limited data. The library of candidate terms is extended for the identification of actuated systems, partial differential equations (PDEs), and implicit differential equations. Robust formulations, including the integral form of SINDy and ensembling techniques, are also implemented to improve performance for real-world data. Finally, we provide a range of new optimization algorithms, including several sparse regression techniques and algorithms to enforce and promote inequality constraints and stability. Together, these updates enable entirely new SINDy model discovery capabilities that have not been reported in the literature, such as constrained PDE identification and ensembling with different sparse regression optimizers.

# Statement of need
Traditionally, the governing laws and equations of nature have been derived from first principles and based on rigorous experimentation and expert intuition. 
In the modern era, cheap and efficient sensors have resulted in an unprecedented growth in the availability of measurement data, opening up the opportunity to perform automated model discovery using data-driven modeling. These data-driven approaches are also increasingly useful for processing and interpreting the information in these large datasets.
A number of such approaches have been developed in recent years, including the dynamic mode decomposition [@schmid2010dynamic;@Kutz2016book], Koopman theory [@Brunton2021koopman], nonlinear autoregressive algorithms [@Billings2013book], neural networks [@pathak2018model;@vlachas2018data;@Raissi2019jcp], Gaussian process regression [@raissi2017machine], operator inference and reduced-order modeling [@Benner2015siamreview;@peherstorfer2016data;@qian2020lift], genetic programming [@Bongard2007pnas;@schmidt_distilling_2009], and sparse regression [@brunton2016pnas].
These approaches have seen many variants and improvements over the years, so data-driven modeling software must be regularly updated to remain useful to the scientific community. The SINDy approach has experienced particularly rapid development, motivating this major update to aggregate these innovations into a single open-source tool that is transparent and easy to use for non-experts or scientists from other fields.

The original `PySINDy` code [@de2020pysindy] provided an implementation of the traditional SINDy method [@brunton2016pnas], which 
assumes that the dynamical evolution of a state variable $\mathbf{q}(t)\in\mathbb{R}^n$ follows an ODE described by a function $\mathbf{f}$,
\begin{equation}\label{eq:sindy_eq}
   \frac{d}{dt} \mathbf{q} = \mathbf{f}(\mathbf{q}).
\end{equation}
SINDy approximates the dynamical system $\mathbf{f}$ in Eq. \eqref{eq:sindy_eq} as a sparse combination of terms from a library of candidate basis functions $\boldsymbol{\theta}(\mathbf{q}) = [\theta_1(\mathbf{q}),\theta_2(\mathbf{q}),\dots,\theta_p(\mathbf{q})]$ 
\begin{equation}\label{eq:sindy_expansion}
\mathbf{f}(\mathbf{q})\approx \sum_{k=1}^{p}\theta_k(\mathbf{q})\boldsymbol\xi_k, \quad \text{or equivalently} \quad \frac{d}{dt}\mathbf{q} \approx \mathbf{\Theta}(\mathbf{q})\mathbf{\Xi},
\end{equation}
where $\boldsymbol{\Xi} = [\boldsymbol\xi_1,\boldsymbol\xi_2,\dots,\boldsymbol\xi_p]$ contain the sparse coefficients. In order for this strategy to be successful, a reasonably accurate approximation of $\mathbf{f}(\mathbf{q})$ should exist as a sparse expansion in the span of $\boldsymbol{\theta}$. Therefore, background scientific knowledge about expected terms in $\mathbf{f}(\mathbf{q})$ can be used to choose the library $\boldsymbol{\theta}$. 
To pose SINDy as a regression problem, we assume we have a set of state measurements sampled at time steps $t_1, ..., t_m$ and rearrange the data into the data matrix $\mathbf{Q} \in \mathbb{R}^{m\times n}$, \begin{eqnarray}\label{eq:Q_matrix}
\mathbf{Q} = \begin{bmatrix}
q_1(t_1) & q_2(t_1) & \cdots & q_n(t_1)\\
q_1(t_2) & q_2(t_2) & \cdots & q_n(t_2)\\
\vdots & \vdots & \ddots & \vdots \\
q_1(t_m) & q_2(t_m) & \cdots & q_n(t_m)
\end{bmatrix}
\label{Eq:DataMatrix}.
\end{eqnarray}
A matrix of derivatives in time, $\mathbf Q_t$, is defined similarly and can be numerically computed from $\mathbf{Q}$. PySINDy defaults to second order finite differences for computing derivatives, although a host of more sophisticated methods are now available, including arbitrary order finite differences, Savitzky-Golay derivatives (i.e. polynomial-filtered derivatives), spectral derivatives with optional filters, arbitrary order spline derivatives, and total variational derivatives [@ahnert2007numerical;@chartrand2011numerical;@tibshirani2011solution].

After $\mathbf Q_t$ is obtained, Eq. \eqref{eq:sindy_expansion} becomes $\mathbf Q_t \approx \mathbf{\Theta}(\mathbf{Q})\mathbf{\Xi}$ and the goal of the SINDy sparse regression problem is to choose a sparse set of coefficients $\mathbf{\Xi}$ that accurately fits the measured data in $\mathbf Q_t$. We can promote sparsity in the identified coefficients via a sparse regularizer $R(\mathbf{\Xi})$, such as the $l_0$ or $l_1$ norm, and use a sparse regression algorithm such as SR3 [@champion2020unified] to solve the resulting optimization problem,
\begin{equation}\label{eq:sindy_regression}
  \text{argmin}_{\boldsymbol\Xi}\|\mathbf Q_t - \boldsymbol\Theta(\mathbf{Q}) \boldsymbol\Xi\|^2 + R(\boldsymbol\Xi).
\end{equation}

The original `PySINDy` package was developed to identify a particular class of systems described by Eq. \eqref{eq:sindy_eq}.
Recent variants of the SINDy method are available that address systems with control inputs and model predictive control (MPC) [@Kaiser2018prsa;@fasel2021sindy], systems with physical constraints [@Loiseau2017jfm;@kaptanoglu2020physics], implicit ODEs [@mangan2016inferring;@kaheman2020sindy], PDEs [@Rudy2017sciadv;@Schaeffer2017prsa], and weak form ODEs and PDEs [@Schaeffer2017pre;@Reinbold2020pre;@messenger2021weakpde]. Other methods, such as ensembling and sub-sampling [@maddu2019stability;@reinbold2021robust;@delahunt2021toolkit], are often vital for making the identification of Eq. \eqref{eq:sindy_eq} more robust. 
In order to incorporate these new developments and accommodate the wide variety of possible dynamical systems, we have extended `PySINDy` to a more general setting and added significant new functionality. Our code\footnote{\url{https://github.com/dynamicslab/pysindy}} is thoroughly documented, contains extensive examples, and integrates a wide range of functionality, some of which may be found in a number of other local SINDy implementations\footnote{\url{https://github.com/snagcliffs/PDE-FIND}, \url{https://github.com/eurika-kaiser/SINDY-MPC},\\ \url{https://github.com/dynamicslab/SINDy-PI}, \url{https://github.com/SchatzLabGT/SymbolicRegression},\\ \url{https://github.com/dynamicslab/databook_python}, \url{https://github.com/sheadan/SINDy-BVP},\\ \url{https://github.com/sethhirsh/BayesianSindy}, \url{https://github.com/racdale/sindyr},\\ \url{https://github.com/SciML/DataDrivenDiffEq.jl}, \url{https://github.com/MathBioCU/WSINDy_PDE},\\ \url{https://github.com/pakreinbold/PDE_Discovery_Weak_Formulation}, \url{https://github.com/ZIB-IOL/CINDy}}. In contrast to some of these existing codes, `PySINDy` is completely open-source, professionally-maintained (for instance, providing unit tests and adhering to PEP8 stylistic standards), and minimally dependent on non-standard Python packages.

# New features
Given spatiotemporal data $\mathbf{Q}(\mathbf{x}, t) \in \mathbb{R}^{m\times n}$, and optional control inputs $\mathbf{u} \in \mathbb{R}^{m \times r}$ (note $m$ has been redefined here to be the product of the number of spatial measurements and the number of time samples), `PySINDy` can now approximate algebraic systems of PDEs (and corresponding weak forms) in an arbitrary number of spatial dimensions. Assuming the system is described by a function $\mathbf{g}$, we have
\begin{equation}\label{eq:pysindy_eq}
    \mathbf{g}(\mathbf{q},\mathbf q_t, \mathbf q_x, \mathbf q_y, \mathbf q_{xx}, ..., \mathbf{u}) = 0.
\end{equation}
ODEs, implicit ODEs, PDEs, and other dynamical systems are subsets of Eq. \eqref{eq:pysindy_eq}. We can accommodate control terms and partial derivatives in the SINDy library by adding them as columns in $\mathbf{\Theta}(\mathbf{Q})$, which becomes $\mathbf{\Theta}(\mathbf{Q}, \mathbf Q_t, \mathbf Q_x, ..., \mathbf{u})$. 

In addition, we have extended `PySINDy` to handle more complex modeling scenarios, including trapping SINDy for provably stable ODE models for fluids [@kaptanoglu2021promoting], models trained using multiple dynamic trajectories, and the generation of many models with sub-sampling and ensembling methods [@fasel2021ensemble] for cross-validation and probabilistic system identification. In order to solve Eq. \eqref{eq:pysindy_eq}, `PySINDy` implements several different sparse regression algorithms. Greedy sparse regression algorithms, including step-wise sparse regression (SSR) [@boninsegna2018sparse] and forward regression orthogonal least squares (FROLS) [@Billings2013book], are now available. For maximally versatile candidate libraries, the new `GeneralizedLibrary` class allows for tensoring, concatenating, and otherwise combining many different candidate libraries, along with optionally specifying a subset of the inputs to use for generating each of the libraries. \autoref{fig:package-structure} illustrates the `PySINDy` code structure, changes, and high-level goals for future work, and [`YouTube` tutorials](https://www.youtube.com/playlist?list=PLN90bHJU-JLoOfEk0KyBs2qLTV7OkMZ25) for this new functionality are available online.

`PySINDy` includes extensive Jupyter notebook tutorials that demonstrate the usage of various features of the package and reproduce nearly the entirety of the examples from the original SINDy paper [@brunton2016pnas], trapping SINDy paper [@kaptanoglu2021promoting], and the PDE-FIND paper [@Rudy2017sciadv]. 
We include an extended example for the quasiperiodic shear-driven cavity flow [@callaham2021role].
As a simple illustration of the new functionality, we demonstrate how SINDy can be used to identify the Kuramoto-Sivashinsky (KS) PDE from data. We train the model on the first 60\% of the data from Rudy et al. [@Rudy2017sciadv], which in total contains 1024 spatial grid points and 251 time steps. The KS model is identified correctly and the prediction for $\dot{\mathbf{q}}$ on the remaining testing data indicates strong performance in \autoref{fig:pde_id}. Lastly, we provide a useful flow chart in \autoref{fig:flow_chart} so that users can make informed choices about which advanced methods are suitable for their datasets. 

# Conclusion
The goal of the `PySINDy` package is to enable anyone with access to measurement data to engage in scientific model discovery. The package is designed to be accessible to inexperienced users, adhere to `scikit-learn` standards, include most of the existing SINDy variations in the literature, and provide a large variety of functionality for more advanced users. We hope that researchers will use and contribute to the code in the future, pushing the boundaries of what is possible in system identification.

# Acknowledgments
`PySINDy` is a fork of [`sparsereg`](https://github.com/Ohjeah/sparsereg) [@markus_quade_sparsereg].
SLB, AAK, KK, and UF acknowledge support from the Army Research Office (ARO  W911NF-19-1-0045). JLC acknowledges support from funding support from the Department of Defense (DoD) through the National Defense Science \& Engineering Graduate (NDSEG) Fellowship Program. ZGN is a Washington Research Foundation Postdoctoral Fellow.

![Summary of SINDy features organized by (a) `PySINDy` structure and (b) functionality. (a) Hierarchy from the sparse regression problem solved by SINDy, to the submodules of `PySINDy`, to the individual optimizers, libraries, and differentiation methods implemented in the code.
(b) Flow chart for organizing the SINDy variants and functionality in the literature. Bright color boxes indicate the features that have been implemented through this work, roughly organized by functionality. Semi-transparent boxes indicate features that have not yet been implemented.\label{fig:package-structure}](Fig1.png)

![`PySINDy` can now be used for PDE identification; we illustrate this new capability by accurately capturing a set of testing data from the Kuramoto-Sivashinsky system, described by $q_t = -qq_x - q_{xx} - q_{xxxx}$. The identified model is $q_t = -0.98qq_x -0.99q_{xx} - 1.0q_{xxxx}$.\label{fig:pde_id}](Fig2.png)

![This flow chart summarizes how `PySINDy` users can start with a dataset and systematically choose the proper candidate library and sparse regression optimizer that are tailored for a specific scientific task. \label{fig:flow_chart}](Fig3.png)

# References
