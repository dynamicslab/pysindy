An introduction to Sparse Identification of Nonlinear Dynamical systems (SINDy)
===============================================================================
We give a gentle introduction to the SINDy method and how different steps
in the algorithm are represented in PySINDy.

Main ideas
-------------
Suppose we have a set of measurements :math:`x(t)\in \mathbb{R}^n` from some physical system
at different points in time :math:`t`.
SINDy seeks to represent the time evolution of :math:`x(t)`` in terms of a nonlinear function
:math:`f`:

.. math::

    \frac{d}{dt}x(t) = f(x(t)).

This equation constitutes a *dynamical system* for the measurements :math:`x(t)`.
The vector :math:`x(t)=[x_1(t), x_2(t), \dots x_n(t)]^\top` gives the state
of the physical system at time :math:`t`.
The function :math:`f(x(t))` constrains how the system evolves in time.

The key idea behind SINDy is that the function :math:`f` is often **sparse**
in the space of an appropriate set of basis functions.
For example, the function

.. math::

    \frac{d}{dt}x = f(x)
    = \begin{bmatrix} f_1(x)\\f_2(x) \end{bmatrix}
    = \begin{bmatrix}1 - x_1 + 3x_1x_2 \\ x_2^2 - 5x_1^3 \end{bmatrix}

is sparse with respect to the set of polynomials of two variables
in the sense that if we were to write an expansion of the component functions
of :math:`f` in this basis
(e.g. :math:`f_{1}(x) = \sum_{i=0}^\infty\sum_{j=0}^\infty a_{i,j}x_1^ix_2^j`),
only a small number of coefficients (:math:`a_{i,j}`) would be nonzero.

SINDy employs **sparse regression** to find a linear combination of basis functions
that best capture the dynamic behavior of the physical system.

Approximation problem
----------------------------
To apply SINDy in practice one needs a set of measurement data collected at times
:math:`t_1, t_2, \dots, t_n`,
and the time derivatives of these measurements
(either measured directly or numerically approximated).
These data are aggregated into the matrices :math:`X` and :math:`\dot X`, respectively:

.. math::

    X = \begin{bmatrix}
        x_1(t_1) & x_2(t_1) & \dots & x_n(t_1) \\
        x_1(t_2) & x_2(t_2) & \dots & x_n(t_2) \\
        \vdots & \vdots & & \vdots \\ x_1(t_m) & x_2(t_m) & \dots & x_n(t_m)
    \end{bmatrix},
    \qquad
    \dot{X} = \begin{bmatrix} \dot{x_1}(t_1) & \dot{x_2}(t_1) & \dots & \dot{x_n}(t_1) \\
        \dot{x_1}(t_2) & \dot{x_2}(t_2) & \dots & \dot{x_n}(t_2) \\
        \vdots & \vdots & & \vdots \\
        \dot{x_1}(t_m) & \dot{x_2}(t_m) & \dots & \dot{x_n}(t_m)
    \end{bmatrix}.

Next, one forms a library matrix :math:`\Theta(X)`
whose columns consist of a chosen set of basis functions applied to the data

.. math::

    \Theta(X) = \begin{bmatrix}
        \mid & \mid & & \mid \\
        \theta_1(X) & \theta_2(X) & \dots & \theta_\ell(X) \\
        \mid & \mid & & \mid
    \end{bmatrix}.

For example, if :math:`\theta_1(x), \theta_2(x), \dots, \theta_\ell(x)`
are monomials (:math:`\theta_i(x) = x^{i-1}`), then

.. math::

    \theta_3(X) = \begin{bmatrix}
        \mid & \mid & & \mid & \mid & & \mid \\
        x_1(t)^2 & x_1(t)x_2(t) & \dots & x_2(t)^2 & x_2(t)x_3(t) & \dots & x_n^2(t) \\
        \mid & \mid & & \mid & \mid & & \mid
    \end{bmatrix},

where vector products and powers are understood to be element-wise.

We seek a set of sparse coefficient vectors (collected into a matrix)

.. math::

    \Xi = \begin{bmatrix}
        \mid & \mid & & \mid \\
        \xi_1 & \xi_2 & \dots & \xi_n \\
        \mid & \mid & & \mid
    \end{bmatrix}.

The vector :math:`\xi_i` provides the coefficients for a linear combination
of basis functions :math:`\theta_1(x), \theta_2(x), \dots, \theta_\ell(x)`
representing the :math:`i`\ th component function of :math:`f`: :math:`f_i(x)`.
That is to say, :math:`f_i(x) = \Theta\left(x^\top\right) \xi_i`,
where :math:`\Theta\left(x^\top\right)` is understood to be a row vector
consisting of symbolic functions
(whereas :math:`\Theta(X)` is a matrix whose entries are numerical values).

With each of the objects :math:`X`, :math:`\dot X`, :math:`\Theta(X)`, and :math:`\Xi`
being defined, we are ready to write down the approximation problem underlying SINDy:


.. math::

    \dot X \approx \Theta(X)\Xi.

Structure of PySINDy
----------------------------
The submodules of PySINDy are each aligned with one of the terms in
the aforementioned approximation equation, :math:`\dot X \approx \Theta(X)\Xi.`

 * ``pysindy.differentiation`` performs numerical differentiation to compute :math:`\dot X` from :math:`X`;
 * ``pysindy.feature_library`` allows the user to specify a set of library functions and handles the formation of :math:`\Theta(X)`;
 * ``pysindy.optimizers`` provides a set of sparse regression solvers for determining :math:`\Xi`.

The ``SINDy`` object encapsulates one class object from each of these three submodules and uses them,
along with a user-supplied data matrix, to find a governing dynamical system.

The `beginning tutorial`_ walks through an example showing how this works using a toy dataset.

.. _beginning tutorial: ./examples/tutorial_1/example
