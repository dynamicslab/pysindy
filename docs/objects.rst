PySINDy Object Model
========================
This document describes the main types of objects in pysindy
and how users tend to interact with them.
It then proceeds to summarize the problems and planned changes to the type system,
as discussed in issues like
`this one <https://github.com/dynamicslab/pysindy/issues/351>`_.
It is most useful for people who want to implement their own variant of SINDy
within the pysindy package.

Current typing system
----------------------------
The PySINDy package revolves around the abstract base class ``_BaseSINDy`` which represents
the problem of fitting a dynamical system :math:`X' = \Xi^T \Theta(X)`.
It implements the basics of printing the discovered system of equations and
fitting the shape of inputs and outputs.
For example, it contains methods like ``equations()``, ``print()``, and  ``_fit_shape()``.
Different subclasses handle how that fitting actually occurs:
As the only current concrete subclass, ``SINDy`` objects follow the traditional approach, comprising a

* ``differentiation_method: BaseDifferentiation``: computes :math:`X'`.
  Subclasses often accept an ``axis`` and ``order`` argument, and are callable objects.
  When creating new differentiation methods, add them to the |derivative|_.
* ``feature_library: BaseFeatureLibrary``: specifies the candidate basis functions to be used to construct :math:`\Theta(X)`.
  Most significantly for the end user, ``fit()`` determines the number and string format
  of the feature library, as applied to the input variables.
  You can see these with ``BaseFeatureLibrary.get_feature_names()`` or ``BaseFeatureLibrary.n_features_out_``.
  One challenge with the straight-pipeline approach is that constraints must be manually constructed as arrays,
  and require knowing the order of the features, which in turn requires the feature library to be fit.
  There is no harm, however, in fitting the feature library on the data before fitting ``SINDy``,
  even though the latter will refit the feature library.
  ``transform()`` is used to actually calculate the feature values on input data.
* ``optimizer: BaseOptimizer``: implements a sparse regression method for solving for :math:`\Xi`.
  These share a common ``fit()`` method, with different implementations going in ``_reduce()``.
  Most notably, they share a ``history_`` of coefficient values and a ``coef_`` array of the final coefficients.
  When subclassing ``BaseOptimizer``, be sure to note whether your approach can be unbiased,
  and if not, raise an error if set to ``True``.


.. |derivative| replace:: ``derivative`` package
.. _derivative: https://derivative.readthedocs.io/en/latest/

Once a ``SINDy`` object has been created it must be fit to measurement data, similar to a ``scikit-learn`` model.
It can then be used to predict derivatives given new measurements in ``predict()``
as well as evolve novel initial conditions forward in time using ``simulate()``.
It can also ``score()`` itself.  Take care, however, as there are different metrics
for a SINDy model (`issue 1`_, `issue 2`_).

.. _issue 1: https://github.com/dynamicslab/pysindy/issues/372

.. _issue 2: https://github.com/scikit-learn/scikit-learn/issues/31360


Problems
---------------------
.. admonition:: A good rule

    Type compatibility should equate to mathematical compatibility

While the single base class ``SINDy`` worked for a while, it ran into problems as different innovations
were added as either differentiation methods, feature libraries, or optimizers,
but not as new types.
Oftentimes the innovations were only compatible with correct decisions on other objects in the SINDy model,
e.g. trapping SINDy is implemented as a ``TrappingSR3`` optimizer, but is only mathematically sensible with a quadratic Polynomial library.
At the same time, the Polynomial library type is not parameterized by polynomial order,
which is just one of the changes that would need to exist in order for the type system to enforce mathematical compatibility.

Similar problems exist in Weak SINDy and SINDy-PI, whose implementations are deeply coupled.

Future type system changes
-----------------------------
Currently weak SINDy is implemented through the ``WeakPDELibrary`` in a basic SINDy model.
However, as it eschews derivative calculation, ``WeakSINDy`` will soon exist as a subclass of ``_BaseSINDy``
for fitting continuous dynamics using the integral form.

Similarly, discrete SINDy, which does not use a differentiation method, will become a subclass of ``_BaseSINDy``
rather than an argument to the ``SINDy`` initialization.

SINDy-PI is a unique problem in that it represents the problem of fitting a dynamical system,
as does ``_BaseSINDy``,
but produces a set of possible coefficient matrices with no ability to choose from them.
Moreover, the equations it attempts to discover are implicit and do not create predictions in a uniform way.
This means that ``predict()``, ``simulate()``, and ``equations()`` do not work.
SINDy-PI is currently implemented across the ``PDELibrary``, ``WeakPDELibrary`` and ``SINDyPIOptimizer``,
but will eventually become its own class that interacts with ``SINDy``, ``WeakSINDy``, Discrete SINDy,
and component objects in a unique way.

``EnsembleOptimizer`` and ``SBR`` are two different optimizers that result in a distribution of coefficients.
The former wraps another optimizer, however it should not wrap ``SBR`` or another ``EnsembleOptimizer``.
This reflects a fundamental difference in types: ``RandomVariableOptimizers`` whose coefficients are understaood to be random variables
and ``DeterministicOptimizers`` whose coefficients are deterministic.
Moreover, post-analysis of random variable optimizers is ad-hoc;
users must access the underlying numpy arrays (``EnsembleOptimizer``)
or numpyro random variables (``SBR``) in order to visualize the distributions.
While that is a smaller problem, it suggests a unified API would support better comparison of these approaches.


Trapping SINDy, as mentioned, requires some spooky action at a distance.
It may become a factory function which chooses the optimizer and feature library for the user,
depending on whether the user wants weak or traditional SINDy.

Differentiation began with ``FiniteDifference``, but quickly moved to methods that
both smooth and differentiate.
For a while pysindy did not use the smooth coordinates, only the smoothed derivatives.
For backwards compatibility, the smoothed coordinates were attached
to the ``BaseDifferentiation`` object, rather than returned.
Using differentiation for PDEs adds additional complexity.
Some differentation/smoothing methods assume a single order of smoothness,
which makes them unsuitable for most PDEs.
Smoothing that only smoothes in one axis as a time does not result in consistent
trajectories when smoothed along different axes.
Moreover, most existing implementations are defined in the ``derivative`` package.
Ideally, ``pysindy`` gets out of the business of derivative implementations,
merely specifying (and correctly using) an API that treats differentiation and smoothing
as two aspects of applying assumptions to a random process.
