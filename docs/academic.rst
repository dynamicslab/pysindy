Use in academic work
================================

Pysindy as a dependency for your research
---------------------------------------------
It's important to specify the dependencies of any project,
and research projects are no exception.
Pysindy on pypi and github have various standard versions, e.g. 2.0.0
that follow `semantic versioning`_.
If you can use one of those versions, it's easy to use standard version ranges,
or pin pysindy to a specific version
(e.g. in pyproject.toml, requirements.txt, or environment.yml).

.. _semantic versioning: https://semver.org/

There are situations when something is wrong or missing in pysindy,
and you want a specific version off of github that does not have an associated
git tag or pypi release.  In those cases, it's possible to use a `direct
reference`_ to a github version, e.g

.. code-block:: text

    pysindy[miosr] @ https://github.com/dynamicslab/pysindy@c1da1f9

.. _direct reference: https://peps.python.org/pep-0440/#direct-references

Most challengingly, you may want to make fixes and changes to pysindy
during the course of a research project.  In these cases, the direct url works,
however you may need to pause your project each time you change pysindy,
as a new direct url will only be created once the pull request is merged.
In these cases, it's possible to include pysindy as a `git submodule`_
of your research project.
This way, your project keeps track of which *local* pysindy
hash is required.
Changes to your local pysindy will appear in the super-project's ``git diff``
so you'll never forget to update your dependencies.
In parallel, you can push your local changes to a fork of pysindy
and pull request them in.
This way, everyone can benefit from improvements you make to pysindy,
while you don't have to wait to use fixes downstream,
and you won't forget to edit your dependencies.

.. _git submodule: https://git-scm.com/book/en/v2/Git-Tools-Submodules

Regardless, in mast cases, you'll want to `cite <Citation>`_ pysindy.  On occasion, you may want to
`add your method to pysindy <Adding your new method to pysindy>`_.

Citation
---------------------
Pysindy is the result of academic work.
If you are using pysindy in a paper, there are a different documents that you can cite:

*   The Journal of Open-Source Software (JOSS) papers (`1 <joss1>`_, `2 <joss2>`_) on pysindy
*   This repository, via the citation.cff
*   The `original SINDy paper`_.
*   Papers introducing innovations that you use, e.g. WeakSINDy, Trapping, or Kalman SINDy.
    References for these can be found in the docstrings for the relevant functionality.

.. _joss1: https://joss.theoj.org/papers/10.21105/joss.02104

.. _joss2: https://joss.theoj.org/papers/10.21105/joss.03994

.. _original SINDy paper: https://www.pnas.org/doi/10.1073/pnas.1517384113

The JOSS papers have the Bibtex:

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


Citation, however, be ambiguous.  See the section on `ambiguity <Ambiguity>`_

Adding your new method to pysindy
----------------------------------
If you're publishing a new approach for learning differential equations from data,
we believe that adding an implementation to pysindy helps promote the new research
and advance the state of the art.
By making the approach API-compatible with existing approaches, it allows researchers
to evaluate new methods against old ones
and facilitates more rapid development of the field.
When things are not API compatible, it means that experiments need to be rewritten
in order to adapt to each different API, coupling the experiment to the method,
and making the experiment more fragile to changes in either dependency.

Although API compatibility is good, there are drawbacks.
If something is added to pysindy, someone needs to take on the maintenance burden.
Not everyone publishing a method needs or wants to develop tests or benchmarks
to make sure that future refactorings and changes don't subtly break the method.
Moreover, if your method is compatible with certain other innovations or shares
functionality, a refactor may be required in order to make maintenance easier.

This is all good and important work, but if its a bridge too far, the
innovation can still be useful as a separate `distribution package`_.
The above difficulties all center around maintenance and dependency management.
If the API is compatible, nothing prevents:

.. _distribution package: https://packaging.python.org/en/latest/discussions/distribution-package-vs-import-package/

.. code-block:: python

    from your_package import YourOptimizer
    import pysindy as ps

    ps.SINDy(optimizer=YourOptimizer())

It is also possible to use `plugins`_ to distribute your method separately but import
via pysindy.
In either case, if nobody is claiming maintenance, it may be useful
to pin the pysindy dependency in your distribution package to a narrow range.
And we would love to to your package in our documentation!

.. _plugins: https://packaging.python.org/en/latest/specifications/entry-points/

So TL;DR: Let us know if you have a new SINDy method!
It could be something we want to merge, or it could be something we want to link to.
But either way, thank you for your contribution to the field!


Ambiguity
----------------------
Rarely, a pull request adds significant functionality and original description,
but either the author of the code was not an author on the relevant paper
(e.g. ``SBR``), or the relevant paper has yet to be published,
or the method is not significant enough to be published on its own,
but provides  a lot of code that would otherwise need to be written for certain applications (eg. ``StabilizedLinearSR3``).
Other times, a separate author fixes mathematically significant implementation flaws;
these occur without publishing the equivalent of a corrigenda or journal letter.
Equally rarely, someone adds a new paper's functionality to pysindy
in order to support an associated manuscript,
but implementing that functionality requires substantial refactoring work
from other package maintainers.

There are largely three ways to give credit to this kind of academic work:
authorship, acknowledgment, and citation.
It's hard to match these levels to the wide range of academic work
in an open source setting.
Ultimately, we believe that sharing wins motivates a better research product
and drives the field forwards.
What that looks like is your decision.

If you want help with git archeaology to understand how features were added,
please add a discussion.

References
----------------------
The following is a partial list of references used in pysindy:

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
