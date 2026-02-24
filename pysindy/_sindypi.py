r"""SINDy-PI: Sparse Identification of Nonlinear Dynamics — Parallel Implicit.

Implements an implicit-SINDy variant where the feature library is applied to both
state variables and their time derivatives, and one sparse model is fit per feature
column of the resulting library matrix.

Reference: Kaheman, K., Kutz, J. N., & Brunton, S. L. (2020).
SINDy-PI: a robust algorithm for parallel implicit sparse identification of
nonlinear dynamics. Proceedings of the Royal Society A, 476(2242), 20200279.
"""
from copy import deepcopy
from typing import Optional
from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Self

from ._core import _adapt_to_multiple_trajectories
from ._core import _check_multiple_trajectories
from ._core import _validate_inputs
from ._core import _zip_like_sequence
from ._core import standardize_shape
from .differentiation import FiniteDifference
from .differentiation.base import BaseDifferentiation
from .feature_library import PolynomialLibrary
from .feature_library.base import BaseFeatureLibrary
from .optimizers import STLSQ
from .optimizers.base import BaseOptimizer
from .utils import AxesArray
from .utils import SampleConcatter


class ParallelImplicitSINDy(BaseEstimator):
    """Sparse Identification of Nonlinear Dynamics — Parallel Implicit (SINDy-PI).

    Builds an implicit-form sparse model:

    Parameters
    ----------
    optimizer : BaseOptimizer, optional
        Template optimizer.  A :func:`copy.deepcopy` is made for every
        column of the feature matrix.  Defaults to ``STLSQ()``.
    feature_library : BaseFeatureLibrary, optional
        Library used to build candidate terms from the concatenated
        ``[x | x_dot]`` data.  Defaults to ``PolynomialLibrary()``.
    differentiation_method : BaseDifferentiation, optional
        Method used to compute time derivatives (and optionally smooth
        ``x``).  Defaults to ``FiniteDifference(axis=-2)``.

    Attributes
    ----------
    optimizers_ : list of BaseOptimizer, length n_output_features_
        One fitted optimizer per column of the feature matrix.
        ``optimizers_[j].coef_`` has shape ``(1, n_output_features_ - 1)``.
    lib_feature_names_ : list of str, length n_output_features_
        Names of all library features, derived from both state and
        derivative variable names.
    feature_names_ : list of str, length n_input_state_features
        Names of the original state variables.
    n_features_in_ : int
        Number of features seen during :meth:`fit` (= 2 × n_state_features).
    n_output_features_ : int
        Number of output features produced by the library.
    """

    def __init__(
        self,
        optimizer: Optional[BaseOptimizer] = None,
        feature_library: Optional[BaseFeatureLibrary] = None,
        differentiation_method: Optional[BaseDifferentiation] = None,
    ):
        self.optimizer = optimizer if optimizer is not None else STLSQ()
        self.feature_library = (
            feature_library if feature_library is not None else PolynomialLibrary()
        )
        if differentiation_method is None:
            differentiation_method = FiniteDifference(axis=-2)
        self.differentiation_method = differentiation_method

    def fit(
        self,
        x,
        t: float | np.ndarray | Sequence[np.ndarray],
        x_dot=None,
        feature_names: Optional[list[str]] = None,
    ) -> Self:
        """Fit one SINDy-PI model per library feature column.

        Parameters
        ----------
        x : array-like or list of array-like, shape (n_samples, n_state_features)
            Training data.  Pass a list for multiple trajectories.
        t : float, array of shape (n_samples,), or list thereof
            If a float, specifies the uniform timestep between samples.
            If an array, specifies the collection times (strictly increasing).
            For multiple trajectories, a matching list of arrays.
        x_dot : array-like or list of array-like, optional
            Pre-computed time derivatives.  If ``None``, derivatives are
            computed from ``x`` using ``differentiation_method``.
        feature_names : list of str, optional
            Names for the *state* variables.  Derivative names are formed by
            appending ``'_dot'``.  Defaults to ``['x0', 'x1', ...]``.

        Returns
        -------
        self
        """
        if not _check_multiple_trajectories(x, t, x_dot, None):
            x, t, x_dot, _ = _adapt_to_multiple_trajectories(x, t, x_dot, None)

        x = [standardize_shape(xi) for xi in x]

        if isinstance(t, np.ScalarType):
            t = [np.arange(0, xi.shape[-2] * t, t) for xi in x]
        t = [standardize_shape(ti) for ti in t]

        if x_dot is not None:
            x_dot = [standardize_shape(xdoti) for xdoti in x_dot]

        _validate_inputs(x, t, x_dot, None)

        if x_dot is None:
            x_smooth, x_dot = zip(
                *[
                    self.feature_library.calc_trajectory(
                        self.differentiation_method, xi, ti
                    )
                    for xi, ti in _zip_like_sequence(x, t)
                ]
            )
        else:
            x_smooth = x
            x_dot = x_dot

        n_state = x_smooth[0].n_coord
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(n_state)]
        self.feature_names_ = feature_names

        dot_names = [f"{name}_dot" for name in feature_names]
        lib_input_names = list(feature_names) + dot_names

        x_combined = [
            np.concatenate((xi, xdoti), axis=xi.ax_coord)
            for xi, xdoti in zip(x_smooth, x_dot)
        ]

        feat_list = self.feature_library.fit_transform(x_combined)
        feats = SampleConcatter().fit_transform(feat_list)

        self.n_features_in_ = self.feature_library.n_features_in_
        self.n_output_features_ = self.feature_library.n_output_features_
        self.lib_feature_names_ = self.feature_library.get_feature_names(
            input_features=lib_input_names
        )

        n_terms = self.n_output_features_
        # feats_arr = np.asarray(feats)
        self.optimizers_: list[BaseOptimizer] = []

        for j in range(n_terms):
            mask = np.ones(n_terms, dtype=bool)
            mask[j] = False
            feat_j = feats[:, mask]
            y_j = feats[:, j]
            opt_j = deepcopy(self.optimizer)
            opt_j.fit(feat_j, y_j)
            self.optimizers_.append(opt_j)

        return self

    def print(self, precision: int = 3, **kwargs) -> None:
        """Print all SINDy-PI candidate models.

        For each library feature ``f_j``, prints the equation::

            (f_j) = c_1 f_1 + ... + c_{j-1} f_{j-1} + c_{j+1} f_{j+1} + ...

        where the coefficients come from ``optimizers_[j]``.

        Parameters
        ----------
        precision : int, optional (default 3)
            Number of decimal places for each coefficient.
        **kwargs
            Additional keyword arguments forwarded to the built-in
            :func:`print`.
        """
        check_is_fitted(self)

        feat_names = self.lib_feature_names_
        n_terms = self.n_output_features_

        def _fmt(c: float, name: str) -> str:
            if np.round(c, precision) == 0:
                return ""
            return f"{c: .{precision}f} {name}"

        for j, opt_j in enumerate(self.optimizers_):
            lhs = feat_names[j]
            rhs_names = [feat_names[k] for k in range(n_terms) if k != j]
            coef = opt_j.coef_.flatten()
            components = [_fmt(c, name) for c, name in zip(coef, rhs_names)]
            rhs = " + ".join(filter(bool, components))
            if not rhs:
                rhs = f"{0:.{precision}f}"
            print(f"({lhs}) = {rhs}", **kwargs)
