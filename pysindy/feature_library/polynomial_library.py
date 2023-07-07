from itertools import chain
from typing import Iterator

import numpy as np
from scipy import sparse
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_is_fitted

from ..utils import AxesArray
from ..utils import comprehend_axes
from ..utils import wrap_axes
from .base import BaseFeatureLibrary
from .base import x_sequence_or_item


class PolynomialLibrary(PolynomialFeatures, BaseFeatureLibrary):
    """Generate polynomial and interaction features.

    This is the same as :code:`sklearn.preprocessing.PolynomialFeatures`,
    but also adds the option to omit interaction features from the library.

    Parameters
    ----------
    degree : integer, optional (default 2)
        The degree of the polynomial features.

    include_interaction : boolean, optional (default True)
        Determines whether interaction features are produced.
        If false, features are all of the form ``x[i] ** k``.

    interaction_only : boolean, optional (default False)
        If true, only interaction features are produced: features that are
        products of at most ``degree`` *distinct* input features (so not
        ``x[1] ** 2``, ``x[0] * x[2] ** 3``, etc.).

    include_bias : boolean, optional (default True)
        If True (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).

    order : str in {'C', 'F'}, optional (default 'C')
        Order of output array in the dense case. 'F' order is faster to
        compute, but may slow down subsequent estimators.

    Attributes
    ----------
    powers_ : array, shape (n_output_features, n_input_features)
        powers_[i, j] is the exponent of the jth input in the ith output.

    n_features_in_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. This number is computed by
        iterating over all appropriately sized combinations of input features.
    """

    def __init__(
        self,
        degree=2,
        include_interaction=True,
        interaction_only=False,
        include_bias=True,
        order="C",
    ):
        super().__init__(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
            order=order,
        )
        self.include_interaction = include_interaction

    @staticmethod
    def _combinations(
        n_features, degree, include_interaction, interaction_only, include_bias
    ) -> Iterator[tuple]:
        if not include_interaction:
            return chain(
                [()] if include_bias else [],
                (
                    exponent * (feat_idx,)
                    for exponent in range(1, degree + 1)
                    for feat_idx in range(n_features)
                ),
            )
        return PolynomialFeatures._combinations(
            n_features=n_features,
            min_degree=int(not include_bias),
            max_degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
        )

    @property
    def powers_(self):
        check_is_fitted(self)
        combinations = self._combinations(
            n_features=self.n_features_in_,
            degree=self.degree,
            include_interaction=self.include_interaction,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        )
        return np.vstack(
            [np.bincount(c, minlength=self.n_features_in_) for c in combinations]
        )

    def get_feature_names(self, input_features=None):
        """Return feature names for output features.

        Parameters
        ----------
        input_features : list of string, length n_features, optional
            String names for input features if available. By default,
            "x0", "x1", ... "xn_features" is used.

        Returns
        -------
        output_feature_names : list of string, length n_output_features
        """
        powers = self.powers_
        if input_features is None:
            input_features = ["x%d" % i for i in range(powers.shape[1])]
        feature_names = []
        for row in powers:
            inds = np.where(row)[0]
            if len(inds):
                name = " ".join(
                    "%s^%d" % (input_features[ind], exp)
                    if exp != 1
                    else input_features[ind]
                    for ind, exp in zip(inds, row[inds])
                )
            else:
                name = "1"
            feature_names.append(name)
        return feature_names

    @x_sequence_or_item
    def fit(self, x_full, y=None):
        """
        Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data.

        Returns
        -------
        self : instance
        """
        if self.degree < 0 or not isinstance(self.degree, int):
            raise ValueError("degree must be a nonnegative integer")
        if (not self.include_interaction) and self.interaction_only:
            raise ValueError(
                "Can't have include_interaction be False and interaction_only"
                " be True"
            )
        n_features = x_full[0].shape[x_full[0].ax_coord]
        combinations = self._combinations(
            n_features,
            self.degree,
            self.include_interaction,
            self.interaction_only,
            self.include_bias,
        )
        self.n_features_in_ = n_features
        self.n_output_features_ = sum(1 for _ in combinations)
        return self

    @x_sequence_or_item
    def transform(self, x_full):
        """Transform data to polynomial features.

        Parameters
        ----------
        x_full : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray or CSR/CSC sparse matrix,
                shape (n_samples, n_output_features)
            The matrix of features, where n_output_features is the number
            of polynomial features generated from the combination of inputs.
        """
        check_is_fitted(self)

        xp_full = []
        for x in x_full:
            if sparse.issparse(x) and x.format not in ["csr", "csc"]:
                # create new with correct sparse
                axes = comprehend_axes(x)
                x = x.asformat("csc")
                wrap_axes(axes, x)
            n_features = x.shape[x.ax_coord]
            if n_features != self.n_features_in_:
                raise ValueError("x shape does not match training shape")

            combinations = self._combinations(
                n_features,
                self.degree,
                self.include_interaction,
                self.interaction_only,
                self.include_bias,
            )
            if sparse.isspmatrix(x):
                columns = []
                for comb in combinations:
                    if comb:
                        out_col = 1
                        for col_idx in comb:
                            out_col = x[..., col_idx].multiply(out_col)
                        columns.append(out_col)
                    else:
                        bias = sparse.csc_matrix(np.ones((x.shape[0], 1)))
                        columns.append(bias)
                xp = sparse.hstack(columns, dtype=x.dtype).tocsc()
            else:
                xp = AxesArray(
                    np.empty(
                        (*x.shape[:-1], self.n_output_features_),
                        dtype=x.dtype,
                        order=self.order,
                    ),
                    x.__dict__,
                )
                for i, comb in enumerate(combinations):
                    xp[..., i] = x[..., comb].prod(-1)
            xp_full = xp_full + [xp]
        return xp_full
