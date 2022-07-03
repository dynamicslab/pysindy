from itertools import chain
from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r

import numpy as np
from scipy import sparse
from sklearn import __version__
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing._csr_polynomial_expansion import _csr_polynomial_expansion
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

    library_ensemble : boolean, optional (default False)
        Whether or not to use library bagging (regress on subset of the
        candidate terms in the library)

    ensemble_indices : integer array, optional (default [0])
        The indices to use for ensembling the library.

    Attributes
    ----------
    powers_ : array, shape (n_output_features, n_input_features)
        powers_[i, j] is the exponent of the jth input in the ith output.

    n_input_features_ : int
        The total number of input features.
        WARNING: This is deprecated in scikit-learn version 1.0 and higher so
        we check the sklearn.__version__ and switch to n_features_in if needed.

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
        library_ensemble=False,
        ensemble_indices=[0],
    ):
        super(PolynomialLibrary, self).__init__(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
            order=order,
        )
        BaseFeatureLibrary.__init__(
            self, library_ensemble=library_ensemble, ensemble_indices=ensemble_indices
        )
        if degree < 0 or not isinstance(degree, int):
            raise ValueError("degree must be a nonnegative integer")
        if (not include_interaction) and interaction_only:
            raise ValueError(
                "Can't have include_interaction be False and interaction_only"
                " be True"
            )
        self.include_interaction = include_interaction

    @staticmethod
    def _combinations(
        n_features, degree, include_interaction, interaction_only, include_bias
    ):
        comb = combinations if interaction_only else combinations_w_r
        start = int(not include_bias)
        if not include_interaction:
            if include_bias:
                return chain(
                    [()],
                    chain.from_iterable(
                        combinations_w_r([j], i)
                        for i in range(1, degree + 1)
                        for j in range(n_features)
                    ),
                )
            else:
                return chain.from_iterable(
                    combinations_w_r([j], i)
                    for i in range(1, degree + 1)
                    for j in range(n_features)
                )
        return chain.from_iterable(
            comb(range(n_features), i) for i in range(start, degree + 1)
        )

    @property
    def powers_(self):
        check_is_fitted(self)
        if float(__version__[:3]) >= 1.0:
            n_features = self.n_features_in_
        else:
            n_features = self.n_input_features_
        combinations = self._combinations(
            n_features,
            self.degree,
            self.include_interaction,
            self.interaction_only,
            self.include_bias,
        )
        return np.vstack([np.bincount(c, minlength=n_features) for c in combinations])

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
        n_features = x_full[0].shape[x_full[0].ax_coord]
        combinations = self._combinations(
            n_features,
            self.degree,
            self.include_interaction,
            self.interaction_only,
            self.include_bias,
        )
        if float(__version__[:3]) >= 1.0:
            self.n_features_in_ = n_features
        else:
            self.n_input_features_ = n_features
        self.n_output_features_ = sum(1 for _ in combinations)
        return self

    @x_sequence_or_item
    def transform(self, x_full):
        """Transform data to polynomial features.

        Parameters
        ----------
        x : array-like or CSR/CSC sparse matrix, shape (n_samples, n_features)
            The data to transform, row by row.
            Prefer CSR over CSC for sparse input (for speed), but CSC is
            required if the degree is 4 or higher. If the degree is less than
            4 and the input format is CSC, it will be converted to CSR, have
            its polynomial features generated, then converted back to CSC.
            If the degree is 2 or 3, the method described in "Leveraging
            Sparsity to Speed Up Polynomial Feature Expansions of CSR Matrices
            Using K-Simplex Numbers" by Andrew Nystrom and John Hughes is
            used, which is much faster than the method used on CSC input. For
            this reason, a CSC input will be converted to CSR, and the output
            will be converted back to CSC prior to being returned, hence the
            preference of CSR.

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
                x = x.asformat("csr")
                wrap_axes(axes, x)

            n_samples = x.shape[x.ax_time]
            n_features = x.shape[x.ax_coord]
            if float(__version__[:3]) >= 1.0:
                if n_features != self.n_features_in_:
                    raise ValueError("x shape does not match training shape")
            else:
                if n_features != self.n_input_features_:
                    raise ValueError("x shape does not match training shape")

            if sparse.isspmatrix_csr(x):
                if self.degree > 3:
                    return sparse.csr_matrix(self.transform(x.tocsc()))
                to_stack = []
                if self.include_bias:
                    to_stack.append(np.ones(shape=(n_samples, 1), dtype=x.dtype))
                to_stack.append(x)
                for deg in range(2, self.degree + 1):
                    xp_next = _csr_polynomial_expansion(
                        x.data,
                        x.indices,
                        x.indptr,
                        x.shape[1],
                        self.interaction_only,
                        deg,
                    )
                    if xp_next is None:
                        break
                    to_stack.append(xp_next)
                xp = sparse.hstack(to_stack, format="csr")
            elif sparse.isspmatrix_csc(x) and self.degree < 4:
                return sparse.csc_matrix(self.transform(x.tocsr()))
            else:
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
        if self.library_ensemble:
            xp_full = self._ensemble(xp_full)
        return xp_full
