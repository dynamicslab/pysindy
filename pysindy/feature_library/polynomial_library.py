import numpy as np
from scipy import sparse
from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.preprocessing import _csr_polynomial_expansion

from pysindy.feature_library import BaseFeatureLibrary


class PolynomialLibrary(PolynomialFeatures, BaseFeatureLibrary):
    """
    Generate polynomial and interaction features. This is the same as
    sklearn.preprocessing.PolynomialFeatures, but also adds the option
    to omit interaction features from the library.
    """

    def __init__(
        self,
        degree=2,
        include_interaction=True,
        interaction_only=False,
        include_bias=True,
        order="C",
    ):
        super(PolynomialLibrary, self).__init__(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=include_bias,
            order=order,
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

        combinations = self._combinations(
            self.n_input_features_,
            self.degree,
            self.include_interaction,
            self.interaction_only,
            self.include_bias,
        )
        return np.vstack(
            [
                np.bincount(c, minlength=self.n_input_features_)
                for c in combinations
            ]
        )

    def get_feature_names(self, input_features=None):
        """
        Return feature names for output features
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

    def fit(self, X, y=None):
        """
        Compute number of output features.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data.
        Returns
        -------
        self : instance
        """
        n_samples, n_features = check_array(X, accept_sparse=True).shape
        combinations = self._combinations(
            n_features,
            self.degree,
            self.include_interaction,
            self.interaction_only,
            self.include_bias,
        )
        self.n_input_features_ = n_features
        self.n_output_features_ = sum(1 for _ in combinations)
        return self

    def transform(self, X):
        """
        Transform data to polynomial features
        Parameters
        ----------
        X : array-like or CSR/CSC sparse matrix, shape [n_samples, n_features]
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
        XP : np.ndarray or CSR/CSC sparse matrix, shape [n_samples, NP]
            The matrix of features, where NP is the number of polynomial
            features generated from the combination of inputs.
        """
        check_is_fitted(self)

        X = check_array(
            X, order="F", dtype=FLOAT_DTYPES, accept_sparse=("csr", "csc")
        )

        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        if sparse.isspmatrix_csr(X):
            if self.degree > 3:
                return self.transform(X.tocsc()).tocsr()
            to_stack = []
            if self.include_bias:
                to_stack.append(np.ones(shape=(n_samples, 1), dtype=X.dtype))
            to_stack.append(X)
            for deg in range(2, self.degree + 1):
                Xp_next = _csr_polynomial_expansion(
                    X.data,
                    X.indices,
                    X.indptr,
                    X.shape[1],
                    self.interaction_only,
                    deg,
                )
                if Xp_next is None:
                    break
                to_stack.append(Xp_next)
            XP = sparse.hstack(to_stack, format="csr")
        elif sparse.isspmatrix_csc(X) and self.degree < 4:
            return self.transform(X.tocsr()).tocsc()
        else:
            combinations = self._combinations(
                n_features,
                self.degree,
                self.include_interaction,
                self.interaction_only,
                self.include_bias,
            )
            if sparse.isspmatrix(X):
                columns = []
                for comb in combinations:
                    if comb:
                        out_col = 1
                        for col_idx in comb:
                            out_col = X[:, col_idx].multiply(out_col)
                        columns.append(out_col)
                    else:
                        bias = sparse.csc_matrix(np.ones((X.shape[0], 1)))
                        columns.append(bias)
                XP = sparse.hstack(columns, dtype=X.dtype).tocsc()
            else:
                XP = np.empty(
                    (n_samples, self.n_output_features_),
                    dtype=X.dtype,
                    order=self.order,
                )
                for i, comb in enumerate(combinations):
                    XP[:, i] = X[:, comb].prod(1)

        return XP
