import numpy as np
from sklearn import __version__
from sklearn.utils.validation import check_is_fitted

from ..utils import AxesArray
from ..utils import comprehend_axes
from .base import BaseFeatureLibrary
from .base import x_sequence_or_item


class FourierLibrary(BaseFeatureLibrary):
    """
    Generate a library with trigonometric functions.

    Parameters
    ----------
    n_frequencies : int, optional (default 1)
        Number of frequencies to include in the library. The library will
        include functions :math:`\\sin(x), \\sin(2x), \\dots
        \\sin(n_{frequencies}x)` for each input feature :math:`x`
        (depending on which of sine and/or cosine features are included).

    include_sin : boolean, optional (default True)
        If True, include sine terms in the library.

    include_cos : boolean, optional (default True)
        If True, include cosine terms in the library.

    library_ensemble : boolean, optional (default False)
        Whether or not to use library bagging (regress on subset of the
        candidate terms in the library)

    ensemble_indices : integer array, optional (default 0)
        The indices to use for ensembling the library.

    Attributes
    ----------
    n_input_features_ : int
        The total number of input features.
        WARNING: This is deprecated in scikit-learn version 1.0 and higher so
        we check the sklearn.__version__ and switch to n_features_in if needed.

    n_output_features_ : int
        The total number of output features. The number of output features
        is ``2 * n_input_features_ * n_frequencies`` if both sines and cosines
        are included. Otherwise it is ``n_input_features * n_frequencies``.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import FourierLibrary
    >>> x = np.array([[0.],[1.],[2.]])
    >>> lib = FourierLibrary(n_frequencies=2).fit(x)
    >>> lib.transform(x)
    array([[ 0.        ,  1.        ,  0.        ,  1.        ],
           [ 0.84147098,  0.54030231,  0.90929743, -0.41614684],
           [ 0.90929743, -0.41614684, -0.7568025 , -0.65364362]])
    >>> lib.get_feature_names()
    ['sin(1 x0)', 'cos(1 x0)', 'sin(2 x0)', 'cos(2 x0)']
    """

    def __init__(
        self,
        n_frequencies=1,
        include_sin=True,
        include_cos=True,
        library_ensemble=False,
        ensemble_indices=[0],
    ):
        super(FourierLibrary, self).__init__(
            library_ensemble=library_ensemble, ensemble_indices=ensemble_indices
        )
        if not (include_sin or include_cos):
            raise ValueError("include_sin and include_cos cannot both be False")
        if n_frequencies < 1 or not isinstance(n_frequencies, int):
            raise ValueError("n_frequencies must be a positive integer")
        self.n_frequencies = n_frequencies
        self.include_sin = include_sin
        self.include_cos = include_cos

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
        check_is_fitted(self)
        if float(__version__[:3]) >= 1.0:
            n_input_features = self.n_features_in_
        else:
            n_input_features = self.n_input_features_
        if input_features is None:
            input_features = ["x%d" % i for i in range(n_input_features)]
        feature_names = []
        for i in range(self.n_frequencies):
            for feature in input_features:
                if self.include_sin:
                    feature_names.append("sin(" + str(i + 1) + " " + feature + ")")
                if self.include_cos:
                    feature_names.append("cos(" + str(i + 1) + " " + feature + ")")
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
        if float(__version__[:3]) >= 1.0:
            self.n_features_in_ = n_features
        else:
            self.n_input_features_ = n_features
        if self.include_sin and self.include_cos:
            self.n_output_features_ = n_features * self.n_frequencies * 2
        else:
            self.n_output_features_ = n_features * self.n_frequencies
        return self

    @x_sequence_or_item
    def transform(self, x_full):
        """Transform data to Fourier features

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape (n_samples, n_output_features)
            The matrix of features, where n_output_features is the number of Fourier
            features generated from the inputs.
        """
        check_is_fitted(self)

        xp_full = []
        for x in x_full:
            # n_samples = x.shape[x.ax_sample]
            n_features = x.shape[x.ax_coord]
            shape = np.array(x.shape)

            if float(__version__[:3]) >= 1.0:
                n_input_features = self.n_features_in_
            else:
                n_input_features = self.n_input_features_
            if n_features != n_input_features:
                raise ValueError("x shape does not match training shape")

            shape[-1] = self.n_output_features_
            xp = np.empty(shape, dtype=x.dtype)
            idx = 0
            for i in range(self.n_frequencies):
                for j in range(n_input_features):
                    if self.include_sin:
                        xp[..., idx] = np.sin((i + 1) * x[..., j])
                        idx += 1
                    if self.include_cos:
                        xp[..., idx] = np.cos((i + 1) * x[..., j])
                        idx += 1
            xp = AxesArray(xp, comprehend_axes(xp))
            xp_full.append(xp)
        if self.library_ensemble:
            xp_full = self._ensemble(xp_full)
        return xp_full
