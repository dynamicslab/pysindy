from itertools import repeat
from typing import Optional
from typing import Sequence
from warnings import warn

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..utils import AxesArray
from .base import _unique
from .base import BaseFeatureLibrary
from .base import x_sequence_or_item
from .weak_pde_library import WeakPDELibrary


class GeneralizedLibrary(BaseFeatureLibrary):
    """Put multiple libraries into one library. All settings
    provided to individual libraries will be applied. Note that this class
    allows one to specifically choose which input variables are used for
    each library, and take tensor products of any pair of libraries. Tensored
    libraries inherit the same input variables specified for the individual
    libraries.

    Parameters
    ----------
    libraries : list of libraries
        Library instances to be applied to the input matrix.

    tensor_array : 2D list of booleans, optional, (default None)
        Default is to not tensor any of the libraries together. Shape
        equal to the # of tensor libraries and the # feature libraries.
        Indicates which pairs of libraries to tensor product together and
        add to the overall library. For instance if you have 5 libraries,
        and want to do two tensor products, you could use the list
        [[1, 0, 0, 1, 0], [0, 1, 0, 1, 1]] to indicate that you want two
        tensored libraries from tensoring libraries 0 and 3 and libraries
        1, 3, and 4.

    inputs_per_library : Sequence of Seqeunces of ints (default None)
        list that specifies which input indexes should be passed as
        inputs for each of the individual feature libraries.
        length must equal the number of feature libraries.  Default is
        that all inputs are used for every library.

    Attributes
    ----------
    self.libraries_full_: list[BaseFeatureLibrary]
        The fitted libraries

    n_features_in_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. The number of output features
        is the sum of the numbers of output features for each of the
        concatenated libraries.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import FourierLibrary, CustomLibrary
    >>> from pysindy.feature_library import GeneralizedLibrary
    >>> x = np.array([[0.,-1],[1.,0.],[2.,-1.]])
    >>> functions = [lambda x : np.exp(x), lambda x,y : np.sin(x+y)]
    >>> lib_custom = CustomLibrary(library_functions=functions)
    >>> lib_fourier = FourierLibrary()
    >>> lib_generalized = GeneralizedLibrary([lib_custom, lib_fourier])
    >>> lib_generalized.fit(x)
    >>> lib_generalized.transform(x)
    """

    def __init__(
        self,
        libraries: list,
        tensor_array=None,
        inputs_per_library: Optional[Sequence[Sequence[int]]] = None,
        exclude_libraries=[],
    ):
        if len(libraries) > 0:
            self.libraries = libraries

            if has_weak(self) and has_nonweak(self):
                raise ValueError(
                    "At least one of the libraries is a weak form library, "
                    "and at least one of the libraries is not, which will "
                    "result in a nonsensical optimization problem. Please use "
                    "all weak form libraries or no weak form libraries."
                )
        else:
            raise ValueError(
                "Empty or nonsensical library list passed to this library."
            )
        if inputs_per_library is not None:
            if len(inputs_per_library) != len(libraries):
                raise ValueError(
                    "If specifying different inputs for each library, then "
                    "first dimension of inputs_per_library must be equal to "
                    "the number of libraries being used."
                )
            if isinstance(inputs_per_library, np.ndarray):
                warn(
                    "inputs_per_library should no longer be passed as a numpy array",
                    UserWarning,
                )
                inputs_per_library = [list(row) for row in inputs_per_library]
            if any(x_ind < 0 for inputs in inputs_per_library for x_ind in inputs):
                raise ValueError(
                    "The inputs_per_library parameter must be a numpy array "
                    "of integers with values between 0 and "
                    "len(input_variables) - 1."
                )

        if tensor_array is not None:
            if np.asarray(tensor_array).ndim != 2:
                raise ValueError("Tensor product array should be 2D list.")
            if np.asarray(tensor_array).shape[-1] != len(libraries):
                raise ValueError(
                    "If specifying tensor products between libraries, then "
                    "last dimension of tensor_array must be equal to the "
                    "number of libraries being used."
                )
            if np.any(np.ravel(tensor_array) > 1) or np.any(np.ravel(tensor_array) < 0):
                raise ValueError(
                    "The tensor_array parameter must be a numpy array "
                    "of booleans, so values must be either 0 or 1."
                )
            for i in range(len(tensor_array)):
                if np.sum(tensor_array[i]) < 2:
                    raise ValueError(
                        "If specifying libraries to tensor together, must "
                        "specify at least two libraries (there should be at "
                        "least two entries with value 1 in the tensor_array)."
                    )
        self.tensor_array = tensor_array
        self.inputs_per_library = inputs_per_library
        self.exclude_libraries = exclude_libraries

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
        self.n_features_in_ = n_features

        # If parameter is not set, use all the inputs
        if self.inputs_per_library is None:
            self.inputs_per_library = list(
                repeat(list(range(n_features)), len(self.libraries))
            )
        else:
            # Check that the numbers in inputs_per_library are sensible
            if any(
                input_ind >= n_features
                for input_list in self.inputs_per_library
                for input_ind in input_list
            ):
                raise ValueError(
                    "Each row in inputs_per_library must consist of integers "
                    "between 0 and the number of total input features - 1. "
                )

        # First fit all libraries separately below, with subset of the inputs
        fitted_libs = [
            lib.fit([x[..., _unique(self.inputs_per_library[i])] for x in x_full], y)
            for i, lib in enumerate(self.libraries)
        ]

        # Next, tensor some libraries and append them to the list
        if self.tensor_array is not None:
            num_tensor_prods = np.shape(self.tensor_array)[0]
            for i in range(num_tensor_prods):
                lib_inds = np.ravel(np.where(self.tensor_array[i]))
                library_subset = np.asarray(fitted_libs)[lib_inds]
                library_full = np.prod(library_subset)
                library_full._set_inputs_per_library(
                    [self.inputs_per_library[lib_ind] for lib_ind in lib_inds]
                )
                library_full.fit(x_full, y)
                fitted_libs.append(library_full)

        # Calculate the sum of output features
        self.n_output_features_ = sum(
            lib.n_output_features_
            for lib in fitted_libs
            if lib not in self.exclude_libraries
        )

        # Save fitted libs
        self.libraries_full_ = fitted_libs

        return self

    @x_sequence_or_item
    def transform(self, x_full):
        """Transform data with libs provided below.

        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape [n_samples, NP]
            The matrix of features, where NP is the number of features
            generated from applying the custom functions to the inputs.

        """
        check_is_fitted(self, attributes=["n_features_in_"])

        xp_full = []
        for x in x_full:
            n_features = x.shape[x.ax_coord]
            n_input_features = self.n_features_in_
            if n_features != n_input_features:
                raise ValueError("x shape does not match training shape")

            xps = []
            for i, lib in enumerate(self.libraries_full_):
                if i < len(self.inputs_per_library):
                    if i not in self.exclude_libraries:
                        xps.append(
                            lib.transform(
                                [x[..., _unique(self.inputs_per_library[i])]]
                            )[0]
                        )
                else:
                    xps.append(lib.transform([x])[0])

            xp = AxesArray(np.concatenate(xps, axis=xps[0].ax_coord), xps[0].__dict__)
            xp_full = xp_full + [xp]
        return xp_full

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
        check_is_fitted(self)
        feature_names = list()
        for i, lib in enumerate(self.libraries_full_):
            if i not in self.exclude_libraries:
                if i < len(self.libraries):
                    if input_features is None:
                        input_features_i = [
                            "x%d" % k for k in _unique(self.inputs_per_library[i])
                        ]
                    else:
                        input_features_i = np.asarray(input_features)[
                            _unique(self.inputs_per_library[i])
                        ].tolist()
                else:
                    # Tensor libraries need all the inputs and then internally
                    # handle the subsampling of the input variables
                    if input_features is None:
                        input_features_i = ["x{k}" for k in range(self.n_features_in_)]
                    else:
                        input_features_i = input_features
                feature_names += lib.get_feature_names(input_features_i)
        return feature_names

    def calc_trajectory(self, diff_method, x, t):
        return self.libraries[0].calc_trajectory(diff_method, x, t)

    def get_spatial_grid(self):
        for lib_k in self.libraries:
            spatial_grid = lib_k.get_spatial_grid()
            if spatial_grid is not None:
                return spatial_grid


def has_weak(lib):
    if isinstance(lib, WeakPDELibrary):
        return True
    elif hasattr(lib, "libraries_"):
        for lib_k in lib.libraries_:
            if has_weak(lib_k):
                return True
    return False


def has_nonweak(lib):
    if hasattr(lib, "libraries_"):
        for lib_k in lib.libraries_:
            if has_nonweak(lib_k):
                return True
    elif not isinstance(lib, WeakPDELibrary):
        return True
    return False
