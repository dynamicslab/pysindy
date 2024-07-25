import warnings
from itertools import product as iproduct
from typing import Optional

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..utils import AxesArray
from ..utils import comprehend_axes
from .base import BaseFeatureLibrary
from .base import x_sequence_or_item
from .polynomial_library import PolynomialLibrary
from pysindy.differentiation import FiniteDifference


class PDELibrary(BaseFeatureLibrary):
    """Generate a PDE library with custom functions.

    Parameters
    ----------
    function_library : BaseFeatureLibrary, optional (default
        PolynomialLibrary(degree=3,include_bias=False))
        SINDy library with output features representing library_functions to include
        in the library, in place of library_functions.

    derivative_order : int, optional (default 0)
        Order of derivative to take on each input variable,
        can be arbitrary non-negative integer.

    spatial_grid : np.ndarray, optional (default None)
        The spatial grid for computing derivatives

    temporal_grid : np.ndarray, optional (default None)
        The temporal grid if using SINDy-PI with PDEs.

    include_bias : boolean, optional (default False)
        If True (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).
        This is hard to do with just lambda functions, because if the system
        is not 1D, lambdas will generate duplicates.

    include_interaction : boolean, optional (default True)
        This is a different than the use for the PolynomialLibrary. If true,
        it generates all the mixed derivative terms. If false, the library
        will consist of only pure no-derivative terms and pure derivative
        terms, with no mixed terms.

    implicit_terms : boolean
        Flag to indicate if SINDy-PI (temporal derivatives) is being used
        for the right-hand side of the SINDy fit.

    multiindices : list of integer arrays,  (default None)
        Overrides the derivative_order to customize the included derivative
        orders. Each integer array indicates the order of differentiation
        along the corresponding axis for each derivative term.

    differentiation_method : callable,  (default FiniteDifference)
        Spatial differentiation method.

     diff_kwargs: dictionary,  (default {})
        Keyword options to supply to differtiantion_method.

    Attributes
    ----------
    n_features_in_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. The number of output features
        is the product of the number of library functions and the number of
        input features.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import PDELibrary
    """

    def __init__(
        self,
        function_library: Optional[BaseFeatureLibrary] = None,
        derivative_order=0,
        spatial_grid=None,
        temporal_grid=None,
        include_bias=False,
        include_interaction=True,
        implicit_terms=False,
        multiindices=None,
        differentiation_method=FiniteDifference,
        diff_kwargs={},
        is_uniform=None,
        periodic=None,
    ):
        self.function_library = function_library
        self.derivative_order = derivative_order
        self.implicit_terms = implicit_terms
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.num_trajectories = 1
        self.differentiation_method = differentiation_method
        self.diff_kwargs = diff_kwargs
        if function_library is None:
            self.function_library = PolynomialLibrary(degree=3, include_bias=False)
        if derivative_order < 0:
            raise ValueError("The derivative order must be >0")

        if is_uniform is not None or periodic is not None:
            # DeprecationWarning are ignored by default...
            warnings.warn(
                "is_uniform and periodic have been deprecated."
                "in favor of differetiation_method and diff_kwargs.",
                UserWarning,
            )

        if (spatial_grid is not None and derivative_order == 0) or (
            spatial_grid is None and derivative_order != 0 and temporal_grid is None
        ):
            raise ValueError(
                "Spatial grid and the derivative order must be "
                "defined at the same time if temporal_grid is not being used."
            )

        if temporal_grid is None and implicit_terms:
            raise ValueError(
                "temporal_grid parameter must be specified if implicit_terms "
                " = True (i.e. if you are using SINDy-PI for PDEs)."
            )
        elif not implicit_terms and temporal_grid is not None:
            raise ValueError(
                "temporal_grid parameter is specified only if implicit_terms "
                " = True (i.e. if you are using SINDy-PI for PDEs)."
            )
        if spatial_grid is not None and spatial_grid.ndim == 1:
            spatial_grid = spatial_grid[:, np.newaxis]

        if temporal_grid is not None and temporal_grid.ndim != 1:
            raise ValueError("temporal_grid parameter must be 1D numpy array.")
        if temporal_grid is not None or spatial_grid is not None:
            if spatial_grid is None:
                spatiotemporal_grid = temporal_grid
                spatial_grid = np.array([])
            elif temporal_grid is None:
                spatiotemporal_grid = spatial_grid[
                    ..., np.newaxis, :
                ]  # append a fake time axis
            else:
                spatiotemporal_grid = np.zeros(
                    (
                        *spatial_grid.shape[:-1],
                        len(temporal_grid),
                        spatial_grid.shape[-1] + 1,
                    )
                )
                for ax in range(spatial_grid.ndim - 1):
                    spatiotemporal_grid[..., ax] = spatial_grid[..., ax][
                        ..., np.newaxis
                    ]
                spatiotemporal_grid[..., -1] = temporal_grid
        else:
            spatiotemporal_grid = np.array([])
            spatial_grid = np.array([])

        self.spatial_grid = spatial_grid

        # list of derivatives
        indices = ()
        if np.array(spatiotemporal_grid).ndim == 1:
            spatiotemporal_grid = np.reshape(
                spatiotemporal_grid, (len(spatiotemporal_grid), 1)
            )

        # if want to include temporal terms -> range(len(dims))
        if self.implicit_terms:
            self.ind_range = spatiotemporal_grid.ndim - 1
        else:
            self.ind_range = spatiotemporal_grid.ndim - 2

        for i in range(self.ind_range):
            indices = indices + (range(derivative_order + 1),)

        if multiindices is None:
            multiindices = []
            for ind in iproduct(*indices):
                current = np.array(ind)
                if np.sum(ind) > 0 and np.sum(ind) <= self.derivative_order:
                    multiindices.append(current)
            multiindices = np.array(multiindices)
        num_derivatives = len(multiindices)

        self.num_derivatives = num_derivatives
        self.multiindices = multiindices
        self.spatiotemporal_grid = AxesArray(
            spatiotemporal_grid, comprehend_axes(spatiotemporal_grid)
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
        check_is_fitted(self)
        n_features = self.n_features_in_

        if input_features is None:
            input_features = ["x%d" % i for i in range(n_features)]
        feature_names = []
        lib_names = []

        # Include constant term
        if self.include_bias:
            feature_names.append("1")
        # Include any non-derivative terms
        lib_names = self.function_library.get_feature_names(input_features)
        feature_names = feature_names + lib_names

        def derivative_string(multiindex):
            ret = ""
            for axis in range(self.ind_range):
                if self.implicit_terms and (axis == self.spatiotemporal_grid.ax_time,):
                    str_deriv = "t"
                else:
                    str_deriv = str(axis + 1)
                for i in range(multiindex[axis]):
                    ret = ret + str_deriv
            return ret

        # Include derivative terms
        derivative_feature_names = []
        for k in range(self.num_derivatives):
            for j in range(n_features):
                derivative_feature_names.append(
                    input_features[j] + "_" + derivative_string(self.multiindices[k])
                )
        feature_names = feature_names + derivative_feature_names

        # Include mixed non-derivative + derivative terms
        if (
            self.include_interaction
            and len(lib_names) > 0
            and len(derivative_feature_names) > 0
        ):
            feature_names = (
                feature_names
                + np.char.add(
                    np.array(lib_names).reshape(1, -1),
                    np.array(derivative_feature_names).reshape(-1, 1),
                )
                .reshape(-1)
                .tolist()
            )

        return feature_names

    @x_sequence_or_item
    def fit(self, x_full, y=None):
        """Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Measurement data.

        Returns
        -------
        self : instance
        """
        x0 = x_full[0]
        n_features = x0.shape[x0.ax_coord]
        self.n_features_in_ = n_features
        n_output_features = 0

        # Count the number of non-derivative terms
        self.function_library.fit(x0.take(0, x0.ax_time))
        n_output_features = self.function_library.n_output_features_

        # Add the mixed derivative library_terms
        if self.include_interaction:
            n_output_features += n_output_features * n_features * self.num_derivatives
        # Add the pure derivative library terms
        n_output_features += n_features * self.num_derivatives

        # If there is a constant term, add 1 to n_output_features
        if self.include_bias:
            n_output_features += 1

        self.n_output_features_ = n_output_features

        # required to generate the function names
        self.get_feature_names()

        return self

    @x_sequence_or_item
    def transform(self, x_full):
        """Transform data to pde features

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            The data to transform, row by row.

        Returns
        -------
        xp : np.ndarray, shape (n_samples, n_output_features)
            The matrix of features, where n_output_features is the number of
            features generated from the tensor product of the derivative terms
            and the library_functions applied to combinations of the inputs.
        """
        check_is_fitted(self)

        xp_full = []
        for x in x_full:
            n_features = x.shape[x.ax_coord]

            if n_features != self.n_features_in_:
                raise ValueError("x shape does not match training shape")

            shape = np.array(x.shape)
            shape[-1] = self.n_output_features_
            xp = np.empty(shape, dtype=x.dtype)

            # derivative terms
            shape[-1] = n_features * self.num_derivatives
            library_derivatives = AxesArray(np.empty(shape, dtype=x.dtype), x.axes)
            library_idx = 0
            for multiindex in self.multiindices:
                derivs = x
                for axis in range(self.ind_range):
                    if multiindex[axis] > 0:
                        s = [0 for dim in self.spatiotemporal_grid.shape]
                        s[axis] = slice(self.spatiotemporal_grid.shape[axis])
                        s[-1] = axis

                        derivs = self.differentiation_method(
                            d=multiindex[axis],
                            axis=axis,
                            **self.diff_kwargs,
                        )._differentiate(derivs, self.spatiotemporal_grid[tuple(s)])
                library_derivatives[
                    ..., library_idx : library_idx + n_features
                ] = derivs
                library_idx += n_features

            # library function terms
            library_functions = self.function_library.fit_transform(x)
            n_library_terms = library_functions.shape[-1]

            library_idx = 0

            # constant term
            if self.include_bias:
                shape[-1] = 1
                xp[..., library_idx] = np.ones(shape[:-1], dtype=x.dtype)
                library_idx += 1

            # library function terms
            xp[..., library_idx : library_idx + n_library_terms] = library_functions
            library_idx += n_library_terms

            # pure derivative terms
            xp[
                ..., library_idx : library_idx + self.num_derivatives * n_features
            ] = library_derivatives
            library_idx += self.num_derivatives * n_features

            # mixed function derivative terms
            shape[-1] = n_library_terms * self.num_derivatives * n_features
            if self.include_interaction:
                xp[
                    ...,
                    library_idx : library_idx
                    + n_library_terms * self.num_derivatives * n_features,
                ] = np.reshape(
                    library_functions[..., "coord", :]
                    * library_derivatives[..., :, "coord"],
                    shape,
                )
                library_idx += n_library_terms * self.num_derivatives * n_features
            xp = AxesArray(xp, comprehend_axes(xp))
            xp_full.append(xp)
        return xp_full

    def get_spatial_grid(self):
        return self.spatial_grid
