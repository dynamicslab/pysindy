from collections.abc import Sequence
from itertools import product as iproduct
from typing import Optional
from typing import Self

import numpy as np
from sklearn.utils.validation import check_is_fitted

from ..utils import AxesArray
from ..utils import comprehend_axes
from .base import BaseFeatureLibrary
from .base import x_sequence_or_item
from pysindy.differentiation import FiniteDifference


class PDELibrary(BaseFeatureLibrary):
    """Generate a PDE library with custom functions.

    Parameters
    ----------
    derivative_order : int, optional (default 1)
        Order of derivative to take on each input variable,
        can be arbitrary non-negative integer.

    spatial_grid : np.ndarray, optional (default None)
        The spatial grid for computing derivatives.  Final argument is
        assumed to be spatial grid dimension.

    include_interaction : boolean, optional (default True)
        This is a different than the use for the PolynomialLibrary. If true,
        it generates all the mixed derivative terms. If false, the library
        will consist of only pure no-derivative terms and pure derivative
        terms, with no mixed terms.

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
        derivative_order=1,
        spatial_grid=None,
        multiindices=None,
        differentiation_method=FiniteDifference,
        diff_kwargs={},
    ):
        self.derivative_order = derivative_order
        self.num_trajectories = 1
        self.differentiation_method = differentiation_method
        self.diff_kwargs = diff_kwargs
        if derivative_order < 0:
            raise ValueError("The derivative order must be >0")

        if spatial_grid is None or (derivative_order == 0 and multiindices is None):
            raise ValueError("Spatial grid and the derivatives must be initialized.")

        if spatial_grid.ndim == 1:
            spatial_grid = spatial_grid[:, np.newaxis]

        self.spatial_grid = spatial_grid

        # list of derivatives
        self.grid_ndim = spatial_grid.ndim - 1
        indices = self.grid_ndim * (range(derivative_order + 1),)

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
        self.spatial_grid = AxesArray(spatial_grid, comprehend_axes(spatial_grid))

    def get_feature_names(
        self, input_features: Optional[list[str]] = None
    ) -> list[str]:
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

        if input_features is None:
            check_is_fitted(self)
            n_features = self.n_features_in_
            input_features = ["x%d" % i for i in range(n_features)]

        return make_pde_feature_names(input_features, self.multiindices)

    @x_sequence_or_item
    def fit(self, x_full: Sequence[AxesArray], y=None) -> Self:
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

        self.n_output_features_ = n_features * self.num_derivatives

        # required to generate the function names
        self.get_feature_names()

        return self

    @x_sequence_or_item
    def transform(self, x_full: Sequence[AxesArray]):
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

            xp = AxesArray(np.empty(shape, dtype=x.dtype), x.axes)
            library_idx = 0
            for multiindex in self.multiindices:
                derivs = x
                for axis in range(self.grid_ndim):
                    order = multiindex[axis]
                    if order > 0:
                        s = [
                            0 if ax != axis else slice(None)
                            for ax in range(self.grid_ndim)
                        ]
                        axis_vals = self.spatial_grid[*s, axis]

                        differentiator = self.differentiation_method(
                            d=order,
                            axis=axis,
                            **self.diff_kwargs,
                        )
                        derivs = differentiator(derivs, axis_vals)
                xp[..., library_idx : library_idx + n_features] = derivs
                library_idx += n_features

            xp_full.append(xp)
        return xp_full

    def get_spatial_grid(self):
        return self.spatial_grid


def make_pde_feature_names(
    input_features: list[str], multiindices: tuple[tuple[int, ...], ...]
) -> list[str]:
    """
    Arguments:
    input_features: String names for system coordinates.
    multiindices: 2D array reflecting the differentiation features
    """

    def derivative_string(multiindex: tuple[int]):
        subscript = ""
        for axis in range(len(multiindex)):
            str_deriv = str(axis + 1)
            subscript += str_deriv * multiindex[axis]
        return subscript

    feat_names = []
    for multiindex in multiindices:
        for sys_coord in input_features:
            feat_names.append(sys_coord + "_" + derivative_string(multiindex))
    return feat_names
