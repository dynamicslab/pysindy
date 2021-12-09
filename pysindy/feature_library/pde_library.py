from itertools import combinations
from itertools import combinations_with_replacement as combinations_w_r

import numpy.random
from numpy import array
from numpy import asarray
from numpy import empty
from numpy import hstack
from numpy import linspace
from numpy import meshgrid
from numpy import ones
from numpy import ravel
from numpy import reshape
from numpy import shape
from numpy import transpose
from numpy import zeros
from numpy import zeros_like
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import RegularGridInterpolator
from scipy.special import comb as n_choose_k
from scipy.special import hyp2f1
from scipy.special import poch
from sklearn import __version__
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .base import BaseFeatureLibrary
from pysindy.differentiation import FiniteDifference


#nearing completion, but we need to debug, test, and validate....
class PDELibrary(BaseFeatureLibrary):
    """Generate a PDE library with custom functions.
    """

    def __init__(
        self,
        library_functions=[],
        derivative_order=0,
        spatial_grid=None,
        temporal_grid=None,
        function_names=None,
        interaction_only=True,
        include_bias=False,
        include_interaction=True,
        is_uniform=False,
        library_ensemble=False,
        ensemble_indices=[0],
    ):
        super(PDELibrary, self).__init__(
            library_ensemble=library_ensemble, ensemble_indices=ensemble_indices
        )
        self.functions = library_functions
        self.derivative_order = derivative_order
        self.spatial_grid = spatial_grid
        self.temporal_grid = temporal_grid
        self.function_names = function_names
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.include_interaction = include_interaction
        self.is_uniform = is_uniform

        if function_names and (len(library_functions) != len(function_names)):
            raise ValueError(
                "library_functions and function_names must have the same"
                " number of elements"
            )
        if derivative_order <= 0:
            raise ValueError("The derivative order must be >0")
            
        if (spatial_grid is None):
            raise ValueError("Spatial grid required")
            
        self.space_ndim = len((self.spatial_grid).shape)
        num_derivatives = 0
        for axis in range(self.space_ndim):
            for j in range(1,derivative_order+1):
                num_derivatives += 1
        self.num_derivatives = num_derivatives

    @staticmethod
    def _combinations(n_features, n_args, interaction_only):
        """Get the combinations of features to be passed to a library function."""
        comb = combinations if interaction_only else combinations_w_r
        return comb(range(n_features), n_args)
    
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
        if float(__version__[:3]) >= 1.0:
            n_features = self.n_features_in_
        else:
            n_features = self.n_input_features
        if input_features is None:
            input_features = ["x%d" % i for i in range(n_features)]
        feature_names = []

        # Include constant term
        if self.include_bias:
            feature_names.append("1")

        # Include any non-derivative terms
        function_lib_len = 0
        if self.functions is not None:
            function_lib_len = len(self.functions)
        for i, f in enumerate(self.functions):
            for c in self._combinations(
                n_features, f.__code__.co_argcount, self.interaction_only
            ):
                feature_names.append(
                    self.function_names[i](*[input_features[j] for j in c])
                )

        if self.space_ndim != 0:
            # Include derivative (integral) terms
            for k in range(self.num_derivatives):
                for j in range(n_features):
                    feature_names.append(
                        self.function_names[function_lib_len + k](input_features[j])
                    )
            # Include mixed non-derivative + derivative (integral) terms
            if self.include_interaction:
                for k in range(self.num_derivatives):
                    for i, f in enumerate(self.functions):
                        for c in self._combinations(
                            n_features,
                            f.__code__.co_argcount,
                            self.interaction_only,
                        ):
                            for jj in range(n_features):
                                feature_names.append(
                                    self.function_names[i](
                                        *[input_features[j] for j in c]
                                    )
                                    + self.function_names[function_lib_len + k](
                                        input_features[jj]
                                    )
                                )
        return feature_names

    def fit(self, x, y=None):
        """Compute number of output features.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Measurement data.

        Returns
        -------
        self : instance
        """
        n_samples, n_features = check_array(x).shape
        if float(__version__[:3]) >= 1.0:
            self.n_features_in_ = n_features
        else:
            self.n_input_features_ = n_features

        n_output_features = 0

        # Count the number of non-derivative terms
        for f in self.functions:
            n_args = f.__code__.co_argcount
            n_output_features += len(
                list(self._combinations(n_features, n_args, self.interaction_only))
            )

        # Add the mixed derivative library_terms
        if self.include_interaction:
            n_output_features += (
                n_output_features * n_features * self.num_derivatives
            )
        # Add the pure derivative library terms
        n_output_features += n_features * self.num_derivatives

        # If there is a constant term, add 1 to n_output_features
        if self.include_bias:
            n_output_features += 1
        self.n_output_features_ = n_output_features
            
        return self

    def transform(self, x):
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

        x = check_array(x)

        n_samples, n_features = x.shape
        if float(__version__[:3]) >= 1.0:
            if n_features != self.n_features_in_:
                raise ValueError("x shape does not match training shape")
        else:
            if n_features != self.n_input_features_:
                raise ValueError("x shape does not match training shape")

        dims=self.spatial_grid.shape
        num_time = n_samples // np.product(dims)
       
        xp = empty((n_samples, self.n_output_features_), dtype=x.dtype)
        library_idx = 0

        # Constant term
        if self.include_bias:
            xp[:, library_idx] = ones(n_samples)
            library_idx += 1
            
        # derivative terms
        # make a list of multiindices with total order less than derivative_order
        multiindices=[]
        for x in itertools.product(*indices):
            current=np.array(x)
            if(np.sum(x)<=derivative_order):
                multiindices.append(current)
        multiindices=np.array(multiindices)
        
        # library terms
        for f in self.functions:
            for c in self._combinations(n_features, f.__code__.co_argcount, self.interaction_only,):
                n_library_terms += 1
        library_idx=0
        for f in self.functions:
            for c in self._combinations(n_features, f.__code__.co_argcount, self.interaction_only,):
                library_functions[:, library_idx] = np.reshape(f(*[x[:, j] for j in c]),(nsamples)
                library_idx += 1
        
        # for each multiindex, calculate the derivative of the term along each axis
        derivative_terms=np.zeros(nsamples,self.n_derivatives,n_features)
        for multiindex in multiindices:
            term=np.copy(x)
            for axis in range(len(dims)):
                if(multiindix[axis]>0):
                    term=FiniteDifference(d=multiindix[axis],axis=axis).differentiate_(term,temporal_grid)
            derivative_terms[:,derivative_ind:derivative_ind+n_features]=np.reshape(term,(nsamples,n_features))
            derivative_ind += n_features
                
        # tensor-product the derivative and the library terms
         xp = np.reshape(library_functions[:,:,np.newaxis]*library_functions[:,np.newaxis,:],(nsamples,n_features*self.n_derivatives*n_library_terms))
        return self._ensemble(xp)
