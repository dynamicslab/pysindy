from .polynomial_library import PolynomialLibrary


class IdentityLibrary(PolynomialLibrary):
    """
    Generate an identity library which maps all input features to
    themselves.

    Attributes
    ----------
    n_features_in_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. The number of output features
        is equal to the number of input features.

    Examples
    --------
    >>> import numpy as np
    >>> from pysindy.feature_library import IdentityLibrary
    >>> x = np.array([[0,-1],[0.5,-1.5],[1.,-2.]])
    >>> lib = IdentityLibrary().fit(x)
    >>> lib.transform(x)
    array([[ 0. , -1. ],
           [ 0.5, -1.5],
           [ 1. , -2. ]])
    >>> lib.get_feature_names()
    ['x0', 'x1']
    """

    def __init__(self):
        super().__init__(degree=1, include_bias=False)
