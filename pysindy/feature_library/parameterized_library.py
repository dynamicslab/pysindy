import numpy as np

from .base import BaseFeatureLibrary
from .generalized_library import GeneralizedLibrary
from .polynomial_library import PolynomialLibrary


class ParameterizedLibrary(GeneralizedLibrary):
    """A tensor product of two libraries with different inputs.
    Typically, this is a feature library of the input data and a
    parameter library of input control, making the SINDyCP method.  If
    the input libraries are weak, the temporal derivatives are
    automatically rescaled by the appropriate domain volumes.

    Parameters
    ----------
    parameter_library : BaseFeatureLibrary, optional (default PolynomialLibrary).
    Specifies the library function to apply to the input control features.

    feature_library : BaseFeatureLibrary, optional (default PolynomialLibrary).
    Specifies the library function to apply to the input data features.

    num_parameters : int, optional (default 3)
    Specifies the number of features in the input control.

    num_features : int, optional (default 3)
    Specifies the number of features in the input data.

    Attributes
    ----------
    see GeneralizedLibrary

    Examples
    ----------
    >>> import numpy as np
    >>> from pysindy.feature_library import ParameterizedLibrary,PolynomialLibrary
    >>> from pysindy import AxesArray
    >>> xs=[np.random.random((5,3)) for n in range(3)]
    >>> us=[np.random.random((5,3)) for n in range(3)]
    >>> feature_lib=PolynomialLibrary(degree=3)
    >>> parameter_lib=PolynomialLibrary(degree=1)
    >>> lib=ParameterizedLibrary(feature_library=feature_lib,
    >>>     parameter_library=parameter_lib,num_features=3,num_parameters=3)
    >>> xus=[AxesArray(np.concatenate([xs[i],us[i]],axis=-1)) for i in range(3)]
    >>> lib.fit(xus)
    >>> lib.transform(xus)
    """

    def __init__(
        self,
        parameter_library: BaseFeatureLibrary = PolynomialLibrary(degree=1),
        feature_library: BaseFeatureLibrary = PolynomialLibrary(),
        num_parameters: int = 3,
        num_features: int = 3,
    ):
        if num_parameters <= 0 or num_features <= 0:
            raise ValueError("Both num_parameter and num_feature must be positive.")
        inputs_per_library = [
            range(num_features, num_parameters + num_features),
            range(num_features),
        ]

        super().__init__(
            libraries=[parameter_library, feature_library],
            tensor_array=[[1, 1]],
            exclude_libraries=[0, 1],
            inputs_per_library=inputs_per_library,
        )

    def calc_trajectory(self, diff_method, x, t):
        # if tensoring weak libraries, add the correction
        if hasattr(self.libraries[0], "K"):
            constants_final = np.ones(self.libraries[0].K)
            for k in range(self.libraries[0].K):
                constants_final[k] = np.sum(self.libraries[0].fullweights0[k])
            x, x_int = self.libraries[0].calc_trajectory(diff_method, x, t)
            return x, x_int * constants_final[:, np.newaxis]
        else:
            return self.libraries[0].calc_trajectory(diff_method, x, t)
