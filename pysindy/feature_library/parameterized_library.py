import numpy as np

from .base import BaseFeatureLibrary
from .generalized_library import GeneralizedLibrary
from .polynomial_library import PolynomialLibrary


class ParameterizedLibrary(GeneralizedLibrary):
    """Construct a SINDyCP library to fit multiple trajectories with variable
    control parameters. The library is composed of a tensor product of a
    feature library, applied to the input data, and a parameter library,
    applied to the input control. If the input libraries are weak, the temporal
    derivatives are automatically rescaled by the appropriate domain volumes.

    Parameters
    ----------
    feature_library : BaseFeatureLibrary, optional (default PolynomialLibrary).
    Specifies the library function to apply to the input data features.

    parameter_library : BaseFeatureLibrary, optional (default PolynomialLibrary).
    Specifies the library function to apply to the input control features.

    num_features : int, optional (default 3)
    Specifies the number of features in the input data.

    num_parameters : int, optional (default 3)
    Specifies the number of features in the input control.

    Attributes
    ----------
    libraries_ : list of libraries
        Library instances to be applied to the input matrix.
        Equal to [parameter_library,feature_library].

    tensor_array_ : 2D list of booleans
        Indicates which pairs of libraries to tensor product together and
        add to the overall library. Equal to [0,1]

    inputs_per_library_ : 2D np.ndarray
        Can be used to specify a subset of the variables to use to generate
        a feature library. Value determined by num_parameters and num_features.

    n_input_features_ : int
        The total number of input features.
        WARNING: This is deprecated in scikit-learn version 1.0 and higher so
        we check the sklearn.__version__ and switch to n_features_in if needed.

    n_output_features_ : int
        The total number of output features. The number of output features
        is the sum of the numbers of output features for each of the
        concatenated libraries.

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
        parameter_library=PolynomialLibrary(degree=1, include_bias=True),
        feature_library=PolynomialLibrary(),
        num_parameters=3,
        num_features=3,
        library_ensemble=False,
        ensemble_indices=[0],
    ):
        if not isinstance(feature_library, BaseFeatureLibrary) or not isinstance(
            parameter_library, BaseFeatureLibrary
        ):
            raise ValueError(
                "Both feature_library and parameter_library must be instances of \
                 BaseFeatureLibrary."
            )

        if num_parameters <= 0 or num_features <= 0:
            raise ValueError("Both num_parameter and num_feature must be positive.")
        libraries = [parameter_library, feature_library]
        tensor_array = [[1, 1]]

        feature_input = np.zeros(num_features + num_parameters, dtype=np.int32)
        feature_input[:num_features] = np.arange(num_features)

        parameter_input = (
            np.ones(num_features + num_parameters, dtype=np.int32) * num_features
        )
        parameter_input[-num_parameters:] = num_features + np.arange(num_parameters)

        inputs_per_libraries = np.array([parameter_input, feature_input])

        super(ParameterizedLibrary, self).__init__(
            libraries,
            tensor_array=tensor_array,
            exclude_libraries=[0, 1],
            inputs_per_library=inputs_per_libraries,
            library_ensemble=library_ensemble,
            ensemble_indices=ensemble_indices,
        )

    def calc_trajectory(self, diff_method, x, t):
        # if tensoring weak libraries, add the correction
        if hasattr(self.libraries_[0], "K"):
            constants_final = np.ones(self.libraries_[0].K)
            for k in range(self.libraries_[0].K):
                constants_final[k] = np.sum(self.libraries_[0].fullweights0[k])
            return (
                self.libraries_[0].calc_trajectory(diff_method, x, t)
                * constants_final[:, np.newaxis]
            )
        else:
            return self.libraries_[0].calc_trajectory(diff_method, x, t)
