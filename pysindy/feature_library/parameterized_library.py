import numpy as np

from .generalized_library import GeneralizedLibrary
from .polynomial_library import PolynomialLibrary


class ParameterizedLibrary(GeneralizedLibrary):
    """

    Parameters
    ----------
    parameter_library : BaseFeatureLibrary, optional (default PolynomialLibrary).

    data_library : BaseFeatureLibrary, optional (default PolynomialLibrary).

    num_features : int, optional (default 3)

    num_parameters : int, optional (default 3)

    Attributes
    ----------

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
