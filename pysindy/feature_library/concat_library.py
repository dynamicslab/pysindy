import numpy as np

from .feature_library import BaseFeatureLibrary

class ConcatLibrary(BaseFeatureLibrary):

    def __init__(self, libraries: list):
        super(ConcatLibrary, self).__init__()

        self._libraries = libraries

    def fit(self, X, y=None):
        # first fit all libs below
        fitted_libs = [lib.fit(X, y) for lib in self._libraries]
        self.n_output_features_ = sum([lib.n_output_features_ for lib in fitted_libs])
        self._libraries = fitted_libs
        return self

    def transform(self, X):

        n_samples = X.shape[0]

        XP = np.zeros((n_samples, self.n_output_features_))

        current_feat = 0
        for lib in self._libraries:
            lib_n_output_features = lib.n_output_features_

            XP[:, current_feat:current_feat+lib_n_output_features] = lib.transform(X)

            current_feat += lib_n_output_features

        return XP

    def get_feature_names(self, input_features=None):
        feature_names = list()
        for lib in self._libraries:
            lib_feat_names = lib.get_feature_names(input_features)
            feature_names += lib_feat_names
        return feature_names
