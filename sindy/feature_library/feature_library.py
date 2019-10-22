"""
Base class for feature library classes.
"""
import abc

class BaseFeatureLibrary():
    """
    Functions that should eventually be implemented:
        -print/get names of features
        -evaluate all features (and store in member variable)
        -addition (concatenate lists of features)
        -
    """
    def __init__(self):
        pass

    # Some kind of function that applies the library
    def fit_transform(self, x):
        pass

    @property
    def size(self):
        return self._size
    

