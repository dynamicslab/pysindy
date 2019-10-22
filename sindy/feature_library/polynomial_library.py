from sindy.feature_library import BaseFeatureLibrary

class PolynomialLibrary(BaseFeatureLibrary):
    """docstring for PolynomialLibrary"""
    def __init__(self, degree):
        super(PolynomialLibrary, self).__init__()
        self.degree = degree
        