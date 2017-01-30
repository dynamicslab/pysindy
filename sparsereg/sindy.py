from sklearn.linear_model import ElasticNet, LinearRegression


class SINDy(ElasticNet):
    def __init__(self, knob=0.01, fit_intercept=False):
        self.knob = knob
        self.fit_intercept = fit_intercept

    def fit(self, x_, y_):
        coefs = LinearRegression(fit_intercept=self.fit_intercep).fit(x, y).coef_

        while True:
            pass
