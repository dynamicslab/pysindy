from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from scipy.integrate import odeint
from numpy import vstack, newaxis, zeros, isscalar, ndim

from pysindy.differentiation import FiniteDifference
from pysindy.optimizers import STLSQ
from pysindy.utils.base import equation, validate_input, drop_nan_rows


class SINDy(BaseEstimator):
    """
    SINDy model object.

    Parameters
    ----------
    optimizer : optimizer object, optional
        Optimization method used to fit the SINDy model. This must be an object
        that extends the sindy.optimizers.BaseOptimizer class. Default is
        sequentially thresholded least squares with a threshold of 0.1.

    feature_library : feature library object, optional
        Default is polynomial features of degree 2.
        TODO: Implement better feature library class.

    differentiation_method : differentiation object, optional
        Method for differentiating the data. This must be an object that
        extends the sindy.differentiation_methods.BaseDifferentiation class.
        Default is centered difference.

    feature_names : list of string, length n_input_features, optional
        Names for the input features. If None, will use ['x0','x1',...].

    discrete_time : boolean, optional (default False)
        If True, dynamical system is treated as a map. Rather than predicting
        derivatives, the right hand side functions step the system forward by
        one time step. If False, dynamical system is assumed to be a flow
        (right hand side functions predict continuous time derivatives).

    n_jobs : int, optional (default 1)
        The number of parallel jobs to use when fitting, predicting with, and
        scoring the model.

    Attributes
    ----------
    model : sklearn.multioutput.MultiOutputRegressor object
        The fitted SINDy model.
    """

    def __init__(
        self,
        optimizer=STLSQ(),
        feature_library=PolynomialFeatures(),
        differentiation_method=FiniteDifference(),
        feature_names=None,
        discrete_time=False,
        n_jobs=1,
    ):
        self.optimizer = optimizer
        self.feature_library = feature_library
        self.differentiation_method = differentiation_method
        self.feature_names = feature_names
        self.discrete_time = discrete_time
        self.n_jobs = n_jobs

    def fit(self, x, t=1, x_dot=None, multiple_trajectories=False):
        """
        Fit the SINDy model.

        Parameters
        ----------
        x: array-like or list of array-like, shape
        (n_samples, n_input_features)
            Training data. If training data contains multiple trajectories,
            x should be a list containing data for each trajectory. Individual
            trajectories may contain different numbers of samples.

        t: float, numpy array of shape [n_samples], or list of numpy arrays,
        optional (default 1)
            If t is a float, it specifies the timestep between each sample.
            If array-like, it specifies the time at which each sample was
            collected.
            In this case the values in t must be strictly increasing.
            In the case of multi-trajectory training data, t may also be a list
            of arrays containing the collection times for each individual
            trajectory.
            Default value is a timestep of 1 between samples.

        x_dot: array-like or list of array-like, shape (n_samples,
        n_input_features), optional (default None)
            Optional pre-computed derivatives of the training data. If not
            provided, the time derivatives of the training data will be
            computed using the specified differentiation method. If x_dot is
            provided, it must match the shape of the training data and these
            values will be used as the time derivatives.

        multiple_trajectories: boolean, optional, (default False)
            Whether or not the training data includes multiple trajectories. If
            True, the training data must be a list of arrays containing data
            for each trajectory. If False, the training data must be a single
            array.

        Returns
        -------
        self: returns an instance of self
        """
        if multiple_trajectories:
            x, x_dot = self.process_multiple_trajectories(x, t, x_dot)
        else:
            x = validate_input(x, t)

            if self.discrete_time:
                if x_dot is None:
                    x_dot = x[1:]
                    x = x[:-1]
                else:
                    x_dot = validate_input(x)
            else:
                if x_dot is None:
                    x_dot = self.differentiation_method(x, t)
                else:
                    x_dot = validate_input(x_dot, t)

        # Drop rows where derivative isn't known
        x, x_dot = drop_nan_rows(x, x_dot)

        steps = [("features", self.feature_library), ("model", self.optimizer)]
        self.model = MultiOutputRegressor(Pipeline(steps), n_jobs=self.n_jobs)

        self.model.fit(x, x_dot)

        self.n_input_features_ = (
            self.model.estimators_[0].steps[0][1].n_input_features_
        )
        self.n_output_features_ = (
            self.model.estimators_[0].steps[0][1].n_output_features_
        )

        if self.feature_names is None:
            feature_names = []
            for i in range(self.n_input_features_):
                feature_names.append("x" + str(i))
            self.feature_names = feature_names

        return self

    def predict(self, x, multiple_trajectories=False):
        """
        Predict the time derivatives using the SINDy model.

        Parameters
        ----------
        x: array-like or list of array-like, shape
        (n_samples, n_input_features)
            Samples

        multiple_trajectories: boolean, optional (default False)
            If True, x contains multiple trajectories and must be a list of
            data from each trajectory. If False, x is a single trajectory.

        Returns
        -------
        x_dot: array-like or list of array-like, shape
        (n_samples, n_input_features)
            Predicted time derivatives
        """
        if hasattr(self, "model"):
            if multiple_trajectories:
                x = [validate_input(xi) for xi in x]
                return [self.model.predict(xi) for xi in x]
            else:
                x = validate_input(x)
                if hasattr(self, "model"):
                    return self.model.predict(x)
        else:
            raise NotFittedError(
                "SINDy model must be fit before predict can be called"
            )

    def equations(self, precision=3):
        """
        Get the right hand sides of the SINDy model equations.

        Parameters
        ----------
        precision: int, optional (default 3)
            Number of decimal points to print for each coefficient in the
            equation.

        Returns
        -------
        equations: list of strings
            Strings containing the SINDy model equation for each input feature.
        """
        if hasattr(self, "model"):
            check_is_fitted(self.model.estimators_[0].steps[-1][1])
            if self.discrete_time:
                base_feature_names = [f + "[k]" for f in self.feature_names]
            else:
                base_feature_names = self.feature_names
            feature_names = (
                self.model.estimators_[0]
                .steps[0][1]
                .get_feature_names(input_features=base_feature_names)
            )
            return [
                equation(
                    est, input_features=feature_names, precision=precision
                )
                for est in self.model.estimators_
            ]
        else:
            raise NotFittedError(
                "SINDy model must be fit before equations can be called"
            )

    def print(self, lhs=None, precision=3):
        """Print the SINDy model equations.
        """
        eqns = self.equations(precision)
        for i, eqn in enumerate(eqns):
            if self.discrete_time:
                print(self.feature_names[i] + "[k+1] = " + eqn)
            elif lhs is None:
                print(self.feature_names[i] + "' = " + eqn)
            else:
                print(lhs[i] + " = " + eqn)

    def score(
        self,
        x,
        t=1,
        x_dot=None,
        multiple_trajectories=False,
        metric=r2_score,
        **metric_kws
    ):
        """
        Returns a score for the time derivative prediction.

        Parameters
        ----------
        x: array-like or list of array-like, shape
        (n_samples, n_input_features)
            Samples

        t: float, numpy array of shape [n_samples], or list of numpy arrays,
        optional
            Time step between samples or array of collection times. Optional,
            used to compute the time derivatives of the samples if x_dot is not
            provided.

        x_dot: array-like or list of array-like, shape
        (n_samples, n_input_features), optional
            Optional pre-computed derivatives of the samples. If provided,
            these values will be used to compute the score. If not provided,
            the time derivatives of the training data will be computed using
            the specified differentiation method.

        multiple_trajectories: boolean, optional (default False)
            If True, x contains multiple trajectories and must be a list of
            data from each trajectory. If False, x is a single trajectory.

        metric: metric function, optional
            Metric function with which to score the prediction. Default is the
            coefficient of determination R^2.

        metric_kws: dict, optional
            Optional keyword arguments to pass to the metric function.

        Returns
        -------
        score: float
            Metric function value for the model prediction of x_dot
        """
        if multiple_trajectories:
            x, x_dot = self.process_multiple_trajectories(
                x, t, x_dot, return_array=True
            )
        else:
            x = validate_input(x, t)
            if x_dot is None:
                if self.discrete_time:
                    x_dot = x[1:]
                    x = x[:-1]
                else:
                    x_dot = self.differentiation_method(x, t)

        if ndim(x_dot) == 1:
            x_dot = x_dot.reshape(-1, 1)

        # Drop rows where derivative isn't known (usually endpoints)
        x, x_dot = drop_nan_rows(x, x_dot)

        x_dot_predict = self.model.predict(x)
        return metric(x_dot_predict, x_dot, **metric_kws)

    def process_multiple_trajectories(self, x, t, x_dot, return_array=True):
        """
        Handle input data that contains multiple trajectories by doing the
        necessary validation, reshaping, and computation of derivatives.
        """
        if not isinstance(x, list):
            raise TypeError("Input x must be a list")

        if self.discrete_time:
            if x_dot is None:
                x_dot = []
                for i in range(len(x)):
                    x_tmp = validate_input(x[i])
                    x[i] = x_tmp[:-1]
                    x_dot.append(x_tmp[1:])
            else:
                if not isinstance(x_dot, list):
                    raise ValueError(
                        "x_dot must be a list if used with x of list type "
                        "(i.e. for multiple trajectories)"
                    )
                x_dot = [validate_input(xd) for xd in x_dot]
        else:
            if x_dot is None:
                if isinstance(t, list):
                    x_dot = []
                    for i in range(len(x)):
                        x[i] = validate_input(x[i], t[i])
                        x_dot.append(self.differentiation_method(x[i], t[i]))
                else:
                    x_dot = []
                    for i in range(len(x)):
                        x[i] = validate_input(x[i], t)
                        x_dot.append(self.differentiation_method(x[i], t))
            else:
                if not isinstance(x_dot, list):
                    raise ValueError(
                        "x_dot must be a list if used with x of list type "
                        "(i.e. for multiple trajectories)"
                    )
                if isinstance(t, list):
                    x_dot = [validate_input(xd, t) for xd, t in zip(x_dot, t)]
                else:
                    x_dot = [validate_input(xd, t) for xd in x_dot]

        if return_array:
            return vstack(x), vstack(x_dot)
        else:
            return x, x_dot

    def differentiate(self, x, t=1, multiple_trajectories=False):
        """
        Apply the model's differentiation method to data

        Parameters
        ----------
        x: array-like or list of array-like, shape
        (n_samples, n_input_features)
            Samples

        t: int, numpy array of shape [n_samples], or list of numpy arrays,
        optional
            Time step between samples or array of collection times. Default is
            a time step of 1 between samples.

        multiple_trajectories: boolean, optional (default False)
            If True, x contains multiple trajectories and must be a list of
            data from each trajectory. If False, x is a single trajectory.

        Returns
        -------
        x_dot: array-like or list of array-like, shape
        (n_samples, n_input_features)
            Time derivatives computed by using the model's differentiation
            method
        """
        if self.discrete_time:
            raise RuntimeError(
                "No differentiation implemented for discrete time model"
            )

        if multiple_trajectories:
            return self.process_multiple_trajectories(
                x, t, None, return_array=False
            )[1]
        else:
            x = validate_input(x, t)
            return self.differentiation_method(x, t)

    def coefficients(self):
        """Return a list of the coefficients learned by SINDy model
        """
        if hasattr(self, "model"):
            check_is_fitted(self.model.estimators_[0].steps[-1][1])
            return vstack([est.steps[-1][1].coef_ for est in self.model.estimators_]).T
        else:
            raise NotFittedError(
                "SINDy model must be fit before coefficients is called"
            )

    def get_feature_names(self):
        """Return a list of names of features used by SINDy model
        """
        if hasattr(self, "model"):
            return (
                self.model.estimators_[0]
                .steps[0][1]
                .get_feature_names(input_features=self.feature_names)
            )
        else:
            raise NotFittedError(
                "SINDy model must be fit before get_feature_names is called"
            )

    def simulate(
        self, x0, t, integrator=odeint, stop_condition=None, **integrator_kws
    ):
        """
        Simulate the SINDy model forward in time.

        Parameters
        ----------
        x0: numpy array, size [n_features]
            Initial condition from which to simulate

        t: int or numpy array of size [n_samples]
            If the model is in continuous time, t must be an array of time
            points at which to simulate. If the model is in discrete time,
            t must be an integer indicating how many steps to predict.

        integrator: function object, optional
            Function to use to integrate the system. Default is scipy's odeint.

        stop_condition: function object, optional
            If model is in discrete time, optional function that gives a
            stopping condition for stepping the simulation forward.

        integrator_kws: dict, optional
            Optional keyword arguments to pass to the integrator

        Returns
        -------
        x: numpy array, size (n_samples, n_features)
            Simulation results
        """

        if self.discrete_time:
            if not isinstance(t, int):
                raise ValueError(
                    "For discrete time model, t must be an integer (indicating"
                    "the number of steps to predict)"
                )

            x = zeros((t, self.n_input_features_))
            x[0] = x0
            for i in range(1, t):
                x[i] = self.predict(x[i - 1: i])
                if stop_condition is not None and stop_condition(x[i]):
                    return x[: i + 1]
            return x
        else:
            if isscalar(t):
                raise ValueError(
                    "For continuous time model, t must be an array of time"
                    " points at which to simulate"
                )

            def rhs(x, t):
                return self.predict(x[newaxis, :])[0]

            return integrator(rhs, x0, t, **integrator_kws)

    @property
    def complexity(self):
        return sum(
            est.steps[1][1].complexity for est in self.model.estimators_
        )
