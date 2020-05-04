import warnings
from typing import Sequence

from numpy import concatenate
from numpy import isscalar
from numpy import ndim
from numpy import newaxis
from numpy import vstack
from numpy import zeros
from scipy.integrate import odeint
from scipy.linalg import LinAlgWarning
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_is_fitted

from pysindy.differentiation import FiniteDifference
from pysindy.optimizers import SINDyOptimizer
from pysindy.optimizers import STLSQ
from pysindy.utils.base import drop_nan_rows
from pysindy.utils.base import equations
from pysindy.utils.base import validate_control_variables
from pysindy.utils.base import validate_input


class SINDy(BaseEstimator):
    """
    SINDy model object.

    Parameters
    ----------
    optimizer : optimizer object, optional
        Optimization method used to fit the SINDy model. This must be an object
        extending the ``sindy.optimizers.BaseOptimizer`` class. Default is
        sequentially thresholded least squares with a threshold of 0.1.

    feature_library : feature library object, optional
        Feature library object used to specify candidate right-hand side features.
        This must be an object extending the
        ``sindy.feature_library.BaseFeatureLibrary`` class.
        Default is polynomial features of degree 2.

    differentiation_method : differentiation object, optional
        Method for differentiating the data. This must be an object extending
        the ``sindy.differentiation_methods.BaseDifferentiation`` class.
        Default is centered difference.

    feature_names : list of string, length n_input_features, optional
        Names for the input features (e.g. ``['x', 'y', 'z']``). If None, will use
        ``['x0', 'x1', ...]``.

    discrete_time : boolean, optional (default False)
        If True, dynamical system is treated as a map. Rather than predicting
        derivatives, the right hand side functions step the system forward by
        one time step. If False, dynamical system is assumed to be a flow
        (right-hand side functions predict continuous time derivatives).

    n_jobs : int, optional (default 1)
        The number of parallel jobs to use when fitting, predicting with, and
        scoring the model.

    Attributes
    ----------
    model : sklearn.multioutput.MultiOutputRegressor object
        The fitted SINDy model.

    n_input_features_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. This number is a function of
        ``self.n_input_features`` and the feature library being used.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> model = SINDy()
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -10.000 1 + 10.000 x0
    x1' = 27.993 1 + -0.999 x0 + -1.000 1 x1
    x2' = -2.666 x1 + 1.000 1 x0
    >>> model.coefficients()
    array([[ 0.        ,  0.        ,  0.        ],
           [-9.99969193, 27.99344519,  0.        ],
           [ 9.99961547, -0.99905338,  0.        ],
           [ 0.        ,  0.        , -2.66645651],
           [ 0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.99990257],
           [ 0.        , -0.99980268,  0.        ],
           [ 0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ]])
    >>> model.score(x, t=t[1]-t[0])
    0.999999985520653

    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> u = lambda t : np.sin(2 * t)
    >>> lorenz_c = lambda z,t : [
                10 * (z[1] - z[0]) + u(t) ** 2,
                z[0] * (28 - z[2]) - z[1],
                z[0] * z[1] - 8 / 3 * z[2],
        ]
    >>> t = np.arange(0,2,0.002)
    >>> x = odeint(lorenz_c, [-8,8,27], t)
    >>> u_eval = u(t)
    >>> model = SINDy()
    >>> model.fit(x, u_eval, t=t[1]-t[0])
    >>> model.print()
    x0' = -10.000 x0 + 10.000 x1 + 1.001 u0^2
    x1' = 27.994 x0 + -0.999 x1 + -1.000 x0 x2
    x2' = -2.666 x2 + 1.000 x0 x1
    >>> model.coefficients()
    array([[ 0.        , -9.99969851,  9.99958359,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  1.00120331],
           [ 0.        , 27.9935177 , -0.99906375,  0.        ,  0.        ,
             0.        ,  0.        , -0.99980455,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        , -2.666437  ,  0.        ,
             0.        ,  0.99990137,  0.        ,  0.        ,  0.        ,
             0.        ,  0.        ,  0.        ,  0.        ,  0.        ]])
    >>> model.score(x, u_eval, t=t[1]-t[0])
    0.9999999855414495
    """

    def __init__(
        self,
        optimizer=None,
        feature_library=None,
        differentiation_method=None,
        feature_names=None,
        discrete_time=False,
        n_jobs=1,
    ):
        if optimizer is None:
            optimizer = STLSQ()
        self.optimizer = optimizer
        if feature_library is None:
            feature_library = PolynomialFeatures()
        self.feature_library = feature_library
        if differentiation_method is None:
            differentiation_method = FiniteDifference()
        self.differentiation_method = differentiation_method
        self.feature_names = feature_names
        self.discrete_time = discrete_time
        self.n_jobs = n_jobs

    def fit(
        self,
        x,
        t=1,
        x_dot=None,
        u=None,
        multiple_trajectories=False,
        unbias=True,
        quiet=False,
    ):
        """
        Fit the SINDy model.

        Parameters
        ----------
        x: array-like or list of array-like, shape (n_samples, n_input_features)
            Training data. If training data contains multiple trajectories,
            x should be a list containing data for each trajectory. Individual
            trajectories may contain different numbers of samples.

        t: float, numpy array of shape [n_samples], or list of numpy arrays, optional \
                (default 1)
            If t is a float, it specifies the timestep between each sample.
            If array-like, it specifies the time at which each sample was
            collected.
            In this case the values in t must be strictly increasing.
            In the case of multi-trajectory training data, t may also be a list
            of arrays containing the collection times for each individual
            trajectory.
            Default value is a timestep of 1 between samples.

        x_dot: array-like or list of array-like, shape (n_samples, n_input_features), \
                optional (default None)
            Optional pre-computed derivatives of the training data. If not
            provided, the time derivatives of the training data will be
            computed using the specified differentiation method. If x_dot is
            provided, it must match the shape of the training data and these
            values will be used as the time derivatives.

        u: array-like or list of array-like, shape (n_samples, n_control_features), \
                optional (default None)
            Control variables/inputs. Include this variable to use sparse
            identification for nonlinear dynamical systems for control (SINDYc).
            If training data contains multiple trajectories (i.e. if x is a list of
            array-like), then u should be a list containing control variable data
            for each trajectory. Individual trajectories may contain different
            numbers of samples.

        multiple_trajectories: boolean, optional, (default False)
            Whether or not the training data includes multiple trajectories. If
            True, the training data must be a list of arrays containing data
            for each trajectory. If False, the training data must be a single
            array.

        unbias: boolean, optional (default True)
            Whether to perform an extra step of unregularized linear regression to
            unbias the coefficients for the identified support.
            If the optimizer (``SINDy.optimizer``) applies any type of regularization,
            that regularization may bias coefficients toward particular values,
            improving the conditioning of the problem but harming the quality of the
            fit. Setting ``unbias=True`` enables an extra step wherein unregularized
            linear regression is applied, but only for the coefficients in the support
            identified by the optimizer. This helps to remove the bias introduced by
            regularization.

        quiet: boolean, optional (default False)
            Whether or not to suppress warnings during model fitting.

        Returns
        -------
        self: returns an instance of self
        """
        if u is None:
            self.n_control_features_ = 0
        else:
            trim_last_point = self.discrete_time and (x_dot is None)
            u = validate_control_variables(
                x,
                u,
                multiple_trajectories=multiple_trajectories,
                trim_last_point=trim_last_point,
            )
            self.n_control_features_ = u.shape[1]

        if multiple_trajectories:
            x, x_dot = self.process_multiple_trajectories(x, t, x_dot)
        else:
            x = validate_input(x, t)

            if self.discrete_time:
                if x_dot is None:
                    x_dot = x[1:]
                    x = x[:-1]
                else:
                    x_dot = validate_input(x_dot)
            else:
                if x_dot is None:
                    x_dot = self.differentiation_method(x, t)
                else:
                    x_dot = validate_input(x_dot, t)

        # Append control variables
        if self.n_control_features_ > 0:
            x = concatenate((x, u), axis=1)

        # Drop rows where derivative isn't known
        x, x_dot = drop_nan_rows(x, x_dot)

        optimizer = SINDyOptimizer(self.optimizer, unbias=unbias)
        steps = [("features", self.feature_library), ("model", optimizer)]
        self.model = Pipeline(steps)

        action = "ignore" if quiet else "default"
        with warnings.catch_warnings():
            warnings.filterwarnings(action, category=ConvergenceWarning)
            warnings.filterwarnings(action, category=LinAlgWarning)
            warnings.filterwarnings(action, category=UserWarning)

            self.model.fit(x, x_dot)

        self.n_input_features_ = self.model.steps[0][1].n_input_features_
        self.n_output_features_ = self.model.steps[0][1].n_output_features_

        if self.feature_names is None:
            feature_names = []
            for i in range(self.n_input_features_ - self.n_control_features_):
                feature_names.append("x" + str(i))
            for i in range(self.n_control_features_):
                feature_names.append("u" + str(i))
            self.feature_names = feature_names

        return self

    def predict(self, x, u=None, multiple_trajectories=False):
        """
        Predict the time derivatives using the SINDy model.

        Parameters
        ----------
        x: array-like or list of array-like, shape (n_samples, n_input_features)
            Samples.

        u: array-like or list of array-like, shape(n_samples, n_control_features), \
                (default None)
            Control variables. If ``multiple_trajectories=True`` then u
            must be a list of control variable data from each trajectory. If the
            model was fit with control variables then u is not optional.

        multiple_trajectories: boolean, optional (default False)
            If True, x contains multiple trajectories and must be a list of
            data from each trajectory. If False, x is a single trajectory.

        Returns
        -------
        x_dot: array-like or list of array-like, shape (n_samples, n_input_features)
            Predicted time derivatives
        """
        check_is_fitted(self, "model")
        if u is None or self.n_control_features_ == 0:
            if self.n_control_features_ > 0:
                raise TypeError(
                    "Model was fit using control variables, so u is required"
                )
            elif u is not None:
                warnings.warn(
                    "Control variables u were ignored because control variables were"
                    " not used when the model was fit"
                )
            if multiple_trajectories:
                x = [validate_input(xi) for xi in x]
                return [self.model.predict(xi) for xi in x]
            else:
                x = validate_input(x)
                return self.model.predict(x)
        else:
            if multiple_trajectories:
                x = [validate_input(xi) for xi in x]
                u = validate_control_variables(
                    x, u, multiple_trajectories=True, return_array=False
                )
                return [
                    self.model.predict(concatenate((xi, ui), axis=1))
                    for xi, ui in zip(x, u)
                ]
            else:
                x = validate_input(x)
                u = validate_control_variables(x, u)
                return self.model.predict(concatenate((x, u), axis=1))

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
        check_is_fitted(self, "model")
        if self.discrete_time:
            base_feature_names = [f + "[k]" for f in self.feature_names]
        else:
            base_feature_names = self.feature_names
        return equations(
            self.model, input_features=base_feature_names, precision=precision
        )

    def print(self, lhs=None, precision=3):
        """Print the SINDy model equations.

        Parameters
        ----------
        lhs: list of strings, optional (default None)
            List of variables to print on the left-hand sides of the learned equations.

        precision: int, optional (default 3)
            Precision to be used when printing out model coefficients.
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
        u=None,
        multiple_trajectories=False,
        metric=r2_score,
        **metric_kws
    ):
        """
        Returns a score for the time derivative prediction.

        Parameters
        ----------
        x: array-like or list of array-like, shape (n_samples, n_input_features)
            Samples

        t: float, numpy array of shape [n_samples], or list of numpy arrays, optional \
                (default 1)
            Time step between samples or array of collection times. Optional,
            used to compute the time derivatives of the samples if x_dot is not
            provided.

        x_dot: array-like or list of array-like, shape (n_samples, n_input_features), \
                optional (default None)
            Optional pre-computed derivatives of the samples. If provided,
            these values will be used to compute the score. If not provided,
            the time derivatives of the training data will be computed using
            the specified differentiation method.

        u: array-like or list of array-like, shape(n_samples, n_control_features), \
                optional (default None)
            Control variables. If ``multiple_trajectories=True`` then u
            must be a list of control variable data from each trajectory.
            If the model was fit with control variables then u is not optional.

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
            Metric function value for the model prediction of x_dot.
        """
        if u is None or self.n_control_features_ == 0:
            if self.n_control_features_ > 0:
                raise TypeError(
                    "Model was fit using control variables, so u is required"
                )
            elif u is not None:
                warnings.warn(
                    "Control variables u were ignored because control variables were"
                    " not used when the model was fit"
                )
        else:
            trim_last_point = self.discrete_time and (x_dot is None)
            u = validate_control_variables(
                x,
                u,
                multiple_trajectories=multiple_trajectories,
                trim_last_point=trim_last_point,
            )

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

        # Append control variables
        if u is not None and self.n_control_features_ > 0:
            x = concatenate((x, u), axis=1)

        # Drop rows where derivative isn't known (usually endpoints)
        x, x_dot = drop_nan_rows(x, x_dot)

        x_dot_predict = self.model.predict(x)
        return metric(x_dot_predict, x_dot, **metric_kws)

    def process_multiple_trajectories(self, x, t, x_dot, return_array=True):
        """
        Handle input data that contains multiple trajectories by doing the
        necessary validation, reshaping, and computation of derivatives.
        """
        if not isinstance(x, Sequence):
            raise TypeError("Input x must be a list")

        if self.discrete_time:
            x = [validate_input(xi) for xi in x]
            if x_dot is None:
                x_dot = [xi[1:] for xi in x]
                x = [xi[:-1] for xi in x]
            else:
                if not isinstance(x_dot, Sequence):
                    raise TypeError(
                        "x_dot must be a list if used with x of list type "
                        "(i.e. for multiple trajectories)"
                    )
                x_dot = [validate_input(xd) for xd in x_dot]
        else:
            if x_dot is None:
                if isinstance(t, Sequence):
                    x = [validate_input(xi, ti) for xi, ti in zip(x, t)]
                    x_dot = [
                        self.differentiation_method(xi, ti) for xi, ti in zip(x, t)
                    ]
                else:
                    x = [validate_input(xi, t) for xi in x]
                    x_dot = [self.differentiation_method(xi, t) for xi in x]
            else:
                if not isinstance(x_dot, Sequence):
                    raise TypeError(
                        "x_dot must be a list if used with x of list type "
                        "(i.e. for multiple trajectories)"
                    )
                if isinstance(t, Sequence):
                    x = [validate_input(xi, ti) for xi, ti in zip(x, t)]
                    x_dot = [validate_input(xd, ti) for xd, ti in zip(x_dot, t)]
                else:
                    x = [validate_input(xi, t) for xi in x]
                    x_dot = [validate_input(xd, t) for xd in x_dot]

        if return_array:
            return vstack(x), vstack(x_dot)
        else:
            return x, x_dot

    def differentiate(self, x, t=1, multiple_trajectories=False):
        """
        Apply the model's differentiation method to data.

        Parameters
        ----------
        x: array-like or list of array-like, shape (n_samples, n_input_features)
            Data to be differentiated.

        t: int, numpy array of shape [n_samples], or list of numpy arrays, optional \
                (default 1)
            Time step between samples or array of collection times. Default is
            a time step of 1 between samples.

        multiple_trajectories: boolean, optional (default False)
            If True, x contains multiple trajectories and must be a list of
            data from each trajectory. If False, x is a single trajectory.

        Returns
        -------
        x_dot: array-like or list of array-like, shape (n_samples, n_input_features)
            Time derivatives computed by using the model's differentiation
            method
        """
        if self.discrete_time:
            raise RuntimeError("No differentiation implemented for discrete time model")

        if multiple_trajectories:
            return self.process_multiple_trajectories(x, t, None, return_array=False)[1]
        else:
            x = validate_input(x, t)
            return self.differentiation_method(x, t)

    def coefficients(self):
        """Return a list of the coefficients learned by SINDy model.
        """
        check_is_fitted(self, "model")
        return self.model.steps[-1][1].coef_

    def get_feature_names(self):
        """Return a list of names of features used by SINDy model.
        """
        check_is_fitted(self, "model")
        return self.model.steps[0][1].get_feature_names(
            input_features=self.feature_names
        )

    def simulate(
        self, x0, t, u=None, integrator=odeint, stop_condition=None, **integrator_kws
    ):
        """
        Simulate the SINDy model forward in time.

        Parameters
        ----------
        x0: numpy array, size [n_features]
            Initial condition from which to simulate.

        t: int or numpy array of size [n_samples]
            If the model is in continuous time, t must be an array of time
            points at which to simulate. If the model is in discrete time,
            t must be an integer indicating how many steps to predict.

        u: function from R^1 to R^{n_control_features} or list/array, optional \
            (default None)
            Control input function.
            If the model is continuous time, i.e. ``self.discrete_time == False``,
            this function should take in a time and output the values of each of
            the n_control_features control features as a list or numpy array.
            If the model is discrete time, i.e. ``self.discrete_time == True``,
            u should be a list (with ``len(u) == t``) or array (with
            ``u.shape[0] == 1``) giving the control inputs at each step.

        integrator: function object, optional
            Function to use to integrate the system. Default is scipy's odeint.

        stop_condition: function object, optional
            If model is in discrete time, optional function that gives a
            stopping condition for stepping the simulation forward.

        integrator_kws: dict, optional
            Optional keyword arguments to pass to the integrator

        Returns
        -------
        x: numpy array, shape (n_samples, n_features)
            Simulation results
        """
        check_is_fitted(self, "model")
        if u is None and self.n_control_features_ > 0:
            raise TypeError("Model was fit using control variables, so u is required")

        if self.discrete_time:
            if not isinstance(t, int):
                raise ValueError(
                    "For discrete time model, t must be an integer (indicating"
                    "the number of steps to predict)"
                )

            if stop_condition is not None:

                def check_stop_condition(xi):
                    return stop_condition(xi)

            else:

                def check_stop_condition(xi):
                    pass

            x = zeros((t, self.n_input_features_ - self.n_control_features_))
            x[0] = x0

            if u is None or self.n_control_features_ == 0:
                if u is not None:
                    warnings.warn(
                        "Control variables u were ignored because control variables"
                        " were not used when the model was fit"
                    )
                for i in range(1, t):
                    x[i] = self.predict(x[i - 1 : i])
                    if check_stop_condition(x[i]):
                        return x[: i + 1]
            else:
                for i in range(1, t):
                    x[i] = self.predict(x[i - 1 : i], u=u[i - 1])
                    if check_stop_condition(x[i]):
                        return x[: i + 1]
            return x
        else:
            if isscalar(t):
                raise ValueError(
                    "For continuous time model, t must be an array of time"
                    " points at which to simulate"
                )

            if u is None or self.n_control_features_ == 0:
                if u is not None:
                    warnings.warn(
                        "Control variables u were ignored because control variables"
                        " were not used when the model was fit"
                    )

                def rhs(x, t):
                    return self.predict(x[newaxis, :])[0]

            else:

                def rhs(x, t):
                    return self.predict(x[newaxis, :], u(t))[0]

            return integrator(rhs, x0, t, **integrator_kws)

    @property
    def complexity(self):
        return self.model.steps[-1][1].complexity
