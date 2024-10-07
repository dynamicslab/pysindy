import sys
import warnings
from abc import ABC
from abc import abstractmethod
from itertools import product
from typing import Collection
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Self

from .differentiation import BaseDifferentiation
from .differentiation import FiniteDifference
from .feature_library import PolynomialLibrary
from .feature_library.base import BaseFeatureLibrary

try:  # Waiting on PEP 690 to lazy import CVXPY
    from .optimizers import SINDyPI

    sindy_pi_flag = True
except ImportError:
    sindy_pi_flag = False
from .optimizers import STLSQ
from .optimizers.base import _BaseOptimizer
from .optimizers.base import BaseOptimizer
from .utils import AxesArray
from .utils import comprehend_axes
from .utils import concat_sample_axis
from .utils import drop_nan_samples
from .utils import SampleConcatter
from .utils import validate_control_variables
from .utils import validate_input
from .utils import validate_no_reshape


class _BaseSINDy(BaseEstimator, ABC):

    feature_library: BaseFeatureLibrary
    optimizer: _BaseOptimizer
    discrete_time: bool
    model: Pipeline
    feature_names: Optional[list[str]]
    # Hacks to remove later
    discrete_time: bool = False
    n_control_features_: int = 0

    @abstractmethod
    def fit(self, x, t, *args, **kwargs) -> Self:
        ...

    def _fit_shape(self):
        """Assign shape attributes for the system that are used post-fit"""
        self.n_features_in_ = self.feature_library.n_features_in_
        self.n_output_features_ = self.feature_library.n_output_features_
        if self.feature_names is None:
            feature_names = []
            for i in range(self.n_features_in_ - self.n_control_features_):
                feature_names.append("x" + str(i))
            for i in range(self.n_control_features_):
                feature_names.append("u" + str(i))
            self.feature_names = feature_names

    def equations(self, precision: int = 3) -> list[str]:
        """
        Get the right hand sides of the SINDy model equations.

        Parameters
        ----------
        precision: int, optional (default 3)
            Number of decimal points to include for each coefficient in the
            equation.

        Returns
        -------
        equations: list of strings
            List of strings representing the SINDy model equations for each
            input feature.
        """
        check_is_fitted(self, "model")
        if self.discrete_time:
            sys_coord_names = [name + "[k]" for name in self.feature_names]
        else:
            sys_coord_names = self.feature_names
        feat_names = self.feature_library.get_feature_names(sys_coord_names)

        def term(c, name):
            rounded_coef = np.round(c, precision)
            if rounded_coef == 0:
                return ""
            else:
                return f"{c:.{precision}f} {name}"

        equations = []
        for coef_row in self.optimizer.coef_:
            components = [term(c, i) for c, i in zip(coef_row, feat_names)]
            eq = " + ".join(filter(bool, components))
            if not eq:
                eq = f"{0:.{precision}f}"
            equations.append(eq)

        return equations

    def print(self, precision: int = 3, **kwargs) -> None:
        """Print the SINDy model equations.

        Parameters
        ----------
        lhs: list of strings, optional (default None)
            List of variables to print on the left-hand sides of the learned equations.
            By default :code:`self.input_features` are used.

        precision: int, optional (default 3)
            Precision to be used when printing out model coefficients.

        **kwargs: Additional keyword arguments passed to the builtin print function
        """
        eqns = self.equations(precision)
        for name, eqn in zip(self.feature_names, eqns, strict=True):
            lhs = f"({name})'"
            print(f"{lhs} = {eqn}", **kwargs)

    def get_feature_names(self):
        """
        Get a list of names of features used by SINDy model.

        Returns
        -------
        feats: list
            A list of strings giving the names of the features in the feature
            library, :code:`self.feature_library`.
        """
        check_is_fitted(self, "model")
        return self.feature_library.get_feature_names(input_features=self.feature_names)


class SINDy(_BaseSINDy):
    """
    Sparse Identification of Nonlinear Dynamical Systems (SINDy).
    Uses sparse regression to learn a dynamical systems model from measurement data.

    Parameters
    ----------
    optimizer : optimizer object, optional
        Optimization method used to fit the SINDy model. This must be a class
        extending :class:`pysindy.optimizers.BaseOptimizer`.
        The default is :class:`STLSQ`.

    feature_library : feature library object, optional
        Feature library object used to specify candidate right-hand side features.
        This must be a class extending
        :class:`pysindy.feature_library.base.BaseFeatureLibrary`.
        The default option is :class:`PolynomialLibrary`.

    differentiation_method : differentiation object, optional
        Method for differentiating the data. This must be a class extending
        :class:`pysindy.differentiation_methods.base.BaseDifferentiation` class.
        The default option is centered difference.

    feature_names : list of string, length n_input_features, optional
        Names for the input features (e.g. ``['x', 'y', 'z']``). If None, will use
        ``['x0', 'x1', ...]``.

    t_default : float, optional (default 1)
        Default value for the time step.

    discrete_time : boolean, optional (default False)
        If True, dynamical system is treated as a map. Rather than predicting
        derivatives, the right hand side functions step the system forward by
        one time step. If False, dynamical system is assumed to be a flow
        (right-hand side functions predict continuous time derivatives).

    Attributes
    ----------
    model : ``sklearn.multioutput.MultiOutputRegressor`` object
        The fitted SINDy model.

    n_input_features_ : int
        The total number of input features.

    n_output_features_ : int
        The total number of output features. This number is a function of
        ``self.n_input_features`` and the feature library being used.

    n_control_features_ : int
        The total number of control input features.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import solve_ivp
    >>> from pysindy import SINDy
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = solve_ivp(lorenz, [-8,8,27], t)
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
    >>> from scipy.integrate import solve_ivp
    >>> from pysindy import SINDy
    >>> u = lambda t : np.sin(2 * t)
    >>> lorenz_c = lambda z,t : [
                10 * (z[1] - z[0]) + u(t) ** 2,
                z[0] * (28 - z[2]) - z[1],
                z[0] * z[1] - 8 / 3 * z[2],
        ]
    >>> t = np.arange(0,2,0.002)
    >>> x = solve_ivp(lorenz_c, [-8,8,27], t)
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
        optimizer: Optional[BaseOptimizer] = None,
        feature_library: Optional[BaseFeatureLibrary] = None,
        differentiation_method: Optional[BaseDifferentiation] = None,
        feature_names: Optional[list[str]] = None,
        t_default: float = 1,
        discrete_time: bool = False,
    ):
        if optimizer is None:
            optimizer = STLSQ()
        self.optimizer = optimizer
        if feature_library is None:
            feature_library = PolynomialLibrary()
        self.feature_library = feature_library
        if differentiation_method is None:
            differentiation_method = FiniteDifference(axis=-2)
        self.differentiation_method = differentiation_method
        if not isinstance(t_default, float) and not isinstance(t_default, int):
            raise ValueError("t_default must be a positive number")
        elif t_default <= 0:
            raise ValueError("t_default must be a positive number")
        else:
            self.t_default = t_default
        self.feature_names = feature_names
        self.discrete_time = discrete_time

    def fit(
        self,
        x,
        t=None,
        x_dot=None,
        u=None,
    ):
        """
        Fit a SINDy model.

        Parameters
        ----------
        x: array-like or list of array-like, shape (n_samples, n_input_features)
            Training data. If training data contains multiple trajectories,
            x should be a list containing data for each trajectory. Individual
            trajectories may contain different numbers of samples.

        t: float, numpy array of shape (n_samples,), or list of numpy arrays, optional \
                (default None)
            If t is a float, it specifies the timestep between each sample.
            If array-like, it specifies the time at which each sample was
            collected.
            In this case the values in t must be strictly increasing.
            In the case of multi-trajectory training data, t may also be a list
            of arrays containing the collection times for each individual
            trajectory.
            If None, the default time step ``t_default`` will be used.

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

        Returns
        -------
        self: a fitted :class:`SINDy` instance
        """

        if t is None:
            t = self.t_default

        if not _check_multiple_trajectories(x, x_dot, u):
            x, t, x_dot, u = _adapt_to_multiple_trajectories(x, t, x_dot, u)
        x, x_dot, u = _comprehend_and_validate_inputs(
            x, t, x_dot, u, self.feature_library
        )

        if u is None:
            self.n_control_features_ = 0
        else:
            u = validate_control_variables(
                x,
                u,
                trim_last_point=(self.discrete_time and x_dot is None),
            )
            self.n_control_features_ = u[0].shape[u[0].ax_coord]
        x, x_dot = self._process_trajectories(x, t, x_dot)

        # Append control variables
        if u is not None:
            x = [np.concatenate((xi, ui), axis=xi.ax_coord) for xi, ui in zip(x, u)]

        steps = [
            ("features", self.feature_library),
            ("shaping", SampleConcatter()),
            ("model", self.optimizer),
        ]
        x_dot = concat_sample_axis(x_dot)
        self.model = Pipeline(steps)
        self.model.fit(x, x_dot)
        self._fit_shape()

        return self

    def predict(self, x, u=None):
        """
        Predict the time derivatives using the SINDy model.

        Parameters
        ----------
        x: array-like or list of array-like, shape (n_samples, n_input_features)
            Samples.

        u: array-like or list of array-like, shape(n_samples, n_control_features), \
                (default None)
            Control variables. If ``multiple_trajectories==True`` then u
            must be a list of control variable data from each trajectory. If the
            model was fit with control variables then u is not optional.

        Returns
        -------
        x_dot: array-like or list of array-like, shape (n_samples, n_input_features)
            Predicted time derivatives
        """
        if not _check_multiple_trajectories(x, None, u):
            x, _, _, u = _adapt_to_multiple_trajectories(x, None, None, u)
            multiple_trajectories = False
        else:
            multiple_trajectories = True

        x, _, u = _comprehend_and_validate_inputs(x, 1, None, u, self.feature_library)

        check_is_fitted(self, "model")
        if self.n_control_features_ > 0 and u is None:
            raise TypeError("Model was fit using control variables, so u is required")
        if self.n_control_features_ == 0 and u is not None:
            warnings.warn(
                "Control variables u were ignored because control variables were"
                " not used when the model was fit"
            )
            u = None
        if self.discrete_time:
            x = [validate_input(xi) for xi in x]
        if u is not None:
            u = validate_control_variables(x, u)
            x = [np.concatenate((xi, ui), axis=xi.ax_coord) for xi, ui in zip(x, u)]
        result = [self.model.predict([xi]) for xi in x]
        result = [
            self.feature_library.reshape_samples_to_spatial_grid(pred)
            for pred in result
        ]

        # Kept for backwards compatibility.
        if not multiple_trajectories:
            return result[0]
        return result

    def print(self, lhs=None, precision=3, **kwargs):
        """Print the SINDy model equations.

        Parameters
        ----------
        lhs: list of strings, optional (default None)
            List of variables to print on the left-hand sides of the learned equations.
            By default :code:`self.input_features` are used.

        precision: int, optional (default 3)
            Precision to be used when printing out model coefficients.

        **kwargs: Additional keyword arguments passed to the builtin print function
        """
        eqns = self.equations(precision)
        if sindy_pi_flag and isinstance(self.optimizer, SINDyPI):
            feature_names = self.get_feature_names()
        else:
            feature_names = self.feature_names
        for i, eqn in enumerate(eqns):
            if self.discrete_time:
                names = f"({feature_names[i]})[k+1]"
            elif lhs is None:
                if not sindy_pi_flag or not isinstance(self.optimizer, SINDyPI):
                    names = f"({feature_names[i]})'"
                else:
                    names = f"({feature_names[i]})"
            else:
                names = f"{lhs[i]}"
            print(f"{names} = {eqn}", **kwargs)

    def score(self, x, t=None, x_dot=None, u=None, metric=r2_score, **metric_kws):
        """
        Returns a score for the time derivative prediction produced by the model.

        Parameters
        ----------
        x: array-like or list of array-like, shape (n_samples, n_input_features)
            Samples from which to make predictions.

        t: float, numpy array of shape (n_samples,), or list of numpy arrays, optional \
                (default None)
            Time step between samples or array of collection times. Optional,
            used to compute the time derivatives of the samples if x_dot is not
            provided.
            If None, the default time step ``t_default`` will be used.

        x_dot: array-like or list of array-like, shape (n_samples, n_input_features), \
                optional (default None)
            Optional pre-computed derivatives of the samples. If provided,
            these values will be used to compute the score. If not provided,
            the time derivatives of the training data will be computed using
            the specified differentiation method.

        u: array-like or list of array-like, shape(n_samples, n_control_features), \
                optional (default None)
            Control variables. If ``multiple_trajectories==True`` then u
            must be a list of control variable data from each trajectory.
            If the model was fit with control variables then u is not optional.

        metric: callable, optional
            Metric function with which to score the prediction. Default is the
            R^2 coefficient of determination.
            See `Scikit-learn \
            <https://scikit-learn.org/stable/modules/model_evaluation.html>`_
            for more options.

        metric_kws: dict, optional
            Optional keyword arguments to pass to the metric function.

        Returns
        -------
        score: float
            Metric function value for the model prediction of x_dot.
        """

        if t is None:
            t = self.t_default

        if not _check_multiple_trajectories(x, x_dot, u):
            x, t, x_dot, u = _adapt_to_multiple_trajectories(x, t, x_dot, u)
        x, x_dot, u = _comprehend_and_validate_inputs(
            x, t, x_dot, u, self.feature_library
        )

        x_dot_predict = self.predict(x, u)

        if self.discrete_time and x_dot is None:
            x_dot_predict = [xd[:-1] for xd in x_dot_predict]

        x, x_dot = self._process_trajectories(x, t, x_dot)

        x_dot = concat_sample_axis(x_dot)
        x_dot_predict = concat_sample_axis(x_dot_predict)

        x_dot, x_dot_predict = drop_nan_samples(x_dot, x_dot_predict)
        return metric(x_dot, x_dot_predict, **metric_kws)

    def _process_trajectories(self, x, t, x_dot):
        """
        Calculate derivatives of input data, iterating through trajectories.

        Parameters
        ----------
        x: list of np.ndarray
            List of measurements, with each entry corresponding to a different
            trajectory.

        t: list of np.ndarray or int
            List of time points for different trajectories.  If a list of ints
            is passed, each entry is assumed to be the timestep for the
            corresponding trajectory in x.  If np.ndarray is passed, it is
            used for each trajectory.

        x_dot: list of np.ndarray
            List of derivative measurements, with each entry corresponding to a
            different trajectory. If None, the derivatives will be approximated
            from x.

        Returns
        -------
        x_out: np.ndarray or list
            Validated version of x. If return_array is True, x_out will be an
            np.ndarray of concatenated trajectories. If False, x_out will be
            a list.

        x_dot_out: np.ndarray or list
            Validated derivative measurements.If return_array is True, x_dot_out
            will be an np.ndarray of concatenated trajectories.
            If False, x_out will be a list.
        """
        if x_dot is None:
            if self.discrete_time:
                x_dot = [xi[1:] for xi in x]
                x = [xi[:-1] for xi in x]
            else:
                x, x_dot = zip(
                    *[
                        self.feature_library.calc_trajectory(
                            self.differentiation_method, xi, ti
                        )
                        for xi, ti in _zip_like_sequence(x, t)
                    ]
                )
        return x, x_dot

    def differentiate(self, x, t=None):
        """
        Apply the model's differentiation method
        (:code:`self.differentiation_method`) to data.

        Parameters
        ----------
        x: array-like or list of array-like, shape (n_samples, n_input_features)
            Data to be differentiated.

        t: int, numpy array of shape (n_samples,), or list of numpy arrays, optional \
                (default None)
            Time step between samples or array of collection times.
            If None, the default time step ``t_default`` will be used.

        Returns
        -------
        x_dot: array-like or list of array-like, shape (n_samples, n_input_features)
            Time derivatives computed by using the model's differentiation
            method
        """
        warnings.warn(
            "SINDy.differentiate is deprecated.  "
            "Call the differentiation_method parameter"
        )
        if t is None:
            t = self.t_default
        if self.discrete_time:
            raise RuntimeError("No differentiation implemented for discrete time model")
        if not _check_multiple_trajectories(x, None, None):
            x, t, _, _ = _adapt_to_multiple_trajectories(x, t, None, None)
            multiple_trajectories = False
        else:
            multiple_trajectories = True
        x, _, _ = _comprehend_and_validate_inputs(
            x, t, None, None, self.feature_library
        )
        result = self._process_trajectories(x, t, None)[1]
        if not multiple_trajectories:
            return result[0]
        return result

    def coefficients(self):
        """
        Get an array of the coefficients learned by SINDy model.

        Returns
        -------
        coef: np.ndarray, shape (n_input_features, n_output_features)
            Learned coefficients of the SINDy model.
            Equivalent to :math:`\\Xi^\\top` in the literature.
        """
        check_is_fitted(self, "model")
        return self.optimizer.coef_

    def simulate(
        self,
        x0,
        t,
        u=None,
        integrator="solve_ivp",
        stop_condition=None,
        interpolator=None,
        integrator_kws={"method": "LSODA", "rtol": 1e-12, "atol": 1e-12},
        interpolator_kws={},
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
            Control inputs.
            If the model is continuous time, i.e. ``self.discrete_time == False``,
            this function should take in a time and output the values of each of
            the n_control_features control features as a list or numpy array.
            Alternatively, if the model is continuous time, ``u`` can also be an
            array of control inputs at each time step. In this case the array is
            fit with the interpolator specified by ``interpolator``.
            If the model is discrete time, i.e. ``self.discrete_time == True``,
            u should be a list (with ``len(u) == t``) or array (with
            ``u.shape[0] == 1``) giving the control inputs at each step.

        integrator: string, optional (default ``solve_ivp``)
            Function to use to integrate the system.
            Default is ``scipy.integrate.solve_ivp``. The only options
            currently supported are solve_ivp and odeint.

        stop_condition: function object, optional
            If model is in discrete time, optional function that gives a
            stopping condition for stepping the simulation forward.

        interpolator: callable, optional (default ``interp1d``)
            Function used to interpolate control inputs if ``u`` is an array.
            Default is ``scipy.interpolate.interp1d``.

        integrator_kws: dict, optional (default {})
            Optional keyword arguments to pass to the integrator

        interpolator_kws: dict, optional (default {})
            Optional keyword arguments to pass to the control input interpolator

        Returns
        -------
        x: numpy array, shape (n_samples, n_features)
            Simulation results
        """
        check_is_fitted(self, "model")
        if u is None and self.n_control_features_ > 0:
            raise TypeError("Model was fit using control variables, so u is required")

        if self.discrete_time:
            if not isinstance(t, int) or t <= 0:
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

            x = np.zeros((t, self.n_features_in_ - self.n_control_features_))
            x[0] = x0

            if u is None or self.n_control_features_ == 0:
                if u is not None:
                    warnings.warn(
                        "Control variables u were ignored because control "
                        "variables were not used when the model was fit"
                    )
                for i in range(1, t):
                    x[i] = self.predict(x[i - 1 : i])
                    if check_stop_condition(x[i]):
                        return x[: i + 1]
            else:
                for i in range(1, t):
                    x[i] = self.predict(x[i - 1 : i], u=u[i - 1, np.newaxis])
                    if check_stop_condition(x[i]):
                        return x[: i + 1]
            return x
        else:
            if np.isscalar(t):
                raise ValueError(
                    "For continuous time model, t must be an array of time"
                    " points at which to simulate"
                )

            if u is None or self.n_control_features_ == 0:
                if u is not None:
                    warnings.warn(
                        "Control variables u were ignored because control "
                        "variables were not used when the model was fit"
                    )

                def rhs(t, x):
                    return self.predict(x[np.newaxis, :])[0]

            else:
                if not callable(u):
                    if interpolator is None:
                        u_fun = interp1d(
                            t, u, axis=0, kind="cubic", fill_value="extrapolate"
                        )
                    else:
                        u_fun = interpolator(t, u, **interpolator_kws)

                    t = t[:-1]
                    warnings.warn(
                        "Last time point dropped in simulation because "
                        "interpolation of control input was used. To avoid "
                        "this, pass in a callable for u."
                    )
                else:
                    u_fun = u

                if u_fun(t[0]).ndim == 1:

                    def rhs(t, x):
                        return self.predict(x[np.newaxis, :], u_fun(t).reshape(1, -1))[
                            0
                        ]

                else:

                    def rhs(t, x):
                        return self.predict(x[np.newaxis, :], u_fun(t))[0]

            # Need to hard-code below, because odeint and solve_ivp
            # have different syntax and integration options.
            if integrator == "solve_ivp":
                return (
                    (solve_ivp(rhs, (t[0], t[-1]), x0, t_eval=t, **integrator_kws)).y
                ).T
            elif integrator == "odeint":
                if integrator_kws.get("method") == "LSODA":
                    integrator_kws = {}
                return odeint(rhs, x0, t, tfirst=True, **integrator_kws)
            else:
                raise ValueError("Integrator not supported, exiting")

    @property
    def complexity(self):
        """
        Complexity of the model measured as the number of nonzero parameters.
        """
        return self.optimizer.complexity


def _zip_like_sequence(x, t):
    """Create an iterable like zip(x, t), but works if t is scalar."""
    if isinstance(t, Sequence):
        return zip(x, t)
    else:
        return product(x, [t])


def _check_multiple_trajectories(x, x_dot, u) -> bool:
    """Determine if data contains multiple trajectories

    Args:
        x: Samples from which to make predictions.
        x_dot: Pre-computed derivatives of the samples.
        u: Control variables

    Returns:
        whether data has multiple trajectories

    Raises:
        TypeError if data contains a mix of single/multiple trajectories
        ValueError if either data different numbers of trajectories

    """
    SequenceOrNone = Union[Sequence, None]
    if sys.version_info.minor < 10:
        mixed_trajectories = (
            isinstance(x, Sequence)
            and (
                not isinstance(x_dot, Sequence)
                and x_dot is not None
                or not isinstance(u, Sequence)
                and u is not None
            )
            or isinstance(x_dot, Sequence)
            and not isinstance(x, Sequence)
            or isinstance(u, Sequence)
            and not isinstance(x, Sequence)
        )
    else:
        mixed_trajectories = (
            isinstance(x, Sequence)
            and (
                not isinstance(x_dot, SequenceOrNone)
                or not isinstance(u, SequenceOrNone)
            )
            or isinstance(x_dot, Sequence)
            and not isinstance(x, Sequence)
            or isinstance(u, Sequence)
            and not isinstance(x, Sequence)
        )
    if mixed_trajectories:
        raise TypeError(
            "If x, x_dot, or u are a Sequence of trajectories, each must be a Sequence"
            " of trajectories or None."
        )
    if isinstance(x, Sequence):
        matching_lengths = (x_dot is None or len(x) == len(x_dot)) and (
            u is None or len(x) == len(u)
        )
        if not matching_lengths:
            raise ValueError("x, x_dot and/or u have mismatched number of trajectories")
        return True
    return False


def _adapt_to_multiple_trajectories(x, t, x_dot, u) -> tuple:
    """Adapt model data to that multiple_trajectories.

    Args:
        x: Samples from which to make predictions.
        t: Time step between samples or array of collection times.
        x_dot: Pre-computed derivatives of the samples.
        u: Control variables

    Returns:
        Tuple of updated x, t, x_dot, u
    """
    x = [x]
    if isinstance(t, Collection):
        t = [t]
    if x_dot is not None:
        x_dot = [x_dot]
    if u is not None:
        u = [u]
    return x, t, x_dot, u


def _comprehend_and_validate_inputs(x, t, x_dot, u, feature_library):
    """Validate input types, reshape arrays, and label axes"""

    def comprehend_and_validate(arr, t):
        arr = AxesArray(arr, comprehend_axes(arr))
        arr = feature_library.correct_shape(arr)
        return validate_no_reshape(arr, t)

    x = [comprehend_and_validate(xi, ti) for xi, ti in _zip_like_sequence(x, t)]
    if x_dot is not None:
        x_dot = [
            comprehend_and_validate(xdoti, ti)
            for xdoti, ti in _zip_like_sequence(x_dot, t)
        ]
    if u is not None:
        reshape_control = False
        for i in range(len(x)):
            if len(x[i].shape) != len(np.array(u[i]).shape):
                reshape_control = True
        if reshape_control:
            try:
                shape = np.array(x[0].shape)
                shape[x[0].ax_coord] = -1
                u = [np.reshape(u[i], shape) for i in range(len(x))]
            except Exception:
                try:
                    if np.isscalar(u[0]):
                        shape[x[0].ax_coord] = 1
                    else:
                        shape[x[0].ax_coord] = len(u[0])
                    u = [np.broadcast_to(u[i], shape) for i in range(len(x))]
                except Exception:
                    raise (
                        ValueError(
                            "Could not reshape control input to match the input data."
                        )
                    )
        correct_shape = True
        for i in range(len(x)):
            for axis in range(x[i].ndim):
                if (
                    axis != x[i].ax_coord
                    and x[i].shape[axis] != np.array(u[i]).shape[axis]
                ):
                    correct_shape = False
        if not correct_shape:
            raise (
                ValueError("Could not reshape control input to match the input data.")
            )
        u = [comprehend_and_validate(ui, ti) for ui, ti in _zip_like_sequence(u, t)]
    return x, x_dot, u
