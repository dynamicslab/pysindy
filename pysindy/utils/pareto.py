from numpy import count_nonzero
from numpy import ravel
from numpy import savetxt
from numpy import zeros
from numpy.linalg import norm
from scipy.integrate import odeint

from ..pysindy import SINDy


def pareto_curve(
    optimizer,
    feature_library,
    differentiation_method,
    feature_names,
    discrete_time,
    thresholds,
    x_fit,
    x_test,
    t_fit,
    t_test,
):
    """
    Function which sweeps out a Pareto Curve in (r, lambda)

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
        Method for differentiating the data. This must be an object that
        extends :class:`pysindy.differentiation_methods.BaseDifferentiation`.
        Default is centered difference.

    feature_names : list of string, length n_input_features, optional
       Names for the input features. If None, will use ['x0','x1',...].

    discrete_time : boolean, optional (default False)
        If True, dynamical system is treated as a map. Rather than predicting
        derivatives, the right hand side functions step the system forward by
        one time step. If False, dynamical system is assumed to be a flow
        (right hand side functions predict continuous time derivatives).

    thresholds: array of floats
        The list of thresholds to change the number of terms available to the
        SINDy model, generating a Pareto curve.

    x_fit: array-like or list of array-like, shape
        (n_samples, n_input_features)
        Training data.

    x_test: array-like or list of array-like, shape
        (n_samples, n_input_features)
        Testing data.

    t_fit: array of floats
        Time slices corresponding to the training data.

    t_test: array of floats
        Time slices corresponding to the testing data.

    """
    model_scores = []
    non_zeros_coeffs = []
    x_err = []
    xdot_err = []

    # Loop over the threshold values of interest
    for j in range(len(thresholds)):
        print("pareto loop, j = ", j)
        optimizer.set_threshold(thresholds[j])
        model = SINDy(
            optimizer=optimizer,
            feature_library=feature_library,
            differentiation_method=differentiation_method(drop_endpoints=True),
            feature_names=feature_names,
            discrete_time=discrete_time,
        )

        # Compute predicted X and Xdot
        model.fit(x_fit, t=t_fit)
        x0 = x_test[0, :]
        x_sim, output = model.simulate(
            x0,
            t_test,
            integrator=odeint,
            stop_condition=None,
            full_output=True,
            rtol=1e-5,
            h0=1e-5,
        )
        xdot_test = model.differentiate(x_test, t=t_test)[1:-2]
        xdot_sim = model.predict(x_test)[1:-2]

        # Compute frobenius error of X and Xdot on the "testing" data
        x_err.append(norm(x_test - x_sim))
        xdot_err.append(norm(xdot_test - xdot_sim))
        num_coeff = len(ravel(model.coefficients()))
        num_nonzero_coeff = count_nonzero(model.coefficients())
        non_zeros_coeffs.append(num_nonzero_coeff / num_coeff * 100)
        model_scores.append(min((1 - min(model.score(x_test, t=t_test), 1)) * 100, 100))

    # Save the errors and other metrics of model performance to file
    errs = zeros((len(model_scores), 5))
    errs[:, 0] = non_zeros_coeffs
    errs[:, 1] = thresholds
    errs[:, 2] = model_scores
    errs[:, 3] = x_err
    errs[:, 4] = xdot_err
    savetxt("pareto_" + str(x_test.shape[1]) + ".txt", errs)
