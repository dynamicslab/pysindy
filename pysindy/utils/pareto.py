from pysindy import SINDy
from numpy import count_nonzero, zeros, ravel, savetxt
from numpy.random import random
from numpy.linalg import norm
from scipy.integrate import odeint
from pysindy.optimizers import ConstrainedSR3


def pareto_curve(optimizer, feature_library,
                 differentiation_method, feature_names,
                 discrete_time, n_jobs, thresholds,
                 x_fit, x_pred, t_fit, t_pred):
    """
    Function which sweeps out a Pareto Curve in (r,lambda)

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

    thresholds: array of floats
        The list of thresholds to change the number of terms available to the
        SINDy model, generating a Pareto curve.

    x_fit: array-like or list of array-like, shape
        (n_samples, n_input_features)
        Training data.

    x_pred: array-like or list of array-like, shape
        (n_samples, n_input_features)
        Testing data for prediction.

    t_fit: array of floats
        Time slices corresponding to the training data.

    t_pred: array of floats
        TTime slices corresponding to the testing data.

    """
    model_scores = []
    non_zeros_coeffs = []
    x_err = []
    xdot_err = []

    # Loop over the threshold values of interest
    for j in range(len(thresholds)):
        print('pareto loop, j = ', j)
        optimizer.set_threshold(thresholds[j])
        model = SINDy(optimizer=optimizer,
                      feature_library=feature_library,
                      differentiation_method=differentiation_method(
                                                drop_endpoints=True),
                      feature_names=feature_names,
                      discrete_time=discrete_time, n_jobs=n_jobs)

        # Compute predicted X and Xdot
        model.fit(x_fit, t=t_fit)
        x0 = x_pred[0, :]
        integrator_kws = {'full_output': True}
        x_sim, output = model.simulate(x0, t_pred, integrator=odeint,
                                       stop_condition=None, full_output=True,
                                       rtol=1e-5, h0=1e-5)
        xdot_pred = model.differentiate(x_pred, t=t_pred)[1:-2]
        xdot_sim = model.predict(x_pred)[1:-2]

        # Compute frobenius error of X and Xdot on the "testing" data
        x_err.append(norm(x_pred-x_sim))
        xdot_err.append(norm(xdot_pred-xdot_sim))
        num_coeff = len(ravel(model.coefficients()))
        num_nonzero_coeff = count_nonzero(model.coefficients())
        non_zeros_coeffs.append(num_nonzero_coeff/num_coeff*100)
        model_scores.append(min((1-min(model.score(
                                    x_pred, t=t_pred), 1))*100, 100))

    # Save the errors and other metrics of model performance to file
    errs = zeros((len(model_scores), 5))
    errs[:, 0] = non_zeros_coeffs
    errs[:, 1] = thresholds
    errs[:, 2] = model_scores
    errs[:, 3] = x_err
    errs[:, 4] = xdot_err
    savetxt('pareto_'+str(x_pred.shape[1])+'.txt', errs)
