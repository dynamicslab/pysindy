import warnings
from typing import Union

import numpy as np
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ridge_regression
from sklearn.utils.validation import check_is_fitted

from .base import BaseOptimizer


class adam_STLSQ(BaseOptimizer):
    """Sequentially thresholded least squares algorithm.
    Defaults to doing Sequentially thresholded Ridge regression.

    Attempts to minimize the objective function
    :math:`\\|y - Xw\\|^2_2 + \\alpha \\|w\\|^2_2`
    by iteratively performing least squares and masking out
    elements of the weight array w that are below a given threshold.

    See the following reference for more details:

        Brunton, Steven L., Joshua L. Proctor, and J. Nathan Kutz.
        "Discovering governing equations from data by sparse
        identification of nonlinear dynamical systems."
        Proceedings of the national academy of sciences
        113.15 (2016): 3932-3937.

    Parameters
    ----------
    threshold : float, optional (default 0.1)
        Minimum magnitude for a coefficient in the weight vector.
        Coefficients with magnitude below the threshold are set
        to zero.

    alpha : float, optional (default 0.05)
        Optional L2 (ridge) regularization on the weight vector.

    max_iter : int, optional (default 20)
        Maximum iterations of the optimization algorithm.

    mom_memory : float, optional (default 0.9)
        0<=mom_memory<=1 is used to calculate the momentum at each time step of
        sequential thresholding as mom_k =
        mom_memory*(mom_k-1) + (1-mom_memory)*coeff_k

    mom_init_iter : int, optional (default 1)
        Iteration number from which momentum is calculated. Until this iteration,
        momentum = coeff.

    ridge_kw : dict, optional (default None)
        Optional keyword arguments to pass to the ridge regression.

    normalize_columns : boolean, optional (default False)
        Normalize the columns of x (the SINDy library terms) before regression
        by dividing by the L2-norm. Note that the 'normalize' option in sklearn
        is deprecated in sklearn versions >= 1.0 and will be removed.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    initial_guess : np.ndarray, shape (n_features) or (n_targets, n_features),
            optional (default None)
        Initial guess for coefficients ``coef_``.
        If None, least-squares is used to obtain an initial guess.

    verbose : bool, optional (default False)
        If True, prints out the different error terms every iteration.

    sparse_ind : list, optional (default None)
        Indices to threshold and perform ridge regression upon.
        If None, sparse thresholding and ridge regression is applied to all
        indices.

    use_mom : bool, optional (default True)
        Use momentum method for sequential thresholding. False case is not implemented yet,
        but will be same as regular STLS

    mom_inplace : bool, optional (default True)
        If true, coefficient at each iteration of STLS will be repalced with momentum.

    variable_thresh : bool, optional (default False)
        If true, a threshold vector will be used for thresholding coeffcieints in STLS.

    threshold_vect : list, optional (default []])
        Initial treshold vector used for variable thresholding. Empty list by default.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s).

    ind_ : array, shape (n_features,) or (n_targets, n_features)
        Array of 0s and 1s indicating which coefficients of the
        weight vector have not been masked out, i.e. the support of
        ``self.coef_``.

    history_ : list
        History of ``coef_``. ``history_[k]`` contains the values of
        ``coef_`` at iteration k of sequentially thresholded least-squares.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import STLSQ
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = STLSQ(threshold=.1, alpha=.5)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -9.999 1 + 9.999 x0
    x1' = 27.984 1 + -0.996 x0 + -1.000 1 x1
    x2' = -2.666 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        threshold=0.1,
        alpha=0.05,
        max_iter=20,
        mom_memory = 0.9,
        mom_init_iter = 1,
        ridge_kw=None,
        normalize_columns=False,
        copy_X=True,
        initial_guess=None,
        verbose=False,
        sparse_ind=None,
        unbias=True,
        use_mom = True,
        mom_inplace = True,
        variable_thresh = False,
        threshold_vect = []
    ):
        super().__init__(
            max_iter=max_iter,
            copy_X=copy_X,
            normalize_columns=normalize_columns,
            unbias=unbias,
        )

        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        if alpha < 0:
            raise ValueError("alpha cannot be negative")

        self.threshold = threshold
        self.alpha = alpha
        self.ridge_kw = ridge_kw
        self.initial_guess = initial_guess
        self.verbose = verbose
        self.sparse_ind = sparse_ind

        self.mom_memory = mom_memory
        self.mom_inplace = mom_inplace
        self.mom_init_iter = mom_init_iter
        if use_mom:
            self.mom_history = []
            assert mom_init_iter >= 1

        self.variable_thresh = variable_thresh
        self.threshold_vect = threshold_vect


    def _sparse_coefficients(
        self, dim: int, ind_nonzero: np.ndarray, coef: np.ndarray, threshold: list
    ) -> (np.ndarray, np.ndarray):
        """Perform thresholding of the weight vector(s) (on specific indices
        if ``self.sparse_ind`` is not None)
        Note threshold is a list of len dim.
        """
        c = np.zeros(dim)
        c[ind_nonzero] = coef
        #Manu Note: this is where the thresholding is happening, we an add the adaptive
        # thresholding step here.
        assert dim == len(threshold), ("Length of threshold vector not same as the "
                                       "length of the feature library")


        big_ind = np.abs(c) >= threshold

        if self.sparse_ind is not None:
            nonsparse_ind_mask = np.ones_like(ind_nonzero)
            nonsparse_ind_mask[self.sparse_ind] = False
            big_ind = big_ind | nonsparse_ind_mask
        c[~big_ind] = 0
        return c, big_ind

    def _updated_momentum(
        self, dim: int, ind_nonzero: np.ndarray, coef: np.ndarray, prev_mom: np.ndarray) -> np.ndarray:
        """Calculate the momentum of parameters"""
        c = np.zeros(dim)
        c[ind_nonzero] = coef

        mom_array = self.mom_memory*prev_mom + ((1 - self.mom_memory)*c)
        return mom_array

    def _regress(self, x: np.ndarray, y: np.ndarray, dim: int, sparse_sub: np.ndarray):
        """Perform the ridge regression (on specific indices if
        ``self.sparse_ind`` is not None)"""
        kw = self.ridge_kw or {}
        if self.sparse_ind is None:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=LinAlgWarning)
                try:
                    coef = ridge_regression(x, y, self.alpha, **kw)
                except LinAlgWarning:
                    # increase alpha until warning stops
                    self.alpha = 2 * self.alpha
            self.iters += 1
            return coef
        if self.sparse_ind is not None:
            alpha_array = np.zeros((dim, dim))
            alpha_array[sparse_sub, sparse_sub] = np.sqrt(self.alpha)
            x_lin = np.concatenate((x, alpha_array), axis=0)
            y_lin = np.concatenate((y, np.zeros((dim,))))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=LinAlgWarning)
                try:
                    coef = (
                        LinearRegression(fit_intercept=False, **kw)
                        .fit(x_lin, y_lin)
                        .coef_
                    )
                except LinAlgWarning:
                    # increase alpha until warning stops
                    self.alpha = 2 * self.alpha
            self.iters += 1
            return coef

    def _no_change(self):
        """Check if the coefficient mask has changed after thresholding"""
        this_coef = self.history_[-1].flatten()
        if len(self.history_) > 1:
            last_coef = self.history_[-2].flatten()
        else:
            last_coef = np.zeros_like(this_coef)
        return all(bool(i) == bool(j) for i, j in zip(this_coef, last_coef))

    def _reduce(self, x, y):
        """Performs at most ``self.max_iter`` iterations of the
        sequentially-thresholded least squares algorithm.

        Assumes an initial guess for coefficients and support are saved in
        ``self.coef_`` and ``self.ind_``.
        """
        if self.initial_guess is not None:
            self.coef_ = self.initial_guess

        ind = self.ind_
        n_samples, n_features = x.shape
        n_targets = y.shape[1]
        n_features_selected = np.sum(ind)
        sparse_sub = [np.array(self.sparse_ind)] * y.shape[1]

        self.threshold_vect = np.array(self.threshold_vect) if len(self.threshold_vect) > 0 \
            else self.threshold*np.ones(n_features)

        # Print initial values for each term in the optimization
        if self.verbose:
            row = [
                "Iteration",
                "|y - Xw|^2",
                "a * |w|_2",
                "|w|_0",
                "Total error: |y - Xw|^2 + a * |w|_2",
            ]
            print(
                "{: >10} ... {: >10} ... {: >10} ... {: >10}"
                " ... {: >10}".format(*row)
            )

        for k in range(self.max_iter):
            if np.count_nonzero(ind) == 0:
                warnings.warn(
                    "Sparsity parameter is too big ({}) and eliminated all "
                    "coefficients".format(self.threshold)
                )
                optvar = np.zeros((n_targets, n_features))
                break

            optvar = np.zeros((n_targets, n_features))
            #Defining momentum vector
            opt_mom = np.zeros((n_targets, n_features))

            for i in range(n_targets):
                if np.count_nonzero(ind[i]) == 0:
                    warnings.warn(
                        "Sparsity parameter is too big ({}) and eliminated all "
                        "coefficients".format(self.threshold)
                    )
                    continue
                coef_i = self._regress(
                    x[:, ind[i]], y[:, i], np.count_nonzero(ind[i]), sparse_sub[i]
                )
                #Calculating momentum
                mom_i = self._updated_momentum(n_features, ind[i], coef_i, self.mom_history[-1][i]) \
                    if k >= self.mom_init_iter else np.copy(coef_i)

                if self.mom_inplace:

                    coef_i, ind_i = self._sparse_coefficients(
                        n_features, np.arange(n_features), mom_i, self.threshold_vect
                    )
                    mom_i = np.copy(coef_i)
                else:
                    coef_i, ind_i = self._sparse_coefficients(
                        n_features, ind[i], coef_i, self.threshold_vect
                    )
                # coef_i, ind_i = self._sparse_coefficients(
                #     n_features, ind[i], coef_i, self.threshold
                #     )
                if self.sparse_ind is not None:
                    vals_to_remove = np.intersect1d(
                        self.sparse_ind, np.where(coef_i == 0)
                    )
                    sparse_sub[i] = _remove_and_decrement(
                        self.sparse_ind, vals_to_remove
                    )
                optvar[i] = coef_i
                opt_mom[i] = mom_i
                ind[i] = ind_i

            self.history_.append(optvar)
            self.mom_history.append(opt_mom)
            if self.verbose:
                R2 = np.sum((y - np.dot(x, optvar.T)) ** 2)
                L2 = self.alpha * np.sum(optvar**2)
                L0 = np.count_nonzero(optvar)
                row = [k, R2, L2, L0, R2 + L2]
                print(
                    "{0:10d} ... {1:10.4e} ... {2:10.4e} ... {3:10d}"
                    " ... {4:10.4e}".format(*row)
                )
            if np.sum(ind) == n_features_selected or self._no_change():
                # could not (further) select important features
                break
        else:
            warnings.warn(
                "STLSQ._reduce did not converge after {} iterations.".format(
                    self.max_iter
                ),
                ConvergenceWarning,
            )
        if self.sparse_ind is None:
            self.coef_ = optvar
            self.ind_ = ind
        else:
            non_sparse_ind = np.setxor1d(self.sparse_ind, range(n_features))
            self.coef_ = optvar[:, self.sparse_ind]
            self.ind_ = ind[:, self.sparse_ind]
            self.optvar_non_sparse_ = optvar[:, non_sparse_ind]
            self.ind_non_sparse_ = ind[:, non_sparse_ind]

    @property
    def complexity(self):
        check_is_fitted(self)

        return np.count_nonzero(self.coef_) + np.count_nonzero(
            [abs(self.intercept_) >= self.threshold]
        )

    def _unbias(self, x: np.ndarray, y: np.ndarray):
        if not self.sparse_ind:
            return super()._unbias(x, y)
        regression_col_mask = np.zeros((y.shape[1], x.shape[1]), dtype=bool)
        regression_col_mask[:, self.sparse_ind] = self.ind_
        non_sparse_ind = np.setxor1d(self.sparse_ind, range(x.shape[1]))
        regression_col_mask[:, non_sparse_ind] = self.ind_non_sparse_

        for i in range(self.ind_.shape[0]):
            if np.any(self.ind_[i]):
                optvar = (
                    LinearRegression(fit_intercept=False)
                    .fit(x[:, regression_col_mask[i]], y[:, i])
                    .coef_
                )
                self.coef_[i] = optvar[self.sparse_ind]
                self.optvar_non_sparse_[i] = optvar[non_sparse_ind]


def _remove_and_decrement(
    existing_vals: Union[np.ndarray, list], vals_to_remove: Union[np.ndarray, list]
) -> np.ndarray:
    """Remove elements from existing values and decrement the elements
    that are greater than the removed elements"""
    for s in reversed(vals_to_remove):
        existing_vals = np.delete(existing_vals, np.where(s == existing_vals))
        existing_vals = np.where(existing_vals > s, existing_vals - 1, existing_vals)
    return existing_vals
