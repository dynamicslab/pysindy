import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ridge_regression

from .base import BaseOptimizer


class SSR(BaseOptimizer):
    """Stepwise sparse regression (SSR) greedy algorithm.

    Attempts to minimize the objective function
    :math:`\\|y - Xw\\|^2_2 + \\alpha \\|w\\|^2_2`
    by iteratively eliminating the smallest coefficient

    See the following reference for more details:

        Boninsegna, Lorenzo, Feliks Nüske, and Cecilia Clementi.
        "Sparse learning of stochastic dynamical equations."
        The Journal of chemical physics 148.24 (2018): 241723.

    Parameters
    ----------

    max_iter : int, optional (default 20)
        Maximum iterations of the optimization algorithm.

    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s).

    history_ : list
        History of ``coef_``. ``history_[k]`` contains the values of
        ``coef_`` at iteration k of SSR

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import SSR
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = SSR(alpha=.5)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    x0' = -9.999 1 + 9.999 x0
    x1' = 27.984 1 + -0.996 x0 + -1.000 1 x1
    x2' = -2.666 x1 + 1.000 1 x0
    """

    def __init__(
        self,
        alpha=0.05,
        max_iter=20,
        ridge_kw=None,
        normalize=False,
        fit_intercept=False,
        copy_X=True,
        criteria="coefficient_value",
    ):
        super(SSR, self).__init__(
            max_iter=max_iter,
            normalize=normalize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
        )

        if alpha < 0:
            raise ValueError("alpha cannot be negative")

        if max_iter <= 0:
            raise ValueError("max iteration must be > 0")

        if criteria != "coefficient_value" and criteria != "model_residual":
            raise ValueError(
                "The only implemented criteria for sparsifying models "
                " are coefficient_value (zeroing out the smallest coefficient)"
                " or model_residual (choosing the N-1 term model with)"
                " the smallest residual error."
            )

        self.criteria = criteria
        self.alpha = alpha
        self.ridge_kw = ridge_kw

    def _coefficient_value(self, coef):
        """Eliminate the smallest element of the weight vector(s)"""
        c = coef
        inds_nonzero = np.ravel(np.nonzero(c))
        c_nonzero = c[inds_nonzero]
        smallest_ind = np.argmin(np.abs(c_nonzero))
        c[inds_nonzero[smallest_ind]] = 0.0
        return c, inds_nonzero[smallest_ind]

    def _model_residual(self, x, y, coef, inds):
        """Choose model with lowest residual error"""
        x_shape = np.shape(x)[1]
        c = np.zeros((x_shape, x_shape - 1))
        err = np.zeros(x_shape)
        for i in range(x_shape):
            mask = np.ones(x_shape, dtype=bool)
            mask[i] = False
            c[i, :] = self._regress(x[:, mask], y)
            err[i] = np.mean((y - x[:, mask] @ c[i, :]) ** 2)
        min_err = np.argmin(err)
        # Figure out where this index is in the larger coef matrix
        total_ind = min_err
        q = -1
        for i in range(len(inds)):
            if q == min_err:
                break
            if not inds[i]:
                total_ind += 1
            else:
                q = q + 1
        cc = coef
        cc[total_ind] = 0.0
        return cc, total_ind

    def _regress(self, x, y):
        """Perform the ridge regression"""
        kw = self.ridge_kw or {}
        coef = ridge_regression(x, y, self.alpha, **kw)
        self.iters += 1
        return coef

    def _reduce(self, x, y):
        """Performs at most ``self.max_iter`` iterations of the
        SSR greedy algorithm.
        """
        n_samples, n_features = x.shape
        n_targets = y.shape[1]

        coef = self._regress(x, y)
        inds = np.ones((n_targets, n_features), dtype=bool)
        for k in range(self.max_iter):
            for i in range(n_targets):
                if not np.allclose(coef[i, :], 0.0):
                    if self.criteria == "coefficient_value":
                        coef[i, :], ind = self._coefficient_value(coef[i, :])
                    else:
                        coef[i, :], ind = self._model_residual(
                            x[:, inds[i, :]], y[:, i], coef[i, :], inds[i, :]
                        )
                    inds[i, ind] = False
                    if not np.any(inds[i, :]):
                        # No terms left to sparsify
                        continue
                    coef[i, inds[i, :]] = self._regress(x[:, inds[i, :]], y[:, i])
            self.history_.append(np.copy(coef))
            if np.sum(np.asarray(inds, dtype=int)) == 3:
                # each equation has one last term
                break
        else:
            warnings.warn(
                "SSR._reduce did not converge after {} iterations.".format(
                    self.max_iter
                ),
                ConvergenceWarning,
            )
            try:
                coef
            except NameError:
                coef = self.coef_
                warnings.warn(
                    "SSR._reduce has no iterations left to determine coef",
                    ConvergenceWarning,
                )
        self.coef_ = coef
