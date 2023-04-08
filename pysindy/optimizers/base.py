"""
Base class for SINDy optimizers.
"""
import abc
import warnings
from typing import Callable
from typing import Tuple

import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.linear_model._base import _preprocess_data
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import check_X_y

from ..utils import AxesArray


def _rescale_data(X, y, sample_weight):
    """Rescale data so as to support sample_weight"""
    n_samples = X.shape[0]
    sample_weight = np.asarray(sample_weight)
    if sample_weight.ndim == 0:
        sample_weight = np.full(n_samples, sample_weight, dtype=sample_weight.dtype)
    sample_weight = np.sqrt(sample_weight)
    sw_matrix = sparse.dia_matrix((sample_weight, 0), shape=(n_samples, n_samples))
    X = safe_sparse_dot(sw_matrix, X)
    y = safe_sparse_dot(sw_matrix, y)
    return X, y


class ComplexityMixin:
    @property
    def complexity(self):
        check_is_fitted(self)
        return np.count_nonzero(self.coef_) + np.count_nonzero(self.intercept_)


class BaseOptimizer(LinearRegression, ComplexityMixin):
    """
    Base class for SINDy optimizers. Subclasses must implement
    a _reduce method for carrying out the bulk of the work of
    fitting a model.

    Parameters
    ----------
    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    normalize_columns : boolean, optional (default False)
        Normalize the columns of x (the SINDy library terms) before regression
        by dividing by the L2-norm. Note that the 'normalize' option in sklearn
        is deprecated in sklearn versions >= 1.0 and will be removed.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    initial_guess : np.ndarray, shape (n_features,) or (n_targets, n_features),
            optional (default None)
        Initial guess for coefficients ``coef_``.
        If None, the initial guess is obtained via a least-squares fit.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s).

    ind_ : array, shape (n_features,) or (n_targets, n_features)
        Array of 0s and 1s indicating which coefficients of the
        weight vector have not been masked out.

    history_ : list
        History of ``coef_`` over iterations of the optimization algorithm.

    Theta_ : np.ndarray, shape (n_samples, n_features)
        The Theta matrix to be used in the optimization. We save it as
        an attribute because access to the full library of terms is
        sometimes needed for various applications.

    """

    def __init__(
        self,
        max_iter=20,
        normalize_columns=False,
        fit_intercept=False,
        initial_guess=None,
        copy_X=True,
    ):
        super(BaseOptimizer, self).__init__(fit_intercept=fit_intercept, copy_X=copy_X)

        if max_iter <= 0:
            raise ValueError("max_iter must be positive")

        self.max_iter = max_iter
        self.iters = 0
        if np.ndim(initial_guess) == 1:
            initial_guess = initial_guess.reshape(1, -1)
        self.initial_guess = initial_guess
        self.normalize_columns = normalize_columns

    # Force subclasses to implement this
    @abc.abstractmethod
    def _reduce(self):
        """
        Carry out the bulk of the work of the fit function.

        Subclass implementations MUST update self.coef_.
        """
        raise NotImplementedError

    def fit(self, x_, y, sample_weight=None, **reduce_kws):
        """
        Fit the model.

        Parameters
        ----------
        x_ : array-like, shape (n_samples, n_features)
            Training data

        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            Target values

        sample_weight : float or numpy array of shape (n_samples,), optional
            Individual weights for each sample

        reduce_kws : dict
            Optional keyword arguments to pass to the _reduce method
            (implemented by subclasses)

        Returns
        -------
        self : returns an instance of self
        """
        x_, y = check_X_y(x_, y, accept_sparse=[], y_numeric=True, multi_output=True)

        x, y, X_offset, y_offset, X_scale = _preprocess_data(
            x_,
            y,
            fit_intercept=self.fit_intercept,
            copy=self.copy_X,
            sample_weight=sample_weight,
        )

        if sample_weight is not None:
            x, y = _rescale_data(x, y, sample_weight)

        self.iters = 0

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        coef_shape = (y.shape[1], x.shape[1])
        self.ind_ = np.ones(coef_shape, dtype=bool)

        self.Theta_ = x

        x_normed = np.copy(x)
        if self.normalize_columns:
            reg = 1 / np.linalg.norm(x, 2, axis=0)
            x_normed = x * reg

        if self.initial_guess is None:
            self.coef_ = np.linalg.lstsq(x_normed, y, rcond=None)[0].T
        else:
            if not self.initial_guess.shape == coef_shape:
                raise ValueError(
                    "initial_guess shape is incompatible with training data. "
                    f"Expected: {coef_shape}. Received: {self.initial_guess.shape}."
                )
            self.coef_ = self.initial_guess

        self.history_ = [self.coef_]

        x_normed = np.asarray(x_normed)

        self._reduce(x_normed, y, **reduce_kws)
        self.ind_ = np.abs(self.coef_) > 1e-14

        # Rescale coefficients to original units
        if self.normalize_columns:
            self.coef_ = np.multiply(reg, self.coef_)
            if hasattr(self, "coef_full_"):
                self.coef_full_ = np.multiply(reg, self.coef_full_)
            for i in range(np.shape(self.history_)[0]):
                self.history_[i] = np.multiply(reg, self.history_[i])

        self._set_intercept(X_offset, y_offset, X_scale)
        return self


class EnsembleOptimizer(BaseOptimizer):
    """Wrapper class for ensembling methods.

    Parameters
    ----------
    opt: BaseOptimizer
        The underlying optimizer to run on each ensemble

    bagging : boolean, optional (default False)
        This parameter is used to allow for "ensembling", i.e. the
        generation of many SINDy models (n_models) by choosing a random
        temporal subset of the input data (n_subset) for each sparse
        regression. This often improves robustness because averages
        (bagging) or medians (bragging) of all the models are usually
        quite high-performing. The user can also generate "distributions"
        of many models, and calculate how often certain library terms
        are included in a model.

    library_ensemble : boolean, optional (default False)
        This parameter is used to allow for "library ensembling",
        i.e. the generation of many SINDy models (n_models) by choosing
        a random subset of the candidate library terms to truncate. So,
        n_models are generated by solving n_models sparse regression
        problems on these "reduced" libraries. Once again, this often
        improves robustness because averages (bagging) or medians
        (bragging) of all the models are usually quite high-performing.
        The user can also generate "distributions" of many models, and
        calculate how often certain library terms are included in a model.

    n_models : int, optional (default 20)
        Number of models to generate via ensemble

    n_subset : int, optional (default len(time base))
        Number of time points to use for ensemble

    n_candidates_to_drop : int, optional (default 1)
        Number of candidate terms in the feature library to drop during
        library ensembling.

    replace : boolean, optional (default True)
        If ensemble true, whether or not to time sample with replacement.

    ensemble_aggregator : callable, optional (default numpy.median)
        Method to aggregate model coefficients across different samples.
        This method argument is only used if ``ensemble`` or ``library_ensemble``
        is True.
        The method should take in a list of 2D arrays and return a 2D
        array of the same shape as the arrays in the list.
        Example: :code:`lambda x: np.median(x, axis=0)`

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Regularized weight vector(s). This is the v in the objective
        function.

    coef_full_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s) that are not subjected to the regularization.
        This is the w in the objective function.

    unbias : boolean
        Whether to perform an extra step of unregularized linear regression
        to unbias the coefficients for the identified support.
        ``unbias`` is automatically set to False if a constraint is used and
        is otherwise left uninitialized.
    """

    def __init__(
        self,
        opt: BaseOptimizer,
        bagging: bool = False,
        library_ensemble: bool = False,
        n_models: int = 20,
        n_subset: int = None,
        n_candidates_to_drop: int = 1,
        replace: bool = True,
        ensemble_aggregator: Callable = None,
    ):
        if not hasattr(opt, "initial_guess"):
            opt.initial_guess = None

        super(EnsembleOptimizer, self).__init__(
            max_iter=opt.max_iter,
            fit_intercept=opt.fit_intercept,
            initial_guess=opt.initial_guess,
            copy_X=opt.copy_X,
        )
        if not bagging and not library_ensemble:
            raise ValueError(
                "If not ensembling data or library terms, use another optimizer"
            )
        if n_subset is not None and n_subset <= 0:
            raise ValueError("n_subset must be a positive integer if bagging")
        if n_candidates_to_drop is not None and n_candidates_to_drop <= 0:
            raise ValueError(
                "n_candidates_to_drop must be a positive integer if ensembling library"
            )
        self.opt = opt
        if n_models is None or n_models == 0:
            warnings.warn(
                "n_models must be a positive integer.  Explicitly initialized to zero"
                " or None, defaulting to 20."
            )
            n_models = 20
        self.n_models = n_models
        self.n_subset = n_subset
        self.bagging = bagging
        self.library_ensemble = library_ensemble
        self.ensemble_aggregator = ensemble_aggregator
        self.replace = replace
        self.n_candidates_to_drop = n_candidates_to_drop
        self.coef_list = []

    def _reduce(self, x: AxesArray, y: np.ndarray) -> None:
        x = AxesArray(np.asarray(x), {"ax_sample": 0, "ax_coord": 1})
        n_samples = x.shape[x.ax_sample]
        if self.bagging and self.n_subset is None:
            self.n_subset = int(0.6 * n_samples)
        if self.bagging and self.n_subset > n_samples and not self.replace:
            warnings.warn(
                "n_subset is larger than sample count without replacement; cannot bag."
            )
            n_subset = n_samples
        else:
            n_subset = self.n_subset

        n_features = x.shape[x.ax_coord]
        if self.library_ensemble and self.n_candidates_to_drop > n_features:
            warnings.warn(
                "n_candidates_to_drop larger than number of features.  Cannot "
                "ensemble library."
            )
            n_candidates_to_drop = 0
        else:
            n_candidates_to_drop = self.n_candidates_to_drop

        for _ in range(self.n_models):
            if self.bagging:
                x_ensemble, y_ensemble = _drop_random_samples(
                    x, y, n_subset, self.replace
                )
            else:
                x_ensemble, y_ensemble = x, y

            keep_inds = np.arange(n_features)
            if self.library_ensemble:
                keep_inds = np.sort(
                    np.random.choice(
                        range(n_features),
                        n_features - n_candidates_to_drop,
                        replace=False,
                    )
                )
                x_ensemble = x_ensemble.take(keep_inds, axis=x.ax_coord)
            self.opt.fit(x_ensemble, y_ensemble)
            new_coefs = np.zeros((y.shape[1], n_features))
            new_coefs[:, keep_inds] = self.opt.coef_
            self.coef_list.append(new_coefs)
        # Get average coefficients
        if self.ensemble_aggregator is None:
            self.coef_ = np.median(self.coef_list, axis=0)
        else:
            self.coef_ = self.ensemble_aggregator(self.coef_list)


def _drop_random_samples(
    x: AxesArray,
    x_dot: AxesArray,
    n_subset: int,
    replace: bool,
) -> Tuple[AxesArray]:
    n_samples = x.shape[x.ax_sample]
    rand_inds = np.random.choice(range(n_samples), n_subset, replace=replace)
    x_new = np.take(x, rand_inds, axis=x.ax_sample)
    x_dot_new = np.take(x_dot, rand_inds, axis=x.ax_sample)

    return x_new, x_dot_new
