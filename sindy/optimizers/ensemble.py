import warnings
from copy import deepcopy

import numpy as np
from sklearn.ensemble.bagging import _generate_indices

from sindy.optimizers import BaseOptimizer, STLSQ


class Ensemble(BaseOptimizer):
    """
    Ensemble of sparse optimizers.

    Forms an ensemble consisting of other sparse optimizers.
    """

    def __init__(
        self,
        base_estimator=STLSQ(),
        n_estimators=10,
        max_samples=1.0,
        bootstrap=True,
        random_state=None,  # TODO
        **kwargs
    ):
        super(Ensemble, self).__init__(**kwargs)

        if n_estimators <= 0:
            raise ValueError('n_estimators must be positive')
        if max_samples <= 0:
            raise ValueError('max_samples must be positive')
        elif isinstance(max_samples, float) and max_samples > 1:
            raise ValueError('If max_samples is a float it must be in [0, 1]')

        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.random_state = random_state

        if isinstance(base_estimator, list):
            self.estimators = [
                deepcopy(self.base_estimator[i % n_estimators])
                for i in range(n_estimators)
            ]
            if len(self.base_estimator) > n_estimators:
                warnings.warn(
                    'Number of estimators passed in exceeds n_estimators. '
                    'Not all estimators will be used in ensemble.'
                )
        else:
            self.estimators = [
                deepcopy(self.base_estimator) for _ in range(n_estimators)
            ]

    def _reduce(self, x, y):
        """
        Train each estimator in the ensemble on subsets of the training data.
        """
        if self.max_samples > x.shape[0]:
            raise ValueError(
                'max_samples cannot exceed number of samples in x'
            )
        elif isinstance(self.max_samples, int):
            n_samples = self.max_samples
        else:
            n_samples = int(self.max_samples * x.shape[0])

        for estimator in self.estimators:
            # Sample data
            sample_indices = _generate_indices(
                self.random_state,
                self.bootstrap,
                1,
                n_samples
            )

            estimator.fit(
                x[sample_indices, :],
                y[sample_indices]
            )

        # TODO: combine votes of individual estimators somehow.
        #  -Vote
        #  -Stack coefficient vectors together?

        # TODO: decide on values of coefficients to use
