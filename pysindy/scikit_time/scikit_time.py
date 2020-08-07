"""
Scikit-time wrapper interface for PySINDy.
"""
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ..pysindy import SINDy


class SINDyEstimator(SINDy):
    """
    Implementation of the Scikit-time API for PySINDy.
    """

    def __init__(
        self,
        optimizer=None,
        feature_library=None,
        differentiation_method=None,
        feature_names=None,
        t_default=1,
        discrete_time=False,
        n_jobs=1,
    ):
        super(SINDyEstimator, self).__init__(
            optimizer=optimizer,
            feature_library=feature_library,
            differentiation_method=differentiation_method,
            feature_names=feature_names,
            t_default=t_default,
            discrete_time=discrete_time,
            n_jobs=n_jobs,
        )
        self._model = None

    def fit(self, x, **kwargs):
        super(SINDyEstimator, self).fit(x, **kwargs)
        self._model = SINDyModel(
            feature_library=self.model.steps[0][1],
            optimizer=self.model.steps[1][1],
            feature_names=self.feature_names,
            t_default=self.t_default,
            discrete_time=self.discrete_time,
            n_control_features_=self.n_control_features_,
        )
        return self

    def fetch_model(self):
        """
        Yields the estimated model. Can be none if :meth:`fit` was note called.

        Returns
        -------
        model: SINDyModel or None
            The estimated SINDy model or none
        """
        return self._model

    @property
    def has_model(self) -> bool:
        """Property reporting whether this estimator contains an estimated
        model. This assumes that the model is initialized with `None` otherwise.
        :type: bool
        """
        return self.model is not None


class SINDyModel(SINDy):
    """
    Implementation of the Scikit-time API for PySINDy.
    """

    def __init__(
        self,
        feature_library,
        optimizer,
        feature_names=None,
        t_default=1,
        discrete_time=False,
        n_control_features_=0,
    ):
        super(SINDyModel, self).__init__(
            feature_library=feature_library,
            optimizer=optimizer,
            feature_names=feature_names,
            t_default=t_default,
            discrete_time=discrete_time,
        )
        self.n_control_features_ = n_control_features_

        check_is_fitted(feature_library)
        check_is_fitted(optimizer)

        steps = [("features", feature_library), ("model", optimizer)]
        self.model = Pipeline(steps)

        self.n_input_features_ = self.model.steps[0][1].n_input_features_
        self.n_output_features_ = self.model.steps[0][1].n_output_features_
