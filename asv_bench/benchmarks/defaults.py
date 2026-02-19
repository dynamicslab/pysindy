from abc import ABC, abstractmethod

import numpy as np
import sindy_exp

import pysindy as ps

class LorenzMixin:
    def make_data(self):
        trajectories, terms = sindy_exp.gen_data(
            "lorenz", 4, noise_abs=0.1, dt=0.03, t_end=10
        )["data"]
        traj = trajectories[0]
        self.true_eqns = terms
        self.x = traj.x_train
        self.x_dot_true = traj.x_train_true_dot
        self.t = traj.t_train


class DefaultBM(ABC):

    @abstractmethod
    def make_data(self) -> None:
        ...

    @abstractmethod
    def make_model(self) -> None:
        ...


    def setup(self):
        self.make_data()
        self.make_model()

    def time_fit(self):
        self.model.fit([self.x], [self.t])

    def peakmem_fit(self):
        self.model.fit([self.x], [self.t])

    def track_score_fit(self):
        self.model.fit([self.x], [self.t])
        true_ode_align, est_ode_align = sindy_exp._utils.unionize_coeff_dicts(
            self.model, self.true_eqns
        )
        metrics = sindy_exp.coeff_metrics(est_ode_align, true_ode_align)
        return metrics["coeff_mae"]

    def time_fit_predict(self):
        self.model.fit([self.x], [self.t])
        preds = self.model.predict(self.x)

    def peakmem_fit_predict(self):
        self.model.fit([self.x], [self.t])
        preds = self.model.predict(self.x)

    def track_score_fit_predict(self):
        self.model.fit([self.x], [self.t])
        preds = self.model.predict(self.x)
        metrics = sindy_exp.pred_metrics(
            self.model, self.x, self.x_dot_true  # type: ignore
        )
        return metrics["pred_l2_fro"]

class SINDyMixin:
    def make_model(self):
        self.model = ps.SINDy()

class SINDyLorenz(LorenzMixin, SINDyMixin, DefaultBM):
    pass

class WeakMixin:
    def make_model(self):
        self.model = ps.WeakSINDy()

class WeakLorenz(LorenzMixin, WeakMixin, DefaultBM):
    pass

# class WeakHeat:

# class WeakBurgers:
