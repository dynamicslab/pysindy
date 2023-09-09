from typing import Annotated
from typing import Sequence

import numpy as np
from scipy import sparse
from scipy.linalg import eigh_tridiagonal
from scipy.linalg import ldl
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from .optimizers import STLSQ
from .pysindy import _adapt_to_multiple_trajectories
from .pysindy import _check_multiple_trajectories
from .pysindy import _comprehend_and_validate_inputs
from .pysindy import _zip_like_sequence
from .pysindy import SINDy
from .utils import AxesArray
from .utils import SampleConcatter
from .utils import validate_control_variables


class SSSINDy(SINDy):
    """Single step SINDy, combining regression and differentiation

    Args:
        optimizer: Optimization method used to fit the SINDy model.
            __init__ must implement the sparse_indices argument.
        alpha: smoothness regularization.

    Other arguments and attributes from superclass
    """

    def __init__(
        self, optimizer: STLSQ = None, alpha: Annotated[float, ">0"] = 1, **kwargs
    ):
        super().__init__(optimizer=optimizer, **kwargs)
        self.alpha = alpha

    def fit(
        self,
        x,
        t=None,
        x_dot=None,
        u=None,
    ):
        if not _check_multiple_trajectories(x, x_dot, u):
            x, t, x_dot, u = _adapt_to_multiple_trajectories(x, t, x_dot, u)
        x, x_dot, u = _comprehend_and_validate_inputs(
            x, t, x_dot, u, self.feature_library
        )
        means, vars = tuple(
            zip(*[_conditional_moments(ti, xi) for ti, xi in _zip_like_sequence(x, t)])
        )

        rtinv_vars = [np.linalg.pinv(psd_root(var)).T for var in vars]
        if u is None:
            self.n_control_features_ = 0
        else:
            u = validate_control_variables(
                x,
                u,
                trim_last_point=(self.discrete_time and x_dot is None),
            )
            self.n_control_features_ = u[0].shape[u[0].ax_coord]
            x = [np.concatenate((xi, ui), axis=xi.ax_coord) for xi, ui in zip(x, u)]

        n_samp = sum(len(rinv) for rinv in rtinv_vars)
        n_tgts = x[0].n_coord

        class _ProblemAssembler(BaseEstimator):
            def __init__(self, rtinv_vars, nx):
                self.rtinv_vars = rtinv_vars
                self.nx = nx

            def fit(self, x, y):
                return self

            def transform(self, feats):
                A11 = sparse.eye(n_samp)
                A12 = feats
                A21 = sparse.block_diag(self.rtinv_vars)
                return sparse.bmat([[A11, A12], [A21, None]]).toarray()

        y_col = np.vstack(
            [
                rtinv_var @ mean.reshape((-1, 1))
                for mean, rtinv_var in zip(means, rtinv_vars)
            ]
        )
        y_col = sparse.vstack((y_col, sparse.csc_array((n_samp, n_tgts))))
        sparse_ind = slice(n_samp, None)
        self.optimizer.set_params(sparse_ind=sparse_ind)

        steps = [
            ("features", self.feature_library),
            ("shaping", SampleConcatter()),
            ("matrix assembly", _ProblemAssembler(rtinv_vars, n_samp)),
            ("model", self.optimizer),
        ]
        self.model = Pipeline(steps)
        self.model.fit(x, y_col.toarray())

        self.n_features_in_ = self.feature_library.n_features_in_
        self.n_output_features_ = self.feature_library.n_output_features_

        if self.feature_names is None:
            feature_names = []
            for i in range(self.n_features_in_ - self.n_control_features_):
                feature_names.append("x" + str(i))
            for i in range(self.n_control_features_):
                feature_names.append("u" + str(i))
            self.feature_names = feature_names

        return self


def _conditional_moments(
    t: float | Sequence | np.ndarray, x: AxesArray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate distribution of x_dot given x for a Kalman process

    Args:
        t: the timestep between states, or the time of each state
        x: the states of a process
    Returns:
        the mean and covariance matrix for the derivative of the process
    """
    if np.isscalar(t):
        dt = t * np.ones(x.n_time - 1)
    else:
        t = t.flatten()
        dt = t[1:] - t[:-1]
    G = _gen_centering_matrix(dt)
    G_dag = np.linalg.pinv(G.toarray())
    Q = _gen_kalman_covariance(dt)
    full_cov = G_dag @ Q @ G_dag.T
    S11 = full_cov[::2, ::2]
    S12 = full_cov[::2, 1::2]
    S22inv = np.linalg.pinv(full_cov[1::2, 1::2])
    reg_coefs = S12 @ S22inv
    x_dot_cov = S11 - reg_coefs @ S12.T
    x_dot_mean = -S22inv @ x
    return x_dot_mean, x_dot_cov


def _gen_centering_matrix(delta_times):
    nt = len(delta_times) + 1
    G_left = sparse.block_diag([-np.array([[1, 0], [dt, 1]]) for dt in delta_times])
    G_right = sparse.eye(2 * (nt - 1))
    align_cols = sparse.csc_array((2 * (nt - 1), 2))
    return sparse.hstack((G_left, align_cols)) + sparse.hstack((align_cols, G_right))


def _gen_kalman_covariance(delta_times):
    Qs = [
        np.array([[dt, dt**2 / 2], [dt**2 / 2, dt**3 / 3]]) for dt in delta_times
    ]
    Q = sparse.block_diag([Q for Q in Qs])
    return (Q + Q.T) / 2  # ensure symmetry


def psd_root(arr: Annotated[np.ndarray, "PSD"]) -> np.ndarray:
    """Calculte a root of a positive semidefinite matrix

    This is faster than a root from eigh.  Does not verify positive
    semidefiniteness
    """
    l, d, _ = ldl(arr)
    w, v = eigh_tridiagonal(np.diag(d), np.diag(d, 1))
    w = np.vstack((w, np.zeros_like(w))).max(axis=0)
    return l @ (v * np.sqrt(w))
