import re
from itertools import chain
from typing import cast
from typing import NamedTuple
from typing import Optional
from typing import TypeVar
from warnings import warn

import jax.numpy as jnp
import numpy as np
import sklearn.metrics
from dysts.flows import Lorenz
from numpy.typing import NBitBase
from numpy.typing import NDArray

import pysindy as ps
from pysindy.sssindy import SSSINDy
from pysindy.sssindy.expressions import JaxPolyLib
from pysindy.sssindy.expressions import JointObjective
from pysindy.sssindy.interpolants import RKHSInterpolant
from pysindy.sssindy.interpolants.kernels import get_gaussianRBF


class SSSINDyLorenzSparsity:
    """
    See that we do decently on the Lorenz system
    """

    def setup(self):
        self.data = gen_lorenz(seed=124, dt=0.02, t_end=5)
        self.sss_model = SSSINDy(
            JointObjective(50, 1, JaxPolyLib(), RKHSInterpolant(get_gaussianRBF(0.05)))
        )

    def time_experiment(self):
        eval_sss(self.data, self.sss_model)

    def track_experiment(self):
        results = eval_sss(self.data, self.sss_model)
        return results

    track_experiment.unit = "MAE"


def eval_sss(data: "ProbData", sss_model: SSSINDy) -> float:
    sss_model.feature_names = data.input_features
    n_coloc = len(data.t_train)
    n_traj = len(data.x_train)
    sss_metrics = []
    t_train = data.t_train
    x_train = [jnp.array(x) for x in data.x_train]
    t_coloc = jnp.linspace(t_train[0], t_train[-1], n_coloc)
    t_coloc = n_traj * [jnp.array(t_coloc)]
    t_train = n_traj * [jnp.array(t_train)]
    sss_model.fit(x_train, t=t_train, t_coloc=t_coloc)

    coeff_true, sss_coeffs, _ = unionize_coeff_matrices(
        sss_model, data.coeff_true, True
    )
    sss_model.print(flush=True)

    x_interp = jnp.hstack(sss_model.x_predict(data.t_train))
    x_true = jnp.hstack(data.x_train_true)
    interp_rel_err = jnp.linalg.norm(x_interp - x_true) / jnp.linalg.norm(x_true)
    sss_metrics.append(
        coeff_metrics(sss_coeffs, coeff_true) | {"interp_rel_err": interp_rel_err}
    )

    return sss_metrics[0]["coeff_mae"]


####################################
# Copied from gen_experiments to avoid circular imports
####################################


NpFlt = np.dtype[np.floating[NBitBase]]
Float1D = np.ndarray[tuple[int], NpFlt]
Float2D = np.ndarray[tuple[int, int], NpFlt]
Shape = TypeVar("Shape", bound=tuple[int, ...])
FloatND = np.ndarray[Shape, np.dtype[np.floating[NBitBase]]]


class ProbData(NamedTuple):
    dt: float
    t_train: Float1D
    x_train: list[FloatND]
    x_test: list[FloatND]
    x_dot_test: list[FloatND]
    x_train_true: list[FloatND]
    input_features: list[str]
    coeff_true: list[dict[str, float]]


def gen_lorenz(
    seed: Optional[int] = None,
    n_trajectories: int = 1,
    ic_stdev: float = 3,
    dt: float = 0.01,
    t_end: float = 10,
) -> ProbData:
    """Generate zero-noise training and test data

    An Experiment step according to the mitosis experiment runner.
    Note that test data has no noise.

    Arguments:
        group: the function to integrate
        seed (int): the random seed for number generation
        n_trajectories (int): number of trajectories of training data
        ic_stdev (float): standard deviation for generating initial conditions
        noise_abs (float): measurement noise standard deviation.
            Defaults to .1 if noise_rel is None.
        noise_rel (float): measurement noise-to-signal power ratio.
            Either noise_abs or noise_rel must be None.  Defaults to
            None.
        dt: time step for sample
        t_end: end time of simulation
        display: Whether to display graphics of generated data.

    Returns:
        dictionary of data and descriptive information
    """
    coeff_true = [
        {"x": -10, "y": 10},
        {"x": 28, "y": -1, "x z": -1},
        {"z": -8 / 3, "x y": 1},
    ]
    input_features = ["x", "y", "z"]
    x0_center = np.array([0, 0, 15])
    rng = np.random.default_rng(seed)
    nt = int(t_end // dt)
    x_train = []
    n_coord = 3
    x0_train = ic_stdev * rng.standard_normal((n_trajectories, n_coord)) + x0_center
    for x0 in x0_train:
        x_train.append(Lorenz().make_trajectory(nt, dt, x0))
    t_train = np.arange(0, t_end, dt, dtype=np.float_)
    x_dot_train = [np.vstack([Lorenz().rhs(xij, 0) for xij in xi]) for xi in x_train]
    return ProbData(
        dt,
        t_train,
        x_train,
        x_train,
        x_dot_train,
        x_train,
        input_features,
        coeff_true,
    )


def unionize_coeff_matrices(
    model: ps._core._BaseSINDy,
    model_true: tuple[list[str], list[dict[str, float]]] | list[dict[str, float]],
    strict: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Reformat true coefficients and coefficient matrix compatibly

    In order to calculate accuracy metrics between true and estimated
    coefficients, this function compares the names of true coefficients
    and a the fitted model's features in order to create comparable
    (i.e. non-ragged) true and estimated coefficient matrices.  In
    a word, it stacks the correct coefficient matrix and the estimated
    coefficient matrix in a matrix that represents the union of true
    features and modeled features.

    Arguments:
        model: fitted model
        model_true: A tuple of (a) a list of input feature names, and
            (b) a list of dicts of format function_name: coefficient,
            one dict for each modeled coordinate/target.  The old format
            of passing one
        strict:
            whether to attempt to translate the model's features into the
            input variable names in the true model.
    Returns:
        Tuple of true coefficient matrix, estimated coefficient matrix,
        and combined feature names

    Warning:
        Does not disambiguate between commutatively equivalent function
        names such as 'x z' and 'z x' or 'x^2' and 'x x'.

    Warning:
        In non-strict mode, when different input variables are detected in the
            SINDy model and in the true model, will attempt to translate true
            features to model inputs, e.g. ``x^2`` -> ``x0^2``.  This is a
            text replacement, not a lexical replacement, so there are edge cases
            where translation fails.  Input variables are sorted alphabetically.
    """
    inputs_model = cast(list[str], model.feature_names)
    if isinstance(model_true, list):
        warn(
            "Passing coeff_true as merely the list of functions is deprecated. "
            " It is now required to pass a tuple of system coordinate variables"
            " as well as the list of functions.",
            DeprecationWarning,
        )
        in_funcs = set.union(*[set(d.keys()) for d in model_true])

        def extract_vars(fname: str) -> set[str]:
            # split on ops like *,^, only accept x, x2, from x * x2 ^ 2, but need x'
            return {
                var for var in re.split(r"[^\w']", fname) if re.match(r"[^\d]", var)
            }

        inputs_set = set.union(*[extract_vars(fname) for fname in in_funcs])
        inputs_true = sorted(inputs_set)
        coeff_true = model_true
    else:
        inputs_true, coeff_true = model_true
    model_features = model.get_feature_names()
    true_features = [set(coeffs.keys()) for coeffs in coeff_true]
    if inputs_true != inputs_model:
        if strict:
            raise ValueError(
                "True model and fit model have different input variable names"
            )
        mapper = dict(zip(inputs_model, inputs_true, strict=True))
        translated_features: list[str] = []
        for feat in model_features:
            for k, v in mapper.items():
                feat = feat.replace(k, v)
            translated_features.append(feat)
        model_features = translated_features

    unmodeled_features = set(chain.from_iterable(true_features)) - set(model_features)
    model_features.extend(list(unmodeled_features))
    est_coeff_mat = model.coefficients()
    new_est_coeff = np.zeros((est_coeff_mat.shape[0], len(model_features)))
    new_est_coeff[:, : est_coeff_mat.shape[1]] = est_coeff_mat
    true_coeff_mat = np.zeros_like(new_est_coeff)
    for row, terms in enumerate(coeff_true):
        for term, coeff in terms.items():
            true_coeff_mat[row, model_features.index(term)] = coeff

    return true_coeff_mat, new_est_coeff, model_features


def coeff_metrics(coefficients, coeff_true):
    metrics = {}
    metrics["coeff_precision"] = sklearn.metrics.precision_score(
        coeff_true.flatten() != 0, coefficients.flatten() != 0
    )
    metrics["coeff_recall"] = sklearn.metrics.recall_score(
        coeff_true.flatten() != 0, coefficients.flatten() != 0
    )
    metrics["coeff_f1"] = sklearn.metrics.f1_score(
        coeff_true.flatten() != 0, coefficients.flatten() != 0
    )
    metrics["coeff_mse"] = sklearn.metrics.mean_squared_error(
        coeff_true.flatten(), coefficients.flatten()
    )
    metrics["coeff_mae"] = sklearn.metrics.mean_absolute_error(
        coeff_true.flatten(), coefficients.flatten()
    )
    metrics["main"] = metrics["coeff_f1"]
    return metrics
