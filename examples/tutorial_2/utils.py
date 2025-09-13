"""This module has several plotting utilities copoied from the
pysindy-experiments package, located at
https://github.com/Jacob-Stevens-Haas/gen-experiments
"""
from typing import Annotated
from typing import Optional
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_coefficients(
    coefficients: Annotated[np.ndarray, "(n_coord, n_features)"],
    input_features: Sequence[str],
    feature_names: Sequence[str],
    ax: Axes,
    **heatmap_kws,
) -> Axes:
    """Plot a set of dynamical system coefficients in a heatmap.

    Args:
        coefficients: A 2D array holding the coefficients of different
            library functions.  System dimension is rows, function index
            is columns
        input_features: system coordinate names, e.g. "x","y","z" or "u","v"
        feature_names: the names of the functions in the library.
        ax: the matplotlib axis to plot on
        **heatmap_kws: additional kwargs to seaborn's styling
    """

    def detex(input: str) -> str:
        if input[0] == "$":
            input = input[1:]
        if input[-1] == "$":
            input = input[:-1]
        return input

    if input_features is None:
        input_features = [r"$\dot x_" + f"{k}$" for k in range(coefficients.shape[0])]
    else:
        input_features = [r"$\dot " + f"{detex(fi)}$" for fi in input_features]

    if feature_names is None:
        feature_names = [f"f{k}" for k in range(coefficients.shape[1])]

    with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):
        heatmap_args = {
            "xticklabels": input_features,
            "yticklabels": feature_names,
            "center": 0.0,
            "cmap": sns.color_palette("vlag", n_colors=20, as_cmap=True),
            "ax": ax,
            "linewidths": 0.1,
            "linecolor": "whitesmoke",
        }
        heatmap_args.update(**heatmap_kws)
        coefficients = np.where(
            coefficients == 0, np.nan * np.empty_like(coefficients), coefficients
        )
        sns.heatmap(coefficients.T, **heatmap_args)

        ax.tick_params(axis="y", rotation=0)

    return ax


def compare_coefficient_plots(
    coefficients_est: Annotated[np.ndarray, "(n_coord, n_feat)"],
    coefficients_true: Annotated[np.ndarray, "(n_coord, n_feat)"],
    input_features: Sequence[str],
    feature_names: Sequence[str],
    scaling: bool = True,
    axs: Optional[Sequence[Axes]] = None,
) -> Figure:
    """Create plots of true and estimated coefficients.

    Args:
        scaling: Whether to scale coefficients so that magnitude of largest to
            smallest (in absolute value) is less than or equal to ten.
        axs: A sequence of axes of at least length two.  Plots are added to the
            first two axes in the list
    """
    n_cols = len(coefficients_est)

    # helps boost the color of small coefficients.  Maybe log is better?
    all_vals = np.hstack((coefficients_est.flatten(), coefficients_true.flatten()))
    nzs = all_vals[all_vals.nonzero()]
    max_val = np.max(np.abs(nzs), initial=0.0)
    min_val = np.min(np.abs(nzs), initial=np.inf)
    if scaling and np.isfinite(min_val) and max_val / min_val > 10:
        pwr_ratio = 1.0 / np.log10(max_val / min_val)
    else:
        pwr_ratio = 1

    def signed_root(x):
        return np.sign(x) * np.power(np.abs(x), pwr_ratio)

    with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):
        if axs is None:
            fig, axs = plt.subplots(
                1, 2, figsize=(1.9 * n_cols, 8), sharey=True, sharex=True
            )
            fig.tight_layout()
        else:
            fig = axs[0].figure

        vmax = signed_root(max_val)

        plot_coefficients(
            signed_root(coefficients_true),
            input_features=input_features,
            feature_names=feature_names,
            ax=axs[0],
            cbar=False,
            vmax=vmax,
            vmin=-vmax,
        )

        plot_coefficients(
            signed_root(coefficients_est),
            input_features=input_features,
            feature_names=feature_names,
            ax=axs[1],
            cbar=False,
            vmax=vmax,
            vmin=-vmax,
        )

        axs[0].set_title("True Coefficients", rotation=45)
        axs[1].set_title("Est. Coefficients", rotation=45)

    return fig
