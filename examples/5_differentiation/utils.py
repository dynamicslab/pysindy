import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

pal = sns.color_palette("Set1")
plot_kws = dict(alpha=0.7, linewidth=3)


def compare_methods(diffs, x, y, y_noisy, y_dot):
    n_methods = len(diffs)
    n_rows = (n_methods // 3) + int(n_methods % 3 > 0)
    fig, axs = plt.subplots(n_rows, 3, figsize=(15, 3 * n_rows), sharex=True)

    for (name, method), ax in zip(diffs, axs.flatten()):
        ax.plot(x, y_dot, label="Exact", color=pal[0], **plot_kws)
        ax.plot(x, method(y, x), ":", label="Approx.", color="black", **plot_kws)
        ax.plot(x, method(y_noisy, x), label="Noisy", color=pal[1], **plot_kws)
        ax.set(title=name)

    axs[0, 0].legend()
    fig.show()

    return axs


def print_equations(equations_clean, equations_noisy):
    print(f"{'':<30} {'Noiseless':<40} {'Noisy':<40}")

    for name in equations_clean.keys():
        print(f"{name:<30} {'':<40} {'':<40}")

        for k, (eq1, eq2) in enumerate(
            zip(equations_clean[name], equations_noisy[name])
        ):
            print(
                f"{'':<30} {'x' + str(k) + '=' + str(eq1):<40} {'x' + str(k) + '=' + str(eq2):<40}"
            )

        print(
            "-------------------------------------------------------------------------------------------"
        )


def plot_coefficients(
    coefficients, input_features=None, feature_names=None, ax=None, **heatmap_kws
):
    if input_features is None:
        input_features = [r"$\dot x_" + f"{k}$" for k in range(coefficients.shape[0])]
    else:
        input_features = [r"$\dot " + f"{fi}$" for fi in input_features]

    if feature_names is None:
        feature_names = [f"f{k}" for k in range(coefficients.shape[1])]

    with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

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

        sns.heatmap(coefficients.T, **heatmap_args)

        ax.tick_params(axis="y", rotation=0)

    return ax


def compare_coefficient_plots(
    coefficients_clean, coefficients_noisy, input_features=None, feature_names=None
):
    n_cols = len(coefficients_clean)

    def signed_sqrt(x):
        return np.sign(x) * np.sqrt(np.abs(x))

    with sns.axes_style(style="white", rc={"axes.facecolor": (0, 0, 0, 0)}):
        fig, axs = plt.subplots(
            2, n_cols, figsize=(1.9 * n_cols, 8), sharey=True, sharex=True
        )

        max_clean = max(np.max(np.abs(c)) for c in coefficients_clean.values())
        max_noisy = max(np.max(np.abs(c)) for c in coefficients_noisy.values())
        max_mag = np.sqrt(max(max_clean, max_noisy))

        for k, name in enumerate(coefficients_clean.keys()):
            plot_coefficients(
                signed_sqrt(coefficients_clean[name]),
                input_features=input_features,
                feature_names=feature_names,
                ax=axs[0, k],
                cbar=False,
                vmax=max_mag,
                vmin=-max_mag,
            )

            plot_coefficients(
                signed_sqrt(coefficients_clean[name]),
                input_features=input_features,
                feature_names=feature_names,
                ax=axs[1, k],
                cbar=False,
            )

            axs[0, k].set_title(name, rotation=45)

        axs[0, 0].set_ylabel("Noiseless", labelpad=10)
        axs[1, 0].set_ylabel("Noisy", labelpad=10)

        fig.tight_layout()


def plot_sho(x_train, x_train_noisy, x_smoothed=None):
    ax = plt.gca()
    ax.plot(x_train[:, 0], x_train[:, 1], ".", label="Clean", color=pal[0], **plot_kws)
    ax.plot(
        x_train_noisy[:, 0],
        x_train_noisy[:, 1],
        ".",
        label="Noisy",
        color=pal[1],
        **plot_kws,
    )
    if x_smoothed is not None:
        ax.plot(
            x_smoothed[:, 0],
            x_smoothed[:, 1],
            ".",
            label="Smoothed",
            color=pal[2],
            **plot_kws,
        )

    ax.set(title="Training data", xlabel="$x_0$", ylabel="$x_1$")
    ax.legend()
    return ax


def plot_lorenz(x_train, x_train_noisy, x_smoothed=None, ax=None):
    if ax is None:
        ax = plt.axes(projection="3d")
    ax.plot(
        x_train[:, 0],
        x_train[:, 1],
        x_train[:, 2],
        color=pal[0],
        label="Clean",
        **plot_kws,
    )

    ax.plot(
        x_train_noisy[:, 0],
        x_train_noisy[:, 1],
        x_train_noisy[:, 2],
        ".",
        color=pal[1],
        label="Noisy",
        alpha=0.3,
    )
    if x_smoothed is not None:
        ax.plot(
            x_smoothed[:, 0],
            x_smoothed[:, 1],
            x_smoothed[:, 2],
            ".",
            color=pal[2],
            label="Smoothed",
            alpha=0.3,
        )
    ax.set(title="Training data", xlabel="$x$", ylabel="$y$", zlabel="$z$")
    ax.legend()
    return ax
