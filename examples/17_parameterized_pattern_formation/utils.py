import os
import timeit

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import pysindy as ps
from pysindy.utils import lorenz

plt.ion()

integrator_keywords = {}
integrator_keywords["rtol"] = 1e-12
integrator_keywords["method"] = "RK45"
integrator_keywords["atol"] = 1e-12


def get_lorenz_trajectories(sigmas, rhos, betas, dt):
    """
    Generate a set of trajectories for the Lorenz ODEs.
    Parameters:
        sigmas: List of the sigma values.
        rhos: List of the rho values.
        betas: List of the beta values.
        dt: time step.
    """

    x_trains = []
    t_trains = []

    for i in range(len(sigmas)):
        t_train = np.arange(0, 10, dt)
        x0_train = [-8, 8, 27]
        t_train_span = (t_train[0], t_train[-1])
        x_train = solve_ivp(
            lorenz,
            t_train_span,
            x0_train,
            args=(sigmas[i], betas[i], rhos[i]),
            t_eval=t_train,
            **integrator_keywords
        ).y.T
        x_trains = x_trains + [x_train]
        t_trains = t_trains + [t_train]
    return x_trains, t_trains


def get_cgle_ic(scale0, scale2, spatial_grid):
    """
    Generate an initial condition for the CGLE.
    Parameters:
        scale0: scale of the random component.
        scale2: scale of the plane-wave component.
        spatial_grid: Spatial grid (assumed to be uniform).
    """
    nx = spatial_grid.shape[0]
    ny = spatial_grid.shape[1]
    ks = np.arange(-2, 3)
    phase_init = np.zeros((nx, ny), dtype=np.complex128) + 1
    kx = 2
    ky = 2
    phase_init = phase_init + scale2 * (
        np.exp(
            1j
            * (
                2 * np.pi * kx / nx * np.arange(nx)[:, np.newaxis]
                + 2 * np.pi * ky / ny * np.arange(ny)[np.newaxis, :]
            )
        )
        - 1
    )

    np.random.seed(100)
    for kx in ks:
        for ky in ks:
            scale = scale0 / (1 + (kx**2 + ky**2)) ** 0.5
            phase_init += (
                scale
                * (np.random.normal(0, 1) + 1j * np.random.normal(0, 1))
                * np.exp(
                    1j
                    * (
                        2 * np.pi * kx / nx * np.arange(nx)[:, np.newaxis]
                        + 2 * np.pi * ky / ny * np.arange(ny)[np.newaxis, :]
                    )
                )
            )

    phase = phase_init.reshape(nx * ny)
    return phase


def get_cgle_trajectory(b, c, x0, t1, t3, dt, spatial_grid):
    """
    Generate an trajectory for the CGLE.
    Parameters:
        b: diffusive CGLE parameter.
        c: nonlinear CGLE parameter.
        x0: initial condition
        t1: total integration time
        t3: time to start recording states
        dt: time step
        spatial_grid: Spatial grid (assumed to be uniform).
    """
    nx = spatial_grid.shape[0]
    ny = spatial_grid.shape[1]
    L = (spatial_grid[1, 0, 0] - spatial_grid[0, 0, 0]) * nx
    nt = int((t1 - t3) / dt)

    def cgle(t, Z, b, c):
        Zxxr = ps.SpectralDerivative(d=2, axis=0)._differentiate(
            np.real(Z).reshape((nx, ny)), 1.0 / nx
        )
        Zyyr = ps.SpectralDerivative(d=2, axis=1)._differentiate(
            np.real(Z).reshape((nx, ny)), 1.0 / ny
        )
        Zxxi = ps.SpectralDerivative(d=2, axis=0)._differentiate(
            np.imag(Z).reshape((nx, ny)), 1.0 / nx
        )
        Zyyi = ps.SpectralDerivative(d=2, axis=1)._differentiate(
            np.imag(Z).reshape((nx, ny)), 1.0 / ny
        )
        lap = (Zxxr + 1j * Zxxi + Zyyr + 1j * Zyyi).reshape(nx * ny)
        return Z - (1 - 1j * c) * Z * Z * Z.conjugate() + (1 + 1j * b) * lap / L**2

    phases = np.zeros([int(t1 / dt), nx * ny], dtype=np.complex128)
    phase = x0.reshape(nx * ny)
    dt1 = dt / 1000
    start = timeit.default_timer()
    for n in range(int(t1 / dt)):
        t = n * dt
        print("%.3f" % (t / t1), end="\r")
        sol = solve_ivp(
            cgle,
            [t, t + dt],
            phase,
            method="RK45",
            args=(b, c),
            rtol=1e-6,
            atol=1e-6,
            first_step=dt1,
        )
        dt1 = np.mean(np.diff(sol.t))
        phase = sol.y[:, -1]
        phases[n] = phase

    stop = timeit.default_timer()
    print(stop - start)
    x = np.zeros((nx, ny, nt, 2))
    x[:, :, :, 0] = np.real(
        phases.reshape((int(t1 / dt), nx, ny))[int(t3 / dt) :].transpose((1, 2, 0))
    )
    x[:, :, :, 1] = np.imag(
        phases.reshape((int(t1 / dt), nx, ny))[int(t3 / dt) :].transpose((1, 2, 0))
    )
    return x


def save_cgle_test_trajectories():
    """
    Save test trajectories with  b=2,1.5 and c=1.5,1.0 if not present.
    """
    nx = 128
    ny = 128
    L = 16
    t1 = 2e2
    t3 = 1.9e2
    dt = 1e-1

    spatial_grid = np.zeros((nx, ny, 2))
    spatial_grid[:, :, 0] = (
        (np.arange(nx) - nx // 2)[:, np.newaxis] * 2 * np.pi * L / nx
    )
    spatial_grid[:, :, 1] = (
        (np.arange(nx) - nx // 2)[np.newaxis, :] * 2 * np.pi * L / nx
    )

    bs = [2.0, 1.5]
    cs = [1.5, 1.0]
    scales = [1e-1, 1e-1]
    scales2 = [1e-2, 1e-2]
    if np.all(
        [os.path.exists("data/cgle/cgle_test_x" + str(i) + ".npy") for i in range(2)]
    ):
        xs_test = [np.load("data/cgle/cgle_test_x" + str(i) + ".npy") for i in range(2)]
    else:
        xs_test = []
        for i in range(len(bs)):
            b = bs[i]
            c = cs[i]
            scale0 = scales[i]
            scale2 = scales2[i]
            x0 = get_cgle_ic(scale0, scale2, spatial_grid)
            x = get_cgle_trajectory(b, c, x0, t1, t3, dt, spatial_grid)
            xs_test = xs_test + [x]
        for i in range(len(xs_test)):
            if not os.path.exists("data/cgle"):
                os.mkdir("data/cgle")
            np.save("data/cgle/cgle_test_x" + str(i), xs_test[i])


def save_cgle_random_trajectories():
    """
    Save training CGLE trajectories with random values of b and c if not present.
    """
    nx = 128
    ny = 128
    L = 16
    t1 = 2e2
    t3 = 1.9e2
    dt = 1e-1

    spatial_grid = np.zeros((nx, ny, 2))
    spatial_grid[:, :, 0] = (
        (np.arange(nx) - nx // 2)[:, np.newaxis] * 2 * np.pi * L / nx
    )
    spatial_grid[:, :, 1] = (
        (np.arange(nx) - nx // 2)[np.newaxis, :] * 2 * np.pi * L / nx
    )

    np.random.seed(100)
    num_trajectories = 5
    bs = np.random.normal(1.5, 0.5, size=num_trajectories)
    cs = np.random.normal(1.0, 0.25, size=num_trajectories)
    scales = [1e0] * 5
    scales2 = [0] * 5

    if np.all(
        [
            os.path.exists("data/cgle/cgle_random_x" + str(i) + ".npy")
            for i in range(num_trajectories)
        ]
    ):
        xs_random = [
            np.load("data/cgle/cgle_random_x" + str(i) + ".npy")
            for i in range(num_trajectories)
        ]
    else:
        xs_random = []
        for i in range(len(bs)):
            b = bs[i]
            c = cs[i]
            scale0 = scales[i]
            scale2 = scales2[i]
            x0 = get_cgle_ic(scale0, scale2, spatial_grid)
            x = get_cgle_trajectory(b, c, x0, t1, t3, dt, spatial_grid)
            xs_random = xs_random + [x]
        for i in range(len(xs_random)):
            if not os.path.exists("data/cgle"):
                os.mkdir("data/cgle")
            np.save("data/cgle/cgle_random_x" + str(i), xs_random[i])


def cgle_noise_sweeps():
    """
    Run the differential SINDyCP fit with varying noise intensities and save the data.
    """
    num = 10
    noisemin = -5
    noisemax = -1
    intensities = 10 ** (noisemin + np.arange(num) / (num - 1) * (noisemax - noisemin))
    nx = 128
    ny = 128
    L = 16
    t1 = 2e2
    t3 = 1.9e2
    dt = 1e-1
    nt = int((t1 - t3) / dt)

    spatial_grid = np.zeros((nx, ny, 2))
    spatial_grid[:, :, 0] = (
        (np.arange(nx) - nx // 2)[:, np.newaxis] * 2 * np.pi * L / nx
    )
    spatial_grid[:, :, 1] = (
        (np.arange(nx) - nx // 2)[np.newaxis, :] * 2 * np.pi * L / nx
    )

    spatiotemporal_grid = np.zeros((nx, ny, nt, 3))
    spatiotemporal_grid[:, :, :, :2] = spatial_grid[:, :, np.newaxis, :]
    spatiotemporal_grid[:, :, :, 2] = dt * np.arange(nt)

    if (
        not os.path.exists("data/cgle/cgle_scores0.npy")
        or not os.path.exists("data/cgle/cgle_scores1.npy")
        or not os.path.exists("data/cgle/cgle_scores2.npy")
    ):
        save_cgle_test_trajectories()
        save_cgle_random_trajectories()

    bs = [2.0, 2.0, 0.5, 1.0]
    cs = [1.0, 0.75, 0.5, 0.75]
    us = [[bs[i], cs[i]] for i in range(len(bs))]
    xs = [np.load("data/cgle/cgle_x" + str(i) + ".npy") for i in range(4)]

    bs = [2.0, 1.5]
    cs = [1.5, 1.0]
    us_test = [[bs[i], cs[i]] for i in range(len(bs))]
    xs_test = [np.load("data/cgle/cgle_test_x" + str(i) + ".npy") for i in range(2)]

    # sweep with training on original data
    nscores = []

    if not os.path.exists("data/cgle/cgle_scores0.npy"):
        start = timeit.default_timer()

        for scale in intensities:
            print(scale)
            library_functions = [
                lambda x: x,
                lambda x: x**3,
                lambda x, y: x**2 * y,
                lambda x, y: y**2 * x,
            ]
            function_names = [
                lambda x: x,
                lambda x: x + x + x,
                lambda x, y: x + x + y,
                lambda x, y: x + y + y,
            ]
            feature_lib = ps.PDELibrary(
                library_functions=library_functions,
                derivative_order=2,
                spatial_grid=spatial_grid,
                include_interaction=False,
                function_names=function_names,
                differentiation_method=ps.SpectralDerivative,
            )
            library_functions = [lambda x: x]
            function_names = [lambda x: x]
            parameter_lib = ps.PDELibrary(
                library_functions=library_functions,
                derivative_order=0,
                include_interaction=False,
                function_names=function_names,
                include_bias=True,
            )
            lib = ps.ParameterizedLibrary(
                parameter_library=parameter_lib,
                feature_library=feature_lib,
                num_parameters=2,
                num_features=2,
            )
            opt = ps.STLSQ(threshold=5e-1, alpha=1e-3, normalize_columns=False)
            xs_noisy = [
                xs[i] + scale * np.random.normal(0, 1, size=xs[i].shape)
                for i in range(len(xs))
            ]
            model = ps.SINDy(
                feature_library=lib, optimizer=opt, feature_names=["x", "y", "b", "c"]
            )
            model.fit(xs_noisy, u=us, t=dt, multiple_trajectories=True)

            nscores = nscores + [
                model.score(xs_test, u=us_test, t=dt, multiple_trajectories=True)
            ]

        stop = timeit.default_timer()
        print(stop - start)

        scores0 = np.concatenate(
            [np.array(intensities)[:, np.newaxis], np.array(nscores)[:, np.newaxis]],
            axis=1,
        )
        np.save("data/cgle/cgle_scores0", scores0)

    # sweep with training on random data
    np.random.seed(100)
    num_trajectories = 5
    bs = np.random.normal(1.5, 0.5, size=num_trajectories)
    cs = np.random.normal(1.0, 0.25, size=num_trajectories)
    us_random = [[bs[i], cs[i]] for i in range(len(bs))]
    xs_random = [np.load("data/cgle/cgle_random_x" + str(i) + ".npy") for i in range(4)]

    nscores = []

    if not os.path.exists("data/cgle/cgle_scores1.npy"):
        start = timeit.default_timer()

        for scale in intensities:
            print(scale)
            library_functions = [
                lambda x: x,
                lambda x: x**3,
                lambda x, y: x**2 * y,
                lambda x, y: y**2 * x,
            ]
            function_names = [
                lambda x: x,
                lambda x: x + x + x,
                lambda x, y: x + x + y,
                lambda x, y: x + y + y,
            ]
            feature_lib = ps.PDELibrary(
                library_functions=library_functions,
                derivative_order=2,
                spatial_grid=spatial_grid,
                include_interaction=False,
                function_names=function_names,
                differentiation_method=ps.SpectralDerivative,
            )
            library_functions = [lambda x: x]
            function_names = [lambda x: x]
            parameter_lib = ps.PDELibrary(
                library_functions=library_functions,
                derivative_order=0,
                include_interaction=False,
                function_names=function_names,
                include_bias=True,
            )
            lib = ps.ParameterizedLibrary(
                parameter_library=parameter_lib,
                feature_library=feature_lib,
                num_parameters=2,
                num_features=2,
            )
            opt = ps.STLSQ(threshold=5e-1, alpha=1e-3, normalize_columns=False)
            xs_noisy = [
                xs_random[i] + scale * np.random.normal(0, 1, size=xs_random[i].shape)
                for i in range(len(xs))
            ]
            model = ps.SINDy(
                feature_library=lib, optimizer=opt, feature_names=["x", "y", "b", "c"]
            )
            model.fit(
                xs_noisy, u=us_random[: len(xs)], t=dt, multiple_trajectories=True
            )

            nscores = nscores + [
                model.score(xs_test, u=us_test, t=dt, multiple_trajectories=True)
            ]

        stop = timeit.default_timer()
        print(stop - start)

        scores1 = np.concatenate(
            [np.array(intensities)[:, np.newaxis], np.array(nscores)[:, np.newaxis]],
            axis=1,
        )
        np.save("data/cgle/cgle_scores1", scores1)
    else:
        scores1 = np.load("data/cgle/cgle_scores1.npy")

    # sweep with varying number and length of random training trajectories
    nums = [2, 3, 4, 5]
    nts = [25, 50, 75, 100]

    if not os.path.exists("data/cgle/cgle_scores2.npy"):
        start = timeit.default_timer()
        scoreses = []
        sampleses = []
        for num in nums:
            scores = []
            samples = []
            for nt in nts:
                print(num, nt, end="  \r")
                library_functions = [
                    lambda x: x,
                    lambda x: x**3,
                    lambda x, y: x**2 * y,
                    lambda x, y: y**2 * x,
                ]
                function_names = [
                    lambda x: x,
                    lambda x: x + x + x,
                    lambda x, y: x + x + y,
                    lambda x, y: x + y + y,
                ]
                feature_lib = ps.PDELibrary(
                    library_functions=library_functions,
                    derivative_order=2,
                    spatial_grid=spatial_grid,
                    include_interaction=False,
                    function_names=function_names,
                    differentiation_method=ps.SpectralDerivative,
                )
                library_functions = [lambda x: x]
                function_names = [lambda x: x]
                parameter_lib = ps.PDELibrary(
                    library_functions=library_functions,
                    derivative_order=0,
                    include_interaction=False,
                    function_names=function_names,
                    include_bias=True,
                )
                lib = ps.ParameterizedLibrary(
                    parameter_library=parameter_lib,
                    feature_library=feature_lib,
                    num_parameters=2,
                    num_features=2,
                )
                opt = ps.STLSQ(threshold=5e-1, alpha=1e-3, normalize_columns=False)
                scale = 1e-3
                shape = np.array(xs_random[0].shape)
                shape[-2] = nt
                xs_noisy = [
                    xs_random[i][:, :, :nt] + scale * np.random.normal(0, 1, size=shape)
                    for i in range(num)
                ]
                model = ps.SINDy(
                    feature_library=lib,
                    optimizer=opt,
                    feature_names=["x", "y", "b", "c"],
                )
                model.fit(xs_noisy, u=us_random, t=dt, multiple_trajectories=True)

                scores = scores + [
                    model.score(xs_test, u=us_test, t=dt, multiple_trajectories=True)
                ]
                samples = samples + [
                    np.sum(
                        [
                            np.product(xs_noisy[i].shape[:3])
                            for i in range(len(xs_noisy))
                        ]
                    )
                ]
            scoreses = scoreses + [scores]
            sampleses = sampleses + [samples]
        stop = timeit.default_timer()
        print(stop - start)
        scores2 = np.concatenate(
            [
                np.array(sampleses)[:, :, np.newaxis],
                np.array(scoreses)[:, :, np.newaxis],
            ],
            axis=2,
        )
        np.save("data/cgle/cgle_scores2", scores2)
    else:
        scores2 = np.load("data/cgle/cgle_scores2.npy")


def cgle_weak_noise_sweeps():
    """
    Run the weak SINDyCP fit with varying noise intensities and save the data.
    """
    num = 10
    noisemin = -5
    noisemax = -1
    intensities = 10 ** (noisemin + np.arange(num) / (num - 1) * (noisemax - noisemin))
    nx = 128
    ny = 128
    L = 16
    t1 = 2e2
    t3 = 1.9e2
    dt = 1e-1
    nt = int((t1 - t3) / dt)

    spatial_grid = np.zeros((nx, ny, 2))
    spatial_grid[:, :, 0] = (
        (np.arange(nx) - nx // 2)[:, np.newaxis] * 2 * np.pi * L / nx
    )
    spatial_grid[:, :, 1] = (
        (np.arange(nx) - nx // 2)[np.newaxis, :] * 2 * np.pi * L / nx
    )

    spatiotemporal_grid = np.zeros((nx, ny, nt, 3))
    spatiotemporal_grid[:, :, :, :2] = spatial_grid[:, :, np.newaxis, :]
    spatiotemporal_grid[:, :, :, 2] = dt * np.arange(nt)

    if (
        not os.path.exists("data/cgle/cgle_weak_scores0.npy")
        or not os.path.exists("data/cgle/cgle_weak_scores1.npy")
        or not os.path.exists("data/cgle/cgle_weak_scores2.npy")
    ):
        save_cgle_test_trajectories()
        save_cgle_random_trajectories()

    bs = [2.0, 2.0, 0.5, 1.0]
    cs = [1.0, 0.75, 0.5, 0.75]
    us = [[bs[i], cs[i]] for i in range(len(bs))]
    xs = [np.load("data/cgle/cgle_x" + str(i) + ".npy") for i in range(len(bs))]
    bs = [2.0, 1.5]
    cs = [1.5, 1.0]
    us_test = [[bs[i], cs[i]] for i in range(len(bs))]
    xs_test = [np.load("data/cgle/cgle_test_x" + str(i) + ".npy") for i in range(2)]

    bs = [2.0, 1.5]
    cs = [1.5, 1.0]
    us_test = [[bs[i], cs[i]] for i in range(len(bs))]
    xs_test = [np.load("data/cgle/cgle_test_x" + str(i) + ".npy") for i in range(2)]
    # sweep with training on original data
    nscores_weak = []

    if not os.path.exists("data/cgle/cgle_weak_scores0.npy"):
        save_cgle_test_trajectories()
        save_cgle_random_trajectories()
        start = timeit.default_timer()

        for scale in intensities:
            print(scale)
            np.random.seed(100)
            library_functions = [
                lambda x: x,
                lambda x: x**3,
                lambda x, y: x**2 * y,
                lambda x, y: y**2 * x,
            ]
            function_names = [
                lambda x: x,
                lambda x: x + x + x,
                lambda x, y: x + x + y,
                lambda x, y: x + y + y,
            ]
            feature_lib = ps.WeakPDELibrary(
                library_functions=library_functions,
                derivative_order=2,
                spatiotemporal_grid=spatiotemporal_grid,
                include_interaction=False,
                function_names=function_names,
                K=500,
                H_xt=[L * 2 * np.pi / 10, L * 2 * np.pi / 10, (t1 - t3) / 10],
            )
            np.random.seed(100)
            library_functions = [lambda x: x]
            function_names = [lambda x: x]
            parameter_lib = ps.WeakPDELibrary(
                library_functions=library_functions,
                spatiotemporal_grid=spatiotemporal_grid,
                derivative_order=0,
                include_interaction=False,
                function_names=function_names,
                include_bias=True,
                K=500,
                H_xt=[L * 2 * np.pi / 10, L * 2 * np.pi / 10, (t1 - t3) / 10],
            )
            lib = ps.ParameterizedLibrary(
                parameter_library=parameter_lib,
                feature_library=feature_lib,
                num_parameters=2,
                num_features=2,
            )
            opt = ps.STLSQ(threshold=5e-1, alpha=1e-3, normalize_columns=False)
            xs_noisy = [
                xs[i] + scale * np.random.normal(0, 1, size=xs[i].shape)
                for i in range(len(xs))
            ]
            model = ps.SINDy(
                feature_library=lib, optimizer=opt, feature_names=["x", "y", "b", "c"]
            )
            model.fit(xs_noisy, u=us, t=dt, multiple_trajectories=True)

            nscores_weak = nscores_weak + [
                model.score(xs_test, u=us_test, t=dt, multiple_trajectories=True)
            ]

        stop = timeit.default_timer()
        print(stop - start)

        weak_scores0 = np.concatenate(
            [
                np.array(intensities)[:, np.newaxis],
                np.array(nscores_weak)[:, np.newaxis],
            ],
            axis=1,
        )
        np.save("data/cgle/cgle_weak_scores0", weak_scores0)
    else:
        weak_scores0 = np.load("data/cgle/cgle_weak_scores0.npy")

    # sweep with training on random data
    np.random.seed(100)
    num_trajectories = 5
    bs = np.random.normal(1.5, 0.5, size=num_trajectories)
    cs = np.random.normal(1.0, 0.25, size=num_trajectories)
    us_random = [[bs[i], cs[i]] for i in range(len(bs))]
    xs_random = [
        np.load("data/cgle/cgle_random_x" + str(i) + ".npy") for i in range(len(bs))
    ]

    nscores_weak = []

    if not os.path.exists("data/cgle/cgle_weak_scores1.npy"):
        start = timeit.default_timer()

        for scale in intensities:
            print(scale)
            np.random.seed(100)
            library_functions = [
                lambda x: x,
                lambda x: x**3,
                lambda x, y: x**2 * y,
                lambda x, y: y**2 * x,
            ]
            function_names = [
                lambda x: x,
                lambda x: x + x + x,
                lambda x, y: x + x + y,
                lambda x, y: x + y + y,
            ]
            feature_lib = ps.WeakPDELibrary(
                library_functions=library_functions,
                derivative_order=2,
                spatiotemporal_grid=spatiotemporal_grid,
                include_interaction=False,
                function_names=function_names,
                K=500,
                H_xt=[L * 2 * np.pi / 10, L * 2 * np.pi / 10, (t1 - t3) / 10],
            )
            np.random.seed(100)
            library_functions = [lambda x: x]
            function_names = [lambda x: x]
            parameter_lib = ps.WeakPDELibrary(
                library_functions=library_functions,
                spatiotemporal_grid=spatiotemporal_grid,
                derivative_order=0,
                include_interaction=False,
                function_names=function_names,
                include_bias=True,
                K=500,
                H_xt=[L * 2 * np.pi / 10, L * 2 * np.pi / 10, (t1 - t3) / 10],
            )
            lib = ps.ParameterizedLibrary(
                parameter_library=parameter_lib,
                feature_library=feature_lib,
                num_parameters=2,
                num_features=2,
            )
            opt = ps.STLSQ(threshold=5e-1, alpha=1e-3, normalize_columns=False)
            xs_noisy = [
                xs_random[i] + scale * np.random.normal(0, 1, size=xs_random[i].shape)
                for i in range(len(xs))
            ]
            model = ps.SINDy(
                feature_library=lib, optimizer=opt, feature_names=["x", "y", "b", "c"]
            )
            model.fit(
                xs_noisy, u=us_random[: len(xs)], t=dt, multiple_trajectories=True
            )

            nscores_weak = nscores_weak + [
                model.score(xs_test, u=us_test, t=dt, multiple_trajectories=True)
            ]

        stop = timeit.default_timer()
        print(stop - start)

        weak_scores1 = np.concatenate(
            [
                np.array(intensities)[:, np.newaxis],
                np.array(nscores_weak)[:, np.newaxis],
            ],
            axis=1,
        )
        np.save("data/cgle/cgle_weak_scores1", weak_scores1)
    else:
        weak_scores1 = np.load("data/cgle/cgle_weak_scores1.npy")

    # sweep with varying number and length of random training trajectories
    nums = [2, 3, 4, 5]
    nts = [25, 50, 75, 100]

    if not os.path.exists("data/cgle/cgle_weak_scores2.npy"):
        start = timeit.default_timer()
        weak_scoreses = []
        weak_sampleses = []
        for num in nums:
            scores = []
            samples = []
            for nt in nts:
                print(num, nt, end="  \r")
                np.random.seed(100)
                library_functions = [
                    lambda x: x,
                    lambda x: x**3,
                    lambda x, y: x**2 * y,
                    lambda x, y: y**2 * x,
                ]
                function_names = [
                    lambda x: x,
                    lambda x: x + x + x,
                    lambda x, y: x + x + y,
                    lambda x, y: x + y + y,
                ]
                T = (
                    spatiotemporal_grid[0, 0, nt - 1, -1]
                    - spatiotemporal_grid[0, 0, 0, -1]
                )
                feature_lib = ps.WeakPDELibrary(
                    library_functions=library_functions,
                    derivative_order=2,
                    spatiotemporal_grid=spatiotemporal_grid[:, :, :nt],
                    include_interaction=False,
                    function_names=function_names,
                    K=500,
                    H_xt=[L * 2 * np.pi / 10, L * 2 * np.pi / 10, T / 10],
                )
                np.random.seed(100)
                library_functions = [lambda x: x]
                function_names = [lambda x: x]
                parameter_lib = ps.WeakPDELibrary(
                    library_functions=library_functions,
                    spatiotemporal_grid=spatiotemporal_grid[:, :, :nt],
                    derivative_order=0,
                    include_interaction=False,
                    function_names=function_names,
                    include_bias=True,
                    K=500,
                    H_xt=[L * 2 * np.pi / 10, L * 2 * np.pi / 10, T / 10],
                )
                lib = ps.ParameterizedLibrary(
                    parameter_library=parameter_lib,
                    feature_library=feature_lib,
                    num_parameters=2,
                    num_features=2,
                )
                opt = ps.STLSQ(threshold=5e-1, alpha=1e-3, normalize_columns=False)
                scale = 1e-3
                shape = np.array(xs_random[0].shape)
                shape[-2] = nt
                xs_noisy = [
                    xs_random[i][:, :, :nt] + scale * np.random.normal(0, 1, size=shape)
                    for i in range(num)
                ]
                model = ps.SINDy(
                    feature_library=lib,
                    optimizer=opt,
                    feature_names=["x", "y", "b", "c"],
                )
                model.fit(xs_noisy, u=us_random, t=dt, multiple_trajectories=True)

                scores = scores + [
                    model.score(xs_test, u=us_test, t=dt, multiple_trajectories=True)
                ]
                samples = samples + [
                    np.sum(
                        [
                            np.product(xs_noisy[i].shape[:3])
                            for i in range(len(xs_noisy))
                        ]
                    )
                ]
            weak_scoreses = weak_scoreses + [scores]
            weak_sampleses = weak_sampleses + [samples]
        stop = timeit.default_timer()
        print(stop - start)
        weak_scores2 = np.concatenate(
            [
                np.array(weak_sampleses)[:, :, np.newaxis],
                np.array(weak_scoreses)[:, :, np.newaxis],
            ],
            axis=2,
        )
        np.save("data/cgle/cgle_weak_scores2", weak_scores2)
    else:
        weak_scores2 = np.load("data/cgle/cgle_weak_scores2.npy")


def get_oregonator_ic(scale0, scale2, spatial_grid):
    """
    Generate an initial condition for the oregonator model.
    Parameters:
        scale0: scale of the random component.
        scale1: scale of the plane wave component.
        spatial grid: Spatial grid, assumed to be uniform.
    """
    n = spatial_grid.shape[0]
    ks = np.arange(-3, 4)
    phase_init = np.zeros((n, n), dtype=np.complex128) + 1
    kx = 2
    ky = 2
    phase_init = phase_init + scale2 * (
        np.exp(
            1j
            * (
                2 * np.pi * kx / n * np.arange(n)[:, np.newaxis]
                + 2 * np.pi * ky / n * np.arange(n)[np.newaxis, :]
            )
        )
        - 1
    )

    np.random.seed(100)
    for kx in ks:
        for ky in ks:
            scale = scale0 / (1 + (kx**2 + ky**2)) ** 0.5
            phase_init += (
                scale
                * (np.random.normal(0, 1) + 1j * np.random.normal(0, 1))
                * np.exp(
                    1j
                    * (
                        2 * np.pi * kx / n * np.arange(n)[:, np.newaxis]
                        + 2 * np.pi * ky / n * np.arange(n)[np.newaxis, :]
                    )
                )
            )
    X0 = np.zeros((n, n)) + 2.99045e-6 * (1 + np.real(phase_init))
    Y0 = np.zeros((n, n)) + 0.000014988 * (1 + np.real(phase_init))
    Z0 = np.zeros((n, n)) + 0.0000380154 * (1 + np.imag(phase_init))
    return X0, Y0, Z0


def get_oregonator_trajectory(u0, b, t1, dt, spatial_grid):
    """
    Generate a trajectory for the Oregonator PDEs.
    Parameters:
        u0: Initial condition.
        b: Parameter value for the control concentration.
        t1: Total integration time.
        dt: time step.
        spatial_grid: Spatial grid, assumed to be uniform
    """
    nt = int(np.round(t1 / dt))
    n = spatial_grid.shape[0]

    def oregonator(
        t,
        u,
        b,
        k1=2,
        k2=1e6,
        k3=10,
        k4=2e3,
        k5=1,
        Dx=1e-5,
        Dy=1.6e-5,
        Dz=0.6e-5,
        H=0.5,
        A=1,
        h=1.0,
    ):
        B = A * 0.786642
        n = spatial_grid.shape[0]
        L = (spatial_grid[1, 0, 0] - spatial_grid[0, 0, 0]) * n
        X = np.reshape(u[: n * n], (n, n))
        Y = np.reshape(u[n * n : 2 * n * n], (n, n))
        Z = np.reshape(u[2 * n * n : 3 * n * n], (n, n))
        dX = ps.SpectralDerivative(d=2, axis=0)._differentiate(
            X, L / n
        ) + ps.SpectralDerivative(d=2, axis=1)._differentiate(X, L / n)
        dY = ps.SpectralDerivative(d=2, axis=0)._differentiate(
            Y, L / n
        ) + ps.SpectralDerivative(d=2, axis=1)._differentiate(Y, L / n)
        dZ = ps.SpectralDerivative(d=2, axis=0)._differentiate(
            Z, L / n
        ) + ps.SpectralDerivative(d=2, axis=1)._differentiate(Z, L / n)
        Xt = (
            k1 * A * H * H * Y
            - k2 * H * X * Y
            + k3 * A * H * X
            - 2 * k4 * X * X
            + Dx * dX
        )
        Yt = -k1 * A * H * H * Y - k2 * H * X * Y + k5 * h * b * B * Z + Dy * dY
        Zt = 2 * k3 * A * H * X - k5 * b * B * Z + Dz * dZ
        ut_updated = np.concatenate([Xt.ravel(), Yt.ravel(), Zt.ravel()])
        return ut_updated

    t = dt * np.arange(nt)
    X = np.zeros((n, n, nt))
    Y = np.zeros((n, n, nt))
    Z = np.zeros((n, n, nt))

    ut = u0
    X[:, :, 0] = ut[: n * n].reshape((n, n))
    Y[:, :, 0] = ut[n * n : 2 * n * n].reshape((n, n))
    Z[:, :, 0] = ut[2 * n * n : 3 * n * n].reshape((n, n))
    dt = t[1] - t[0] / 2
    try:
        for i in range(len(t) - 1):
            print("%.3f\t\r" % (i / len(t)), end="", flush=True)
            usol = solve_ivp(
                oregonator,
                (t[i], t[i + 1]),
                args=(b,),
                y0=ut,
                first_step=dt,
                **integrator_keywords
            )
            if not usol.success:
                print(usol.message)
                break

            dt = np.diff(usol.t)[-1]
            ut = usol.y[:, -1]
            X[:, :, i + 1] = ut[: n * n].reshape((n, n))
            Y[:, :, i + 1] = ut[n * n : 2 * n * n].reshape((n, n))
            Z[:, :, i + 1] = ut[2 * n * n : 3 * n * n].reshape((n, n))
    except KeyboardInterrupt:
        print("keyboard!")
    return X, Y, Z


def get_homogeneous_oregonator_trajectory(b, t1, dt):
    """
    Generate a trajectory for the homogeneous Oregonator ODEs.
    Parameters:
        b: Parameter value for the control concentration.
        t1: Total integration time.
        dt: time step.
    """
    nt = int(np.round(t1 / dt))
    t = dt * np.arange(nt)
    X = np.zeros((nt))
    Y = np.zeros((nt))
    Z = np.zeros((nt))

    def oregonator_homogeneous(
        t, u, b, k1=2, k2=1e6, k3=10, k4=2e3, k5=1, H=0.5, A=1, h=1.0
    ):
        B = A * 0.786642
        X = u[0]
        Y = u[1]
        Z = u[2]
        Xt = k1 * A * H * H * Y - k2 * H * X * Y + k3 * A * H * X - 2 * k4 * X * X
        Yt = -k1 * A * H * H * Y - k2 * H * X * Y + k5 * h * b * B * Z
        Zt = 2 * k3 * A * H * X - k5 * b * B * Z
        return np.array([Xt, Yt, Zt])

    u0 = np.array([2.99045e-6, 0.000014988, 0.000037639]) * 1.1
    ut = u0
    X[0] = ut[0]
    Y[0] = ut[1]
    Z[0] = ut[2]
    dt = t[1] - t[0] / 2
    print("%.2f\r" % (b), end="", flush=True)
    for i in range(len(t) - 1):
        usol = solve_ivp(
            oregonator_homogeneous,
            (t[i], t[i + 1]),
            args=(b,),
            y0=ut,
            first_step=dt,
            **integrator_keywords
        )
        dt = np.diff(usol.t)[-1]
        ut = usol.y[:, -1]
        X[i + 1] = ut[0]
        Y[i + 1] = ut[1]
        Z[i + 1] = ut[2]
    return X, Y, Z


def animate_oregonator():
    """
    Save an animation for the oregonator model
    """
    if not os.path.exists("animation"):
        os.mkdir("animation")

    bs = np.linspace(0.88, 0.98, 6)
    xs = [np.load("data/oregonator/canard_" + str(0) + ".npy")]
    xs = xs + [
        np.load("data/oregonator/oregonator_" + str(i) + ".npy")
        for i in range(0, len(bs), 2)
    ]

    Nt = xs[0].shape[2]
    num = len(xs)

    for i in range(Nt):
        plt.figure(figsize=(8, 8))
        print(i, end="\r")
        for j in range(num):
            plt.subplot(2, 2, j + 1)
            plt.imshow(xs[j][:, :, i, 0])
            plt.clim(2e-6, 8e-6)
            plt.xticks([])
            plt.yticks([])
        plt.savefig("animation/%04d.png" % (i))
        plt.close()


def animate_clge(xs, us):
    """
    Save an animation for the cgle model
    """
    if not os.path.exists("animation"):
        os.mkdir("animation")
    xs = [np.load("data/cgle/cgle_x" + str(i) + ".npy") for i in range(4)]
    us = [np.load("data/cgle/cgle_u" + str(i) + ".npy") for i in range(4)]
    bs = [us[i][0, 0, 0, 0] for i in range(4)]
    cs = [us[i][0, 0, 0, 1] for i in range(4)]
    nx = 128
    ny = 128
    L = 16
    t1 = 2e2
    t3 = 1.9e2
    dt = 1e-1

    spatial_grid = np.zeros((nx, ny, 2))
    spatial_grid[:, :, 0] = (
        (np.arange(nx) - nx // 2)[:, np.newaxis] * 2 * np.pi * L / nx
    )
    spatial_grid[:, :, 1] = (
        (np.arange(nx) - nx // 2)[np.newaxis, :] * 2 * np.pi * L / nx
    )

    xs_animation = []
    t1 = 100
    t3 = 0
    dt = 0.1

    for i in range(len(bs)):
        x0 = xs[i][:, :, -1, 0] + 1j * xs[i][:, :, -1, 1]
        x = get_cgle_trajectory(x0, bs[i], cs[i], t1, t3, dt, spatial_grid)
        xs_animation.append(x)

    if not os.path.exists("animation"):
        os.mkdir("animation")

    start = timeit.default_timer()
    for n in range(xs_animation[0].shape[2]):
        print(n, end="  \r")
        fig, axs = plt.subplots(2, 2, figsize=(5, 4.5), constrained_layout=True)
        for i in range(len(xs_animation)):
            plt.subplot(2, 2, i + 1)
            phase = (
                np.arctan2(xs_animation[i][:, :, n, 0], xs_animation[i][:, :, n, 1])
                + np.pi
            ) / (2 * np.pi)
            amp = (
                xs_animation[i][:, :, n, 0] ** 2 + xs_animation[i][:, :, n, 1] ** 2
            ) ** 0.5 / 1.5
            plot = np.triu(amp) + np.tril(phase, k=-1)
            pl = plt.pcolormesh(plot, vmin=0, vmax=1, cmap="twilight")
            plt.gca().set_xticks([0, 32, 64, 96, 128])
            plt.gca().set_xticklabels(["$-L$", "$-L/2$", "0", "$L/2$", "$L$"])
            plt.gca().set_yticks([0, 32, 64, 96, 128])
            plt.gca().set_yticklabels(["$-L$", "$-L/2$", "0", "$L/2$", "$L$"])
            if i == 2 or i == 3:
                plt.xlabel("$x$")
            else:
                plt.gca().set_xticklabels([])
            if i == 0 or i == 2:
                plt.ylabel("$y$")
            else:
                plt.gca().set_yticklabels([])

        plt.colorbar(
            pl,
            ax=axs[:, :],
            orientation="horizontal",
            location="top",
            ticks=[0, 0.25, 0.5, 0.75, 1],
            label=r"$\phi/2\pi$",
            shrink=0.6,
        )
        plt.colorbar(
            pl,
            ax=axs[:, :],
            orientation="vertical",
            ticks=[0, 0.25, 0.5, 0.75, 1],
            label="$r/1.5$",
            shrink=0.6,
        )
        plt.savefig(
            "animation/%04d.png" % (n), pad_inches=0.5, dpi=200, bbox_inches="tight"
        )
        plt.close()

    stop = timeit.default_timer()
    print(stop - start)


def get_sh_ic(scale0, spatial_grid):
    """
    Generate an initial condition for the Swift Hohenberg equation.
    Parameters:
        scale0: scale of the random component.
        spatial_grid: Spatial grid (assumed to be uniform).
    """
    nx = spatial_grid.shape[0]

    ks = np.arange(-20, 21)
    u0 = np.zeros((nx), dtype=np.complex128)
    for kx in ks:
        scale = scale0 / (1 + np.abs(kx) ** 0.5)
        u0 += (
            scale
            * (np.random.normal(0, 1) + 1j * np.random.normal(0, 1))
            * np.exp(1j * (2 * np.pi * kx / nx * np.arange(nx)))
        )

    return np.real(u0)


def get_sh_trajectory(u0, r, b3, b5, t1, dt, spatial_grid):
    """
    Generate an trajectory for the Swift Hohenberg equation.
    Parameters:
        u0: initial condition
        r: linear parameter.
        b3: cubic parameter.
        b5: quintic parameter
        t1: total integration time
        dt: time step
        spatial_grid: Spatial grid (assumed to be uniform).
    """
    nx = spatial_grid.shape[0]
    L = (spatial_grid[1, 0] - spatial_grid[0, 0]) * nx
    us = np.zeros((int(t1 / dt), nx))

    def sh(t, u, r, b3, b5):
        uxx = ps.SpectralDerivative(d=2, axis=0)._differentiate(u, L / nx)
        uxxxx = ps.SpectralDerivative(d=4, axis=0)._differentiate(u, L / nx)
        return r * u - uxxxx - 2 * uxx - u + b3 * u**3 - b5 * u**5

    t = 0
    u = u0
    for n in range(int(t1 / dt)):
        t = n * dt
        print("%.1f" % (t / t1), end="\r")
        sol = solve_ivp(
            sh,
            [t, t + dt],
            u,
            method="RK45",
            args=(r, b3, b5),
            rtol=1e-6,
            atol=1e-6,
            first_step=dt / 100,
        )
        u = sol.y[:, -1]
        us[n] = u

    return np.transpose(us)[:, :, np.newaxis]
