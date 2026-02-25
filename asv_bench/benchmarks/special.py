from itertools import product as iproduct

import numpy as np
import sindy_exp
import sympy as sp

import pysindy as ps


class PDEFindSteady:
    r"""Recover Poisson's equation on a unit square with zero BCs.

    .. math::

        \nabla^2 u = 1

    The eigenfunctions of the Laplacian with zero BCs are sine functions
    with odd frequencies.  The coefficients of the eigenfunctions decay as

    .. math::

        \alpha_{m,n} = \frac{16}{\pi^4} \frac{1}{m n (m^2 + n^2)}

    """

    def setup(self):
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        spatial_grid = np.stack(np.meshgrid(y, x, indexing="ij"), axis=-1)
        spatial_grid = ps.AxesArray(
            spatial_grid, axes={"ax_spatial": [0, 1], "ax_coord": 2}
        )
        u = np.zeros_like(spatial_grid[..., 0])
        M = 10
        N = 10
        # only odd numbers because of the zero BCs
        # Bases are sine functions
        for m, n in iproduct(range(1, M + 1, 2), range(1, N + 1, 2)):
            u += (
                16
                / np.pi**4
                * 1
                / (m * n * (m**2 + n**2))
                * np.sin(m * np.pi * spatial_grid[..., 0])
                * np.sin(n * np.pi * spatial_grid[..., 1])
            )
        self.U = u[..., "time", "coord"]
        self.spatial_grid = spatial_grid
        st_grid = spatial_grid[..., "time", :]
        time_mesh = np.zeros_like(st_grid[..., :1])
        self.st_grid = np.concat((st_grid, time_mesh), axis=-1)
        self.lhs = np.ones_like(self.st_grid[..., :1])
        self.differentiation_method = ps.FiniteDifference(order=2)
        self.feature_library = ps.PDELibrary(
            spatial_grid=spatial_grid, derivative_order=3
        )
        self.optimizer = ps.STLSQ(threshold=0.1)
        self.true_coeff = [{sp.parse_expr("u_22"): -1.0, sp.parse_expr("u_11"): -1.0}]

    def track_pdefind_steady(self):
        model = ps.SINDy(
            differentiation_method=self.differentiation_method,
            feature_library=self.feature_library,
            optimizer=self.optimizer,
        )
        model.fit(self.U, t=0.01, x_dot=self.lhs, feature_names=["u"])
        true_coeff, model_coeff = sindy_exp._utils.unionize_coeff_dicts(
            model, self.true_coeff
        )
        return sindy_exp.coeff_metrics(model_coeff, true_coeff)["coeff_mae"]

    def time_pdefind_steady(self):
        model = ps.SINDy(
            differentiation_method=self.differentiation_method,
            feature_library=self.feature_library,
            optimizer=self.optimizer,
        )
        model.fit(self.U, t=0.01, x_dot=self.lhs, feature_names=["u"])

    def peakmem_pdefind_steady(self):
        model = ps.SINDy(
            differentiation_method=self.differentiation_method,
            feature_library=self.feature_library,
            optimizer=self.optimizer,
        )
        model.fit(self.U, t=0.01, x_dot=self.lhs, feature_names=["u"])

    def time_weak_steady(self):
        model = ps.WeakSINDy(
            differentiation_method=self.differentiation_method,
            feature_library=self.feature_library,
            optimizer=ps.STLSQ(alpha=1e-6, threshold=0.1),
        )
        model.fit(self.U, st_grids=self.st_grid, x_dot=self.lhs, feature_names=["u"])

    def peakmem_weak_steady(self):
        model = ps.WeakSINDy(
            differentiation_method=self.differentiation_method,
            feature_library=self.feature_library,
            optimizer=ps.STLSQ(alpha=1e-6, threshold=0.1),
        )
        model.fit(self.U, st_grids=self.st_grid, x_dot=self.lhs, feature_names=["u"])

    def track_weak_steady(self):
        model = ps.WeakSINDy(
            differentiation_method=self.differentiation_method,
            feature_library=self.feature_library,
            optimizer=ps.STLSQ(alpha=1e-6, threshold=0.1),
        )
        model.fit(self.U, st_grids=self.st_grid, x_dot=self.lhs, feature_names=["u"])
        true_coeff, model_coeff = sindy_exp._utils.unionize_coeff_dicts(
            model, self.true_coeff
        )
        return sindy_exp.coeff_metrics(model_coeff, true_coeff)["coeff_mae"]
