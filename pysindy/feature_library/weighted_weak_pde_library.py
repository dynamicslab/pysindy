import numpy as np
from .weak_pde_library import WeakPDELibrary
from ..utils import AxesArray


class WeightedWeakPDELibrary(WeakPDELibrary):
    """
    WeakPDELibrary with GLS whitening via a Cholesky factor built from the
    variance field on the spatiotemporal grid.

    Parameters
    ----------
    spatiotemporal_weights : ndarray, shape = spatiotemporal_grid.shape[:-1]
        Pointwise noise variances σ^2 on the grid (no feature axis).
        The covariance of the weak RHS is Cov[V] = M_y diag(σ^2) M_y^T.

    Notes
    -----
    The whitener W = L^{-1}, with L L^T = Cov[V], is left-applied to both Θ and V.
    This implements min_x || W(Θ x - V) ||_2^2, i.e., GLS in the weak space.
    """

    def __init__(self, *args, spatiotemporal_weights=None, **kwargs):
        self.spatiotemporal_weights = spatiotemporal_weights
        self._L_chol = None  # lower-triangular Cholesky factor of Cov[V]
        super().__init__(*args, **kwargs)

    # ------------------------------ core whitening ------------------------------

    def _build_whitener_from_variance(self):
        """
        Construct L such that Cov[V] = L L^T with
        Cov[V]_{kℓ} = sum_{g ∈ grid} ( w_k[g] * w_ℓ[g] * σ^2[g] ),
        where w_k are the time-derivative weak weights on domain k.
        """
        if self.spatiotemporal_weights is None:
            self._L_chol = None
            return

        # --- robust weight-field shape handling ---
        base_grid = np.asarray(self.spatiotemporal_grid)
        expected = tuple(base_grid.shape[:-1])           # e.g. (Nx, Nt) for a 2D grid
        var_grid = np.asarray(self.spatiotemporal_weights)

        if var_grid.shape == expected + (1,):
            var_grid = var_grid[..., 0]
        elif var_grid.shape != expected:
            raise ValueError(
                f"spatiotemporal_weights must have shape {expected} or {expected + (1,)}, "
                f"got {var_grid.shape}"
            )

        # Flattened variance for convenient indexing
        var_flat = var_grid.ravel(order="C")
        grid_shape = expected
        K = self.K

        idx_lists = []
        val_lists = []
        for k in range(K):
            # local multi-index grids (can be 1D, 2D, 3D… arrays)
            inds_axes = [np.asarray(ax, dtype=np.intp) for ax in self.inds_k[k]]
            grids = np.meshgrid(*inds_axes, indexing="ij")

            # linearize to 1D!
            lin_idx = np.ravel_multi_index(tuple(grids), dims=grid_shape, order="C")
            lin_idx = lin_idx.ravel(order="C")

            # corresponding weak RHS weights, flattened to 1D
            wk = np.asarray(self.fulltweights[k], dtype=float).ravel(order="C")

            if wk.shape[0] != lin_idx.shape[0]:
                raise RuntimeError(
                    f"Weight/variance size mismatch on cell {k}: "
                    f"wk has {wk.shape[0]} entries, indices have {lin_idx.shape[0]}"
                )

            vals = wk * np.sqrt(var_flat[lin_idx])

            idx_lists.append(lin_idx)
            val_lists.append(vals)

        # Build Cov[V] = B B^T with B_{k,i} = w_k[i] * sqrt(var[i])
        Cov = np.zeros((K, K), dtype=float)
        for k in range(K):
            vk = val_lists[k]
            Cov[k, k] = np.dot(vk, vk)
            idx_k = idx_lists[k]
            
            map_k = dict(zip(idx_k.tolist(), vk.tolist()))
            for ell in range(k + 1, K):
                s = 0.0
                idx_e = idx_lists[ell]
                v_e = val_lists[ell]
                map_e = dict(zip(idx_e.tolist(), v_e.tolist()))
                if len(map_k) <= len(map_e):
                    for j, vkj in map_k.items():
                        ve = map_e.get(j)
                        if ve is not None:
                            s += vkj * ve
                else:
                    for j, ve in map_e.items():
                        vk_j = map_k.get(j)
                        if vk_j is not None:
                            s += vk_j * ve
                Cov[k, ell] = s
                Cov[ell, k] = s

        avg_diag = np.trace(Cov) / max(K, 1)
        nugget = 1e-12 * avg_diag
        Cov.flat[:: K + 1] += nugget

        try:
            self._L_chol = np.linalg.cholesky(Cov)
        except np.linalg.LinAlgError:
            # inflate nugget and retry once
            Cov.flat[:: K + 1] += max(1e-10, 1e-6 * avg_diag)
            self._L_chol = np.linalg.cholesky(Cov)

    def _apply_whitener(self, A):
        """Return L^{-1} A without forming L^{-1} explicitly."""
        if self._L_chol is None:
            return A
        return np.linalg.solve(self._L_chol, A)

    # ------------------------------ hooks ------------------------------

    def _weak_form_setup(self):
        # parent builds inds_k and the weak weight tensors
        super()._weak_form_setup()
        # then build the GLS whitener from the variance field
        if self.spatiotemporal_weights is not None:
            self._build_whitener_from_variance()

    def convert_u_dot_integral(self, u):
        Vy = super().convert_u_dot_integral(u)     # (K, 1)
        Vy_w = self._apply_whitener(np.asarray(Vy))
        return AxesArray(Vy_w, {"ax_sample": 0, "ax_coord": 1})

    def transform(self, x_full):
        VTheta_list = super().transform(x_full)    # list of (K, n_features)
        if self._L_chol is None:
            return VTheta_list
        out = []
        for VTheta in VTheta_list:
            A = np.asarray(VTheta)
            A_w = self._apply_whitener(A)          # (K, m)
            out.append(AxesArray(A_w, {"ax_sample": 0, "ax_coord": 1}))
        return out

