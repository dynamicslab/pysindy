import os

import numpy as np

import pysindy as ps

for filebase in ["odd", "even", "periodic"]:
    sols = np.load(filebase + "_sols.npy")
    pars = np.load(filebase + "_pars.npy")
    if not os.path.exists(filebase + "/"):
        os.mkdir(filebase)

    evals = []
    evecs = []
    for i in range(len(sols)):
        print(filebase, i, end="\r")
        t, u = sols[i].T
        epsilon, r, b3, b5, T, norm = pars[i]
        u = u[::4]
        t = t[::4]
        t = T * t
        u = np.concatenate([u, -np.flip(u)[1:]])
        t = np.concatenate([t, t[-1] + np.cumsum(np.flip(np.diff(t)))])

        fd = ps.FiniteDifference(d=2, axis=0, order=4, periodic=True)
        interior_coeffs = fd._coefficients(t)
        interior_inds = fd.stencil_inds
        slice_interior = slice((fd.n_stencil - 1) // 2, -(fd.n_stencil - 1) // 2)
        slice_boundary = np.concatenate(
            [
                np.arange(0, (fd.n_stencil - 1) // 2),
                -np.flip(1 + np.arange(1, (fd.n_stencil - 1) // 2)),
                np.array([-1]),
            ]
        )
        boundary_coeffs = fd._coefficients_boundary_periodic(t)
        boundary_inds = fd.stencil_inds

        dxx = np.zeros((len(t), len(t)))
        for i in range(len(interior_inds)):
            dxx[slice_interior][
                np.arange(len(interior_inds[i])), interior_inds[i]
            ] = interior_coeffs[:, i].T
        for i in range(len(boundary_inds)):
            dxx[slice_boundary][
                np.arange(len(boundary_inds[i])), boundary_inds[i]
            ] = boundary_coeffs[:, i].T

        fd = ps.FiniteDifference(d=4, axis=0, order=4, periodic=True)
        interior_coeffs = fd._coefficients(t)
        interior_inds = fd.stencil_inds
        slice_interior = slice((fd.n_stencil - 1) // 2, -(fd.n_stencil - 1) // 2)
        slice_boundary = np.concatenate(
            [
                np.arange(0, (fd.n_stencil - 1) // 2),
                -np.flip(1 + np.arange(1, (fd.n_stencil - 1) // 2)),
                np.array([-1]),
            ]
        )
        boundary_coeffs = fd._coefficients_boundary_periodic(t)
        boundary_inds = fd.stencil_inds

        dxxxx = np.zeros((len(t), len(t)))
        for i in range(len(interior_inds)):
            dxxxx[slice_interior][
                np.arange(len(interior_inds[i])), interior_inds[i]
            ] = interior_coeffs[:, i].T
        for i in range(len(boundary_inds)):
            dxxxx[slice_boundary][
                np.arange(len(boundary_inds[i])), boundary_inds[i]
            ] = boundary_coeffs[:, i].T

        vals, vecs = np.linalg.eig(
            (r - 1 + 3 * b3 * u**2 - 5 * b5 * u**4) * np.eye(len(t))
            - dxxxx
            - 2 * dxx
        )
        evals = evals + [vals]
        evecs = evecs + [vecs]

    inds = np.argsort(np.real(evals))[:, -5:]
    inds2 = [
        inds[i][np.argsort(np.linalg.norm(evecs[i][:, inds[i]], ord=1, axis=0))[-3:]]
        for i in range(len(inds))
    ]
    stableinds = np.where(
        np.all(
            np.real(
                np.array(evals)[tuple([np.arange(len(evals))[:, np.newaxis], inds2])]
            )
            < 1e-1,
            axis=1,
        )
    )[0]
    unstableinds = np.where(
        np.any(
            np.real(
                np.array(evals)[tuple([np.arange(len(evals))[:, np.newaxis], inds2])]
            )
            > 1e-1,
            axis=1,
        )
    )[0]
    stable_xs = []
    stable_ys = []
    unstable_xs = []
    unstable_ys = []
    for inds in np.split(stableinds, np.where(np.diff(stableinds) != 1)[0] + 1):
        stable_xs.append(pars[inds, 0])
        stable_ys.append(pars[inds, -1])
    for inds in np.split(unstableinds, np.where(np.diff(unstableinds) != 1)[0] + 1):
        unstable_xs.append(pars[inds, 0])
        unstable_ys.append(pars[inds, -1])
    for i in range(len(stable_xs)):
        branch = np.array([stable_xs[i], stable_ys[i]])
        np.save(filebase + "/stable_" + str(i), branch)
    for i in range(len(unstable_xs)):
        branch = np.array([unstable_xs[i], unstable_ys[i]])
        np.save(filebase + "/unstable_" + str(i), branch)
