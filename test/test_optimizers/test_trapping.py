import numpy as np
import pytest
import scipy

from pysindy import PolynomialLibrary
from pysindy import TrappingSR3
from pysindy.optimizers.trapping_sr3 import _antisymm_double_constraint
from pysindy.optimizers.trapping_sr3 import _antisymm_triple_constraints
from pysindy.optimizers.trapping_sr3 import _make_constraints


@pytest.fixture
def poly_lib_terms_coef_bias(scope="session"):
    lib = PolynomialLibrary(2, include_bias=True).fit(np.zeros((1, 2)))
    # terms are [1, x, y, x^2 , xy, y^2]
    polyterms = [(t_ind, exps) for t_ind, exps in enumerate(lib.powers_)]
    coeffs = np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    return lib, polyterms, coeffs


@pytest.fixture
def poly_lib_terms_coef_nobias(scope="session"):
    lib = PolynomialLibrary(2, include_bias=False).fit(np.zeros((1, 2)))
    # terms are [x, y, x^2 , xy, y^2]
    polyterms = [(t_ind, exps) for t_ind, exps in enumerate(lib.powers_)]
    coeffs = np.array([[2, 3, 4, 5, 6], [8, 9, 10, 11, 12]])
    return lib, polyterms, coeffs


@pytest.mark.parametrize(
    "lib_terms_coeffs",
    (
        pytest.lazy_fixture("poly_lib_terms_coef_bias"),
        pytest.lazy_fixture("poly_lib_terms_coef_nobias"),
    ),
)
def test_PL(lib_terms_coeffs):
    _, terms, coeffs = lib_terms_coeffs
    PL_symm, PL_unsymm = TrappingSR3._build_PL(terms)

    expected_symm = np.array([[2.0, 5.5], [5.5, 9.0]])
    expected_unsymm = np.array([[2.0, 3.0], [8.0, 9.0]])
    result = np.einsum("ijkl,kl", PL_symm, coeffs)
    np.testing.assert_array_equal(result, expected_symm)
    result = np.einsum("ijkl,kl", PL_unsymm, coeffs)
    np.testing.assert_array_equal(result, expected_unsymm)


@pytest.mark.parametrize(
    "lib_terms_coeffs",
    (
        pytest.lazy_fixture("poly_lib_terms_coef_bias"),
        pytest.lazy_fixture("poly_lib_terms_coef_nobias"),
    ),
)
def test_PQ(lib_terms_coeffs):
    _, terms, coeffs = lib_terms_coeffs
    PQ = TrappingSR3._build_PQ(terms)
    expected = np.array([[[4.0, 2.5], [2.5, 6]], [[10.0, 5.5], [5.5, 12.0]]])
    result = np.einsum("ijklm,lm", PQ, coeffs)
    np.testing.assert_array_equal(result, expected)


def test_enstrophy_constraints_imply_enstrophy_symmetry():
    n_tgts = 4
    root = np.random.normal(size=(n_tgts, n_tgts))
    mod_matrix = root @ root.T
    bias = False
    lib = PolynomialLibrary(2, include_bias=bias).fit(np.ones((1, n_tgts)))
    terms = [(t_ind, exps) for t_ind, exps in enumerate(lib.powers_)]
    PQ = TrappingSR3._build_PQ(terms)

    _, constraint_lhs = _make_constraints(n_tgts, include_bias=bias)
    constraint_lhs = np.tensordot(constraint_lhs, mod_matrix, axes=1)
    n_constraint, n_features, _ = constraint_lhs.shape
    constraint_mat = constraint_lhs.reshape((n_constraint, -1))
    coeff_basis = scipy.linalg.null_space(constraint_mat)
    _, constraint_nullity = coeff_basis.shape
    coeffs = coeff_basis @ np.random.normal(size=(constraint_nullity, 1))
    coeffs = coeffs.reshape((n_features, n_tgts))

    Q = np.tensordot(PQ, coeffs, axes=([4, 3], [0, 1]))
    Q_tilde = np.tensordot(mod_matrix, Q, axes=1)
    Q_permsum = (
        Q_tilde + np.transpose(Q_tilde, [1, 2, 0]) + np.transpose(Q_tilde, [2, 0, 1])
    )
    np.testing.assert_allclose(np.zeros_like(Q_permsum), Q_permsum, atol=1e-14)


def test_enstrophy_symmetry_implies_enstrophy_constraints():
    n_tgts = 4
    root = np.random.normal(size=(n_tgts, n_tgts))
    mod_matrix = root @ root.T
    u, _, vt = np.linalg.svd(mod_matrix)
    mod_matrix = u @ vt
    mod_inv = np.linalg.inv(mod_matrix)
    bias = False
    lib = PolynomialLibrary(2, include_bias=bias).fit(np.ones((1, n_tgts)))
    terms = [(t_ind, exps) for t_ind, exps in enumerate(lib.powers_)]
    PQ = TrappingSR3._build_PQ(terms)
    PQinv = np.zeros_like(PQ)
    PQinv[np.where(PQ != 0)] = 1

    Q_tilde = np.random.normal(size=(n_tgts, n_tgts, n_tgts))
    Q_tilde[(range(n_tgts),) * 3] = 0
    Q_tilde = (Q_tilde + np.transpose(Q_tilde, [0, 2, 1])) / 2
    Q_tilde -= (
        Q_tilde + np.transpose(Q_tilde, [1, 2, 0]) + np.transpose(Q_tilde, [2, 0, 1])
    ) / 3
    # Assert symmetry
    Qperm = (
        Q_tilde + np.transpose(Q_tilde, [1, 2, 0]) + np.transpose(Q_tilde, [2, 0, 1])
    )
    np.testing.assert_allclose(Qperm, np.zeros_like(Qperm), atol=1e-15)
    Q = np.tensordot(mod_inv, Q_tilde, axes=1)

    # transpose:  make_constraints is (features, targets), but PQ is (targets, features)
    coeffs = np.tensordot(PQinv, Q, axes=([0, 1, 2], [0, 1, 2])).T
    expected, constraint_lhs = _make_constraints(n_tgts, include_bias=bias)
    constraint_lhs = np.tensordot(constraint_lhs, mod_matrix, axes=1)
    n_constraints, _, _ = constraint_lhs.shape
    result = constraint_lhs.reshape((n_constraints, -1)) @ coeffs.flatten()
    np.testing.assert_allclose(result, expected, atol=1e-15)


def test_constraints_imply_symmetry():
    n_tgts = 4
    bias = False
    lib = PolynomialLibrary(2, include_bias=bias).fit(np.ones((1, n_tgts)))
    terms = [(t_ind, exps) for t_ind, exps in enumerate(lib.powers_)]
    PQ = TrappingSR3._build_PQ(terms)

    _, constraint_lhs = _make_constraints(n_tgts, include_bias=bias)
    n_constraint, n_features, _ = constraint_lhs.shape
    constraint_mat = constraint_lhs.reshape((n_constraint, -1))
    coeff_basis = scipy.linalg.null_space(constraint_mat)
    _, constraint_nullity = coeff_basis.shape
    coeffs = coeff_basis @ np.random.normal(size=(constraint_nullity, 1))
    coeffs = coeffs.reshape((n_features, n_tgts))

    Q = np.tensordot(PQ, coeffs, axes=([4, 3], [0, 1]))
    Q_permsum = Q + np.transpose(Q, [1, 2, 0]) + np.transpose(Q, [2, 0, 1])
    np.testing.assert_allclose(np.zeros_like(Q_permsum), Q_permsum, atol=1e-15)


def test_symmetry_implies_constraints():
    n_tgts = 4
    bias = False
    lib = PolynomialLibrary(2, include_bias=bias).fit(np.ones((1, n_tgts)))
    terms = [(t_ind, exps) for t_ind, exps in enumerate(lib.powers_)]
    PQ = TrappingSR3._build_PQ(terms)
    PQinv = np.zeros_like(PQ)
    PQinv[np.where(PQ != 0)] = 1

    Q = np.random.normal(size=(n_tgts, n_tgts, n_tgts))
    Q[(range(n_tgts),) * 3] = 0
    Q = (Q + np.transpose(Q, [0, 2, 1])) / 2
    Q -= (Q + np.transpose(Q, [1, 2, 0]) + np.transpose(Q, [2, 0, 1])) / 3
    # Assert symmetry
    Qperm = Q + np.transpose(Q, [1, 2, 0]) + np.transpose(Q, [2, 0, 1])
    np.testing.assert_allclose(Qperm, np.zeros_like(Qperm), atol=1e-15)

    # transpose:  make_constraints is (features, targets), but PQ is (targets, features)
    coeffs = np.tensordot(PQinv, Q, axes=([0, 1, 2], [0, 1, 2])).T
    expected, constraint_lhs = _make_constraints(n_tgts, include_bias=bias)
    n_constraints, _, _ = constraint_lhs.shape
    result = constraint_lhs.reshape((n_constraints, -1)) @ coeffs.flatten()
    np.testing.assert_allclose(result, expected, atol=1e-15)


@pytest.mark.parametrize("include_bias", (True, False))
def test_trapping_constraints(include_bias):
    # x, y, x^2, xy, y^2
    constraint_rhs, constraint_lhs = _make_constraints(2, include_bias=include_bias)
    stable_coefs = np.array([[0, 0, 0, 1, -1], [0, 0, -1, 1, 0]])
    if include_bias:
        stable_coefs = np.concatenate(([[0], [0]], stable_coefs), axis=1)
    result = np.tensordot(constraint_lhs, stable_coefs, ((1, 2), (1, 0)))
    np.testing.assert_array_equal(constraint_rhs, result)
    _, lg_constraint = _make_constraints(4, include_bias=include_bias)
    # constraint should be full-rank
    expected = len(lg_constraint)
    result = np.linalg.matrix_rank(lg_constraint.reshape((expected, -1)))
    assert result == expected


def test_trapping_triple_mixed_constraint():
    # xy, xz, yz
    stable_coefs = np.array([[0, 0, -1], [0, 0.5, 0], [0.5, 0, 0]])
    mixed_terms = {frozenset((0, 1)): 0, frozenset((0, 2)): 1, frozenset((1, 2)): 2}
    constraint_lhs = _antisymm_triple_constraints(3, 3, mixed_terms)
    result = np.tensordot(constraint_lhs, stable_coefs, ((1, 2), (1, 0)))
    np.testing.assert_array_equal(result, np.zeros_like(result))


def test_trapping_double_constraint():
    stable_coefs = np.array(
        [
            # w^2, wx, wy, wz, x^2, xy, xz, y^2, yz, z^2
            [0, 1, 2, 3, -4, 0, 0, -8, 0, -9],  # w
            [-1, 4, 0, 0, 0, 5, 6, -10, 0, -11],  # x
            [-2, 0, 8, 0, -5, 10, 0, 0, 7, -12],  # y
            [-3, 0, 0, 9, -6, 0, 11, -7, 12, 0],  # z
        ]
    )
    pure_terms = {0: 0, 1: 4, 2: 7, 3: 9}
    mixed_terms = {
        frozenset((0, 1)): 1,
        frozenset((0, 2)): 2,
        frozenset((0, 3)): 3,
        frozenset((1, 2)): 5,
        frozenset((1, 3)): 6,
        frozenset((2, 3)): 8,
    }
    constraint_lhs = _antisymm_double_constraint(4, 10, pure_terms, mixed_terms)
    result = np.tensordot(constraint_lhs, stable_coefs, ((1, 2), (1, 0)))
    np.testing.assert_array_equal(result, np.zeros_like(result))
    # constraint should be full-rank
    expected = len(constraint_lhs)
    result = np.linalg.matrix_rank(constraint_lhs.reshape((expected, -1)))
    assert result == expected
