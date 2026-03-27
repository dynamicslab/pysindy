from functools import partial
from types import ModuleType
from typing import Any
from typing import Callable

import jax
import jax.numpy as jnp
from jax import grad


def diagpart(M):
    return jnp.diag(jnp.diag(M))


def vectorize_kfunc(k):
    return jax.vmap(jax.vmap(k, in_axes=(None, 0)), in_axes=(0, None))


def op_k_apply(k: Callable[[float, float], float], L_op, R_op):
    return R_op(L_op(k, 0), 1)


def make_block(k, L_op, R_op):
    return vectorize_kfunc(op_k_apply(k, L_op, R_op))


def get_kernel_block_ops(
    k, ops_left, ops_right, output_dim=1, type_pkg: ModuleType = jnp
):
    def k_super(x, y):
        I_mat = type_pkg.eye(output_dim)
        blocks = [
            [
                type_pkg.kron(make_block(k, L_op, R_op)(x, y), I_mat)
                for R_op in ops_right
            ]
            for L_op in ops_left
        ]
        return type_pkg.block(blocks)

    return k_super


def eval_k(k, index):
    return k


def diff_k(k, index):
    return grad(k, index)


def diff2_k(k, index):
    return grad(grad(k, index), index)


def get_selected_grad(k, index, selected_index):
    gradf = grad(k, index)

    def selgrad(*args):
        return gradf(*args)[selected_index]

    return selgrad


def dx_k(k, index):
    return get_selected_grad(k, index, 1)


def dxx_k(k, index):
    return get_selected_grad(get_selected_grad(k, index, 1), index, 1)


def dt_k(k, index):
    return get_selected_grad(k, index, 0)


def nth_derivative_1d(k: Callable, index: int, n: int) -> Callable:
    """
    Computes derivative of order n of k with respect to index and returns the resulting
    function as a callable
    """
    result = k
    for _ in range(n):
        result = jax.grad(result, argnums=index)
    return result


def nth_derivative_operator_1d(n):
    """
    Computes the operator associated to the nth derivative, which maps functions to
    functions. These now match the format of the operators defined above, like diff_k,
    diff2_k.
    """
    return partial(nth_derivative_1d, n=n)
