"""Utilities to convert a `dysts` dynamical system object's rhs to SymPy.

This module inspects the source of an object's RHS method (by default
named ``rhs``), parses the function using ``ast``, and converts the
returned expression(s) into SymPy expressions.

The conversion is intentionally conservative and aims to handle common
patterns used in simple rhs implementations, e.g. returning a tuple/list
of arithmetic expressions, using indexing into a state vector (``x[0]``),
and calls to common ``numpy``/``math`` functions (``np.sin``, ``math.exp``, ...).

Limitations:
- It does not execute arbitrary code from the inspected function.
- Complex control flow, loops, or non-trivial Python constructs may not
  be fully supported.

Example
-------
from dysts.flows import Lorenz
from inspect_to_sympy import object_to_sympy_rhs

lor = Lorenz()
symbols, exprs, lambda_rhs = object_to_sympy_rhs(lor)
# `symbols` is a list of SymPy symbols for the state vector
# `exprs` is a list of SymPy expressions for the RHS
# `lambda_rhs` is a SymPy Lambda mapping state symbols -> rhs expressions
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import sympy as sp


def _is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


class _ASTToSympy(ast.NodeVisitor):
    def __init__(
        self,
        state_name: str,
        state_symbols: List[sp.Symbol],
        locals_map: Dict[str, Any],
    ):
        self.state_name = state_name
        self.state_symbols = state_symbols
        self.locals = dict(locals_map)

    def generic_visit(self, node):
        raise NotImplementedError(f"AST node not supported: {node!r}")

    def visit_Constant(self, node: ast.Constant):
        return sp.sympify(node.value)

    def visit_Num(self, node: ast.Num):
        return sp.sympify(node.n)

    def visit_Name(self, node: ast.Name):
        if node.id in self.locals:
            return self.locals[node.id]
        return sp.Symbol(node.id)

    def visit_Tuple(self, node: ast.Tuple):
        elems = []
        for elt in node.elts:
            val = self.visit(elt)
            if isinstance(val, (list, tuple)):
                elems.extend(list(val))
            else:
                elems.append(val)
        return tuple(elems)

    def visit_List(self, node: ast.List):
        elems = []
        for elt in node.elts:
            val = self.visit(elt)
            if isinstance(val, (list, tuple)):
                elems.extend(list(val))
            else:
                elems.append(val)
        return elems

    def visit_Starred(self, node: ast.Starred):
        # Handle starred expressions like `*x` in list/tuple literals.
        # If the starred value is the state vector name, expand to state symbols.
        if isinstance(node.value, ast.Name) and node.value.id == self.state_name:
            return tuple(self.state_symbols)
        # Otherwise, evaluate the value and if it is a sequence, return its items
        val = self.visit(node.value)
        if isinstance(val, (list, tuple)):
            return tuple(val)
        raise NotImplementedError(
            "Unsupported starred expression; cannot expand non-iterable"
        )

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        if isinstance(node.op, ast.Mod):
            return left % right
        raise NotImplementedError(f"Binary op not supported: {node.op!r}")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise NotImplementedError(f"Unary op not supported: {node.op!r}")

    def visit_Call(self, node: ast.Call):
        # Determine function name
        func = node.func
        func_name = None
        mod_name = None

        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            # e.g. np.sin or math.exp
            if isinstance(func.value, ast.Name):
                mod_name = func.value.id
            func_name = func.attr
        else:
            raise NotImplementedError(
                f"Call to unsupported func node: {ast.dump(func)}"
            )

        # Map common numpy/math functions to sympy
        func_map = {
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "exp": sp.exp,
            "log": sp.log,
            "sqrt": sp.sqrt,
            "abs": sp.Abs,
            "atan": sp.atan,
            "asin": sp.asin,
            "acos": sp.acos,
        }

        args = [self.visit(a) for a in node.args]

        # Special-case array constructors: return underlying list/tuple
        if func_name in ("array", "asarray") and mod_name in ("np", "numpy"):
            # expect a single positional arg that's a list/tuple
            if len(args) == 1:
                return args[0]

        if func_name in func_map:
            return func_map[func_name](*args)

        # Unknown function: create a Sympy Function
        symf = sp.Function(func_name)
        return symf(*args)

    def visit_Subscript(self, node: ast.Subscript):
        # Support patterns like x[0] where x is the state vector name
        value = node.value
        # handle simple constant index
        if _is_name(value, self.state_name):
            # Python >=3.9: slice is directly the node.slice
            idx_node = node.slice
            if isinstance(idx_node, ast.Constant):
                idx = idx_node.value
            elif isinstance(idx_node, ast.Index) and isinstance(
                idx_node.value, ast.Constant
            ):
                idx = idx_node.value.value
            else:
                raise NotImplementedError(
                    "Only constant indices into state vector supported"
                )
            return self.state_symbols[idx]

        # If it's something else, try to evaluate generically
        base = self.visit(value)
        # slice may be constant
        if isinstance(node.slice, ast.Constant):
            key = node.slice.value
            return base[key]
        raise NotImplementedError("Unsupported subscript pattern")


def object_to_sympy_rhs(
    obj: Any, func_name: str = "_rhs"
) -> Tuple[List[sp.Symbol], List[sp.Expr], sp.Lambda]:
    """Inspect ``obj`` for a method named ``func_name`` and return a SymPy
    representation of its RHS.

    Returns a tuple ``(state_symbols, exprs, lambda_rhs)`` where ``state_symbols``
    is a list of SymPy symbols for the state vector, ``exprs`` is a list of
    SymPy expressions for the RHS components, and ``lambda_rhs`` is a SymPy
    Lambda mapping the state symbols to the RHS vector.
    """

    if not hasattr(obj, func_name):
        raise AttributeError(f"Object has no attribute {func_name!r}")

    func = getattr(obj, func_name)
    src = inspect.getsource(func)
    src = textwrap.dedent(src)

    parsed = ast.parse(src)

    # Find first FunctionDef
    fndef = None
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef):
            fndef = node
            break
    if fndef is None:
        raise RuntimeError("No function definition found in source")

    # Determine state argument names. Common dysts signature:
    # (self, *states, t, *parameters). Prefer obj.dimension when available.
    arg_names = [a.arg for a in fndef.args.args]
    if len(arg_names) == 0:
        raise RuntimeError("Function has no arguments")

    start_idx = 0
    if arg_names[0] == "self":
        start_idx = 1

    vector_mode = False
    state_arg_names: List[str]
    t_idx = None
    if "t" in arg_names:
        t_idx = arg_names.index("t")

    if hasattr(obj, "dimension") and isinstance(getattr(obj, "dimension"), int):
        n_state = int(getattr(obj, "dimension"))
        if t_idx is not None:
            potential = arg_names[start_idx:t_idx]
            if len(potential) >= n_state:
                state_arg_names = potential[:n_state]
            else:
                state_arg_names = [arg_names[start_idx]]
                vector_mode = True
        else:
            potential = arg_names[start_idx:]
            if len(potential) >= n_state:
                state_arg_names = potential[:n_state]
            else:
                state_arg_names = [arg_names[start_idx]]
                vector_mode = True
    else:
        if t_idx is not None:
            state_arg_names = arg_names[start_idx:t_idx]
            if len(state_arg_names) == 0:
                state_arg_names = [arg_names[start_idx]]
                vector_mode = True
            elif len(state_arg_names) == 1:
                # single name could be vector or scalar; assume vector-mode
                vector_mode = True
        else:
            state_arg_names = [arg_names[start_idx]]
            vector_mode = True

    # If vector_mode, inspect AST for subscript/index usage or tuple unpacking
    if vector_mode:
        state_name = state_arg_names[0]
        max_index = -1
        unpack_size = None
        for node in ast.walk(fndef):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id == state_name
            ):
                sl = node.slice
                if isinstance(sl, ast.Constant) and isinstance(sl.value, int):
                    if sl.value > max_index:
                        max_index = sl.value
                elif (
                    isinstance(sl, ast.Index)
                    and isinstance(sl.value, ast.Constant)
                    and isinstance(sl.value.value, int)
                ):
                    if sl.value.value > max_index:
                        max_index = sl.value.value
            if isinstance(node, ast.Assign):
                if isinstance(node.value, ast.Name) and node.value.id == state_name:
                    targets = node.targets
                    if len(targets) == 1 and isinstance(
                        targets[0], (ast.Tuple, ast.List)
                    ):
                        unpack_size = len(targets[0].elts)

        if unpack_size is not None:
            n_state = unpack_size
        elif max_index >= 0:
            n_state = max_index + 1
        else:
            n_state = int(getattr(obj, "dimension", 3))

        state_symbols = [sp.Symbol(f"x{i}") for i in range(n_state)]
        primary_state_name = state_name
    else:
        # individual state args -> use their arg names as symbol names
        state_symbols = [sp.Symbol(n) for n in state_arg_names]
        primary_state_name = state_arg_names[0] if len(state_arg_names) > 0 else "x"

    # Build locals mapping from known state arg names and parameters
    locals_map: Dict[str, Any] = {}
    for i, name in enumerate(state_arg_names):
        if i < len(state_symbols):
            locals_map[name] = state_symbols[i]

    # map parameters (if present) to numeric values or symbols
    if hasattr(obj, "parameters") and isinstance(getattr(obj, "parameters"), dict):
        params = getattr(obj, "parameters")
        if t_idx is not None:
            param_arg_names = arg_names[t_idx + 1 :]
        else:
            param_arg_names = []
        for pname in param_arg_names:
            if pname in params:
                locals_map[pname] = sp.sympify(params[pname])
            else:
                locals_map[pname] = sp.Symbol(pname)

    converter = _ASTToSympy(primary_state_name, state_symbols, locals_map)

    return_expr = None
    # Walk through function body statements, handle Assign and Return
    for stmt in fndef.body:
        if isinstance(stmt, ast.Assign):
            # only simple single-target assignments supported
            if len(stmt.targets) != 1:
                raise ValueError("Only single-target assignments supported")
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                value_expr = converter.visit(stmt.value)
                locals_map[target.id] = value_expr
            elif (
                isinstance(target, (ast.Tuple, ast.List))
                and isinstance(stmt.value, ast.Name)
                and stmt.value.id == state_name
            ):
                # unpacking like a,b,c = x -> map names to state symbols
                for i, elt in enumerate(target.elts):
                    if isinstance(elt, ast.Name):
                        locals_map[elt.id] = state_symbols[i]
        elif isinstance(stmt, ast.Return):
            return_expr = stmt.value

    if return_expr is None:
        raise ValueError("No return statement found in function")
        # maybe last statement is an Expr with list construction;
        # try to find a Return node deep
        for node in ast.walk(fndef):
            if isinstance(node, ast.Return):
                return_expr = node.value
                break

    if return_expr is None:
        raise RuntimeError("No return expression found in function body")

    # Refresh converter with updated locals
    converter = _ASTToSympy(primary_state_name, state_symbols, locals_map)
    rhs_val = converter.visit(return_expr)

    # Normalize rhs_val to a list/tuple of expressions
    if isinstance(rhs_val, (list, tuple)):
        exprs = list(rhs_val)
    else:
        # single expression: treat as 1-dim RHS
        exprs = [rhs_val]

    lambda_rhs = sp.Lambda(tuple(state_symbols), sp.Matrix(exprs))

    return state_symbols, exprs, lambda_rhs


__all__ = ["object_to_sympy_rhs"]
