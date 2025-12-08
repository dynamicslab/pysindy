import jax
import sympy as sym
import sympy2jax
from jax import custom_jvp
from sympy import factorial


def make_custom_jvp_function(f, fprime):
    """Return a function with custom JVP defined by (fprime)."""

    @jax.custom_jvp
    def f_wrapped(x):
        return f(x)

    @f_wrapped.defjvp
    def f_jvp(primals, tangents):
        (x,) = primals
        (x_dot,) = tangents
        return f(x), fprime(x) * x_dot

    return f_wrapped


def make_sympy_callable(expr):
    def inner(d):
        return sympy2jax.SymbolicModule(expr)(d=d)

    return inner


def get_sympy_matern(p):
    d2 = sym.symbols("d2", positive=True, real=True)
    exp_multiplier = -sym.sqrt(2 * p + 1)
    coefficients = [
        (factorial(p) / factorial(2 * p))
        * (factorial(p + i) / (factorial(i) * factorial(p - i)))
        * (sym.sqrt(8 * p + 4)) ** (p - i)
        for i in range(p + 1)
    ]
    powers = list(range(p, -1, -1))
    matern = sum(
        [c * sym.sqrt((d2**power)) for c, power in zip(coefficients, powers)]
    ) * sym.exp(exp_multiplier * sym.sqrt(d2))
    return d2, matern


def build_matern_core(p):
    d2, matern = get_sympy_matern(p)
    d = sym.var("d", pos=True, real=True)

    maternd = sym.powdenest(matern.subs(d2, d**2))
    subrule = {
        d * sym.DiracDelta(d): 0,
        sym.Abs(d) * sym.DiracDelta(d): 0,
        sym.Abs(d) * sym.sign(d): d,
        d * sym.sign(d): sym.Abs(d),
    }

    def compute_next_derivative(expr):
        return sym.powdenest(sym.expand(expr.diff(d).subs(subrule))).subs(subrule)

    derivatives = [compute_next_derivative(maternd)]
    for k in range(2 * p - 1):
        derivatives.append(compute_next_derivative(derivatives[-1]))

    jax_derivatives = [make_sympy_callable(f) for f in derivatives]

    wrapped_derivatives = [
        make_custom_jvp_function(f, fprime)
        for f, fprime in zip(jax_derivatives[:-1], jax_derivatives[1:])
    ]

    matern_func_raw = sympy2jax.SymbolicModule(maternd)
    core_matern = custom_jvp(lambda d: matern_func_raw(d=d))

    @core_matern.defjvp
    def core_matern_jvp(primals, tangents):
        (x,) = primals
        (x_dot,) = tangents
        ans = core_matern(x)
        ans_dot = wrapped_derivatives[0](x) * x_dot
        return ans, ans_dot

    return core_matern
