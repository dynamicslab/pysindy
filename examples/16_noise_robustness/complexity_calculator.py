import numpy as np
import dysts.flows as flows
from sympy.parsing.sympy_parser import parse_expr
from sympy import count_ops
import inspect


def compute_medl(systems_list, all_sols_train, param_list):
    """
    Computes the mean-error-description-length (MEDL) of all
    systems in the given system list.

    Attributes
    ----------
    systems_list: list
        a list of system's names whose to be computed
    all_sols_train: dictionary
        a dictionary of systems' trajectories
    param_list:
        a list of dictionaries which contains the parameter values
        used to generate the system

    Returns
    -------
        a list MEDL of the systems in the same order as systems in
        the system list
    """
    compl_list = []
    for i, system in enumerate(systems_list):
        x_train = all_sols_train[system]
        params = param_list[i]
        std_eqs = get_stand_expr(x_train, system, params)
        curr_compl = 0
        for eq in std_eqs:
            curr_compl += get_expr_complexity(eq)
        compl_list.append(curr_compl)
    return compl_list


def get_stand_expr(x_train, system, params):
    """
    Turns python functions of a given system into SymPy standard strings.
    """
    system_str = inspect.getsource(getattr(flows, system))
    cut1 = system_str.find("return")
    system_str = system_str[: cut1 - 1]
    cut2 = system_str.rfind("):")
    system_str = system_str[cut2 + 5:]
    chunks = system_str.split("\n")[:-1]
    for j, chunk in enumerate(chunks):
        cind = chunk.rfind("=")
        chunk = chunk[cind + 1:]
        for key in params.keys():
            if "Lorenz" in system and "rho" in params.keys():
                chunk = chunk.replace("rho", str(params["rho"]), 10)
            if "Bouali2" in system:
                chunk = chunk.replace("bb", "0", 10)
            chunk = chunk.replace(key, str(params[key]), 10)
        # print(chunk)
        chunk = chunk.replace("--", "", 10)
        # get all variables into (x, y, z, w) form
        chunk = chunk.replace("q1", "x", 10)
        chunk = chunk.replace("q2", "y", 10)
        chunk = chunk.replace("p1", "z", 10)
        chunk = chunk.replace("p2", "w", 10)
        chunk = chunk.replace("px", "z", 10)
        chunk = chunk.replace("py", "w", 10)

        # Do any unique ones
        chunk = chunk.replace("(-10 + -4)", "-14", 10)
        chunk = chunk.replace("(-10 * -4)", "40", 10)
        chunk = chunk.replace("3.0 * 1.0", "3", 10)
        chunk = chunk.replace(" - 0 * z", "", 10)
        chunk = chunk.replace("(28 - 35)", "-7", 10)
        chunk = chunk.replace("(1 / 0.2 - 0.001)", "4.999", 10)
        chunk = chunk.replace("- (1.0 - 1.0) * x^2 ", "", 10)
        chunk = chunk.replace("(26 - 37)", "-11", 10)
        chunk = chunk.replace("64^2", "4096", 10)
        chunk = chunk.replace("64**2", "4096", 10)
        chunk = chunk.replace("3 / np.sqrt(2) * 0.55", "1.166726189", 10)
        chunk = chunk.replace("3 * np.sqrt(2) * 0.55", "2.333452378", 10)
        chunk = chunk.replace("+ -", "- ", 10)
        chunk = chunk.replace("-1.5 * -0.0026667", "0.00400005", 10)

        chunk = chunk.replace("- 0.0026667 * 0xz", "", 10)
        chunk = chunk.replace("1/4096", "0.000244140625", 10)
        chunk = chunk.replace("10/4096", "0.00244140625", 10)
        chunk = chunk.replace("28/4096", "0.0068359375", 10)
        chunk = chunk.replace("2.667/4096", "0.000651123046875", 10)
        chunk = chunk.replace("0.2 * 9", "1.8", 10)
        chunk = chunk.replace(" - 3 * 0", "", 10)
        chunk = chunk.replace("2 * 1", "2", 10)
        chunk = chunk.replace("3 * 2.1 * 0.49", "3.087", 10)
        chunk = chunk.replace("2 * 2.1", "4.2", 10)
        chunk = chunk.replace("-40 / -14", "2.85714285714", 10)
        # change notation of squared and cubed terms
        chunk = chunk.replace(" 1x", " x", 10)
        chunk = chunk.replace(" 1y", " y", 10)
        chunk = chunk.replace(" 1z", " z", 10)
        chunk = chunk.replace(" 1w", " w", 10)
        chunks[j] = chunk
        chunk = chunk.replace(" ", "", 400)
        chunk = chunk.replace("-x", "-1x", 10)
        chunk = chunk.replace("-y", "-1y", 10)
        chunk = chunk.replace("-z", "-1z", 10)
        chunk = chunk.replace("-w", "-1w", 10)
        chunk = chunk.replace("--", "-", 20)
    return chunks


###############################################################################
# Methods, bestApproximation, get_numberDL_scanned, and get_expr_complexity,  #
# are retrieved from https://github.com/SJ001/AI-Feynman.                     #
#                                                                             #
# We modified get_expr_complexity to make it behave correctly with powers.    #
###############################################################################
"""
Citation:
@article{udrescu2020ai,
  title={AI Feynman: A physics-inspired method for symbolic regression},
  author={Udrescu, Silviu-Marian and Tegmark, Max},
  journal={Science Advances},
  volume={6},
  number={16},
  pages={eaay2631},
  year={2020},
  publisher={American Association for the Advancement of Science}
}

@article{udrescu2020ai,
  title={AI Feynman 2.0:
  Pareto-optimal symbolic regression exploiting graph modularity},
  author={Udrescu, Silviu-Marian and Tan, Andrew and Feng, Jiahai and Neto,
  Orisvaldo and Wu, Tailin and Tegmark, Max},
  journal={arXiv preprint arXiv:2006.10782},
  year={2020}
}
"""


def get_expr_complexity(expr):
    expr = parse_expr(expr, evaluate=True)
    compl = 0

    def is_atomic_number(expr):
        return expr.is_Atom and expr.is_number

    numbers_expr = [subexpression for subexpression in expr.args
                    if is_atomic_number(subexpression)]
    variables_expr = [subexpression for subexpression in expr.args
                      if not(is_atomic_number(subexpression))]

    for j in numbers_expr:
        try:
            compl = compl + get_number_DL_snapped(float(j))
        except Exception as e:
            compl = compl + 1000000
            print(e.message, e.args)

    # compute n, k: n basis functions appear k times
    n_uniq_vars = len(expr.free_symbols)
    n_uniq_ops = len(count_ops(expr, visual=True).free_symbols)

    N = n_uniq_vars + n_uniq_ops

    n_ops = count_ops(expr)
    n_vars = len(variables_expr)

    n_power_addional = 0
    for subexpression in variables_expr:
        if subexpression.is_Pow:
            b, e = subexpression.as_base_exp()
            if b.is_Symbol and e.is_Integer:
                n_power_addional += (e - 1) * 2 - 1

    K = n_ops + n_vars + n_power_addional

    if n_uniq_ops != 0 or n_uniq_ops != 0:
        compl = compl + K * np.log2(N)

    return compl


def bestApproximation(x, imax):
    def float2contfrac(x, nmax):
        x = float(x)
        c = [np.floor(x)]
        y = x - np.floor(x)
        k = 0
        while np.abs(y) != 0 and k < nmax:
            y = 1 / float(y)
            i = np.floor(y)
            c.append(i)
            y = y - i
            k = k + 1
        return c

    def contfrac2frac(seq):
        num, den = 1, 0
        for u in reversed(seq):
            num, den = den + num * u, num
        return num, den

    def contFracRationalApproximations(c):
        return np.array(list(contfrac2frac(c[:i+1]) for i in range(len(c))))

    def contFracApproximations(c):
        q = contFracRationalApproximations(c)
        return q[:, 0] / float(q[:, 1])

    def truncateContFrac(q, imax):
        k = 0
        while k < len(q) and np.maximum(np.abs(q[k, 0]), q[k, 1]) <= imax:
            k = k + 1
        return q[:k]

    def pval(p):
        p = p.astype(float)
        return 1 - np.exp(-p ** 0.87 / 0.36)

    xsign = np.sign(x)
    q = truncateContFrac(
        contFracRationalApproximations(float2contfrac(abs(x), 20)), imax)

    if len(q) > 0:
        p = np.abs(q[:, 0] / q[:, 1] - abs(x)).astype(float)\
            * (1 + np.abs(q[:, 0])) * q[:, 1]
        p = pval(p)
        i = np.argmin(p)
        return (xsign * q[i, 0] / float(q[i, 1]),
                xsign * q[i, 0], q[i, 1], p[i])
    else:
        return (None, 0, 0, 1)


def get_number_DL_snapped(n):
    epsilon = 1e-10
    n = float(n)
    if np.isnan(n):
        return 1000000
    elif np.abs(n - int(n)) < epsilon:
        return np.log2(1 + abs(int(n)))
    elif np.abs(n - bestApproximation(n, 10000)[0]) < epsilon:
        _, numerator, denominator, _ = bestApproximation(n, 10000)
        return np.log2((1 + abs(numerator)) * abs(denominator))
    elif np.abs(n - np.pi) < epsilon:
        return np.log2(1 + 3)
    else:
        PrecisionFloorLoss = 1e-14
        return np.log2(1 + (float(n) / PrecisionFloorLoss) ** 2) / 2
