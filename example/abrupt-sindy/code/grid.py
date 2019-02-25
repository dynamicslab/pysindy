from adaptive_sindy import STRidge2 as STRidge
from sklearn.metrics import explained_variance_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import check_random_state


def is_macosx():
    import platform

    return platform.system() == "Darwin"


def get_n_jobs():
    return 1 if is_macosx() else -1


seed = 42
n_splits = 5
alpha = [0, 0.2, 0.4, 0.6, 0.8, 0.95]
threshold = [0.1, 0.2, 0.4]
degree = [2, 3]

pipe = MultiOutputRegressor(
    make_pipeline(PolynomialFeatures(include_bias=False), STRidge(normalize=False, threshold_intercept=True))
)

scorer = make_scorer(explained_variance_score, multioutput="uniform_average")

parameters = {
    "estimator__stridge2__threshold": threshold,
    "estimator__stridge2__alpha": alpha,
    "estimator__polynomialfeatures__degree": degree,
}


n_jobs = get_n_jobs()
rng = check_random_state(seed)
cv = KFold(n_splits=n_splits, random_state=rng, shuffle=True)

grid = GridSearchCV(pipe, parameters, scoring=scorer, cv=cv, n_jobs=n_jobs, error_score=0)


def math_mode(txt):
    try:
        txt_ = map(str, txt)
    except TypeError:
        txt_ = str(txt)
    return f"${','.join(txt_)}$"


def param_table():
    alphas = math_mode(alpha)
    lambdas = math_mode(threshold)
    degrees = math_mode(degree)
    seeds = f"${seed}$"
    n_splitss = math_mode(n_splits)

    table = f"""
\\begin{{tabular}}{{l|l}}
    Parameter & Value \\\\ \\hline
    $\\alpha$ & {alphas} \\\\
    $\\gamma $ & {lambdas}\\\\
    $n_{{\\text{{degree}}}}$ & {degrees} \\\\
    $n_{{\\text{{fold}}}}$ & {n_splitss} \\\\
    Seed &  {seeds} \\\\
    CV & k-fold \\\\
    Score & explained variance score
\end{{tabular}}
"""
    with open("../tables/grid.tex", "w") as f:
        f.writelines(table)


if __name__ == "__main__":
    param_table()
