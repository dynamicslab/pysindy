import time
import warnings

import dysts.flows as flows
import matplotlib.pyplot as plt
import numpy as np
from dysts.equation_utils import compute_medl
from dysts.equation_utils import make_dysts_true_coefficients
from dysts.equation_utils import nonlinear_terms_from_coefficients
from utils import load_data
from utils import normalized_RMSE
from utils import Pareto_scan_ensembling
from utils import total_coefficient_error_normalized
from utils import weakform_reorder_coefficients

import pysindy as ps

# bad code but allows us to ignore warnings
warnings.filterwarnings("ignore")

t1 = time.time()

# Arneodo does not have the Lyapunov spectrum calculated so omit it.
# HindmarshRose and AtmosphericRegime seem to be poorly sampled
# by the dt and dominant time scales used in the database, so we omit them.
systems_list = [
    "Aizawa",
    "Bouali2",
    "GenesioTesi",
    "HyperBao",
    "HyperCai",
    "HyperJha",
    "HyperLorenz",
    "HyperLu",
    "HyperPang",
    "Laser",
    "Lorenz",
    "LorenzBounded",
    "MooreSpiegel",
    "Rossler",
    "ShimizuMorioka",
    "HenonHeiles",
    "GuckenheimerHolmes",
    "Halvorsen",
    "KawczynskiStrizhak",
    "VallisElNino",
    "RabinovichFabrikant",
    "NoseHoover",
    "Dadras",
    "RikitakeDynamo",
    "NuclearQuadrupole",
    "PehlivanWei",
    "SprottTorus",
    "SprottJerk",
    "SprottA",
    "SprottB",
    "SprottC",
    "SprottD",
    "SprottE",
    "SprottF",
    "SprottG",
    "SprottH",
    "SprottI",
    "SprottJ",
    "SprottK",
    "SprottL",
    "SprottM",
    "SprottN",
    "SprottO",
    "SprottP",
    "SprottQ",
    "SprottR",
    "SprottS",
    "Rucklidge",
    "Sakarya",
    "RayleighBenard",
    "Finance",
    "LuChenCheng",
    "LuChen",
    "QiChen",
    "ZhouChen",
    "BurkeShaw",
    "Chen",
    "ChenLee",
    "WangSun",
    "DequanLi",
    "NewtonLiepnik",
    "HyperRossler",
    "HyperQi",
    "Qi",
    "LorenzStenflo",
    "HyperYangChen",
    "HyperYan",
    "HyperXu",
    "HyperWang",
    "Hadley",
]
alphabetical_sort = np.argsort(systems_list)
systems_list = np.array(systems_list)[alphabetical_sort]

# attributes list
attributes = [
    "maximum_lyapunov_estimated",
    "lyapunov_spectrum_estimated",
    "embedding_dimension",
    "parameters",
    "dt",
    "hamiltonian",
    "period",
    "unbounded_indices",
]

# Get attributes
all_properties = dict()
for i, equation_name in enumerate(systems_list):
    eq = getattr(flows, equation_name)()
    attr_vals = [getattr(eq, item, None) for item in attributes]
    all_properties[equation_name] = dict(zip(attributes, attr_vals))
    t1 = time.time()

# Arneodo does not have the Lyapunov spectrum calculated so omit it.
# HindmarshRose and AtmosphericRegime seem to be poorly sampled
# by the dt and dominant time scales used in the database, so we omit them.
systems_list = [
    "Aizawa",
    "Bouali2",
    "GenesioTesi",
    "HyperBao",
    "HyperCai",
    "HyperJha",
    "HyperLorenz",
    "HyperLu",
    "HyperPang",
    "Laser",
    "Lorenz",
    "LorenzBounded",
    "MooreSpiegel",
    "Rossler",
    "ShimizuMorioka",
    "HenonHeiles",
    "GuckenheimerHolmes",
    "Halvorsen",
    "KawczynskiStrizhak",
    "VallisElNino",
    "RabinovichFabrikant",
    "NoseHoover",
    "Dadras",
    "RikitakeDynamo",
    "NuclearQuadrupole",
    "PehlivanWei",
    "SprottTorus",
    "SprottJerk",
    "SprottA",
    "SprottB",
    "SprottC",
    "SprottD",
    "SprottE",
    "SprottF",
    "SprottG",
    "SprottH",
    "SprottI",
    "SprottJ",
    "SprottK",
    "SprottL",
    "SprottM",
    "SprottN",
    "SprottO",
    "SprottP",
    "SprottQ",
    "SprottR",
    "SprottS",
    "Rucklidge",
    "Sakarya",
    "RayleighBenard",
    "Finance",
    "LuChenCheng",
    "LuChen",
    "QiChen",
    "ZhouChen",
    "BurkeShaw",
    "Chen",
    "ChenLee",
    "WangSun",
    "DequanLi",
    "NewtonLiepnik",
    "HyperRossler",
    "HyperQi",
    "Qi",
    "LorenzStenflo",
    "HyperYangChen",
    "HyperYan",
    "HyperXu",
    "HyperWang",
    "Hadley",
]
alphabetical_sort = np.argsort(systems_list)
systems_list = np.array(systems_list)[alphabetical_sort]

# attributes list
attributes = [
    "maximum_lyapunov_estimated",
    "lyapunov_spectrum_estimated",
    "embedding_dimension",
    "parameters",
    "dt",
    "hamiltonian",
    "period",
    "unbounded_indices",
]

# Get attributes
all_properties = dict()
for i, equation_name in enumerate(systems_list):
    eq = getattr(flows, equation_name)()
    attr_vals = [getattr(eq, item, None) for item in attributes]
    all_properties[equation_name] = dict(zip(attributes, attr_vals))

# Get training and testing trajectories for all the experimental systems
n = 1000  # Trajectories with 1000 points
pts_per_period = 100  # sample them with 100 points per period
n_trajectories = 5  # generate 5 trajectories starting from different initial conditions on the attractor
all_sols_train, all_t_train, all_sols_test, all_t_test = load_data(
    systems_list,
    all_properties,
    n=n,
    pts_per_period=pts_per_period,
    n_trajectories=n_trajectories,
)
t2 = time.time()
print("Took ", t2 - t1, " seconds to load the systems")
print("# of training trajectories = ", n_trajectories)
print("# of points per period = ", pts_per_period)
print("# of points per trajectory = ", n)

num_attractors = len(systems_list)

# Calculate some dynamical properties
lyap_list = []
dimension_list = []
param_list = []

# Calculate various definitions of scale separation
scale_list_avg = []
scale_list = []

for system in systems_list:
    lyap_list.append(all_properties[system]["maximum_lyapunov_estimated"])
    dimension_list.append(all_properties[system]["embedding_dimension"])
    param_list.append(all_properties[system]["parameters"])

    # Ratio of dominant (average) to smallest timescales
    scale_list_avg.append(
        all_properties[system]["period"] / all_properties[system]["dt"]
    )


# Get the true coefficients for each system
true_coefficients = make_dysts_true_coefficients(
    systems_list, all_sols_train, dimension_list, param_list
)

# Compute all the different nonlinear terms from the true coefficients
nonlinearities = nonlinear_terms_from_coefficients(true_coefficients)

# Compute various dynamical properties
count = 0
for i, system in enumerate(systems_list):
    sorted_spectrum = np.sort(
        (np.array(all_properties[system]["lyapunov_spectrum_estimated"]))
    )
    lambda_max = sorted_spectrum[-1]
    lambda_min = sorted_spectrum[0]
    if np.all(
        np.array(all_properties[system]["lyapunov_spectrum_estimated"][0:2]) > 0.0
    ):
        count += 1
    scale_list.append(lambda_max / lambda_min)

print(
    "Number of hyper-chaotic (at least two positive Lyapunov exponents) systems = ",
    count,
)

# Lastly, compute the syntactic complexity of all the dynamical system equations,
# as measured by the mean-equation description length (MEDL). More informationn
# can be found in the AI Feynman 2.0 paper
medl_list = compute_medl(systems_list, param_list)

# Shorten some of the dynamical system names to make nicer plots
systems_list_cleaned = []
for i, system in enumerate(systems_list):
    if system == "GuckenheimerHolmes":
        systems_list_cleaned.append("GuckenHolmes")
    elif system == "NuclearQuadrupole":
        systems_list_cleaned.append("NuclearQuad")
    elif system == "RabinovichFabrikant":
        systems_list_cleaned.append("RabFabrikant")
    elif system == "KawczynskiStrizhak":
        systems_list_cleaned.append("KawcStrizhak")
    elif system == "RikitakeDynamo":
        systems_list_cleaned.append("RikiDynamo")
    elif system == "ShimizuMorioka":
        systems_list_cleaned.append("ShMorioka")
    elif system == "HindmarshRose":
        systems_list_cleaned.append("Hindmarsh")
    elif system == "RayleighBenard":
        systems_list_cleaned.append("RayBenard")
    else:
        systems_list_cleaned.append(system)

# Make two summary plots of all the dynamical system properties
medl_levels = np.logspace(1, 3, 40)
lyap_levels = np.logspace(-2, 2, 40)
scale_levels = np.logspace(2, 5, 40)

plt.figure()
plt.hist(lyap_list, bins=lyap_levels, ec="k")
plt.hist(scale_list_avg, bins=scale_levels, ec="k")
plt.hist(medl_list, bins=medl_levels, ec="k")
plt.xlabel("")
plt.ylabel("# of systems")
plt.legend(
    ["Max Lyapunov exponent", "Scale separation", "Description length"], framealpha=1.0
)
plt.xscale("log")
plt.grid(True)
ax = plt.gca()
ax.set_axisbelow(True)

plt.figure(figsize=(30, 6))
plt.imshow(nonlinearities.T, aspect="equal", origin="lower", cmap="Oranges")
plt.xticks(np.arange(num_attractors), rotation="vertical", fontsize=20)
ax = plt.gca()
plt.xlim(-0.5, num_attractors - 0.5)
ax.set_xticklabels(np.array(systems_list_cleaned))
plt.yticks(np.arange(5), fontsize=22)
plt.colorbar(shrink=0.3, pad=0.01).ax.tick_params(labelsize=16)
plt.ylabel("Poly\n order", rotation=0, fontsize=22)
ax.yaxis.set_label_coords(-0.045, 0.3)

# if using the weak form, this makes the Pareto curve based on the "strong"
# or regular RMSE error instead of the RMSE error of the weak formulation.
strong_rmse = True

# List of algorithms and noise levels to sweep through
algorithms = ["STLSQ", "SR3", r"SR3 ($\nu = 0.1$)", "Lasso", "MIOSR"]
noise_levels = [0.0, 0.1, 1.0]
weak_form_flags = [False, True]

for weak_form in weak_form_flags:
    # if weak_form = True, need to reorder the coefficients because the
    # weak form uses a library with different term ordering
    true_coefficients = weakform_reorder_coefficients(
        systems_list, dimension_list, true_coefficients
    )

    for algorithm in algorithms:
        for noise_level in noise_levels:
            print("Algorithm: ", algorithm)
            print("Weak form: ", weak_form)
            print("Noise Level: ", noise_level, "%")
            t1 = time.time()

            # Note, defaults to using the AIC to decide the Pareto-optimal model
            n_models = 10
            (
                xdot_rmse_errors,
                xdot_coef_errors,
                AIC,
                x_dot_tests,
                x_dot_test_preds,
                predicted_coefficients,
                best_threshold_values,
                models,
                condition_numbers,
            ) = Pareto_scan_ensembling(
                systems_list,
                dimension_list,
                true_coefficients,
                all_sols_train,
                all_t_train,
                all_sols_test,
                all_t_test,
                normalize_columns=False,  # whether to normalize the SINDy matrix
                noise_level=noise_level,  # amount of noise to add to the training data
                n_models=n_models,  # number of models to train using EnsemblingOptimizer functionality
                n_subset=int(
                    0.5 * len(all_t_train["HyperBao"][0])
                ),  # subsample 50% of the training data for each model
                replace=False,  # Do the subsampling without replacement
                weak_form=weak_form,  # use the weak form or not
                algorithm=algorithm,  # optimization algorithm
                strong_rmse=strong_rmse,  # use the strong rmse with the weak form or not
            )
            t2 = time.time()
            print("Total time to compute = ", t2 - t1, " seconds")

            avg_rmse_error = np.zeros(num_attractors)
            std_rmse_error = np.zeros(num_attractors)
            coef_avg_error = np.zeros((num_attractors, n_models))
            best_thresholds = np.zeros(num_attractors)
            for i, attractor_name in enumerate(systems_list):
                for j in range(n_models):
                    coef_avg_error[i, j] = total_coefficient_error_normalized(
                        true_coefficients[i],
                        np.array(predicted_coefficients[attractor_name])[0, j, :, :],
                    )
                avg_rmse_error[i] = np.mean(
                    np.ravel(abs(np.array(xdot_rmse_errors[attractor_name])))
                )
                std_rmse_error[i] = np.std(
                    np.ravel(abs(np.array(xdot_rmse_errors[attractor_name])))
                )
                best_thresholds[i] = best_threshold_values[attractor_name][0]

            if weak_form:
                rmse_error_strong = {}

                for system in systems_list:
                    rmse_error_strong[system] = list()

                # Compute the "strong" RMSE errors using a non-weak model, since otherwise
                # the RMSE from the weak model is computed from the subdomains. This is
                # normally fine, but we would like to compare the same RMSE error between the
                # traditional and weak SINDy methods.
                models_strong = []
                for i, attractor_name in enumerate(systems_list):
                    x_test_list = []
                    t_test_list = []
                    for j in range(n_trajectories):
                        x_test_list.append(all_sols_test[attractor_name][j])
                        t_test_list.append(all_t_test[attractor_name][j])
                    poly_lib = ps.PolynomialLibrary(degree=4)
                    if dimension_list[i] == 3:
                        feature_names = ["x", "y", "z"]
                    else:
                        feature_names = ["x", "y", "z", "w"]
                    optimizer = ps.STLSQ(
                        threshold=0.0,
                        alpha=1e-5,
                        max_iter=100,
                        normalize_columns=False,
                        ridge_kw={"tol": 1e-10},
                    )
                    model = ps.SINDy(
                        feature_library=poly_lib,
                        optimizer=optimizer,
                        feature_names=feature_names,
                    )
                    model.fit(np.zeros(all_sols_train[attractor_name][0].shape))

                    rmses = []
                    for coef_temp in predicted_coefficients[attractor_name][0]:
                        if dimension_list[i] == 3:
                            model.optimizer.coef_ = coef_temp[:, :]
                        else:
                            model.optimizer.coef_ = coef_temp[:, :]
                        # model.print()
                        models_strong.append(model)
                        x_dot_test = model.differentiate(
                            x_test_list, t=t_test_list, multiple_trajectories=True
                        )
                        x_dot_test_pred = model.predict(
                            x_test_list, multiple_trajectories=True
                        )
                        rmses = rmses + [
                            normalized_RMSE(
                                np.array(x_dot_test).reshape(
                                    n_trajectories * n, dimension_list[i]
                                ),
                                np.array(x_dot_test_pred).reshape(
                                    n_trajectories * n, dimension_list[i]
                                ),
                            )
                        ]
                    rmse_error_strong[attractor_name].append(np.mean(rmses))

            np.savetxt(
                "data/avg_coef_errors_"
                + algorithm
                + "_noise{0:.2f}".format(noise_level)
                + "_weakform"
                + str(weak_form),
                np.mean(coef_avg_error, axis=-1),
            )
            np.savetxt(
                "data/std_coef_errors_"
                + algorithm
                + "_noise{0:.2f}".format(noise_level)
                + "_weakform"
                + str(weak_form),
                np.std(coef_avg_error, axis=-1),
            )
            np.savetxt(
                "data/avg_rmse_errors_"
                + algorithm
                + "_noise{0:.2f}".format(noise_level)
                + "_weakform"
                + str(weak_form),
                avg_rmse_error,
            )
            np.savetxt(
                "data/std_rmse_errors_"
                + algorithm
                + "_noise{0:.2f}".format(noise_level)
                + "_weakform"
                + str(weak_form),
                std_rmse_error,
            )
