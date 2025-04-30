from itertools import combinations

import numpy as np


def sparse_penalty_fe(Fs, lmbda, num_terms):
    """
    Applies a sparse penalty to the given feature matrix.

    Parameters:
    - Fs (numpy.ndarray): The feature matrix.
    - lmbda (float): The sparse penalty parameter.
    - num_terms (int): The number of terms in the penalty.

    Returns:
    - numpy.ndarray: The feature matrix with the sparse penalty applied.
    """
    return Fs + lmbda * num_terms


def free_energy(C, V, rho, gamma):
    """
    Calculate the free energy.

    Parameters:
    - C (numpy.ndarray): Covariance matrix.
    - V (numpy.ndarray): Data matrix.
    - rho (float): Resolution (distribution standard deviation) parameter.
    - gamma (list): List of indices of library terms to be included.

    Returns:
    - float: The calculated free energy.
    """
    subC = C[gamma][:, gamma]
    subV = V[gamma].reshape(-1, 1)

    tempF = (
        -len(gamma) * 0.5 * np.log(2 * np.pi * rho**2)
        + 0.5 * np.linalg.slogdet(subC)[1]
        - 0.5 / rho**2 * (subV.T @ np.linalg.inv(subC) @ subV)
    )
    return tempF.reshape(-1)


def free_energy_coefs(C, V, rho, num_terms, num_feats, dim):
    """
    Calculate the free energy coefficients for a given set of inputs.

    Parameters:
    - C (numpy.ndarray): Covariance matrix.
    - V (numpy.ndarray): Data matrix.
    - rho (float): The resolution parameter.
    - num_terms (int): The number of terms.
    - num_feats (int): The number of features.
    - dim (int): The variable index to consider.

    Returns:
    - gammas (dict): A dictionary containing the index combinations.
    - Fs (dict): A dictionary containing the free energy values for each index combination.
    - mean_coefs (dict): A dictionary containing the mean coefficients for each index combination.
    """
    gammas = get_idx_combinations(num_feats, num_terms)
    Fs = {key: None for key in gammas}
    mean_coefs = {key: None for key in gammas}
    for i, gamma in enumerate(gammas):
        lgamma = list(gamma)
        Fs[gamma] = free_energy(C, V[:, dim], rho, lgamma)
        mean_coefs[gamma] = np.linalg.inv(C[lgamma][:, lgamma]) @ V[
            lgamma, dim
        ].reshape(-1, 1)

    return gammas, Fs, mean_coefs


def get_idx_combinations(list_len, num_terms):
    """
    Returns all possible combinations of length num_terms from a list of length list_len
    """
    return [i for i in combinations(range(list_len), num_terms)]


def lowest_n_combs(num_lowest, Fs, feat_names):
    """
    Calculate the lowest energies of given number of combinations.

    Parameters:


    Returns:

    """
    myList = sorted(Fs.items())
    x, y = zip(*myList)
    y = np.array(y).reshape(-1)
    lowest_n = np.argsort(y)[:num_lowest]

    Fs_n = []
    combs_n = []
    for i in range(num_lowest):
        a = list(x[lowest_n[i]])
        c_temp = []
        for c in a:
            c_temp.append(feat_names[c])
        combs_n.append(" ".join(c_temp))
        Fs_n.append(lowest_n[i])

    lowest_n = np.argsort(y)[:num_lowest]
    comb_n = []
    Fs_n = []
    for i in range(len(lowest_n)):
        comb_n.append(comb_n[lowest_n[i]])
        Fs_n.append(lowest_n[i])

    return comb_n, Fs_n


def dict_to_lists(Fs, mean_coefs, gammas):
    Fs_list = []
    coef_list = []
    for g in gammas:
        Fs_list.append(Fs[g])
        coef_list.append(mean_coefs[g])
    return Fs_list, coef_list
