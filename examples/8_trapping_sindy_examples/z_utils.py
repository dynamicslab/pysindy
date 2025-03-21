from itertools import combinations

import numpy as np


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
    print(V.shape)
    subC = C[gamma][:, gamma]
    subV = V[gamma]
    print(subV.shape)

    tempF = (
        -len(gamma) * 0.5 * np.log(2 * np.pi * rho**2)
        + 0.5 * np.linalg.slogdet(subC)[1]
        - 0.5 / rho**2 * (subV.T @ np.linalg.inv(subC) @ subV)
    )
    return tempF


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
        mean_coefs[gamma] = np.linalg.inv(C[lgamma][:, lgamma]) @ V[lgamma, dim]

    return gammas, Fs, mean_coefs


def get_idx_combinations(list_len, num_terms):
    """
    Returns all possible combinations of length num_terms from a list of length list_len
    """
    return [i for i in combinations(range(list_len), num_terms)]
