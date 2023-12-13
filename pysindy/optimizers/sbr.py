import jax.numpy as jnp
import numpy as np
import numpyro
from jax import random
from numpyro.distributions import Exponential
from numpyro.distributions import HalfCauchy
from numpyro.distributions import InverseGamma
from numpyro.distributions import Normal
from numpyro.infer import MCMC
from numpyro.infer import NUTS

from .base import BaseOptimizer


class SBR(BaseOptimizer):
    def __init__(
        self,
        tau_0=0.1,
        nu=4,
        s=2,
        lamb=1,
        normalize_columns=False,
        copy_X=True,
        **mcmc_kwargs,
    ):
        super().__init__(
            copy_X=copy_X,
            normalize_columns=normalize_columns,
        )
        # set the hyperparameters
        self.tau_0 = tau_0
        self.nu = nu
        self.s = s
        self.lamb = lamb

        # store the MCMC kwargs.
        self.mcmc_kwargs = mcmc_kwargs

    def _reduce(self, x, y):
        # set up a sparse regression and sample.
        regression = BayesianSparseRegression(self.tau_0, self.nu, self.s, self.lamb)
        self.mcmc = regression.fit(x, y, **self.mcmc_kwargs)

        # get the variable names and the mean values from the samples.
        beta_names = [
            varname
            for varname in self.mcmc.get_samples().keys()
            if varname.startswith("beta")
        ]
        beta = np.array(
            [self.mcmc.get_samples()[varname].mean().item() for varname in beta_names]
        ).reshape(y.shape[1], -1)

        # set the mean values as the coefficients.
        self.coef_ = beta


class BayesianSparseRegression:
    def __init__(self, tau_0=0.1, nu=4, s=2, lamb=1):
        # set hyperparameters
        self.tau_0 = tau_0
        self.nu = nu
        self.s = s
        self.lamb = lamb

    def _model(self, x, y):
        # get the dimensionality of the problem.
        n_features = x.shape[1]
        n_targets = y.shape[1]

        # sample the hyperparameters.
        tau, c_sq = sample_reg_horseshoe_hyper()

        # sample the parameters compute the predicted outputs.
        beta = []
        for i in range(n_targets):
            beta_i = []
            for j in range(n_features):
                beta_i.append(sample_reg_horseshoe(i, j, tau, c_sq))
            beta.append(beta_i)
        beta = jnp.array(beta)
        mu = jnp.dot(x, beta.T)

        # compute the likelihood.
        sigma = numpyro.sample("sigma", Exponential(1.0))
        numpyro.sample("obs", Normal(mu, sigma), obs=y)

    def fit(self, x, y, **kwargs):
        # set up a jax random key.
        seed = kwargs.pop("seed", 0)
        rng_key = random.PRNGKey(seed)
        rng_key, rng_key_ = random.split(rng_key)

        # run the MCMC
        kernel = NUTS(self._model)
        num_warmup = kwargs.pop("num_warmup", 2000)
        num_samples = kwargs.pop("num_samples", 5000)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, **kwargs)
        mcmc.run(rng_key_, x=x, y=y)

        # extract the MCMC samples and compute the UQ-SINDy parameters.
        return mcmc


def sample_reg_horseshoe_hyper(tau_0=0.1, nu=4, s=2):
    """
    For details on this prior, please refer to:
    Hirsh, S. M., Barajas-Solano, D. A., & Kutz, J. N. (2021).
    parsifying Priors for Bayesian Uncertainty Quantification in
    Model Discovery (arXiv:2107.02107). arXiv. http://arxiv.org/abs/2107.02107
    """
    tau = numpyro.sample("tau", HalfCauchy(tau_0))
    c_sq = numpyro.sample("c_sq", InverseGamma(nu / 2, nu / 2 * s**2))
    return tau, c_sq


def sample_reg_horseshoe(i, j, tau, c_sq):
    """
    For details on this prior, please refer to:
    Hirsh, S. M., Barajas-Solano, D. A., & Kutz, J. N. (2021).
    parsifying Priors for Bayesian Uncertainty Quantification in
    Model Discovery (arXiv:2107.02107). arXiv. http://arxiv.org/abs/2107.02107
    """
    lambda_i_j = numpyro.sample("lambda_{}_{}".format(i, j), HalfCauchy(1.0))
    lambda_i_j_squiggle = (
        jnp.sqrt(c_sq) * lambda_i_j / jnp.sqrt(c_sq + tau**2 * lambda_i_j**2)
    )
    beta_i_j = numpyro.sample(
        "beta_{}_{}".format(i, j),
        Normal(0.0, jnp.sqrt(lambda_i_j_squiggle**2 * tau**2)),
    )
    return beta_i_j
