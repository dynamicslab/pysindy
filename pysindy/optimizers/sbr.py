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
        sparsity_coef_tau0=0.1,
        slab_shape_nu=4,
        slab_shape_s=2,
        noise_hyper_lambda=1,
        normalize_columns=False,
        copy_X=True,
        **mcmc_kwargs,
    ):
        super().__init__(
            copy_X=copy_X,
            normalize_columns=normalize_columns,
        )
        # set the hyperparameters
        self.sparsity_coef_tau0 = sparsity_coef_tau0
        self.slab_shape_nu = slab_shape_nu
        self.slab_shape_s = slab_shape_s
        self.noise_hyper_lambda = noise_hyper_lambda

        # store the MCMC kwargs.
        self.mcmc_kwargs = mcmc_kwargs

    def _reduce(self, x, y):
        # set up a sparse regression and sample.
        self.mcmc = self._run_mcmc(x, y, **self.mcmc_kwargs)

        # set the mean values as the coefficients.
        self.coef_ = np.array(self.mcmc.get_samples()["beta"].mean(axis=0))

    def _numpyro_model(self, x, y):
        # get the dimensionality of the problem.
        n_features = x.shape[1]
        n_targets = y.shape[1]

        # sample the hyperparameters.
        tau, c_sq = sample_reg_horseshoe_hyper(
            self.sparsity_coef_tau0, self.slab_shape_nu, self.slab_shape_s
        )

        # sample the parameters compute the predicted outputs.
        beta = sample_reg_horseshoe(tau, c_sq, (n_targets, n_features))
        mu = jnp.dot(x, beta.T)

        # compute the likelihood.
        sigma = numpyro.sample("sigma", Exponential(self.noise_hyper_lambda))
        numpyro.sample("obs", Normal(mu, sigma), obs=y)

    def _run_mcmc(self, x, y, **kwargs):
        # set up a jax random key.
        seed = kwargs.pop("seed", 0)
        rng_key = random.PRNGKey(seed)
        rng_key, rng_key_ = random.split(rng_key)

        # run the MCMC
        kernel = NUTS(self._numpyro_model)
        num_warmup = kwargs.pop("num_warmup", 2000)
        num_samples = kwargs.pop("num_samples", 5000)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, **kwargs)
        mcmc.run(rng_key_, x=x, y=y)

        # extract the MCMC samples and compute the UQ-SINDy parameters.
        return mcmc


def sample_reg_horseshoe_hyper(tau0=0.1, nu=4, s=2):
    """
    For details on this prior, please refer to:
    Hirsh, S. M., Barajas-Solano, D. A., & Kutz, J. N. (2021).
    parsifying Priors for Bayesian Uncertainty Quantification in
    Model Discovery (arXiv:2107.02107). arXiv. http://arxiv.org/abs/2107.02107
    """
    tau = numpyro.sample("tau", HalfCauchy(tau0))
    c_sq = numpyro.sample("c_sq", InverseGamma(nu / 2, nu / 2 * s**2))
    return tau, c_sq


def sample_reg_horseshoe(tau, c_sq, shape):
    """
    For details on this prior, please refer to:
    Hirsh, S. M., Barajas-Solano, D. A., & Kutz, J. N. (2021).
    parsifying Priors for Bayesian Uncertainty Quantification in
    Model Discovery (arXiv:2107.02107). arXiv. http://arxiv.org/abs/2107.02107
    """
    lamb = numpyro.sample("lambda", HalfCauchy(1.0), sample_shape=shape)
    lamb_squiggle = jnp.sqrt(c_sq) * lamb / jnp.sqrt(c_sq + tau**2 * lamb**2)
    beta = numpyro.sample(
        "beta",
        Normal(jnp.zeros_like(lamb_squiggle), jnp.sqrt(lamb_squiggle**2 * tau**2)),
    )
    return beta
