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
    """
    Sparse Bayesian Regression (SBR) optimizer. This uses the regularised
    horseshoe prior over the SINDy coefficients to achieve sparsification.

    The horseshoe prior contains a "spike" of nonzero probability at the
    origin, and a "slab" of distribution in cases where a coefficient is nonzero.

    The SINDy coefficients are set as the posterior means of the MCMC NUTS samples.
    Additional statistics can be computed from the MCMC samples stored in
    the mcmc_ attribute using e.g. ArviZ.

    See the following reference for more details:

        Hirsh, S. M., Barajas-Solano, D. A., & Kutz, J. N. (2021).
        parsifying Priors for Bayesian Uncertainty Quantification in
        Model Discovery (arXiv:2107.02107). arXiv. http://arxiv.org/abs/2107.02107

    Parameters
    ----------
    sparsity_coef_tau0 : float, optional (default 0.1)
        Sparsity coefficient for regularised horseshoe hyper-prior. Lower
        value increases the sparsity of the SINDy coefficients.

    slab_shape_nu : float, optional (default 4)
        Controls spread of slab.  For values less than 4,
        the kurtosis of of nonzero coefficients is undefined.  As value increases
        past 4, For higher values, the
        variance and kurtosis approach :math:`s` and :math:`s^2`, respectively

    slab_shape_s : float, optional (default 2)
        Controls spread of slab.  Higher values lead to more spread
        out nonzero coefficients.

    noise_hyper_lambda : float, optional (default 1)
        Rate hyperparameter for the exponential prior distribution over
        the noise standard deviation.

    normalize_columns : boolean, optional (default False)
        Normalize the columns of x (the SINDy library terms) before regression
        by dividing by the L2-norm. Note that the 'normalize' option in sklearn
        is deprecated in sklearn versions >= 1.0 and will be removed.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    **mcmc_kwargs
        Instructions for MCMC sampling.
        Keyword arguments are passed to numpyro.infer.MCMC

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Posterior means of the SINDy coefficients.

    mcmc : numpyro.infer.MCMC
        Complete traces of the posterior samples.
    """

    def __init__(
        self,
        sparsity_coef_tau0=0.1,
        slab_shape_nu=4,
        slab_shape_s=2,
        noise_hyper_lambda=1,
        num_warmup=1000,
        num_samples=5000,
        mcmc_kwargs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # set the hyperparameters
        self.sparsity_coef_tau0 = sparsity_coef_tau0
        self.slab_shape_nu = slab_shape_nu
        self.slab_shape_s = slab_shape_s
        self.noise_hyper_lambda = noise_hyper_lambda

        # set MCMC sampling parameters.
        self.num_warmup = num_warmup
        self.num_samples = num_samples

        # set the MCMC kwargs.
        if mcmc_kwargs is not None:
            self.mcmc_kwargs = mcmc_kwargs
        else:
            self.mcmc_kwargs = {}

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
        beta = _sample_reg_horseshoe(tau, c_sq, (n_targets, n_features))
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
        mcmc = MCMC(
            kernel, num_warmup=self.num_warmup, num_samples=self.num_samples, **kwargs
        )
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


def _sample_reg_horseshoe(tau, c_sq, shape):
    lamb = numpyro.sample("lambda", HalfCauchy(1.0), sample_shape=shape)
    lamb_squiggle = jnp.sqrt(c_sq) * lamb / jnp.sqrt(c_sq + tau**2 * lamb**2)
    beta = numpyro.sample(
        "beta",
        Normal(jnp.zeros_like(lamb_squiggle), jnp.sqrt(lamb_squiggle**2 * tau**2)),
    )
    return beta
