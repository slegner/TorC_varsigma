"""TorC — torsion-condensate cosmology against DESI DR2 BAO, in JAX.

Canonical anchoring convention: noH0 (a0 = 1, today := a = 1); see
torc.theory. The H0-renormalised convention is archived in legacy/H0_renorm.
"""
from .theory import (rhopsol_jax, omega_r, OMEGA_R_DEFAULT,
                     N_EARLY, N_LATE, Z_SPLIT)
from .likelihood import loglikelihood, model_vector
from .priors import (PRIOR_BOUNDS, logprior_fn, sample_prior,
                     loglikelihood_bounded)

__all__ = ["rhopsol_jax", "omega_r", "OMEGA_R_DEFAULT", "N_EARLY", "N_LATE",
           "Z_SPLIT", "loglikelihood", "model_vector", "PRIOR_BOUNDS",
           "logprior_fn", "sample_prior", "loglikelihood_bounded"]
